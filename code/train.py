# -- coding: utf-8 --
"""
Training and evaluation pipeline for the MASP framework.

Supports:
  - Joint pre-training of HeterBERT (MLM + contrastive objectives)
  - Fine-tuning of MotivationAwarePreference with BPR loss
  - Evaluation with HR@K and NDCG@K metrics
  - Model checkpointing and incremental (warm-start) updates
"""

import argparse
import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from data_loader import BaseSessionDataset, build_dataloader
from HeterBERT import HeterBERT, HeterBERTPreTrainer
from metrics import compute_metrics
from motivation_prediction import MotivationAwarePreference

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    extra: Optional[Dict] = None,
) -> None:
    """Save a model checkpoint to *path*."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
    logger.info("Checkpoint saved to %s (epoch %d, best_metric=%.4f)", path, epoch, best_metric)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float]:
    """
    Load a checkpoint from *path*.

    Returns (epoch, best_metric) from the saved state so that training can
    resume seamlessly (incremental update).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch = state.get("epoch", 0)
    best_metric = state.get("best_metric", 0.0)
    logger.info("Checkpoint loaded from %s (epoch %d, best_metric=%.4f)", path, epoch, best_metric)
    return epoch, best_metric


# ---------------------------------------------------------------------------
# HeterBERT pre-training
# ---------------------------------------------------------------------------

def pretrain_heter_bert(
    model: HeterBERTPreTrainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    save_dir: str = "./checkpoints",
    checkpoint_path: Optional[str] = None,
) -> HeterBERT:
    """
    Pre-train HeterBERT with MLM and (optionally) contrastive objectives.

    The DataLoader is expected to yield batches of dicts with at minimum:
        input_ids:      (B, S) — token IDs
        type_ids:       (B, S) — type IDs per token
        attention_mask: (B, S) — optional padding mask

    Args:
        model:            HeterBERTPreTrainer instance
        dataloader:       training DataLoader
        optimizer:        optimiser (e.g. AdamW)
        device:           compute device
        epochs:           number of pre-training epochs
        save_dir:         directory to write checkpoints
        checkpoint_path:  if given, resume from this checkpoint before training
    Returns:
        Trained HeterBERT backbone extracted from the pre-trainer.
    """
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = 0
    best_loss = float("inf")

    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, device)

    model.to(device)
    model.train()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            type_ids = batch["type_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            special_tokens_mask = batch.get("special_tokens_mask")
            if special_tokens_mask is not None:
                special_tokens_mask = special_tokens_mask.to(device)

            # Optional second view for contrastive learning
            input_ids_2 = batch.get("input_ids_2")
            type_ids_2 = batch.get("type_ids_2")
            attention_mask_2 = batch.get("attention_mask_2")
            if input_ids_2 is not None:
                input_ids_2 = input_ids_2.to(device)
                type_ids_2 = type_ids_2.to(device)
                if attention_mask_2 is not None:
                    attention_mask_2 = attention_mask_2.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                type_ids=type_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                input_ids_2=input_ids_2,
                type_ids_2=type_ids_2,
                attention_mask_2=attention_mask_2,
            )
            loss = outputs["total_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(dataloader), 1)
        elapsed = time.time() - t0
        logger.info(
            "Pre-train epoch %d/%d | loss=%.4f | %.1fs",
            epoch + 1,
            epochs,
            avg_loss,
            elapsed,
        )

        # Checkpoint on improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = os.path.join(save_dir, "heter_bert_best.pt")
            save_checkpoint(ckpt, model, optimizer, epoch, best_loss)

    return model.bert


# ---------------------------------------------------------------------------
# Preference model fine-tuning
# ---------------------------------------------------------------------------

def train_preference_model(
    model: MotivationAwarePreference,
    train_loader: DataLoader,
    val_dataset: BaseSessionDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50,
    k_values: Tuple[int, ...] = (5, 10, 20),
    save_dir: str = "./checkpoints",
    checkpoint_path: Optional[str] = None,
    early_stop_patience: int = 5,
) -> MotivationAwarePreference:
    """
    Fine-tune MotivationAwarePreference using BPR loss.

    Args:
        model:                MotivationAwarePreference
        train_loader:         DataLoader yielding (group_id, pos_feat, neg_feat, …)
        val_dataset:          validation dataset for HR/NDCG evaluation
        optimizer:            optimiser
        device:               compute device
        epochs:               maximum number of fine-tuning epochs
        k_values:             cut-offs for HR@K / NDCG@K
        save_dir:             checkpoint directory
        checkpoint_path:      optional warm-start checkpoint
        early_stop_patience:  stop training after this many epochs without improvement
    Returns:
        Fine-tuned model.
    """
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = 0
    best_hr = 0.0
    patience_counter = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, best_hr = load_checkpoint(checkpoint_path, model, optimizer, device)

    model.to(device)
    item_features = val_dataset.item_features.to(device)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            pos_feat = batch["pos_item_feat"].to(device)  # (B, item_dim)
            neg_feat = batch["neg_item_feat"].to(device)  # (B, item_dim)

            # Build a minimal interaction history tensor (single positive item)
            hist = pos_feat.unsqueeze(1)  # (B, 1, item_dim)

            optimizer.zero_grad()
            outputs = model(
                item_embeddings_hist=hist,
                candidate_item_embeddings=item_features,
                top_k=10,
            )
            user_profile = outputs["user_profile"]  # (B, d_model)

            # BPR loss requires d_model-dimensioned item embeddings; project features
            loss = model.compute_bpr_loss(user_profile, pos_feat, neg_feat)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        # Validation
        metrics = evaluate_preference_model(model, val_dataset, device, k_values=list(k_values))
        hr_main = metrics.get(f"HR@{k_values[0]}", 0.0)

        logger.info(
            "Epoch %d/%d | loss=%.4f | %s | %.1fs",
            epoch + 1,
            epochs,
            avg_loss,
            "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
            elapsed,
        )

        if hr_main > best_hr:
            best_hr = hr_main
            patience_counter = 0
            ckpt = os.path.join(save_dir, "preference_best.pt")
            save_checkpoint(ckpt, model, optimizer, epoch, best_hr)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", epoch + 1)
                break

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_preference_model(
    model: MotivationAwarePreference,
    dataset: BaseSessionDataset,
    device: torch.device,
    k_values: Optional[list] = None,
) -> Dict[str, float]:
    """
    Evaluate MotivationAwarePreference on a held-out dataset split.

    For each (group, positive_items, all_items) tuple the model scores all
    items, ranks them, and computes HR@K / NDCG@K.

    Args:
        model:    trained MotivationAwarePreference
        dataset:  dataset object (val or test split)
        device:   compute device
        k_values: list of K values; defaults to [5, 10, 20]
    Returns:
        dict of metric name → value
    """
    if k_values is None:
        k_values = [5, 10, 20]

    model.eval()
    item_features = dataset.item_features.to(device)
    eval_data = dataset.get_eval_data()

    all_ranked: list = []
    all_gt: list = []

    with torch.no_grad():
        for group_id, pos_indices, _ in eval_data:
            # Build a dummy history from the positive items (batch_size=1)
            hist_feats = item_features[pos_indices].unsqueeze(0)  # (1, len, D)
            outputs = model(
                item_embeddings_hist=hist_feats,
                candidate_item_embeddings=item_features,
                top_k=max(k_values),
            )
            ranked_ids = outputs["candidate_ids"][0].cpu().tolist()
            all_ranked.append(ranked_ids)
            all_gt.append(pos_indices)

    return compute_metrics(all_ranked, all_gt, k_values)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MASP training & evaluation")
    p.add_argument("--dataset", choices=["yelp", "yelp_l", "douban"], default="yelp")
    p.add_argument("--data_root", type=str, required=True, help="Path to dataset directory")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--mode", choices=["pretrain", "finetune", "eval"], default="finetune")
    # Model hyper-parameters
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--num_types", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training hyper-parameters
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    train_loader, train_dataset = build_dataloader(
        args.dataset, args.data_root, split="train",
        batch_size=args.batch_size, feature_dim=args.d_model, seed=args.seed,
    )
    _, val_dataset = build_dataloader(
        args.dataset, args.data_root, split="val",
        batch_size=args.batch_size, feature_dim=args.d_model, seed=args.seed,
    )
    _, test_dataset = build_dataloader(
        args.dataset, args.data_root, split="test",
        batch_size=args.batch_size, feature_dim=args.d_model, seed=args.seed,
    )

    feature_dim = train_dataset.feature_dim

    # ------------------------------------------------------------------
    # Pre-training mode
    # ------------------------------------------------------------------
    if args.mode == "pretrain":
        vocab_size = train_dataset.num_items + 10  # +10 for special tokens
        pre_trainer = HeterBERTPreTrainer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_types=args.num_types,
            dropout=args.dropout,
        )
        optimizer = torch.optim.AdamW(
            pre_trainer.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        pretrain_heter_bert(
            pre_trainer,
            train_loader,
            optimizer,
            device,
            epochs=args.epochs,
            save_dir=args.save_dir,
            checkpoint_path=args.resume,
        )
        return

    # ------------------------------------------------------------------
    # Fine-tuning / evaluation mode
    # ------------------------------------------------------------------
    model = MotivationAwarePreference(
        item_dim=feature_dim,
        d_model=args.d_model,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.mode == "finetune":
        train_preference_model(
            model,
            train_loader,
            val_dataset,
            optimizer,
            device,
            epochs=args.epochs,
            save_dir=args.save_dir,
            checkpoint_path=args.resume,
            early_stop_patience=args.patience,
        )

    # Always run final evaluation on test set
    if args.resume:
        try:
            load_checkpoint(args.resume, model, device=device)
        except FileNotFoundError:
            best_ckpt = os.path.join(args.save_dir, "preference_best.pt")
            if os.path.exists(best_ckpt):
                load_checkpoint(best_ckpt, model, device=device)

    test_metrics = evaluate_preference_model(model, test_dataset, device)
    logger.info("=== Test Results ===")
    for name, value in test_metrics.items():
        logger.info("  %s = %.4f", name, value)


if __name__ == "__main__":
    main()
