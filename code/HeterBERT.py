# -- coding: utf-8 --
"""
HeterBERT: BERT-based model for heterogeneous social platform data.

Implements the following components from the MASP paper:
  - Recurrence Positional Encoding (handles variable-length attributes)
  - Type Encoding (captures attribute type information)
  - Type Attention Layer (double threshold mechanism)
  - C-Merge Layer (type-attribute correlation)
  - MLM and contrastive learning pre-training tasks
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Index used to mark masked positions during MLM pre-training.
# Index 0 is reserved as the padding token; 1 is the [MASK] token.
MASK_TOKEN_ID: int = 1


class RecurrentPositionalEncoding(nn.Module):
    """
    Recurrence-based positional encoding for variable-length attribute sequences.

    Uses a GRU to produce position-aware incremental encodings, which are added
    to the input as a residual, allowing the model to handle sequences of
    varying lengths without a fixed positional table.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model) position-enriched representations
        """
        pos_enc, _ = self.gru(x)
        pos_enc = self.proj(pos_enc)
        return self.dropout(x + pos_enc)


class TypeEncoding(nn.Module):
    """
    Type encoding model to capture type information for heterogeneous attributes.

    Each token (attribute value) is assigned a type ID (e.g., text, numeric,
    categorical, location).  The learned type embeddings are added to the token
    representations, giving the model awareness of attribute semantics.
    """

    def __init__(self, num_types: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Index 0 is reserved as padding / unknown type
        self.type_embedding = nn.Embedding(num_types + 1, d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.normal_(self.type_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        (batch_size, seq_len, d_model) — token representations
            type_ids: (batch_size, seq_len) — integer type ID per token
        Returns:
            (batch_size, seq_len, d_model) — type-enriched representations
        """
        type_emb = self.type_embedding(type_ids)
        return self.dropout(self.norm(x + type_emb))


class TypeAttentionLayer(nn.Module):
    """
    Type attention layer with a double threshold mechanism.

    After computing standard scaled-dot-product attention weights, values outside
    the [alpha, beta] range are zeroed out and the remaining weights are
    re-normalized.  This filters uninformative or over-dominant attention
    patterns and makes the model focus on relevant type interactions.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        alpha: float = 0.1,
        beta: float = 0.9,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.alpha = alpha
        self.beta = beta
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:              (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len) — 1 for valid, 0 for padding
        Returns:
            output:          (batch_size, seq_len, d_model)
            attn_weights:    (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = _split_heads(self.q_proj(x))
        K = _split_heads(self.k_proj(x))
        V = _split_heads(self.v_proj(x))

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply padding mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Double threshold: keep attention values in [alpha, beta], zero out the rest
        threshold_mask = (attn_weights >= self.alpha) & (attn_weights <= self.beta)
        attn_weights = attn_weights * threshold_mask.float()

        # Re-normalize surviving weights
        attn_sum = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn_weights = attn_weights / attn_sum
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        return output, attn_weights


class CMergeLayer(nn.Module):
    """
    C-Merge layer for type-attribute correlation.

    A gated fusion mechanism that correlates type-level representations with
    attribute-level representations, producing a joint embedding that captures
    both semantic content (attribute) and structural role (type).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.W_gate = nn.Linear(d_model * 2, d_model)
        self.W_attr = nn.Linear(d_model, d_model)
        self.W_type = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, attr_repr: torch.Tensor, type_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attr_repr: (batch_size, seq_len, d_model) — attribute representations
            type_repr: (batch_size, seq_len, d_model) — type representations
        Returns:
            (batch_size, seq_len, d_model) — merged representations
        """
        gate = torch.sigmoid(self.W_gate(torch.cat([attr_repr, type_repr], dim=-1)))
        merged = gate * self.W_type(type_repr) + (1.0 - gate) * self.W_attr(attr_repr)
        output = self.act(self.W_out(merged))
        # Residual connection from attr_repr
        return self.norm(self.dropout(output) + attr_repr)


class HeterBERTLayer(nn.Module):
    """
    Single transformer layer of HeterBERT.

    Stacks TypeEncoding → TypeAttention (with double threshold) → C-Merge → FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_types: int,
        alpha: float = 0.1,
        beta: float = 0.9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.type_encoding = TypeEncoding(num_types, d_model, dropout)
        self.type_attention = TypeAttentionLayer(d_model, num_heads, alpha, beta, dropout)
        self.c_merge = CMergeLayer(d_model, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Type encoding to get type-enriched input
        type_repr = self.type_encoding(x, type_ids)

        # 2. Type attention with double threshold
        attn_out, _ = self.type_attention(type_repr, attention_mask)
        x = self.norm1(x + attn_out)

        # 3. C-Merge: correlate attribute and type representations
        merged = self.c_merge(x, type_repr)

        # 4. Feed-forward network with residual
        ffn_out = self.ffn(merged)
        x = self.norm2(merged + ffn_out)
        return x


class HeterBERT(nn.Module):
    """
    HeterBERT: BERT-variant for heterogeneous social platform data.

    Key differences from standard BERT:
    - Recurrence positional encoding (instead of sinusoidal / learned fixed PE)
    - Per-layer TypeEncoding + TypeAttention + C-Merge instead of plain self-attention
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 512,
        num_types: int = 10,
        max_seq_len: int = 512,
        alpha: float = 0.1,
        beta: float = 0.9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = RecurrentPositionalEncoding(d_model, dropout)

        self.layers = nn.ModuleList(
            [
                HeterBERTLayer(d_model, num_heads, d_ff, num_types, alpha, beta, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch_size, seq_len) — token IDs
            type_ids:       (batch_size, seq_len) — type IDs per token
            attention_mask: (batch_size, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            (batch_size, seq_len, d_model) — contextualized representations
        """
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, type_ids, attention_mask)

        return self.norm(x)

    def get_cls_representation(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the [CLS] token representation (position 0) for retrieval/classification."""
        return self.forward(input_ids, type_ids, attention_mask)[:, 0, :]


class MLMHead(nn.Module):
    """Masked Language Model prediction head for HeterBERT pre-training."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, vocab_size) — logits over vocabulary
        """
        x = self.act(self.dense(hidden_states))
        x = self.norm(x)
        return self.decoder(x)


class ContrastiveHead(nn.Module):
    """
    Contrastive learning head using InfoNCE loss.

    Projects representations to a lower-dimensional unit hypersphere and
    computes a symmetric cross-entropy loss between two augmented views of
    the same instance.
    """

    def __init__(self, d_model: int, proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (batch_size, d_model) — representation of view 1
            z2: (batch_size, d_model) — representation of view 2
        Returns:
            scalar InfoNCE loss
        """
        z1 = F.normalize(self.proj(z1), dim=-1)
        z2 = F.normalize(self.proj(z2), dim=-1)
        batch_size = z1.shape[0]
        sim = torch.mm(z1, z2.T) / self.temperature  # (B, B)
        labels = torch.arange(batch_size, device=z1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss


class HeterBERTPreTrainer(nn.Module):
    """
    HeterBERT wrapper that combines MLM and contrastive learning pre-training.

    Pre-training procedure:
    1. Randomly mask tokens and predict them (MLM objective).
    2. Generate two views of each sequence and maximise their agreement
       in a projected space (contrastive objective).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 512,
        num_types: int = 10,
        max_seq_len: int = 512,
        alpha: float = 0.1,
        beta: float = 0.9,
        dropout: float = 0.1,
        proj_dim: int = 128,
        mlm_probability: float = 0.15,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.mlm_probability = mlm_probability

        self.bert = HeterBERT(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            num_types=num_types,
            max_seq_len=max_seq_len,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
        )
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.contrastive_head = ContrastiveHead(d_model, proj_dim, temperature)

    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a masked copy of input_ids for the MLM objective.

        Strategy per masked position:
          - 80 % → replaced with [MASK] token (index 1)
          - 10 % → replaced with a random token
          - 10 % → kept unchanged

        Returns:
            masked_input_ids: input with masked positions replaced
            labels:           original tokens at masked positions; -100 elsewhere
        """
        labels = input_ids.clone()
        prob_matrix = torch.full(input_ids.shape, self.mlm_probability, device=input_ids.device)

        if special_tokens_mask is not None:
            prob_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)

        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100

        # 80 % → [MASK]
        replace_mask = torch.bernoulli(
            torch.full(input_ids.shape, 0.8, device=input_ids.device)
        ).bool() & masked_indices
        input_ids[replace_mask] = MASK_TOKEN_ID

        # 10 % → random token (50 % of the remaining masked)
        random_mask = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5, device=input_ids.device)).bool()
            & masked_indices
            & ~replace_mask
        )
        random_tokens = torch.randint(
            self.bert.vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device
        )
        input_ids[random_mask] = random_tokens[random_mask]

        return input_ids, labels

    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        input_ids_2: Optional[torch.Tensor] = None,
        type_ids_2: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids / type_ids / attention_mask: primary view
            special_tokens_mask: positions that must not be masked (CLS, SEP, PAD)
            input_ids_2 / type_ids_2 / attention_mask_2: optional second view for
                contrastive loss
        Returns:
            dict with keys 'mlm_loss', optional 'contrastive_loss', and 'total_loss'
        """
        results: Dict[str, torch.Tensor] = {}

        # --- MLM ---
        masked_input_ids, mlm_labels = self.mask_tokens(input_ids.clone(), special_tokens_mask)
        hidden_states = self.bert(masked_input_ids, type_ids, attention_mask)
        logits = self.mlm_head(hidden_states)
        mlm_loss = F.cross_entropy(
            logits.view(-1, self.bert.vocab_size),
            mlm_labels.view(-1),
            ignore_index=-100,
        )
        results["mlm_loss"] = mlm_loss

        # --- Contrastive ---
        if input_ids_2 is not None:
            z1 = self.bert.get_cls_representation(input_ids, type_ids, attention_mask)
            z2 = self.bert.get_cls_representation(input_ids_2, type_ids_2, attention_mask_2)
            contrastive_loss = self.contrastive_head(z1, z2)
            results["contrastive_loss"] = contrastive_loss
            results["total_loss"] = mlm_loss + contrastive_loss
        else:
            results["total_loss"] = mlm_loss

        return results
