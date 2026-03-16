# -- coding: utf-8 --
"""
Motivation-Aware Preference Prediction module.

Implements:
  - Activity Category Arrangement (ACA) algorithm
  - M-Merge layers for dynamic user/item representation fusion
  - UserProfileConstructor: builds a user profile from interaction history
  - CandidateItemGenerator: generates a ranked candidate item list
  - MotivationAwarePreference: top-level prediction module
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activity Category Arrangement (ACA) algorithm
# ---------------------------------------------------------------------------

def aca_algorithm(
    user_history: List[Dict],
    category_embeddings: np.ndarray,
    num_categories: int,
    decay: float = 0.9,
) -> np.ndarray:
    """
    Activity Category Arrangement (ACA) algorithm.

    Ranks activity categories for a user based on interaction frequency and
    the similarity between the user's aggregate preference vector and each
    category embedding.  A recency decay factor down-weights older interactions.

    Args:
        user_history:        list of interaction dicts, each containing at least
                             ``{'category': int, 'timestamp': float}``.
                             Items should be ordered oldest-first.
        category_embeddings: (num_categories, emb_dim) float array of category
                             prototype embeddings.
        num_categories:      total number of activity categories.
        decay:               exponential decay factor for recency weighting.

    Returns:
        sorted_category_ids: (num_categories,) array of category indices sorted
                             by descending relevance score.
    """
    if len(user_history) == 0:
        return np.arange(num_categories)

    category_counts = np.zeros(num_categories, dtype=np.float64)
    n = len(user_history)
    for idx, item in enumerate(user_history):
        cat = int(item.get("category", 0))
        if 0 <= cat < num_categories:
            # More recent interactions get higher weight
            weight = decay ** (n - 1 - idx)
            category_counts[cat] += weight

    # Normalise to preference distribution
    total = category_counts.sum()
    category_prefs = category_counts / total if total > 0 else np.ones(num_categories) / num_categories

    # Aggregate preference vector as a weighted sum of category embeddings
    user_pref_emb = category_prefs @ category_embeddings  # (emb_dim,)

    # Cosine similarity between user preference and each category
    norms = np.linalg.norm(category_embeddings, axis=1, keepdims=True) + 1e-9
    norm_cats = category_embeddings / norms
    user_norm = user_pref_emb / (np.linalg.norm(user_pref_emb) + 1e-9)
    similarities = norm_cats @ user_norm  # (num_categories,)

    # Final score combines preference distribution and embedding similarity
    scores = 0.5 * category_prefs + 0.5 * (similarities + 1.0) / 2.0
    return np.argsort(-scores)


# ---------------------------------------------------------------------------
# M-Merge Layer
# ---------------------------------------------------------------------------

class MMergeLayer(nn.Module):
    """
    M-Merge layer for dynamic user/item representation fusion.

    A gated fusion module that merges a user profile representation with an
    item representation, conditioned on a (optional) motivation vector.  The
    output can be used directly as a preference score feature.
    """

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.W_user = nn.Linear(user_dim, d_model)
        self.W_item = nn.Linear(item_dim, d_model)
        self.W_gate = nn.Linear(d_model * 2, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        motivation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            user_repr:  (..., user_dim) — user profile embedding
            item_repr:  (..., item_dim) — item feature embedding
            motivation: (...,) or (..., 1) — optional scalar motivation weight
                        in [0, 1] that modulates the fusion gate
        Returns:
            (..., d_model) — fused representation
        """
        u = self.W_user(user_repr)
        v = self.W_item(item_repr)

        gate = torch.sigmoid(self.W_gate(torch.cat([u, v], dim=-1)))

        if motivation is not None:
            m = motivation.unsqueeze(-1) if motivation.dim() == u.dim() - 1 else motivation
            gate = gate * m

        merged = gate * u + (1.0 - gate) * v
        output = self.act(self.W_out(merged))
        return self.norm(self.dropout(output) + u)


# ---------------------------------------------------------------------------
# User Profile Constructor
# ---------------------------------------------------------------------------

class UserProfileConstructor(nn.Module):
    """
    Builds a dynamic user profile from a sequence of past interactions.

    Encodes interaction history (items + their attribute embeddings) with an
    attention pooling mechanism that weighs recent and high-rated items more
    heavily.
    """

    def __init__(self, item_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.item_proj = nn.Linear(item_dim, d_model)
        # Attention score for each interaction
        self.attn_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        interaction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            item_embeddings:  (batch_size, history_len, item_dim) — embedded items
            interaction_mask: (batch_size, history_len) — 1 for valid, 0 for padding
        Returns:
            (batch_size, d_model) — user profile vector
        """
        proj = self.item_proj(item_embeddings)  # (B, L, d_model)
        scores = self.attn_score(proj)  # (B, L, 1)

        if interaction_mask is not None:
            scores = scores.masked_fill(interaction_mask.unsqueeze(-1) == 0, float("-inf"))

        weights = F.softmax(scores, dim=1)  # (B, L, 1)
        profile = (weights * proj).sum(dim=1)  # (B, d_model)
        return self.norm(self.dropout(profile))


# ---------------------------------------------------------------------------
# Candidate Item Generator
# ---------------------------------------------------------------------------

class CandidateItemGenerator(nn.Module):
    """
    Generates a ranked candidate item list for a user.

    Computes preference scores for all items in a pool, applies category
    ordering from the ACA algorithm, and returns the top-K candidates.
    """

    def __init__(self, d_model: int, item_dim: int, dropout: float = 0.1,
                 category_bonus_weight: float = 0.1):
        super().__init__()
        self.category_bonus_weight = category_bonus_weight
        self.score_head = nn.Sequential(
            nn.Linear(d_model + item_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        user_profile: torch.Tensor,
        item_embeddings: torch.Tensor,
        top_k: int = 20,
        category_order: Optional[np.ndarray] = None,
        item_categories: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            user_profile:    (batch_size, d_model)
            item_embeddings: (num_items, d_model)
            top_k:           number of candidate items to return
            category_order:  (num_categories,) sorted category IDs from ACA
            item_categories: (num_items,) integer category ID per item
        Returns:
            candidate_ids:  (batch_size, top_k) item indices
            scores:         (batch_size, top_k) corresponding preference scores
        """
        batch_size = user_profile.shape[0]
        num_items = item_embeddings.shape[0]

        # Expand user profile to score all items at once
        u = user_profile.unsqueeze(1).expand(-1, num_items, -1)  # (B, N, d)
        v = item_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, d)
        scores = self.score_head(torch.cat([u, v], dim=-1)).squeeze(-1)  # (B, N)

        # Apply category-order boost from ACA if available
        if category_order is not None and item_categories is not None:
            # Build a rank-based bonus: best category gets bonus 1, worst gets 0
            num_cats = len(category_order)
            cat_rank = torch.zeros(num_cats, device=scores.device)
            for rank, cat_id in enumerate(category_order):
                cat_rank[int(cat_id)] = 1.0 - rank / max(num_cats - 1, 1)
            item_bonus = cat_rank[item_categories]  # (N,)
            scores = scores + self.category_bonus_weight * item_bonus.unsqueeze(0)

        top_scores, top_ids = torch.topk(scores, k=min(top_k, num_items), dim=1)
        return top_ids, top_scores


# ---------------------------------------------------------------------------
# Top-level Motivation-Aware Preference Prediction
# ---------------------------------------------------------------------------

class MotivationAwarePreference(nn.Module):
    """
    Motivation-Aware Preference Prediction model.

    Integrates:
    - UserProfileConstructor for building dynamic user profiles
    - MMergeLayer for motivation-conditioned user-item fusion
    - CandidateItemGenerator for final candidate ranking
    """

    def __init__(
        self,
        item_dim: int,
        d_model: int,
        num_m_merge_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.item_dim = item_dim
        self.profile_constructor = UserProfileConstructor(item_dim, d_model, dropout)

        # Stack of M-Merge layers for progressive preference refinement
        self.m_merge_layers = nn.ModuleList(
            [MMergeLayer(d_model, item_dim, d_model, dropout) for _ in range(num_m_merge_layers)]
        )
        self.candidate_generator = CandidateItemGenerator(d_model, item_dim, dropout)
        # Projection from item_dim to d_model for BPR loss computation
        self.item_proj = nn.Linear(item_dim, d_model)

    def build_user_profile(
        self,
        item_embeddings_hist: torch.Tensor,
        interaction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct a user profile from their interaction history.

        Args:
            item_embeddings_hist: (batch_size, history_len, item_dim)
            interaction_mask:     (batch_size, history_len)
        Returns:
            (batch_size, d_model)
        """
        return self.profile_constructor(item_embeddings_hist, interaction_mask)

    def forward(
        self,
        item_embeddings_hist: torch.Tensor,
        candidate_item_embeddings: torch.Tensor,
        interaction_mask: Optional[torch.Tensor] = None,
        motivation: Optional[torch.Tensor] = None,
        top_k: int = 20,
        category_order: Optional[np.ndarray] = None,
        item_categories: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            item_embeddings_hist:      (batch_size, history_len, item_dim)
            candidate_item_embeddings: (num_candidates, item_dim)
            interaction_mask:          (batch_size, history_len)
            motivation:                (batch_size,) scalar motivation scores
            top_k:                     number of items to return per user
            category_order:            (num_categories,) sorted category IDs from ACA
            item_categories:           (num_candidates,) category per candidate
        Returns:
            dict with 'candidate_ids', 'scores', 'user_profile'
        """
        # Build user profile from history
        user_profile = self.build_user_profile(item_embeddings_hist, interaction_mask)

        # Progressively refine with M-Merge layers
        # We use the mean candidate embedding as a general "item context"
        item_context = candidate_item_embeddings.mean(dim=0).unsqueeze(0).expand(
            user_profile.shape[0], -1
        )
        for layer in self.m_merge_layers:
            user_profile = layer(user_profile, item_context, motivation)

        # Generate candidate list
        candidate_ids, scores = self.candidate_generator(
            user_profile,
            candidate_item_embeddings,
            top_k=top_k,
            category_order=category_order,
            item_categories=item_categories,
        )

        return {
            "candidate_ids": candidate_ids,
            "scores": scores,
            "user_profile": user_profile,
        }

    def compute_bpr_loss(
        self,
        user_profiles: torch.Tensor,
        pos_item_embeddings: torch.Tensor,
        neg_item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bayesian Personalised Ranking loss for training.

        Args:
            user_profiles:       (batch_size, d_model)
            pos_item_embeddings: (batch_size, item_dim)
            neg_item_embeddings: (batch_size, item_dim)
        Returns:
            scalar BPR loss
        """
        pos = self.item_proj(pos_item_embeddings)
        neg = self.item_proj(neg_item_embeddings)
        pos_scores = (user_profiles * pos).sum(dim=-1)
        neg_scores = (user_profiles * neg).sum(dim=-1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss
