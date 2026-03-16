# -- coding: utf-8 --
"""
Performance metrics for the MASP framework.

Implements:
  - hr_at_k         — Hit Rate @ K
  - ndcg_at_k       — Normalised Discounted Cumulative Gain @ K
  - spearman_correlation — Spearman rank correlation between two orderings
  - compute_metrics  — convenience wrapper that computes HR@K and NDCG@K
                       for multiple K values at once
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Hit Rate @ K
# ---------------------------------------------------------------------------

def hr_at_k(
    ranked_items: Sequence[int],
    ground_truth: Sequence[int],
    k: int,
) -> float:
    """
    Hit Rate @ K.

    Returns 1.0 if any of the top-K ``ranked_items`` appears in
    ``ground_truth``, else 0.0.

    Args:
        ranked_items:  ordered list of recommended item IDs (best first)
        ground_truth:  set of relevant item IDs for this user/group
        k:             cut-off rank
    Returns:
        0.0 or 1.0
    """
    gt_set = set(ground_truth)
    top_k = ranked_items[:k]
    return 1.0 if any(item in gt_set for item in top_k) else 0.0


# ---------------------------------------------------------------------------
# NDCG @ K
# ---------------------------------------------------------------------------

def ndcg_at_k(
    ranked_items: Sequence[int],
    ground_truth: Sequence[int],
    k: int,
) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.

    Binary relevance: an item is relevant (gain = 1) if it appears in
    ``ground_truth``.

    Args:
        ranked_items:  ordered list of recommended item IDs (best first)
        ground_truth:  set of relevant item IDs
        k:             cut-off rank
    Returns:
        NDCG in [0, 1]
    """
    gt_set = set(ground_truth)
    top_k = ranked_items[:k]

    dcg = sum(
        1.0 / math.log2(rank + 2)  # rank is 0-indexed → denominator = log2(rank+2)
        for rank, item in enumerate(top_k)
        if item in gt_set
    )

    # Ideal DCG: all relevant items are ranked first
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Spearman Rank Correlation
# ---------------------------------------------------------------------------

def spearman_correlation(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    """
    Spearman rank correlation coefficient.

    Args:
        x: first ranking / score sequence
        y: second ranking / score sequence (same length)
    Returns:
        Spearman ρ in [-1, 1]; returns 0.0 if the input is degenerate.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(arr: Sequence[float]) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        order = np.argsort(a)
        ranks = np.empty(n, dtype=np.float64)
        # Handle ties by averaging ranks
        i = 0
        while i < n:
            j = i + 1
            while j < n and a[order[j]] == a[order[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg_rank
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d = rx - ry
    d_sq_sum = np.dot(d, d)

    # Standard Spearman rank correlation: ρ = 1 - (6 * Σd²) / (n*(n²-1))
    rho = 1.0 - 6.0 * d_sq_sum / (n * (n * n - 1))
    return float(np.clip(rho, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------

def compute_metrics(
    all_ranked_items: List[Sequence[int]],
    all_ground_truths: List[Sequence[int]],
    k_values: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Compute HR@K and NDCG@K averaged over all evaluation instances.

    Args:
        all_ranked_items:  list of per-user ranked item lists
        all_ground_truths: list of per-user ground-truth item sets
        k_values:          which K values to evaluate; defaults to [5, 10, 20]
    Returns:
        dict with keys like 'HR@5', 'NDCG@5', 'HR@10', …
    """
    if k_values is None:
        k_values = [5, 10, 20]

    assert len(all_ranked_items) == len(all_ground_truths), (
        "ranked_items and ground_truths must have the same number of entries"
    )

    metrics: Dict[str, float] = {f"HR@{k}": 0.0 for k in k_values}
    metrics.update({f"NDCG@{k}": 0.0 for k in k_values})

    n = len(all_ranked_items)
    if n == 0:
        return metrics

    for ranked, gt in zip(all_ranked_items, all_ground_truths):
        for k in k_values:
            metrics[f"HR@{k}"] += hr_at_k(ranked, gt, k)
            metrics[f"NDCG@{k}"] += ndcg_at_k(ranked, gt, k)

    for key in metrics:
        metrics[key] /= n

    return metrics
