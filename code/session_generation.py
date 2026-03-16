# -- coding: utf-8 --
"""
Multi-Constraints Session Generation module.

Implements:
  - NaiveMCSG  — exhaustive multi-constraint session generation
  - ILG        — Item List Generation optimisation algorithm
  - GreedyMCSG — efficient greedy approximation of MCSG
  - SessionGenerator — unified interface
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _travel_time(dist_km: float, speed_kmh: float = 30.0) -> float:
    """Estimated travel time in hours."""
    return dist_km / speed_kmh if speed_kmh > 0 else float("inf")


# ---------------------------------------------------------------------------
# Session constraints data class
# ---------------------------------------------------------------------------

class SessionConstraints:
    """
    Container for multi-constraint session planning parameters.

    Attributes:
        max_distance_km:     maximum total travel distance for the session
        max_duration_hours:  maximum total duration (travel + activity dwell time)
        min_session_length:  minimum number of items (activities) in the session
        max_session_length:  maximum number of items in the session
        start_location:      (lat, lon) tuple for the session starting point
        dwell_time_hours:    default dwell time per activity in hours
        speed_kmh:           average travel speed (for time estimation)
        required_categories: set of category IDs that must appear in the session
        excluded_item_ids:   set of item IDs that must not appear in the session
    """

    def __init__(
        self,
        max_distance_km: float = 50.0,
        max_duration_hours: float = 8.0,
        min_session_length: int = 2,
        max_session_length: int = 8,
        start_location: Optional[Tuple[float, float]] = None,
        dwell_time_hours: float = 1.0,
        speed_kmh: float = 30.0,
        required_categories: Optional[set] = None,
        excluded_item_ids: Optional[set] = None,
    ):
        self.max_distance_km = max_distance_km
        self.max_duration_hours = max_duration_hours
        self.min_session_length = min_session_length
        self.max_session_length = max_session_length
        self.start_location = start_location
        self.dwell_time_hours = dwell_time_hours
        self.speed_kmh = speed_kmh
        self.required_categories = required_categories or set()
        self.excluded_item_ids = excluded_item_ids or set()


# ---------------------------------------------------------------------------
# Naive MCSG
# ---------------------------------------------------------------------------

class NaiveMCSG:
    """
    Naive Multi-Constraint Session Generation.

    Enumerates all subsets of the candidate item list up to max_session_length,
    scores each feasible session, and returns the best one.

    Time complexity: O(2^n) — use GreedyMCSG for large candidate sets.
    """

    def __init__(self, constraints: SessionConstraints):
        self.constraints = constraints

    def _session_distance(
        self,
        session: List[Dict],
        start_loc: Optional[Tuple[float, float]],
    ) -> float:
        """Total travel distance (km) for the ordered session."""
        total = 0.0
        locs = []
        if start_loc is not None:
            locs.append(start_loc)
        for item in session:
            loc = item.get("location")
            if loc is not None:
                locs.append(loc)
        for i in range(1, len(locs)):
            total += _haversine_km(*locs[i - 1], *locs[i])
        return total

    def _session_duration(self, session: List[Dict], dist_km: float) -> float:
        """Total session duration in hours (travel + dwell times)."""
        travel = _travel_time(dist_km, self.constraints.speed_kmh)
        dwell = sum(item.get("dwell_time", self.constraints.dwell_time_hours) for item in session)
        return travel + dwell

    def _is_feasible(self, session: List[Dict]) -> bool:
        """Check whether a session satisfies all constraints."""
        c = self.constraints
        if not (c.min_session_length <= len(session) <= c.max_session_length):
            return False
        ids = {item["id"] for item in session}
        if ids & c.excluded_item_ids:
            return False
        dist = self._session_distance(session, c.start_location)
        if dist > c.max_distance_km:
            return False
        if self._session_duration(session, dist) > c.max_duration_hours:
            return False
        cats = {item.get("category") for item in session}
        if not c.required_categories.issubset(cats):
            return False
        return True

    def generate(
        self,
        candidate_items: List[Dict],
        scores: Optional[List[float]] = None,
    ) -> Optional[List[Dict]]:
        """
        Args:
            candidate_items: list of item dicts (id, location, category, …)
            scores:          preference score per candidate (same order)
        Returns:
            Best feasible session as an ordered list of item dicts, or None.
        """
        if scores is None:
            scores = [1.0] * len(candidate_items)

        best_session: Optional[List[Dict]] = None
        best_score = -float("inf")
        n = len(candidate_items)

        for mask in range(1, 1 << n):
            session = []
            session_score = 0.0
            for i in range(n):
                if mask & (1 << i):
                    session.append(candidate_items[i])
                    session_score += scores[i]
            if self._is_feasible(session):
                if session_score > best_score:
                    best_score = session_score
                    best_session = session

        return best_session


# ---------------------------------------------------------------------------
# ILG — Item List Generation optimisation
# ---------------------------------------------------------------------------

class ILG:
    """
    Item List Generation (ILG) optimisation algorithm.

    Prunes the candidate item pool before session planning using a constraint
    propagation approach:
    1. Remove items that are individually infeasible (excluded IDs, over-distance
       from the start, etc.).
    2. Sort surviving items by a combined reachability-preference score.
    3. Return a pruned list of at most ``max_candidates`` items.

    This pre-filtering makes downstream MCSG / GreedyMCSG much faster.
    """

    def __init__(
        self,
        constraints: SessionConstraints,
        max_candidates: int = 50,
        reachability_bonus_weight: float = 0.2,
    ):
        self.constraints = constraints
        self.max_candidates = max_candidates
        self.reachability_bonus_weight = reachability_bonus_weight

    def _item_score(self, item: Dict, pref_score: float) -> float:
        """Combined score: preference weight + reachability bonus."""
        c = self.constraints
        bonus = 0.0
        if c.start_location is not None:
            loc = item.get("location")
            if loc is not None:
                dist = _haversine_km(*c.start_location, *loc)
                # Higher bonus for items closer to the start (normalised by max_distance)
                bonus = max(0.0, 1.0 - dist / c.max_distance_km)
        return pref_score + self.reachability_bonus_weight * bonus

    def run(
        self,
        candidate_items: List[Dict],
        preference_scores: Optional[List[float]] = None,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Prune and rank candidate items.

        Args:
            candidate_items:   raw candidate list from MotivationAwarePreference
            preference_scores: model preference score per item
        Returns:
            (pruned_items, pruned_scores)
        """
        if preference_scores is None:
            preference_scores = [1.0] * len(candidate_items)

        c = self.constraints
        filtered: List[Tuple[float, Dict, float]] = []

        for item, pref in zip(candidate_items, preference_scores):
            # Hard constraint: excluded items
            if item.get("id") in c.excluded_item_ids:
                continue
            # Soft constraint: items too far from start are deprioritised (not hard-removed)
            score = self._item_score(item, pref)
            filtered.append((score, item, pref))

        # Sort by combined score descending
        filtered.sort(key=lambda t: t[0], reverse=True)
        filtered = filtered[: self.max_candidates]

        pruned_items = [t[1] for t in filtered]
        pruned_scores = [t[2] for t in filtered]
        return pruned_items, pruned_scores


# ---------------------------------------------------------------------------
# Greedy MCSG
# ---------------------------------------------------------------------------

class GreedyMCSG:
    """
    Greedy Multi-Constraint Session Generation.

    Incrementally adds the highest-scoring feasible item to the session.
    Each step:
    1. Compute the marginal travel distance / time added by each remaining item.
    2. Select the item with the highest preference score whose addition keeps
       all constraints satisfied.
    3. Repeat until no more items can be added or max_session_length is reached.

    Time complexity: O(n^2) — efficient for large candidate sets.
    """

    def __init__(self, constraints: SessionConstraints):
        self.constraints = constraints

    def generate(
        self,
        candidate_items: List[Dict],
        scores: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Args:
            candidate_items: list of item dicts (id, location, category, …)
            scores:          preference score per candidate
        Returns:
            Ordered session as a list of item dicts (may be empty).
        """
        if scores is None:
            scores = [1.0] * len(candidate_items)

        c = self.constraints
        session: List[Dict] = []
        used_ids: set = set()
        current_dist = 0.0
        current_duration = 0.0
        # Current location starts at the session start location
        current_loc = c.start_location

        # Sort by score descending for greedy selection
        sorted_items = sorted(
            zip(candidate_items, scores), key=lambda t: t[1], reverse=True
        )

        for item, score in sorted_items:
            if len(session) >= c.max_session_length:
                break
            if item.get("id") in used_ids or item.get("id") in c.excluded_item_ids:
                continue

            # Compute incremental distance
            item_loc = item.get("location")
            if current_loc is not None and item_loc is not None:
                delta_dist = _haversine_km(*current_loc, *item_loc)
            else:
                delta_dist = 0.0

            dwell = item.get("dwell_time", c.dwell_time_hours)
            delta_time = _travel_time(delta_dist, c.speed_kmh) + dwell

            # Check constraints
            if current_dist + delta_dist > c.max_distance_km:
                continue
            if current_duration + delta_time > c.max_duration_hours:
                continue

            # Accept item
            session.append(item)
            used_ids.add(item.get("id"))
            current_dist += delta_dist
            current_duration += delta_time
            if item_loc is not None:
                current_loc = item_loc

        return session


# ---------------------------------------------------------------------------
# Session Generator (unified interface)
# ---------------------------------------------------------------------------

class SessionGenerator(nn.Module):
    """
    Unified session generation interface.

    Combines the ILG pruning step with GreedyMCSG (default) or NaiveMCSG
    (for small candidate sets) to produce a multi-constraint session.

    A learnable scoring layer refines the model preference scores before
    passing them to the generation algorithm.
    """

    def __init__(
        self,
        d_model: int,
        constraints: Optional[SessionConstraints] = None,
        use_naive: bool = False,
        ilg_max_candidates: int = 50,
        naive_threshold: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.constraints = constraints or SessionConstraints()
        self.use_naive = use_naive
        self.naive_threshold = naive_threshold

        self.ilg = ILG(self.constraints, max_candidates=ilg_max_candidates)
        self.greedy_mcsg = GreedyMCSG(self.constraints)
        self.naive_mcsg = NaiveMCSG(self.constraints)

        # Learnable score refinement head
        self.score_refine = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, 1),
        )

    def refine_scores(self, item_embeddings: torch.Tensor) -> List[float]:
        """
        Apply learned score refinement to item embeddings.

        Args:
            item_embeddings: (num_items, d_model)
        Returns:
            list of float scores
        """
        with torch.no_grad():
            scores = self.score_refine(item_embeddings).squeeze(-1)
        return scores.cpu().tolist()

    def generate_session(
        self,
        candidate_items: List[Dict],
        item_embeddings: Optional[torch.Tensor] = None,
        preference_scores: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Generate a session from a candidate item list.

        Args:
            candidate_items:   list of item dicts
            item_embeddings:   (num_items, d_model) — if given, scores are refined
            preference_scores: raw preference scores (used if embeddings not given)
        Returns:
            Ordered list of item dicts forming the session.
        """
        if item_embeddings is not None:
            preference_scores = self.refine_scores(item_embeddings)
        elif preference_scores is None:
            preference_scores = [1.0] * len(candidate_items)

        # ILG pruning
        pruned_items, pruned_scores = self.ilg.run(candidate_items, preference_scores)

        if self.use_naive and len(pruned_items) <= self.naive_threshold:
            session = self.naive_mcsg.generate(pruned_items, pruned_scores) or []
        else:
            session = self.greedy_mcsg.generate(pruned_items, pruned_scores)

        return session

    def forward(
        self,
        candidate_items: List[Dict],
        item_embeddings: Optional[torch.Tensor] = None,
        preference_scores: Optional[List[float]] = None,
    ) -> Dict:
        """
        Forward call — generates and returns session with metadata.

        Returns:
            dict with keys:
              'session'          — list of item dicts
              'session_length'   — int
              'total_distance'   — float (km)
              'total_duration'   — float (hours)
        """
        session = self.generate_session(candidate_items, item_embeddings, preference_scores)

        # Compute session stats
        total_dist = 0.0
        total_dur = 0.0
        c = self.constraints
        current_loc = c.start_location
        for item in session:
            loc = item.get("location")
            if current_loc is not None and loc is not None:
                d = _haversine_km(*current_loc, *loc)
                total_dist += d
                total_dur += _travel_time(d, c.speed_kmh)
                current_loc = loc
            total_dur += item.get("dwell_time", c.dwell_time_hours)

        return {
            "session": session,
            "session_length": len(session),
            "total_distance": total_dist,
            "total_duration": total_dur,
        }
