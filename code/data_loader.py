# -- coding: utf-8 --
"""
Data loaders for Yelp, Yelp-L, and Douban datasets.

Each dataset exposes the same interface (PyTorch Dataset subclasses) so that
a single DataLoader can be used across all three without any change to the
training loop.

Expected directory layout (mirrors data/readme.txt):
    <root>/
        groupid_events.dat   — Group→Events  (tab-separated)
        groupid_users.dat    — Group→Members (tab-separated)
        user_events.dat      — User→Events   (tab-separated)
        social.dat           — User→Friends  (tab-separated)
        [item_features.dat]  — Item metadata (optional, dataset-specific)

Each .dat line format:
    ID<TAB>val1,val2,...
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Low-level file parsers
# ---------------------------------------------------------------------------

def _parse_id_list_file(path: str) -> Dict[int, List[int]]:
    """Parse files of the form  ``ID<TAB>v1,v2,...`` into a dict."""
    result: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            key = int(parts[0])
            values = [int(v) for v in parts[1].split(",") if v.strip()] if len(parts) > 1 else []
            result[key] = values
    return result


def _parse_item_features(path: str) -> Dict[int, np.ndarray]:
    """
    Parse optional item feature file.

    Expected format (space- or tab-separated):
        item_id  feat1  feat2  ...
    """
    features: Dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            item_id = int(vals[0])
            feats = np.array([float(v) for v in vals[1:]], dtype=np.float32)
            features[item_id] = feats
    return features


# ---------------------------------------------------------------------------
# Base Dataset
# ---------------------------------------------------------------------------

class BaseSessionDataset(Dataset):
    """
    Base class for MASP session datasets.

    Builds (group_id, pos_event_id, neg_event_id) training triples using
    BPR-style negative sampling.  Optionally attaches item feature vectors.

    Parameters
    ----------
    root : str
        Path to dataset directory.
    split : 'train' | 'val' | 'test'
        Which split to load.  The first 80 % of groups form the training set,
        next 10 % validation, last 10 % test.
    feature_dim : int
        Dimensionality of random feature vectors generated when
        ``item_features.dat`` is absent.
    neg_samples : int
        Number of negative items per positive (only used in training split).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        feature_dim: int = 64,
        neg_samples: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.feature_dim = feature_dim
        self.neg_samples = neg_samples
        self.rng = np.random.default_rng(seed)

        self._load_data()
        self._build_splits()
        self._build_triples()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        ge_path = os.path.join(self.root, "groupid_events.dat")
        gu_path = os.path.join(self.root, "groupid_users.dat")
        ue_path = os.path.join(self.root, "user_events.dat")
        so_path = os.path.join(self.root, "social.dat")
        fe_path = os.path.join(self.root, "item_features.dat")

        self.group_events: Dict[int, List[int]] = (
            _parse_id_list_file(ge_path) if os.path.exists(ge_path) else {}
        )
        self.group_users: Dict[int, List[int]] = (
            _parse_id_list_file(gu_path) if os.path.exists(gu_path) else {}
        )
        self.user_events: Dict[int, List[int]] = (
            _parse_id_list_file(ue_path) if os.path.exists(ue_path) else {}
        )
        self.social: Dict[int, List[int]] = (
            _parse_id_list_file(so_path) if os.path.exists(so_path) else {}
        )

        # Gather all item IDs
        all_items: set = set()
        for evts in self.group_events.values():
            all_items.update(evts)
        for evts in self.user_events.values():
            all_items.update(evts)
        self.all_item_ids = sorted(all_items)
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.all_item_ids)}
        self.num_items = len(self.all_item_ids)

        # Item feature matrix
        if os.path.exists(fe_path):
            raw_feats = _parse_item_features(fe_path)
            feat_dim = next(iter(raw_feats.values())).shape[0] if raw_feats else self.feature_dim
            self.feature_dim = feat_dim
            feat_mat = np.zeros((self.num_items, feat_dim), dtype=np.float32)
            for iid, feat in raw_feats.items():
                idx = self.item_id_to_idx.get(iid)
                if idx is not None:
                    feat_mat[idx] = feat
            self.item_features = torch.tensor(feat_mat)
        else:
            # Generate random features (for debugging / when features are absent)
            rng = np.random.default_rng(0)
            feat_mat = rng.standard_normal((self.num_items, self.feature_dim)).astype(np.float32)
            self.item_features = torch.tensor(feat_mat)

        self.all_group_ids = sorted(self.group_events.keys())
        self.num_groups = len(self.all_group_ids)

    def _build_splits(self) -> None:
        n = self.num_groups
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if self.split == "train":
            self.split_groups = self.all_group_ids[:n_train]
        elif self.split == "val":
            self.split_groups = self.all_group_ids[n_train : n_train + n_val]
        else:
            self.split_groups = self.all_group_ids[n_train + n_val :]

    def _build_triples(self) -> None:
        """Build (group_idx, pos_item_idx, neg_item_idx) triples."""
        self.triples: List[Tuple[int, int, int]] = []
        all_item_set = set(self.all_item_ids)

        for group_id in self.split_groups:
            pos_events = self.group_events.get(group_id, [])
            if not pos_events:
                continue
            pos_set = set(pos_events)
            neg_pool = list(all_item_set - pos_set)
            if not neg_pool:
                continue
            for pos_id in pos_events:
                for _ in range(self.neg_samples):
                    neg_id = neg_pool[int(self.rng.integers(len(neg_pool)))]
                    self.triples.append(
                        (
                            group_id,
                            self.item_id_to_idx[pos_id],
                            self.item_id_to_idx[neg_id],
                        )
                    )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        group_id, pos_idx, neg_idx = self.triples[idx]
        return {
            "group_id": torch.tensor(group_id, dtype=torch.long),
            "pos_item_idx": torch.tensor(pos_idx, dtype=torch.long),
            "neg_item_idx": torch.tensor(neg_idx, dtype=torch.long),
            "pos_item_feat": self.item_features[pos_idx],
            "neg_item_feat": self.item_features[neg_idx],
        }

    def get_eval_data(self) -> List[Tuple[int, List[int], List[int]]]:
        """
        Return evaluation tuples: (group_id, pos_item_indices, all_item_indices).
        """
        eval_data = []
        for group_id in self.split_groups:
            pos_events = self.group_events.get(group_id, [])
            pos_indices = [self.item_id_to_idx[e] for e in pos_events if e in self.item_id_to_idx]
            if pos_indices:
                eval_data.append((group_id, pos_indices, list(range(self.num_items))))
        return eval_data


# ---------------------------------------------------------------------------
# Concrete datasets
# ---------------------------------------------------------------------------

class YelpDataset(BaseSessionDataset):
    """
    Yelp dataset loader (standard-size split from the MASP paper).

    Uses the same file format as BaseSessionDataset.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        feature_dim: int = 64,
        neg_samples: int = 1,
        seed: int = 42,
    ):
        super().__init__(root, split, feature_dim, neg_samples, seed)


class YelpLDataset(BaseSessionDataset):
    """
    Yelp-L (large-scale) dataset loader.

    Identical file format to YelpDataset but typically larger data volumes.
    The larger vocabulary / feature dimension can be configured via
    ``feature_dim``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        feature_dim: int = 128,
        neg_samples: int = 4,
        seed: int = 42,
    ):
        super().__init__(root, split, feature_dim, neg_samples, seed)


class DoubanDataset(BaseSessionDataset):
    """
    Douban dataset loader.

    The Douban social network uses the same data layout; this class exists as a
    distinct type so that dataset-specific hyper-parameter defaults (e.g., a
    higher default feature dimension) can differ from the Yelp variants.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        feature_dim: int = 64,
        neg_samples: int = 1,
        seed: int = 42,
    ):
        super().__init__(root, split, feature_dim, neg_samples, seed)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

_DATASET_REGISTRY = {
    "yelp": YelpDataset,
    "yelp_l": YelpLDataset,
    "douban": DoubanDataset,
}


def build_dataloader(
    dataset_name: str,
    root: str,
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 0,
    feature_dim: int = 64,
    neg_samples: int = 1,
    seed: int = 42,
    shuffle: Optional[bool] = None,
) -> Tuple[DataLoader, BaseSessionDataset]:
    """
    Build a DataLoader for one of the supported datasets.

    Args:
        dataset_name: one of 'yelp', 'yelp_l', 'douban'
        root:         path to dataset directory
        split:        'train', 'val', or 'test'
        batch_size:   mini-batch size
        num_workers:  DataLoader worker processes
        feature_dim:  item feature dimensionality
        neg_samples:  BPR negative samples per positive
        seed:         random seed
        shuffle:      if None, shuffle only when split == 'train'
    Returns:
        (dataloader, dataset)
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from {list(_DATASET_REGISTRY)}"
        )
    cls = _DATASET_REGISTRY[dataset_name]
    dataset = cls(
        root=root,
        split=split,
        feature_dim=feature_dim,
        neg_samples=neg_samples,
        seed=seed,
    )
    if shuffle is None:
        shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset
