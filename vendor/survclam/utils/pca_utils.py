#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils_pca.py

Fold-specific Incremental PCA utilities for CLAMFamily-style pipelines.

Goal:
  - Fit IncrementalPCA on TRAIN split only (per fold)
  - Save PCA model to disk (so eval.py can reuse it)
  - Wrap any Dataset so that __getitem__ returns projected embeddings on-the-fly
    (no need to write pt_files_pca on disk)

Assumptions:
  - Your dataset __getitem__(idx) returns a tuple whose first element is
    a 2D tensor/array of shape [n_tiles, D] (float-like).
  - Variable-length bags (n_tiles differs) are OK; we stream rows into IPCA.

Dependencies:
  - scikit-learn (IncrementalPCA)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.decomposition import IncrementalPCA
except Exception as e:
    IncrementalPCA = None
    _SKLEARN_IMPORT_ERR = e


# -----------------------------------------------------------------------------
# Core transformer (numpy storage; torch in/out)
# -----------------------------------------------------------------------------

@dataclass
class PCATransformer:
    mean: np.ndarray          # [D]
    components: np.ndarray    # [k, D]
    explained_variance: Optional[np.ndarray] = None  # [k]
    explained_variance_ratio: Optional[np.ndarray] = None  # [k]

    @property
    def k(self) -> int:
        return int(self.components.shape[0])

    @property
    def dim_in(self) -> int:
        return int(self.components.shape[1])

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """
        X: [N, D] float32/float64
        returns: [N, k] float32
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D X, got {X.shape}")
        if X.shape[1] != self.dim_in:
            raise ValueError(f"Dim mismatch: X has D={X.shape[1]} but PCA expects D={self.dim_in}")

        Xc = X.astype(np.float32, copy=False) - self.mean.reshape(1, -1).astype(np.float32, copy=False)
        Z = Xc @ self.components.T.astype(np.float32, copy=False)
        return Z.astype(np.float32, copy=False)

    def transform_torch(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: torch.Tensor [N, D]
        returns: torch.Tensor [N, k]
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {tuple(X.shape)}")
        if int(X.shape[1]) != self.dim_in:
            raise ValueError(f"Dim mismatch: X has D={int(X.shape[1])} but PCA expects D={self.dim_in}")

        # keep on CPU for deterministic + avoid GPU mem spikes; caller can move later
        if X.is_cuda:
            X = X.detach().cpu()
        Xn = X.detach().to(torch.float32)

        mean = torch.from_numpy(self.mean.astype(np.float32, copy=False)).view(1, -1)
        comps = torch.from_numpy(self.components.astype(np.float32, copy=False))  # [k, D]
        Z = (Xn - mean) @ comps.t()
        return Z.to(torch.float32)


# -----------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------

def save_pca_npz(pca: PCATransformer, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        mean=pca.mean.astype(np.float32),
        components=pca.components.astype(np.float32),
        explained_variance=(None if pca.explained_variance is None else pca.explained_variance.astype(np.float32)),
        explained_variance_ratio=(None if pca.explained_variance_ratio is None else pca.explained_variance_ratio.astype(np.float32)),
    )


def load_pca_npz(path: str) -> PCATransformer:
    data = np.load(path, allow_pickle=True)
    mean = data["mean"].astype(np.float32)
    components = data["components"].astype(np.float32)

    ev = None
    evr = None
    if "explained_variance" in data and data["explained_variance"].dtype != object:
        ev = data["explained_variance"].astype(np.float32)
    if "explained_variance_ratio" in data and data["explained_variance_ratio"].dtype != object:
        evr = data["explained_variance_ratio"].astype(np.float32)

    return PCATransformer(mean=mean, components=components, explained_variance=ev, explained_variance_ratio=evr)


# -----------------------------------------------------------------------------
# Dataset wrapper (projects on-the-fly)
# -----------------------------------------------------------------------------

class PCADatasetWrapper(Dataset):
    """
    Transparent proxy wrapper:
    - Delegates ALL unknown attributes/methods to base dataset
    - Only modifies __getitem__ to PCA-project the first returned element (features)
    """
    def __init__(self, base: Dataset, pca: PCATransformer):
        self.base = base
        self.pca = pca

        # Optional: eagerly expose commonly-used metadata fields so downstream code
        # that checks wrapper.__dict__ (or hasattr) behaves like the base dataset.
        for attr in [
            "slide_data", "patient_data", "patient_dict",
            "patient_strat", "patient_voting", "bag_level",
            "pt_id_col", "case_id_col", "data_dir",
            "return_institution", "institution_col",
        ]:
            if hasattr(base, attr):
                try:
                    setattr(self, attr, getattr(base, attr))
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]

        if isinstance(item, tuple):
            feats = item[0]
            rest = item[1:]
        else:
            feats = item
            rest = tuple()

        feats_t = _as_2d_float_tensor(feats)
        proj = self.pca.transform_torch(feats_t)

        if isinstance(item, tuple):
            return (proj, *rest)
        return proj

    def __getattr__(self, name: str):
        """
        If an attribute isn't found on the wrapper, fetch it from the base dataset.
        This is the key to preserving slide_id/case_id logic downstream.
        """
        # Called only if normal attribute lookup fails
        try:
            return getattr(self.base, name)
        except AttributeError as e:
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'") from e

    def get_base(self) -> Dataset:
        return self.base


def wrap_dataset_with_pca(ds: Dataset, pca: PCATransformer) -> Dataset:
    # Avoid double-wrapping
    if isinstance(ds, PCADatasetWrapper):
        return ds
    return PCADatasetWrapper(ds, pca)



def wrap_dataset_with_pca(ds: Dataset, pca: PCATransformer) -> Dataset:
    return PCADatasetWrapper(ds, pca)


# -----------------------------------------------------------------------------
# Fitting IncrementalPCA on TRAIN split (stream tiles)
# -----------------------------------------------------------------------------

def fit_incremental_pca_on_dataset(
    train_dataset: Dataset,
    n_components: int,
    batch_rows: int = 200_000,
    num_workers: int = 0,
    max_total_rows: Optional[int] = None,
    seed: int = 1,
    verbose: bool = True,
) -> PCATransformer:
    """
    Fits IncrementalPCA on ALL tiles from train_dataset by streaming.

    train_dataset yields one bag at a time: [n_tiles, D].
    We stream each bag in row-chunks of size batch_rows into ipca.partial_fit.

    max_total_rows:
      Optional cap on total #rows used for fitting (useful if train is massive).
      If set, we stop once we have fed that many rows.

    Returns:
      PCATransformer
    """
    if IncrementalPCA is None:
        raise ImportError(f"scikit-learn not available; ImportError: {_SKLEARN_IMPORT_ERR}")

    # Determinism for optional subsampling behavior
    rng = np.random.default_rng(seed)

    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_keep_single,
        pin_memory=False,
    )

    ipca = IncrementalPCA(n_components=int(n_components))

    total_rows = 0
    n_bags = 0

    if verbose:
        print(f"[IPCA] Fitting IncrementalPCA(k={n_components})")
        print(f"[IPCA] batch_rows={batch_rows}  max_total_rows={max_total_rows}")

    for item in loader:
        feats = item[0] if isinstance(item, tuple) else item
        X = _as_2d_float_numpy(feats)  # [N, D] float32

        # stream row chunks
        N = X.shape[0]
        if N == 0:
            continue

        # Optional: if max_total_rows is set and this bag would exceed it,
        # we only take the remaining rows (random subset for fairness).
        if max_total_rows is not None and (total_rows + N) > max_total_rows:
            remaining = max_total_rows - total_rows
            if remaining <= 0:
                break
            if remaining < N:
                # random subset of rows
                idx = rng.choice(N, size=int(remaining), replace=False)
                X = X[idx, :]
                N = X.shape[0]

        for start in range(0, N, batch_rows):
            chunk = X[start:start + batch_rows, :]
            ipca.partial_fit(chunk)
            total_rows += chunk.shape[0]
            if max_total_rows is not None and total_rows >= max_total_rows:
                break

        n_bags += 1
        if max_total_rows is not None and total_rows >= max_total_rows:
            break

    if verbose:
        print(f"[IPCA] Done. bags={n_bags}  rows_used={total_rows}")

    # Build our lightweight transformer
    mean = getattr(ipca, "mean_", None)
    comps = getattr(ipca, "components_", None)
    if mean is None or comps is None:
        raise RuntimeError("IncrementalPCA did not fit properly (missing mean_/components_).")

    ev = getattr(ipca, "explained_variance_", None)
    evr = getattr(ipca, "explained_variance_ratio_", None)

    return PCATransformer(
        mean=np.asarray(mean, dtype=np.float32),
        components=np.asarray(comps, dtype=np.float32),
        explained_variance=(None if ev is None else np.asarray(ev, dtype=np.float32)),
        explained_variance_ratio=(None if evr is None else np.asarray(evr, dtype=np.float32)),
    )


def fit_and_save_fold_pca(
    train_dataset: Dataset,
    out_npz: str,
    n_components: int,
    batch_rows: int = 200_000,
    num_workers: int = 0,
    max_total_rows: Optional[int] = None,
    seed: int = 1,
    verbose: bool = True,
) -> PCATransformer:
    """
    Convenience: fit on train_dataset, save to out_npz, return transformer.
    """
    pca = fit_incremental_pca_on_dataset(
        train_dataset=train_dataset,
        n_components=n_components,
        batch_rows=batch_rows,
        num_workers=num_workers,
        max_total_rows=max_total_rows,
        seed=seed,
        verbose=verbose,
    )
    save_pca_npz(pca, out_npz)
    if verbose:
        print(f"[IPCA] Saved PCA model -> {out_npz}")
    return pca


# -----------------------------------------------------------------------------
# Utilities (conversions + collate)
# -----------------------------------------------------------------------------

def _collate_keep_single(batch):
    # batch is a list of length 1 (because batch_size=1)
    return batch[0]


def _as_2d_float_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        # try torch.tensor
        t = torch.tensor(x)

    if t.ndim != 2:
        raise ValueError(f"Expected 2D features [N,D], got {tuple(t.shape)}")
    return t.to(torch.float32)


def _as_2d_float_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features [N,D], got {arr.shape}")
    return arr.astype(np.float32, copy=False)
