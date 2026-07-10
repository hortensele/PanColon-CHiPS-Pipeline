#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create leave-one-group-out (LOO) split CSVs from a dataset_csv.

Outputs (in out_dir):
  - splits_0.csv, splits_1.csv, ...  (CLAM-style: columns train/val/test with IDs)
  - fold_to_group.csv                (maps fold -> left-out group value)
  - split_settings.txt               (records args)

Typical use:
  python create_splits_loo.py \
    --dataset_csv dataset_csv/colon_united_pca_univ2_20x.csv \
    --out_dir results/colon_united_exp42 \
    --loo_col source \
    --id_col slide_id \
    --task_type survival \
    --val_frac 0.1 \
    --seed 1

Optional val stratification:
  --stratify_cols institution
  --stratify_cols institution,sex
"""

import os
import argparse
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _write_settings(out_dir: str, args: argparse.Namespace):
    path = os.path.join(out_dir, "split_settings.txt")
    with open(path, "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")


def _pad_columns(train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> pd.DataFrame:
    """
    Create a DataFrame with columns train/val/test, padded with empty strings
    to equal length (typical CLAM split CSV format).
    """
    n = max(len(train_ids), len(val_ids), len(test_ids), 1)

    def pad(x):
        x = list(map(str, x))
        if len(x) < n:
            x = x + [""] * (n - len(x))
        return x

    return pd.DataFrame({
        "train": pad(train_ids),
        "val":   pad(val_ids),
        "test":  pad(test_ids),
    })


def _parse_list(csv_str: str) -> List[str]:
    if csv_str is None or str(csv_str).strip() == "":
        return []
    return [c.strip() for c in str(csv_str).split(",") if c.strip()]


def _normalize_strat_value(x) -> str:
    """
    Stable token for stratification values.
    - NaN/None/empty -> "na"
    """
    if x is None:
        return "na"
    try:
        if pd.isna(x):
            return "na"
    except Exception:
        pass
    s = str(x).strip()
    return s if s != "" else "na"


def _validate_stratify_cols(df: pd.DataFrame, stratify_cols: List[str]) -> List[str]:
    if not stratify_cols:
        return []
    missing = [c for c in stratify_cols if c not in df.columns]
    if missing:
        msg = f"Missing stratify_cols in CSV: {missing}. Available columns: {list(df.columns)}"
        print(f"[WARN] {msg} -> ignoring missing columns.")
        stratify_cols = [c for c in stratify_cols if c in df.columns]
    return stratify_cols


def _make_joint_strata_labels(
    df: pd.DataFrame,
    *,
    label_col: Optional[str],
    stratify_cols: List[str],
) -> np.ndarray:
    """
    Build a 1D array of stratum labels (strings) for stratified split.

    If label_col is provided, the label is included in the stratum key.
    Always includes stratify_cols if provided.
    """
    if not stratify_cols and label_col is None:
        raise ValueError("Need at least one of label_col or stratify_cols to build strata.")

    parts = []
    if label_col is not None:
        # keep as string to avoid issues with categorical labels
        parts.append(df[label_col].astype(str).values)

    if stratify_cols:
        clin = []
        for c in stratify_cols:
            arr = df[c].map(_normalize_strat_value)
    
            # force missing -> "NA" (important so dtype stays consistent)
            arr = arr.fillna("NA")
    
            # IMPORTANT: to_numpy(dtype=str) guarantees a unicode array, not object
            clin.append(arr.to_numpy(dtype=str))
    
        # start (no need to add "")
        clin_join = clin[0].copy()
    
        for arr in clin[1:]:
            # make sure every arr is also unicode, just in case
            arr = np.asarray(arr, dtype=str)
            clin_join = np.char.add(np.char.add(clin_join, "||"), arr)

    parts.append(clin_join)


    # combine all parts
    y = parts[0]
    for p in parts[1:]:
        y = np.char.add(np.char.add(y, "__"), p)
    return y


def _stratified_val_split_generic(
    df_trainval: pd.DataFrame,
    id_col: str,
    val_frac: float,
    seed: int,
    strata_y: np.ndarray,
) -> Tuple[List[str], List[str]]:
    """
    Stratified split: select val_frac of TRAINVAL for val using strata_y.
    Falls back to random split if stratification is impossible (e.g., too many singletons).
    """
    if val_frac <= 0:
        return df_trainval[id_col].tolist(), []

    from sklearn.model_selection import StratifiedShuffleSplit

    X = np.arange(len(df_trainval))

    # If any class has <2 samples, StratifiedShuffleSplit can fail.
    # We'll attempt, and fall back to random.
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(sss.split(X, strata_y))
        train_ids = df_trainval.iloc[train_idx][id_col].tolist()
        val_ids   = df_trainval.iloc[val_idx][id_col].tolist()
        return train_ids, val_ids
    except Exception as e:
        print(f"[WARN] Stratified val split failed ({type(e).__name__}: {e}); falling back to random.")
        return _random_val_split(df_trainval=df_trainval, id_col=id_col, val_frac=val_frac, seed=seed)


def _stratified_val_split_classification(
    df_trainval: pd.DataFrame,
    id_col: str,
    val_frac: float,
    seed: int,
    label_col: str,
    stratify_cols: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Classification: stratified split. If stratify_cols provided, stratify by (label + clinical tuple).
    Else stratify by label only (original behavior).
    """
    if val_frac <= 0:
        return df_trainval[id_col].tolist(), []

    if stratify_cols:
        strata_y = _make_joint_strata_labels(df_trainval, label_col=label_col, stratify_cols=stratify_cols)
        return _stratified_val_split_generic(df_trainval, id_col, val_frac, seed, strata_y)
    else:
        # original: label-only stratification
        from sklearn.model_selection import StratifiedShuffleSplit
        y = df_trainval[label_col].astype(int).values
        X = np.arange(len(df_trainval))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(sss.split(X, y))
        train_ids = df_trainval.iloc[train_idx][id_col].tolist()
        val_ids   = df_trainval.iloc[val_idx][id_col].tolist()
        return train_ids, val_ids


def _random_val_split(
    df_trainval: pd.DataFrame,
    id_col: str,
    val_frac: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Random split (survival/regression): select val_frac of TRAINVAL for val.
    """
    if val_frac <= 0:
        return df_trainval[id_col].tolist(), []

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df_trainval))
    rng.shuffle(idx)

    n_val = int(round(val_frac * len(df_trainval)))
    n_val = max(min(n_val, len(df_trainval)), 0)

    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    train_ids = df_trainval.iloc[tr_idx][id_col].tolist()
    val_ids   = df_trainval.iloc[val_idx][id_col].tolist()
    return train_ids, val_ids


def _val_split_survival_or_regression(
    df_trainval: pd.DataFrame,
    id_col: str,
    val_frac: float,
    seed: int,
    stratify_cols: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Survival/Regression:
    - If stratify_cols provided: stratify val split by clinical tuple.
    - Else: random (original behavior).
    """
    if val_frac <= 0:
        return df_trainval[id_col].tolist(), []

    if stratify_cols:
        strata_y = _make_joint_strata_labels(df_trainval, label_col=None, stratify_cols=stratify_cols)
        return _stratified_val_split_generic(df_trainval, id_col, val_frac, seed, strata_y)

    return _random_val_split(df_trainval=df_trainval, id_col=id_col, val_frac=val_frac, seed=seed)


def main():
    parser = argparse.ArgumentParser("Create LOO split CSVs (create_splits_loo.py)")

    parser.add_argument("--dataset_csv", type=str, required=True,
                        help="Path to dataset CSV (your dataset_csv/<task>.csv).")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory to write splits_*.csv into.")
    parser.add_argument("--loo_col", type=str, required=True,
                        help="Column name to leave-one-group-out on (e.g. 'source' or 'institution').")

    parser.add_argument("--id_col", type=str, default="slide_id",
                        help="Which ID to write into splits files. Common: slide_id or case_id.")
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["classification", "survival", "regression"],
                        help="Controls how val split is chosen (stratified for classification).")

    parser.add_argument("--label_col", type=str, default="label",
                        help="Classification label column (only used if task_type=classification).")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of remaining (non-test) to use as VAL within each fold.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for val selection.")
    parser.add_argument("--dropna_loo", action="store_true", default=True,
                        help="Drop rows where loo_col is NaN/empty. (Default True)")

    # NEW: optional stratification for the TRAIN/VAL split within each fold
    parser.add_argument("--stratify_cols", type=str, default="",
                        help="Optional: comma-separated clinical column(s) to stratify TRAIN/VAL split by "
                             "(e.g., 'institution' or 'institution,sex'). "
                             "For classification, stratification becomes (label + these cols). "
                             "For survival/regression, stratification uses these cols only.")

    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    _write_settings(args.out_dir, args)

    df = pd.read_csv(args.dataset_csv)

    if args.id_col not in df.columns:
        raise ValueError(f"--id_col '{args.id_col}' not found in CSV columns: {list(df.columns)}")
    if args.loo_col not in df.columns:
        raise ValueError(f"--loo_col '{args.loo_col}' not found in CSV columns: {list(df.columns)}")

    stratify_cols = _parse_list(args.stratify_cols)
    stratify_cols = _validate_stratify_cols(df, stratify_cols)

    # Clean group column
    g = df[args.loo_col]
    # treat empty strings as NaN
    g = g.replace(r"^\s*$", np.nan, regex=True)
    df = df.copy()
    df[args.loo_col] = g

    if args.dropna_loo:
        df = df.dropna(subset=[args.loo_col]).reset_index(drop=True)

    # Make groups stable order (sorted string repr)
    group_vals = sorted(df[args.loo_col].astype(str).unique().tolist())
    if len(group_vals) < 2:
        raise ValueError(
            f"Need at least 2 unique values in loo_col='{args.loo_col}'. "
            f"Found: {group_vals}"
        )

    # fold mapping
    fold_map_rows = []

    for fold, held_out in enumerate(group_vals):
        df_test = df[df[args.loo_col].astype(str) == str(held_out)].copy()
        df_trainval = df[df[args.loo_col].astype(str) != str(held_out)].copy()

        test_ids = df_test[args.id_col].astype(str).tolist()

        # Choose val
        seed_fold = int(args.seed) + int(fold)

        if args.task_type == "classification":
            if args.label_col not in df_trainval.columns:
                raise ValueError(
                    f"task_type=classification but label_col '{args.label_col}' not found in CSV."
                )
            train_ids, val_ids = _stratified_val_split_classification(
                df_trainval=df_trainval,
                id_col=args.id_col,
                val_frac=float(args.val_frac),
                seed=seed_fold,
                label_col=args.label_col,
                stratify_cols=stratify_cols,
            )
        else:
            train_ids, val_ids = _val_split_survival_or_regression(
                df_trainval=df_trainval,
                id_col=args.id_col,
                val_frac=float(args.val_frac),
                seed=seed_fold,
                stratify_cols=stratify_cols,
            )

        # Write split CSV
        out_csv = os.path.join(args.out_dir, f"splits_{fold}.csv")
        split_df = _pad_columns(train_ids, val_ids, test_ids)
        split_df.to_csv(out_csv, index=False)

        fold_map_rows.append({
            "fold": fold,
            "held_out_value": str(held_out),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        })

        extra = ""
        if stratify_cols:
            extra = f" | val_stratify_cols={','.join(stratify_cols)}"
        print(
            f"[fold {fold:03d}] held_out {args.loo_col}={held_out} | "
            f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} -> {out_csv}{extra}"
        )

    fold_map = pd.DataFrame(fold_map_rows)
    fold_map_path = os.path.join(args.out_dir, "fold_to_group.csv")
    fold_map.to_csv(fold_map_path, index=False)

    print(f"\nWrote fold mapping -> {fold_map_path}")
    print(f"Done. Total folds: {len(group_vals)}")


if __name__ == "__main__":
    main()