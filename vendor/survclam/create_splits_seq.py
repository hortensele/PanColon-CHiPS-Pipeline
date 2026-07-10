#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
import pandas as pd

from dataset_modules.dataset_generic import Generic_WSIFamilyDataset


# ----------------------------
# helpers
# ----------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _safe_token(s: str) -> str:
    s = str(s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def _parse_list(csv_str):
    if csv_str is None or str(csv_str).strip() == "":
        return []
    return [c.strip() for c in str(csv_str).split(",") if c.strip()]


def _build_label_dict(label_map_str):
    if label_map_str is None or str(label_map_str).strip() == "":
        return {}
    out = {}
    for pair in str(label_map_str).split(","):
        k, v = pair.split(":")
        out[k.strip()] = int(v.strip())
    return out


def _parse_ignore(ignore_str):
    if ignore_str is None or str(ignore_str).strip() == "":
        return []
    return [tok.strip() for tok in str(ignore_str).split(",") if tok.strip()]


def make_run_signature_simple(task_type: str, time_col: str = None, covariate_cols=None, bag_level: str = "slide") -> str:
    covariate_cols = covariate_cols or []
    cov_tag = (
        "no_covariates"
        if len(covariate_cols) == 0
        else "_".join(_safe_token(c) for c in covariate_cols) + "_covariates"
    )
    level_tag = "patient_level" if bag_level == "patient" else "slide_level"

    if task_type == "survival":
        if time_col is None or str(time_col).strip() == "":
            raise ValueError("time_col is required for survival run signature")
        mid = _safe_token(time_col)
    elif task_type == "classification":
        mid = "classification"
    elif task_type == "regression":
        mid = "regression"
    else:
        mid = _safe_token(task_type)

    return f"{_safe_token(task_type)}__{mid}__{cov_tag}__{level_tag}"


def _pad_columns(train_ids, val_ids, test_ids):
    n = max(len(train_ids), len(val_ids), len(test_ids), 1)

    def pad(x):
        x = list(map(str, x))
        if len(x) < n:
            x = x + [""] * (n - len(x))
        return x

    return pd.DataFrame({"train": pad(train_ids), "val": pad(val_ids), "test": pad(test_ids)})


def _bool_style_df(train_ids, val_ids, test_ids):
    all_ids = sorted(set(list(train_ids) + list(val_ids) + list(test_ids)))
    df = pd.DataFrame(False, index=all_ids, columns=["train", "val", "test"])
    df.loc[list(train_ids), "train"] = True
    df.loc[list(val_ids), "val"] = True
    df.loc[list(test_ids), "test"] = True
    return df.reset_index().rename(columns={"index": "bag_id"})


def _sample_val_patients_stratified(patient_cls_ids, val_frac, seed):
    """
    patient_cls_ids: list of arrays/lists of patient indices per stratum
    Returns: (train_pat_idxs, val_pat_idxs)
    """
    rng = np.random.default_rng(seed)

    val_pat = []
    train_pat = []

    for ids in patient_cls_ids:
        ids = np.asarray(ids, dtype=int)
        if ids.size == 0:
            continue

        rng.shuffle(ids)

        n_val = int(round(val_frac * ids.size))
        # keep it sane: if stratum has >=2, ensure at least 1 train remains
        if ids.size >= 2:
            n_val = min(max(n_val, 1 if val_frac > 0 else 0), ids.size - 1)
        else:
            # only one patient in stratum -> must go to train
            n_val = 0

        val_ids = ids[:n_val]
        tr_ids = ids[n_val:]

        val_pat.extend(val_ids.tolist())
        train_pat.extend(tr_ids.tolist())

    # dedupe + sort
    val_pat = sorted(set(val_pat))
    train_pat = sorted(set(train_pat))

    return train_pat, val_pat


def _normalize_strat_value(x) -> str:
    """
    Convert a clinical stratifier value to a stable string token.
    - NaN/None/empty -> "na"
    - numbers -> canonical string
    - everything else -> stripped string
    """
    if x is None:
        return "na"
    # pandas missing
    try:
        if pd.isna(x):
            return "na"
    except Exception:
        pass
    s = str(x).strip()
    if s == "":
        return "na"
    return s


def _build_combined_patient_strata(patient_cls_ids, patient_df: pd.DataFrame, stratify_cols):
    """
    Combine existing patient strata (patient_cls_ids) with 1+ clinical stratify columns.

    patient_cls_ids: list[list[int]] giving patient indices per base stratum
    patient_df: dataset.patient_data (must align with patient indices)
    stratify_cols: list[str] clinical columns in patient_df

    Returns: list[np.ndarray] patient indices per combined stratum
    """
    n_pat = len(patient_df)
    if n_pat == 0:
        raise RuntimeError("patient_data is empty; cannot build combined strata.")

    # Validate columns
    missing = [c for c in stratify_cols if c not in patient_df.columns]
    if missing:
        msg = (
            f"Missing stratify_cols in dataset.patient_data: {missing}. "
            f"Available columns: {list(patient_df.columns)}"
        )
        print(f"[WARN] {msg} -> proceeding with available columns only.")
        stratify_cols = [c for c in stratify_cols if c in patient_df.columns]

    if len(stratify_cols) == 0:
        # nothing to do
        return [np.asarray(ids, dtype=int) for ids in patient_cls_ids]

    # Map each patient index -> base stratum id (from patient_cls_ids)
    base = np.full(n_pat, -1, dtype=int)
    for s, ids in enumerate(patient_cls_ids):
        ids = np.asarray(ids, dtype=int)
        if ids.size == 0:
            continue
        base[ids] = s

    # If any patients are not assigned, fall back to their own stratum id to avoid losing them
    unassigned = np.where(base < 0)[0]
    if unassigned.size > 0:
        # put each unassigned in a new unique base id bucket
        start = (base.max() + 1) if base.max() >= 0 else 0
        for k, idx in enumerate(unassigned.tolist()):
            base[idx] = start + k
        print(f"[WARN] {unassigned.size} patients were not in patient_cls_ids; assigned unique base strata.")

    # Build combined key -> indices
    buckets = {}
    for i in range(n_pat):
        clin_tuple = tuple(_normalize_strat_value(patient_df.iloc[i][c]) for c in stratify_cols)
        key = (int(base[i]),) + clin_tuple
        buckets.setdefault(key, []).append(i)

    combined = [np.asarray(v, dtype=int) for v in buckets.values() if len(v) > 0]

    # Some light reporting
    sizes = sorted([len(v) for v in combined], reverse=True)
    n_small = sum(1 for s in sizes if s < 2)
    print(
        f"[INFO] Built {len(combined)} combined strata from base_strata={int(base.max()+1)} "
        f"and stratify_cols={stratify_cols}. "
        f"Strata sizes (top 10)={sizes[:10]}; singleton_strata={n_small}."
    )
    return combined


def _holdout_stratified(ids, frac, rng):
    """
    Shuffle `ids` and peel off ~frac as a held-out subset, stratum-wise.
    Returns (held, rest) as int arrays. Guarantees at least 1 stays in `rest`
    when the stratum has >=2 members and frac>0.
    """
    ids = np.asarray(ids, dtype=int).copy()
    if ids.size == 0:
        return ids[:0], ids[:0]
    rng.shuffle(ids)
    n = ids.size
    n_hold = int(round(float(frac) * n))
    if n >= 2:
        n_hold = min(max(n_hold, 1 if frac > 0 else 0), n - 1)
    else:
        n_hold = 0
    return ids[:n_hold], ids[n_hold:]


def _quantile_bin_strata(targets, n_bins):
    """
    Build quantile-based strata (patient indices) from continuous regression
    targets so k-fold splits are balanced across the outcome distribution.
    Patients with non-finite targets are dropped from all strata.
    """
    targets = np.asarray(targets, dtype=float)
    idx = np.arange(targets.size)
    valid = np.isfinite(targets)
    vals = targets[valid]
    if vals.size == 0:
        return [idx]
    uniq = np.unique(vals)
    eff = int(max(1, min(int(n_bins), uniq.size)))
    if eff <= 1:
        return [idx[valid]]
    cut = np.quantile(vals, np.linspace(0.0, 1.0, eff + 1))
    interior = cut[1:-1]
    bin_ids = np.digitize(targets, interior, right=False)  # 0..eff-1
    strata = []
    for b in range(eff):
        sel = idx[valid & (bin_ids == b)]
        if sel.size > 0:
            strata.append(sel)
    return strata


def _build_all_strata(dataset, task_type, stratify_cols, patient_df, reg_bins):
    """Return list of np.ndarray patient-index strata for any task_type."""
    if task_type in ("classification", "survival"):
        if not hasattr(dataset, "patient_cls_ids") or dataset.patient_cls_ids is None:
            raise RuntimeError("Expected dataset.patient_cls_ids for patient-level stratification.")
        if len(stratify_cols) > 0:
            return _build_combined_patient_strata(
                patient_cls_ids=dataset.patient_cls_ids,
                patient_df=patient_df,
                stratify_cols=stratify_cols,
            )
        return [np.asarray(ids, dtype=int) for ids in dataset.patient_cls_ids if len(ids) > 0]
    elif task_type == "regression":
        if "target" not in patient_df.columns:
            raise RuntimeError("Expected patient-level 'target' column for regression strata.")
        return _quantile_bin_strata(patient_df["target"].values, reg_bins)
    raise ValueError(f"Unsupported task_type: {task_type}")


def _write_split_files(split_dir, fold, train_ids, val_ids, test_ids):
    out_csv = os.path.join(split_dir, f"splits_{fold}.csv")
    out_bool = os.path.join(split_dir, f"splits_{fold}_bool.csv")
    _pad_columns(train_ids, val_ids, test_ids).to_csv(out_csv, index=False)
    _bool_style_df(train_ids, val_ids, test_ids).to_csv(out_bool, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser("Create splits: k=1 final split or k>1 stratified CV (fixed test set)")

    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--clinical_csv", type=str, required=True)
    ap.add_argument("--runs_root", type=str, required=True)

    ap.add_argument("--task_type", type=str, required=True, choices=["classification", "survival", "regression"])
    ap.add_argument("--bag_level", type=str, default="slide", choices=["slide", "patient"])
    ap.add_argument("--slide_level_split", action="store_true", default=False)
    ap.add_argument("--patient_voting", type=str, default="max", choices=["max", "maj"])

    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--label_map", type=str, default="")
    ap.add_argument("--ignore_labels", type=str, default="")

    ap.add_argument("--time_col", type=str, default="time")
    ap.add_argument("--event_col", type=str, default="event")
    ap.add_argument("--target_col", type=str, default="target")

    ap.add_argument("--covariate_cols", type=str, default="")
    ap.add_argument("--reg_bins", type=int, default=4)

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_frac", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)

    # NEW: optional clinical stratification (patient-level)
    ap.add_argument(
        "--stratify_cols",
        type=str,
        default="",
        help="Optional: comma-separated clinical column(s) to stratify splits by "
             "(e.g., 'institution' or 'institution,sex'). "
             "For classification/survival this is combined with the existing label/event strata.",
    )

    # important for your new dataset_generic.py
    ap.add_argument(
        "--pt_id_col",
        type=str,
        default="slide_id",
        help="Column that matches .pt filename stem (not used for splitting, but validated).",
    )

    args = ap.parse_args()

    if not os.path.isfile(args.clinical_csv):
        raise FileNotFoundError(f"Missing clinical_csv: {args.clinical_csv}")

    covariate_cols = _parse_list(args.covariate_cols)
    stratify_cols = _parse_list(args.stratify_cols)

    label_dict = _build_label_dict(args.label_map)
    ignore_list = _parse_ignore(args.ignore_labels)

    # splitting semantics
    if args.bag_level == "patient":
        effective_patient_strat = True
    else:
        effective_patient_strat = (not args.slide_level_split)

    dataset = Generic_WSIFamilyDataset(
        csv_path=args.clinical_csv,
        task_type=args.task_type,
        label_col=args.label_col,
        label_dict=(label_dict if args.task_type == "classification" else {}),
        ignore=ignore_list,
        time_col=args.time_col,
        event_col=args.event_col,
        target_col=args.target_col,
        covariate_cols=covariate_cols,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        patient_strat=effective_patient_strat,
        patient_voting=args.patient_voting,
        bag_level=args.bag_level,
        reg_bins=args.reg_bins,
        pt_id_col=args.pt_id_col,
    )

    if int(args.k) < 1:
        raise ValueError(f"--k must be >= 1, got {args.k}.")

    if args.bag_level != "patient":
        raise ValueError("This splitter currently supports patient-level splits. Please run with --bag_level patient.")

    # run signature (same as your training code)
    run_sig = make_run_signature_simple(
        task_type=args.task_type,
        time_col=(args.time_col if args.task_type == "survival" else None),
        covariate_cols=covariate_cols,
        bag_level=args.bag_level,
    )

    # output directory: label_frac_XXX convention
    lf_pct = int(round(float(args.label_frac) * 100))
    split_dir = os.path.join(
        args.runs_root,
        args.dataset_name,
        run_sig,
        "splits",
        f"label_frac_{lf_pct:03d}",
    )
    _ensure_dir(split_dir)

    # patient-level metadata (indices align with strata)
    patient_df = dataset.patient_data if isinstance(dataset.patient_data, pd.DataFrame) else pd.DataFrame(dataset.patient_data)
    case_ids = np.asarray(patient_df["case_id"]).astype(str)

    folds_written = []
    fold_descs = []

    if int(args.k) == 1:
        # -------------- single FINAL split (k=1, no test set) --------------
        if float(args.test_frac) != 0.0:
            raise ValueError("For --k 1 (final split), please run with --test_frac 0.0.")

        if args.task_type in ("classification", "survival"):
            patient_cls_ids = dataset.patient_cls_ids
            if len(stratify_cols) > 0:
                patient_cls_ids = _build_combined_patient_strata(
                    patient_cls_ids=patient_cls_ids,
                    patient_df=patient_df,
                    stratify_cols=stratify_cols,
                )
            train_pat, val_pat = _sample_val_patients_stratified(
                patient_cls_ids=patient_cls_ids,
                val_frac=float(args.val_frac),
                seed=int(args.seed),
            )
        elif args.task_type == "regression":
            rng = np.random.default_rng(int(args.seed))
            all_pat = np.arange(len(patient_df["case_id"]), dtype=int)
            rng.shuffle(all_pat)
            n_val = int(round(float(args.val_frac) * all_pat.size))
            if all_pat.size >= 2:
                n_val = min(max(n_val, 1 if args.val_frac > 0 else 0), all_pat.size - 1)
            else:
                n_val = 0
            val_pat = sorted(all_pat[:n_val].tolist())
            train_pat = sorted(all_pat[n_val:].tolist())
        else:
            raise ValueError(f"Unsupported task_type: {args.task_type}")

        if len(train_pat) == 0:
            raise RuntimeError("Train split ended up empty. Decrease val_frac or check strata sizes.")

        train_ids = [case_ids[i] for i in train_pat]
        val_ids = [case_ids[i] for i in val_pat]
        out_csv = _write_split_files(split_dir, 0, train_ids, val_ids, [])
        folds_written.append(out_csv)
        fold_descs.append({
            "fold": 0, "k": 1, "bag_level": args.bag_level,
            "patient_strat": effective_patient_strat,
            "stratify_cols": ",".join(stratify_cols) if stratify_cols else "",
            "n_train_patients": len(train_pat), "n_val_patients": len(val_pat),
            "n_test_patients": 0, "val_frac": float(args.val_frac),
            "test_frac": float(args.test_frac), "seed": int(args.seed),
        })

    else:
        # -------------- k-fold stratified CV with a FIXED test set --------------
        # Strata are built once; the test set is held out once (fixed across folds);
        # each fold draws a different stratified val partition from the remaining pool.
        strata = _build_all_strata(dataset, args.task_type, stratify_cols, patient_df, args.reg_bins)

        rng_test = np.random.default_rng(int(args.seed))
        test_pat = []
        pool_per_stratum = []
        for st in strata:
            held, rest = _holdout_stratified(st, float(args.test_frac), rng_test)
            test_pat.extend(held.tolist())
            pool_per_stratum.append(rest)
        test_ids = [case_ids[i] for i in sorted(test_pat)]

        for fold in range(int(args.k)):
            rng_val = np.random.default_rng(int(args.seed) + 1000 * (fold + 1))
            train_pat, val_pat = [], []
            for rest in pool_per_stratum:
                held_val, tr = _holdout_stratified(rest, float(args.val_frac), rng_val)
                val_pat.extend(held_val.tolist())
                train_pat.extend(tr.tolist())

            if len(train_pat) == 0:
                raise RuntimeError(
                    f"Fold {fold}: train split empty. Decrease val_frac/test_frac or check strata sizes."
                )

            train_ids = [case_ids[i] for i in sorted(train_pat)]
            val_ids = [case_ids[i] for i in sorted(val_pat)]
            out_csv = _write_split_files(split_dir, fold, train_ids, val_ids, test_ids)
            folds_written.append(out_csv)
            fold_descs.append({
                "fold": fold, "k": int(args.k), "bag_level": args.bag_level,
                "patient_strat": effective_patient_strat,
                "stratify_cols": ",".join(stratify_cols) if stratify_cols else "",
                "n_train_patients": len(train_pat), "n_val_patients": len(val_pat),
                "n_test_patients": len(test_pat), "val_frac": float(args.val_frac),
                "test_frac": float(args.test_frac), "seed": int(args.seed),
            })

    # shared descriptor + run_config.json for reproducibility
    pd.DataFrame(fold_descs).to_csv(os.path.join(split_dir, "splits_descriptor.csv"), index=False)

    run_base = os.path.join(args.runs_root, args.dataset_name, run_sig)
    _ensure_dir(run_base)
    run_cfg = {
        "dataset_name": args.dataset_name,
        "clinical_csv": args.clinical_csv,
        "task_type": args.task_type,
        "bag_level": args.bag_level,
        "slide_level_split": bool(args.slide_level_split),
        "patient_voting": args.patient_voting,
        "pt_id_col": args.pt_id_col,
        "time_col": args.time_col,
        "event_col": args.event_col,
        "target_col": args.target_col,
        "covariate_cols": covariate_cols,
        "k": int(args.k),
        "seed": int(args.seed),
        "label_frac": float(args.label_frac),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "stratify_cols": stratify_cols,
        "reg_bins": int(args.reg_bins),
    }
    with open(os.path.join(run_base, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2, sort_keys=True)

    print(f"[OK] Wrote {len(folds_written)} split file(s) to: {split_dir}")
    for p in folds_written:
        print(f"     - {os.path.basename(p)}")
    if int(args.k) > 1:
        print(f"[OK] k-fold CV: k={args.k}, fixed test_frac={args.test_frac}, val_frac={args.val_frac}")
    if len(stratify_cols) > 0:
        print(f"[OK] Clinical stratification enabled on: {stratify_cols}")


if __name__ == "__main__":
    main()