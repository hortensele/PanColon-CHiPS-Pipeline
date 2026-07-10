#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import hashlib
import numpy as np
import pandas as pd

# internal imports
from utils.utils import seed_torch
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSIFamilyDataset

# PCA utilities (fold-wise IncrementalPCA)
from utils.pca_utils import fit_and_save_fold_pca, wrap_dataset_with_pca, load_pca_npz


###############################################################################
# helpers
###############################################################################

def _build_label_dict(label_map_str):
    """
    label_map_str: e.g. "LGG:0,GBM:1" or "" if not classification.
    Returns dict {raw_label: int_label}.
    """
    if label_map_str is None or label_map_str.strip() == "":
        return {}
    out = {}
    pairs = label_map_str.split(",")
    for p in pairs:
        k, v = p.split(":")
        out[k.strip()] = int(v.strip())
    return out


def _parse_ignore(ignore_str):
    """
    ignore_str: e.g. "classX,classY" or "".
    Returns list of label names to ignore/drop.
    """
    if ignore_str is None or ignore_str.strip() == "":
        return []
    return [tok.strip() for tok in ignore_str.split(",") if tok.strip()]


def _parse_list(csv_str):
    """
    Parses "age,sex,stage" -> ["age","sex","stage"].
    Returns [] if empty.
    """
    if csv_str is None or csv_str.strip() == "":
        return []
    return [c.strip() for c in csv_str.split(",") if c.strip()]


def _safe_token(s: str) -> str:
    s = str(s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def make_run_signature_simple(task_type: str,
                              time_col: str = None,
                              covariate_cols=None,
                              bag_level: str = "slide",
                              domain_adapt: bool = False,
                              domain_level: str = "bag") -> str:
    """
    Your requested format:
      survival__os_event_data__age_sex_covariates__patient_level
    + optional domain tag appended:
      ...__domain_bag   / ...__domain_tile / ...__domain_both
    """
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

    sig = f"{_safe_token(task_type)}__{mid}__{cov_tag}__{level_tag}"
    return sig


def resolve_run_dirs(runs_root: str,
                     dataset_name: str,
                     run_sig: str,
                     label_frac: float,
                     exp_code: str,
                     seed: int):
    """
    Layout:
      {runs_root}/{dataset_name}/{run_sig}/
        run_config.json
        splits/label_frac_100/splits_0.csv ...
        results/{exp_code}_s{seed}/
        eval/{exp_code}_s{seed}/
    """
    base = os.path.join(runs_root, dataset_name, run_sig)
    frac_tag = int(round(label_frac * 100))

    split_dir = os.path.join(base, "splits", f"label_frac_{frac_tag:03d}")
    results_dir = os.path.join(base, "results", f"{exp_code}_s{seed}")
    eval_dir = os.path.join(base, "eval", f"{exp_code}_s{seed}")

    return base, split_dir, results_dir, eval_dir


def resolve_feature_leaf_dir(features_root: str,
                             dataset_name: str,
                             feature_key: str):
    """
    Feature-store layout (RAW pt only):
      {features_root}/{dataset_name}/{feature_key}/pt_files
    Returns the LEAF directory containing *.pt.
    """
    return os.path.join(features_root, dataset_name, feature_key, "pt_files")


def _metric_column_names(task_type: str):
    if task_type == 'survival':
        return ('train_cindex', 'val_cindex', 'test_cindex')
    elif task_type == 'classification':
        return ('train_auc', 'val_auc', 'test_auc')
    else:
        return ('train_r2', 'val_r2', 'test_r2')


def _read_fold_metrics(results_dir: str, fold_idx: int, task_type: str):
    metrics_path = os.path.join(results_dir, f"fold_{fold_idx}_metrics.csv")
    if not os.path.exists(metrics_path):
        return (float('nan'), float('nan'), float('nan'))

    df = pd.read_csv(metrics_path)

    if task_type == 'survival':
        metric_col = 'cindex'
    elif task_type == 'classification':
        metric_col = 'auc'
    else:
        metric_col = 'r2'

    def _get_metric(split_name):
        row = df[df['split'] == split_name]
        if len(row) == 0 or metric_col not in row.columns:
            return float('nan')
        return float(row[metric_col].values[0])

    return (_get_metric('train'), _get_metric('val'), _get_metric('test'))


def _save_train_predictions_subset(results_dir: str, fold_idx: int):
    pred_path = os.path.join(results_dir, f"fold_{fold_idx}_predictions.csv")
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path)
    if 'split' not in df.columns:
        return

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    out_path = os.path.join(results_dir, f"fold_{fold_idx}_train_predictions.csv")
    train_df.to_csv(out_path, index=False)


def _load_best_alpha_from_disk(results_dir: str):
    p = os.path.join(results_dir, "best_alpha.txt")
    if os.path.exists(p):
        with open(p, "r") as f:
            return float(f.read().strip())
    return None


def make_pca_signature(task_type: str,
                       time_col: str = None,
                       covariate_cols=None,
                       bag_level: str = "slide") -> str:
    """
    PCA cache signature format you want:
      survival_os_event_data__no_covariates__patient_level
    (note the single underscore between task_type and time_col)
    """
    covariate_cols = covariate_cols or []
    cov_tag = (
        "no_covariates"
        if len(covariate_cols) == 0
        else "_".join(_safe_token(c) for c in covariate_cols) + "_covariates"
    )
    level_tag = "patient_level" if bag_level == "patient" else "slide_level"

    if task_type == "survival":
        if time_col is None or str(time_col).strip() == "":
            raise ValueError("time_col is required for survival PCA signature")
        mid = _safe_token(time_col)
        return f"{_safe_token(task_type)}_{mid}__{cov_tag}__{level_tag}"

    if task_type == "classification":
        return f"classification__{cov_tag}__{level_tag}"

    if task_type == "regression":
        return f"regression__{cov_tag}__{level_tag}"

    return f"{_safe_token(task_type)}__{cov_tag}__{level_tag}"


###############################################################################
# Domain-adversarial (institution) helpers
###############################################################################

def _infer_num_institutions_from_csv(csv_path: str, institution_col: str) -> int:
    if institution_col is None or institution_col.strip() == "":
        return 0
    df = pd.read_csv(csv_path, usecols=[institution_col])
    vals = df[institution_col].dropna().astype(str).unique().tolist()
    return int(len(vals))


def _split_csv_values(split_csv_path: str):
    """Return a flat list of all non-null entries from splits_{k}.csv."""
    df = pd.read_csv(split_csv_path)
    vals = []
    for c in df.columns:
        col = df[c].dropna().tolist()
        vals.extend(col)
    out = []
    for v in vals:
        s = str(v).strip()
        if s != "" and s.lower() != "nan":
            out.append(s)
    return out


def _infer_from_id_for_split(dataset, split_csv_path: str, pt_id_col: str) -> bool:
    vals = _split_csv_values(split_csv_path)
    if len(vals) == 0:
        return True

    if any(not re.fullmatch(r"-?\d+", v) for v in vals[: min(2000, len(vals))]):
        return True

    int_vals = [int(v) for v in vals[: min(2000, len(vals))]]
    mn, mx = min(int_vals), max(int_vals)
    if mn >= 0 and mx < len(dataset):
        return False

    return True


def _copy_id_metadata(dst, src):
    """
    Some PCA wrappers accidentally drop metadata. This copies common ID fields over.
    Safe no-ops if attributes don't exist.
    """
    for attr in [
        "slide_data", "patient_data", "patient_strat", "patient_voting",
        "bag_level", "pt_id_col", "case_id_col", "slide_ids", "case_ids"
    ]:
        if hasattr(src, attr) and not hasattr(dst, attr):
            try:
                setattr(dst, attr, getattr(src, attr))
            except Exception:
                pass
    return dst


###############################################################################
# PCA cache helpers
###############################################################################

def _sha1_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fingerprint_splits_dir(split_dir: str, max_files: int = 5000) -> str:
    """
    Hash contents of splits_*.csv to make PCA cache safe across different split configs.
    """
    if split_dir is None or not os.path.isdir(split_dir):
        return "no_splits"

    files = sorted(
        fn for fn in os.listdir(split_dir)
        if fn.startswith("splits_") and fn.endswith(".csv")
    )[:max_files]

    h = hashlib.sha1()
    h.update(os.path.abspath(split_dir).encode("utf-8"))
    for fn in files:
        p = os.path.join(split_dir, fn)
        h.update(fn.encode("utf-8"))
        try:
            h.update(_sha1_file(p).encode("utf-8"))
        except Exception:
            st = os.stat(p)
            h.update(str(st.st_size).encode("utf-8"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))

    return h.hexdigest()[:12]


def _fingerprint_features_dir(feature_dir_leaf: str, feature_key: str = "") -> str:
    """
    Fingerprint based on the leaf feature directory (and feature_key, if provided).
    This keeps cache stable across results_dir changes but different feature sets.
    """
    leaf = os.path.abspath(str(feature_dir_leaf))
    key = str(feature_key or "").strip()

    h = hashlib.sha1()
    h.update(leaf.encode("utf-8"))
    h.update(key.encode("utf-8"))
    return h.hexdigest()[:12]


def resolve_pca_cache_dir(args) -> str:
    """
    Save PCA models under:
      <pca_cache_root>/<dataset>/<feature_key>/<pca_sig>__label_frac_XXX/ipca_k{K}_max{max}
    If --pca_cache_root is empty: default legacy <results_dir>/pca_models.
    """
    if getattr(args, "pca_cache_root", None) and str(args.pca_cache_root).strip() != "":
        root = os.path.abspath(str(args.pca_cache_root).strip())

        if args.feature_key is None or str(args.feature_key).strip() == "":
            raise ValueError(
                "For semantic PCA cache paths, you must provide --feature_key "
                "(e.g., TITAN_20x or univ2_20x)."
            )

        covariate_cols = _parse_list(getattr(args, "covariate_cols", ""))
        pca_sig = make_pca_signature(
            task_type=args.task_type,
            time_col=(args.time_col if args.task_type == "survival" else None),
            covariate_cols=covariate_cols,
            bag_level=args.bag_level,
        )

        frac_tag = int(round(float(args.label_frac) * 100))
        frac_dir = f"{pca_sig}__label_frac_{frac_tag:03d}"

        pca_k = int(args.pca_k)
        max_rows_tag = "none" if args.pca_max_rows in [None, 0] else str(int(args.pca_max_rows))
        subdir = f"ipca_k{pca_k}_max{max_rows_tag}"

        return os.path.join(
            root,
            args.dataset_name,
            str(args.feature_key).strip(),
            frac_dir,
            subdir
        )

    return os.path.join(args.results_dir, "pca_models")


###############################################################################
# PCA: fold-specific fit on TRAIN split and wrap (optional)
###############################################################################

def apply_fold_pca(args, fold_idx: int, train_split, val_split, test_split):
    if args.pca_k is None or int(args.pca_k) <= 0:
        return train_split, val_split, test_split

    pca_k = int(args.pca_k)

    pca_dir = resolve_pca_cache_dir(args)
    os.makedirs(pca_dir, exist_ok=True)

    pca_path = os.path.join(pca_dir, f"fold_{fold_idx}_ipca_k{pca_k}.npz")

    if os.path.exists(pca_path) and (not args.refit_pca):
        print(f"[PCA] Loading existing PCA: {pca_path}")
        pca = load_pca_npz(pca_path)
    else:
        print(f"[PCA] Fitting fold PCA on TRAIN only: fold={fold_idx} k={pca_k}")
        print(f"[PCA] Saving PCA to: {pca_path}")
        pca = fit_and_save_fold_pca(
            train_dataset=train_split,
            out_npz=pca_path,
            n_components=pca_k,
            batch_rows=int(args.pca_batch_rows),
            num_workers=0,
            max_total_rows=(None if args.pca_max_rows in [None, 0] else int(args.pca_max_rows)),
            seed=int(args.seed),
            verbose=True,
        )

    train_orig, val_orig, test_orig = train_split, val_split, test_split

    train_split = wrap_dataset_with_pca(train_split, pca)
    val_split   = wrap_dataset_with_pca(val_split, pca)
    test_split  = wrap_dataset_with_pca(test_split, pca) if test_split is not None else None

    train_split = _copy_id_metadata(train_split, train_orig)
    val_split   = _copy_id_metadata(val_split, val_orig)
    if test_split is not None:
        test_split = _copy_id_metadata(test_split, test_orig)

    return train_split, val_split, test_split


###############################################################################
# cross-fold training / alpha sweep
###############################################################################

def run_training_over_folds(args, dataset):
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)

    # ---------------------------
    # skip sweep if requested
    # ---------------------------
    # The alpha sweep only applies to survival (cox-additive alpha). For
    # classification/regression there is a single fixed alpha, so sweeping just
    # trains every fold an extra time for nothing -> skip straight to the single
    # training pass below.
    run_sweep = (args.task_type == "survival") and not getattr(args, "retrain_only", False)
    if not run_sweep:
        if args.task_type == "survival" and getattr(args, "retrain_only", False):
            disk_alpha = _load_best_alpha_from_disk(args.results_dir)
            if disk_alpha is not None:
                args.alpha = disk_alpha
            print(f"[Retrain-only] Skipping alpha sweep. Using alpha={args.alpha}")
        else:
            print(f"[No sweep] task_type={args.task_type}: single alpha={getattr(args, 'alpha', 0.0)}; skipping sweep.")
        overall_best_alpha = getattr(args, "alpha", 0.0)
        overall_best_mean_val = float("nan")
    else:
        alpha_list = [1e-5, 1e-4, 1e-3, 1e-2]

        overall_best_alpha = None
        overall_best_mean_val = -np.inf

        for alpha in alpha_list:
            print(f"\n========== Alpha sweep pass: alpha={alpha} ==========\n")

            args.alpha = alpha

            this_alpha_train_metrics = []
            this_alpha_val_metrics = []
            this_alpha_test_metrics = []

            for fold_idx in folds:
                print(f"\n----- Fold {fold_idx} -----")
                seed_torch(args.seed)

                split_csv_path = os.path.join(args.split_dir, f'splits_{fold_idx}.csv')

                if args.bag_level == "patient":
                    from_id = True
                else:
                    from_id = _infer_from_id_for_split(dataset, split_csv_path, args.pt_id_col)

                print(f"[Splits] fold={fold_idx} using from_id={from_id} (bag_level='{args.bag_level}', pt_id_col='{args.pt_id_col}')")

                train_split, val_split, test_split = dataset.return_splits(
                    from_id=from_id,
                    csv_path=split_csv_path,
                    bag_level=args.bag_level
                )

                train_split, val_split, test_split = apply_fold_pca(
                    args, fold_idx, train_split, val_split, test_split
                )

                print("bag_level:", args.bag_level, "| patient_strat:", getattr(dataset, "patient_strat", None))
                print("split sizes -> train:", len(train_split), "val:", len(val_split), "test:",
                      (len(test_split) if test_split is not None else 0))

                val_metric, test_metric = train(
                    (train_split, val_split, test_split),
                    fold_idx,
                    args
                )

                train_m, val_m_csv, test_m_csv = _read_fold_metrics(
                    args.results_dir,
                    fold_idx,
                    args.task_type
                )

                if not np.isfinite(val_m_csv):
                    val_m_csv = val_metric
                if not np.isfinite(test_m_csv):
                    test_m_csv = test_metric

                this_alpha_train_metrics.append(train_m)
                this_alpha_val_metrics.append(val_m_csv)
                this_alpha_test_metrics.append(test_m_csv)

            mean_val_metric = float(np.nanmean(this_alpha_val_metrics))
            print(f"\nAlpha {alpha}: mean validation main metric across folds = {mean_val_metric:.4f}")

            if mean_val_metric > overall_best_mean_val:
                overall_best_mean_val = mean_val_metric
                overall_best_alpha = alpha

    args.alpha = overall_best_alpha
    print(f"\n*** Best alpha: {overall_best_alpha} with mean val metric {overall_best_mean_val:.4f} ***\n")

    final_train_metrics = []
    final_val_metrics = []
    final_test_metrics = []

    for fold_idx in folds:
        print(f"\n[Retrain best alpha] Fold {fold_idx}")

        split_csv_path = os.path.join(args.split_dir, f'splits_{fold_idx}.csv')

        if args.bag_level == "patient":
            from_id = True
        else:
            from_id = _infer_from_id_for_split(dataset, split_csv_path, args.pt_id_col)

        print(f"[Splits] fold={fold_idx} using from_id={from_id} (bag_level='{args.bag_level}', pt_id_col='{args.pt_id_col}')")

        train_split, val_split, test_split = dataset.return_splits(
            from_id=from_id,
            csv_path=split_csv_path,
            bag_level=args.bag_level
        )

        train_split, val_split, test_split = apply_fold_pca(
            args, fold_idx, train_split, val_split, test_split
        )

        print("bag_level:", args.bag_level, "| patient_strat:", getattr(dataset, "patient_strat", None))
        print("split sizes -> train:", len(train_split), "val:", len(val_split), "test:",
              (len(test_split) if test_split is not None else 0))

        _ = train((train_split, val_split, test_split), fold_idx, args)

        train_m, val_m, test_m = _read_fold_metrics(args.results_dir, fold_idx, args.task_type)
        final_train_metrics.append(train_m)
        final_val_metrics.append(val_m)
        final_test_metrics.append(test_m)

        _save_train_predictions_subset(args.results_dir, fold_idx)

    train_col, val_col, test_col = _metric_column_names(args.task_type)

    summary_df = pd.DataFrame({
        'fold': folds,
        train_col: final_train_metrics,
        val_col: final_val_metrics,
        test_col: final_test_metrics,
    })

    return summary_df, overall_best_alpha


###############################################################################
# main()
###############################################################################

def main(args):
    covariate_cols = _parse_list(args.covariate_cols)

    run_sig = make_run_signature_simple(
        task_type=args.task_type,
        time_col=args.time_col if args.task_type == "survival" else None,
        covariate_cols=covariate_cols,
        bag_level=args.bag_level,
        domain_adapt=bool(getattr(args, "domain_adapt", False)),
        domain_level=str(getattr(args, "domain_level", "bag")),
    )

    run_base, split_dir, results_dir, eval_dir = resolve_run_dirs(
        runs_root=args.runs_root,
        dataset_name=args.dataset_name,
        run_sig=run_sig,
        label_frac=args.label_frac,
        exp_code=args.exp_code,
        seed=args.seed,
    )

    os.makedirs(run_base, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if args.clinical_csv is None or args.clinical_csv.strip() == "":
        raise ValueError("--clinical_csv is required (task-agnostic clinical master).")

    dataset_csv_path = args.clinical_csv
    if not os.path.isfile(dataset_csv_path):
        raise FileNotFoundError(f"Missing clinical_csv: {dataset_csv_path}")

    # Determine patient strat behavior
    if args.bag_level == "patient":
        effective_patient_strat = True
    else:
        effective_patient_strat = (not args.slide_level_split)

    # Domain-adversarial sanity checks
    if bool(getattr(args, "domain_adapt", False)):
        if args.institution_col is None or args.institution_col.strip() == "":
            raise ValueError("--domain_adapt requires --institution_col (column name in clinical CSV).")

        if getattr(args, "num_institutions", 0) in [0, None]:
            args.num_institutions = _infer_num_institutions_from_csv(dataset_csv_path, args.institution_col)

        if int(args.num_institutions) <= 1:
            raise ValueError(
                f"domain_adapt enabled but inferred num_institutions={args.num_institutions}. "
                f"Check --institution_col '{args.institution_col}'."
            )

    dataset = Generic_WSIFamilyDataset(
        csv_path=dataset_csv_path,
        task_type=args.task_type,
        label_col=args.label_col,
        label_dict=_build_label_dict(args.label_map),
        ignore=_parse_ignore(args.ignore_labels),
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
        pt_id_col=args.pt_id_col,

        institution_col=getattr(args, "institution_col", "institution"),
        return_institution=bool(getattr(args, "domain_adapt", False)),
    )

    # attach feature leaf dir
    if args.feature_dir is not None and args.feature_dir.strip() != "":
        dataset.data_dir = args.feature_dir
    else:
        if args.features_root is None or args.features_root.strip() == "":
            raise ValueError("Provide either --feature_dir OR --features_root + --feature_key.")
        if args.feature_key is None or args.feature_key.strip() == "":
            raise ValueError("Provide either --feature_dir OR --features_root + --feature_key.")
        dataset.data_dir = resolve_feature_leaf_dir(
            features_root=args.features_root,
            dataset_name=args.dataset_name,
            feature_key=args.feature_key,
        )

    if not os.path.isdir(dataset.data_dir):
        raise FileNotFoundError(f"Feature dir not found: {dataset.data_dir}")

    if args.bag_level == "patient":
        dataset.filter_patients_with_missing_embeddings(pt_id_col=args.pt_id_col, verbose=True)

    dataset.bag_level = args.bag_level

    # splits dir (must exist)
    args.split_dir = split_dir
    print("split_dir:", args.split_dir)
    assert os.path.isdir(args.split_dir), f"Missing split_dir {args.split_dir}"

    # results dir
    args.results_dir = results_dir
    os.makedirs(args.results_dir, exist_ok=True)

    # ✅ expose feature leaf dir to PCA cache resolver
    args.feature_dir_leaf = dataset.data_dir

    # Optional: print cache destination once (informative)
    if args.pca_k is not None and int(args.pca_k) > 0:
        cache_dir = resolve_pca_cache_dir(args)
        print(f"[PCA] Cache directory resolved to: {cache_dir}")

    # sanity check pt_id_col
    if hasattr(dataset, "slide_data") and len(dataset.slide_data) > 0:
        sample_row = dataset.slide_data.iloc[0]
        if args.pt_id_col not in dataset.slide_data.columns:
            raise KeyError(f"--pt_id_col '{args.pt_id_col}' not found in clinical CSV columns.")
        print(f"[DEBUG] pt_id_col='{args.pt_id_col}' sample value:", str(sample_row[args.pt_id_col]))

    # infer n_classes, cov_dim, alpha defaults
    if args.task_type == 'classification':
        args.n_classes = dataset.num_classes
    else:
        args.n_classes = getattr(args, "n_classes", 1)

    args.cov_dim = len(covariate_cols)

    if args.task_type != 'survival' and not hasattr(args, 'alpha'):
        args.alpha = 0.0

    settings = {
        'dataset_name': args.dataset_name,
        'clinical_csv': dataset_csv_path,
        'run_signature': run_sig,

        'task_type': args.task_type,
        'bag_level': args.bag_level,
        'pt_id_col': args.pt_id_col,
        'slide_level_split': bool(args.slide_level_split),
        'patient_strat': effective_patient_strat,
        'patient_voting': args.patient_voting,

        # Domain-adversarial settings
        'domain_adapt': bool(getattr(args, "domain_adapt", False)),
        'institution_col': getattr(args, "institution_col", None),
        'num_institutions': int(getattr(args, "num_institutions", 0)),
        'domain_level': str(getattr(args, "domain_level", "bag")),
        'domain_lambda_max': float(getattr(args, "domain_lambda_max", 1.0)),
        'domain_warmup_epochs': int(getattr(args, "domain_warmup_epochs", 5)),
        'domain_loss_weight': float(getattr(args, "domain_loss_weight", 1.0)),
        'domain_tile_k': int(getattr(args, "domain_tile_k", 256)),
        'domain_lambda_start_epoch': int(getattr(args, "domain_lambda_start_epoch", 3)),
        'domain_lambda_ramp_epochs': int(getattr(args, "domain_lambda_ramp_epochs", 12)),
        'grad_clip_norm': float(getattr(args, "grad_clip_norm", 1.0)),

        'domain_min_count': int(getattr(args, "domain_min_count", 30)),
        'domain_class_weighting': str(getattr(args, "domain_class_weighting", "sqrt_inv")),
        'domain_label_smoothing': float(getattr(args, "domain_label_smoothing", 0.05)),
        'domain_tile_lambda_mult': float(getattr(args, "domain_tile_lambda_mult", 0.2)),

        'use_plateau': bool(getattr(args, "use_plateau", True)),
        'plateau_factor': float(getattr(args, "plateau_factor", 0.5)),
        'plateau_patience': int(getattr(args, "plateau_patience", 2)),
        'plateau_threshold': float(getattr(args, "plateau_threshold", 1e-4)),
        'plateau_cooldown': int(getattr(args, "plateau_cooldown", 0)),
        'plateau_min_lr': float(getattr(args, "plateau_min_lr", 1e-6)),

        'num_splits': args.k,
        'k_start': args.k_start,
        'k_end': args.k_end,
        'max_epochs': args.max_epochs,

        'results_dir': args.results_dir,
        'split_dir': args.split_dir,

        'lr': args.lr,
        'opt': args.opt,
        'reg': args.reg,
        'label_frac': args.label_frac,
        'seed': args.seed,

        'model_type': args.model_type,
        'model_size': args.model_size,
        'embed_dim': args.embed_dim,
        'drop_out': args.drop_out,
        'bag_loss': args.bag_loss,
        'bag_weight': args.bag_weight,
        'inst_loss': args.inst_loss,
        'B': args.B,
        'subtyping': args.subtyping,
        'weighted_sample': args.weighted_sample,

        'covariate_cols': covariate_cols,
        'cov_dim': args.cov_dim,
        'alpha': args.alpha,
        'cov_fusion': args.cov_fusion,

        'features_root': args.features_root,
        'feature_key': args.feature_key,
        'feature_dir_leaf': dataset.data_dir,

        'pca_k': args.pca_k,
        'pca_batch_rows': int(args.pca_batch_rows),
        'pca_max_rows': (None if args.pca_max_rows in [None, 0] else int(args.pca_max_rows)),
        'refit_pca': bool(args.refit_pca),

        'pca_cache_root': (str(getattr(args, "pca_cache_root", "")).strip() or None),
        'pca_cache_dir_resolved': (resolve_pca_cache_dir(args) if (args.pca_k is not None and int(args.pca_k) > 0) else None),

        # -------------------------
        # NEW: survival extras + EMA
        # -------------------------
        'attn_entropy_lambda': float(getattr(args, "attn_entropy_lambda", 0.0)),
        'rank_lambda': float(getattr(args, "rank_lambda", 0.0)),
        'rank_margin': float(getattr(args, "rank_margin", 0.0)),

        'use_ema': bool(getattr(args, "use_ema", False)),
        'ema_decay': float(getattr(args, "ema_decay", 0.999)),
        'ema_device': (str(getattr(args, "ema_device", "")).strip() or None),
        'use_ema_eval': int(getattr(args, "use_ema_eval", 1)),
        'save_ema_ckpt': int(getattr(args, "save_ema_ckpt", 1)),
    }

    exp_txt = os.path.join(args.results_dir, f"experiment_{args.exp_code}.txt")
    with open(exp_txt, 'w') as f:
        for k, v in settings.items():
            f.write(f"{k}: {v}\n")

    with open(os.path.join(run_base, "run_config.json"), "w") as f:
        json.dump(settings, f, indent=2, sort_keys=True)

    print("################# Settings ###################")
    for key, val in settings.items():
        print(f"{key}: {val}")

    if args.task_type == "survival":
        print(
            f"[Surv extras] attn_entropy_lambda={getattr(args,'attn_entropy_lambda',0.0)} | "
            f"rank_lambda={getattr(args,'rank_lambda',0.0)} (margin={getattr(args,'rank_margin',0.0)}) | "
            f"use_ema={bool(getattr(args,'use_ema',False))} (decay={getattr(args,'ema_decay',0.999)}, "
            f"eval={int(getattr(args,'use_ema_eval',1))}, save_ckpt={int(getattr(args,'save_ema_ckpt',1))})"
        )

    summary_df, best_alpha = run_training_over_folds(args, dataset)

    with open(os.path.join(args.results_dir, "best_alpha.txt"), "w") as f:
        f.write(f"{best_alpha}\n")

    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)

    summary_name = f"summary_partial_{start}_{end}.csv" if len(folds) != args.k else "summary.csv"
    summary_df.to_csv(os.path.join(args.results_dir, summary_name), index=False)

    print("Finished all folds.")
    return summary_df


###############################################################################
# CLI
###############################################################################

parser = argparse.ArgumentParser(description='Configurations for WSI Training (CLAMFamily, run-sig paths)')

# Core
parser.add_argument('--dataset_name', type=str, required=True,
                    help='e.g. colon_united / colon_tcga / colon_avant')
parser.add_argument('--clinical_csv', type=str, required=True,
                    help='Path to task-agnostic clinical master CSV for this dataset')
parser.add_argument('--runs_root', type=str, required=True,
                    help='Root dir for runs (splits/results/eval), e.g. /gpfs/scratch/leh06/CLAMFamily/runs')

# Features (RAW pt)
parser.add_argument('--features_root', type=str, default=None,
                    help='Root dir of feature store, e.g. /gpfs/scratch/leh06/CLAMFamily/datasets')
parser.add_argument('--feature_key', type=str, default=None,
                    help="Feature subdir name, e.g. 'univ2_20x' (matches save_embeddings scripts)")
parser.add_argument('--feature_dir', type=str, default=None,
                    help='Explicit LEAF directory that contains *.pt (overrides features_root/feature_key).')

parser.add_argument('--pt_id_col', type=str, default='slide_id',
                    help=(
                        "Column in clinical CSV that matches the .pt filename stem. "
                        "If your CSV's 'slide_id' actually holds case_ids, set this to the real per-slide pt id column "
                        "(e.g. 'wsi_id', 'pt_id', 'slide_pt_id')."
                    ))

# Fold PCA (optional, train-only fit; applied on-the-fly)
parser.add_argument('--pca_k', type=int, default=None,
                    help='If set, fit fold-specific IncrementalPCA on TRAIN split and project train/val/test.')
parser.add_argument('--pca_batch_rows', type=int, default=200_000,
                    help='Row-chunk size for IncrementalPCA.partial_fit (tiles per chunk).')
parser.add_argument('--pca_max_rows', type=int, default=0,
                    help='Optional cap on total #tiles used to fit PCA per fold (0 = no cap).')
parser.add_argument('--refit_pca', action='store_true', default=False,
                    help='If set, refit PCA even if fold PCA model already exists on disk.')

parser.add_argument('--pca_cache_root', type=str, default='',
                    help=(
                        "Root directory for PCA cache. "
                        "PCA models will be saved under: <pca_cache_root>/<dataset>/<features_fp>/<splits_fp>/ipca_.../"
                        "If empty, defaults to <results_dir>/pca_models."
                    ))

# Task type
parser.add_argument('--task_type', type=str, required=True,
                    choices=['classification', 'survival', 'regression'],
                    help='Prediction target type.')

# Splitting semantics
parser.add_argument('--slide_level_split', action='store_true', default=False,
                    help='if set, *and* bag_level="slide", split by slide_id instead of case_id')
parser.add_argument('--patient_voting', type=str, default='max',
                    choices=['max', 'maj'],
                    help='how to aggregate slide labels to patient label for stratification in classification tasks')

# CSV schema
parser.add_argument('--label_col', type=str, default='label',
                    help='column in CSV for classification labels')
parser.add_argument('--label_map', type=str, default='',
                    help='mapping "raw_label:int_id,raw_label2:int_id2" for classification')
parser.add_argument('--ignore_labels', type=str, default='',
                    help='comma-separated list of labels to drop for classification')
parser.add_argument('--time_col', type=str, default='time',
                    help='column in CSV for survival time')
parser.add_argument('--event_col', type=str, default='event',
                    help='column in CSV for survival event (1=event,0=censored)')
parser.add_argument('--target_col', type=str, default='target',
                    help='column in CSV for regression targets')
parser.add_argument('--covariate_cols', type=str, default='',
                    help='comma-separated covariate columns, e.g. "age,stage_int,grade_code"')

# Domain-adversarial CLI
parser.add_argument('--domain_adapt', action='store_true', default=False,
                    help='Enable domain-adversarial learning to remove institution signal.')
parser.add_argument('--institution_col', type=str, default='institution',
                    help='Column in clinical CSV containing institution/site ID/name.')
parser.add_argument('--num_institutions', type=int, default=0,
                    help='Number of institutions (0 = infer from CSV). Must match model head K.')
parser.add_argument('--domain_level', type=str, default='bag',
                    choices=['bag', 'tile', 'both'],
                    help='Use bag-level domain head, tile-level head, or both.')
parser.add_argument('--domain_lambda_max', type=float, default=0.05,
                    help='Max GRL lambda (gradient reversal strength).')
parser.add_argument('--domain_warmup_epochs', type=int, default=5,
                    help='Warmup epochs to ramp GRL lambda to lambda_max.')
parser.add_argument('--domain_loss_weight', type=float, default=1.0,
                    help='Weight applied to domain CE loss in total objective.')
parser.add_argument('--domain_tile_k', type=int, default=256,
                    help='If tile-level/both: number of top-attention tiles used for domain head.')
parser.add_argument('--domain_lambda_start_epoch', type=int, default=3,
                    help='Epoch to start applying GRL/domain adaptation (default: 3).')
parser.add_argument('--domain_lambda_ramp_epochs', type=int, default=12,
                    help='Number of epochs to ramp GRL lambda to lambda_max (default: 12).')
parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                    help='Global grad norm clipping. Set 0 to disable. Typical: 0.5–5.0.')

parser.add_argument('--domain_min_count', type=int, default=30,
                    help='Institutions with < this many TRAIN patients will be masked out of domain loss.')
parser.add_argument('--domain_class_weighting', type=str, default='sqrt_inv',
                    choices=['none', 'sqrt_inv', 'inv', 'count', 'sqrt'],
                    help='Weight domain CE by institution size (TRAIN-only).')
parser.add_argument('--domain_label_smoothing', type=float, default=0.05,
                    help='Label smoothing for domain CE (0 disables).')
parser.add_argument('--domain_tile_lambda_mult', type=float, default=0.2,
                    help='Tile lambda = domain_tile_lambda_mult * domain_lambda_max.')

# Scheduler knobs
parser.add_argument('--use_plateau', action='store_true', default=True,
                    help='Use ReduceLROnPlateau scheduler (default: True).')
parser.add_argument('--plateau_factor', type=float, default=0.5,
                    help='LR multiplier when plateauing (default: 0.5).')
parser.add_argument('--plateau_patience', type=int, default=2,
                    help='Epochs with no improvement before LR drop (default: 2).')
parser.add_argument('--plateau_threshold', type=float, default=1e-4,
                    help='Minimum change in monitored metric to count as improvement.')
parser.add_argument('--plateau_cooldown', type=int, default=0,
                    help='Cooldown epochs after LR reduction.')
parser.add_argument('--plateau_min_lr', type=float, default=1e-6,
                    help='Minimum LR allowed by scheduler.')

# Training hyperparams
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--label_frac', type=float, default=1.0)
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay / L2 for optimizer')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--k', type=int, default=10,
                    help='total number of CV folds')
parser.add_argument('--k_start', type=int, default=-1,
                    help='which fold to start at (-1 means 0)')
parser.add_argument('--k_end', type=int, default=-1,
                    help='which fold to end at (-1 means k)')
parser.add_argument('--log_data', action='store_true', default=False)
parser.add_argument('--testing', action='store_true', default=False,
                    help='debug mode, may affect loaders')
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--es_patience', type=int, default=15,
                    help='EarlyStopping patience (epochs with no improvement).')
parser.add_argument('--es_min_epoch', type=int, default=20,
                    help='EarlyStopping: do not start counting before this epoch.')
parser.add_argument('--lambda_pred_l2', type=float, default=0.01,
                    help='regression: L2 penalty on the (standardized) prediction; '
                         'shrinks toward the target mean. Set 0 to disable.')
parser.add_argument('--standardize_target', action='store_true', default=False,
                    help='regression: z-score the target using train-fold mean/std '
                         '(baked into the model as buffers; eval auto-consistent).')
parser.add_argument('--target_transform', type=str, choices=['none', 'log1p'], default='none',
                    help='regression: optional target transform applied before '
                         'standardization (log1p compresses a right-skewed target; '
                         'forward applies expm1 so predictions stay in raw units).')
parser.add_argument('--reg_loss', type=str, choices=['mse', 'huber'], default='mse',
                    help='regression training loss (huber = robust to tail outliers).')
parser.add_argument('--huber_delta', type=float, default=1.0,
                    help='delta for Huber/smooth-L1 regression loss (standardized space).')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25)
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                    help='classification loss at bag level')
parser.add_argument('--model_type', type=str,
                    choices=['clam_family', 'clam_sb', 'clam_mb', 'mil', 'clam_sb_surv', 'mil_mc'],
                    default='clam_family',
                    help='which MIL/CLAM-style model variant to build')
parser.add_argument('--exp_code', type=str, default='exp',
                    help='experiment code subdir name')
parser.add_argument('--weighted_sample', action='store_true', default=False)
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
parser.add_argument('--subtyping', action='store_true', default=False)
parser.add_argument('--cov_fusion', type=str,
                    choices=['concat', 'cox_additive'],
                    default='concat',
                    help='How to fuse covariates with image bag.')

parser.add_argument('--bag_level', type=str,
                    choices=['slide', 'patient'],
                    default='slide',
                    help=(
                        'MIL bag granularity:\n'
                        '  "slide"   = 1 WSI per bag (classic). Stratification will still be patient-level\n'
                        '              unless you also pass --slide_level_split.\n'
                        '  "patient" = concatenate all slides from the same case_id into one bag, and we\n'
                        '              ALWAYS stratify by patient.'))

# CLAM / instance-level knobs
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                    help='disable instance-level clustering (if implemented)')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None)
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss')
parser.add_argument('--B', type=int, default=8,
                    help='num positive/negative patches for instance clustering')

# Cox regularization alpha (swept for survival)
parser.add_argument('--alpha', type=float, default=0.0,
                    help='L2-style regularization strength for Cox head (survival only)')
parser.add_argument('--retrain_only', action='store_true', default=False,
                    help='Skip alpha sweep and only run the retrain step using --alpha or best_alpha.txt.')

# --- attention entropy reg ---
parser.add_argument("--attn_entropy_lambda", type=float, default=0.0,
                    help="Attention entropy regularizer weight (encourage high-entropy attention). Try 1e-3 to 1e-2.")

# --- pairwise rank loss ---
parser.add_argument("--rank_lambda", type=float, default=0.0,
                    help="Pairwise ranking loss weight. Try 0.05–0.5 (start 0.1).")
parser.add_argument("--rank_margin", type=float, default=0.0,
                    help="Hinge margin for pairwise ranking loss (default 0.0).")

# --- EMA ---
parser.add_argument("--use_ema", action="store_true", help="Enable EMA of model weights")
parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (e.g. 0.99–0.9999)")
parser.add_argument("--ema_device", type=str, default=None, help="Store EMA weights on this device (e.g. 'cpu')")
parser.add_argument("--use_ema_eval", type=int, default=1, help="1=eval/export EMA weights, 0=use raw")
parser.add_argument("--save_ema_ckpt", type=int, default=1, help="1=save EMA checkpoint at end of fold")

args = parser.parse_args()
seed_torch(args.seed)

if __name__ == "__main__":
    results_df = main(args)
    print("finished!")
    print("end script")