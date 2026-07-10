#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLAMFamily evaluation / external inference script.

Key behaviors:
- Rebuilds the same run-signature directory structure used in training to locate checkpoints.
- Loads a *new* clinical CSV (e.g., external cohort) and a pt feature directory (via --feature_dir).
- Supports survival/classification/regression.
- Supports optional fold PCA (train-only fit) projection at eval time.
- IMPORTANT CHANGE vs your pasted version:
  - If --split all, we do NOT require splits_dir to exist and we do not attempt to load split CSVs.
    This is what you want for external inference on a single cohort without train/val/test splits.

Assumptions:
- utils.eval_utils.evaluate_model(eval_dataset, args, ckpt_path) exists and can handle args.task_type.
- dataset_modules.dataset_generic.Generic_WSIFamilyDataset exists and supports the args used below.
- utils.pca_utils.load_pca_npz and wrap_dataset_with_pca exist (your fold-PCA wrapper).
"""

from __future__ import print_function

import argparse
import os
import re
import json
import numpy as np
import pandas as pd
import torch

from utils.eval_utils import evaluate_model
from dataset_modules.dataset_generic import Generic_WSIFamilyDataset

# Fold PCA (train-only fit) + projection wrapper
from utils.pca_utils import load_pca_npz, wrap_dataset_with_pca


###############################################################################
# helpers (match main.py behavior)
###############################################################################

def _build_label_dict(label_map_str):
    if label_map_str is None or label_map_str.strip() == "":
        return {}
    out = {}
    for pair in label_map_str.split(","):
        k, v = pair.split(":")
        out[k.strip()] = int(v.strip())
    return out


def _parse_ignore(ignore_str):
    if ignore_str is None or ignore_str.strip() == "":
        return []
    return [tok.strip() for tok in ignore_str.split(",") if tok.strip()]


def _parse_list(csv_str):
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
                              bag_level: str = "slide") -> str:
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


def resolve_run_dirs(runs_root: str,
                     dataset_name: str,
                     run_sig: str,
                     label_frac: float,
                     exp_code: str,
                     seed: int):
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
    RAW pt only:
      {features_root}/{dataset_name}/pt_files/{feature_key}/pt_files
    """
    return os.path.join(features_root, dataset_name, "pt_files", feature_key, "pt_files")


def resolve_ckpt_path(models_dir: str, fold: int) -> str:
    """
    Supports both:
      A) {models_dir}/s_{fold}_checkpoint.pt
      B) {models_dir}/{fold}/s_{fold}_checkpoint.pt
    """
    candidates = [
        os.path.join(models_dir, f"s_{fold}_checkpoint.pt"),
        os.path.join(models_dir, str(fold), f"s_{fold}_checkpoint.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    msg = "Missing checkpoint. Tried:\n" + "\n".join(candidates)
    raise FileNotFoundError(msg)


###############################################################################
# main eval
###############################################################################

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = device  # reserved; evaluate_model likely manages device usage

    # covariates
    covariate_cols = _parse_list(args.covariate_cols)
    label_dict = _build_label_dict(args.label_map)
    ignore_list = _parse_ignore(args.ignore_labels)

    # patient_strat (match training)
    if args.bag_level == "patient":
        effective_patient_strat = True
    else:
        effective_patient_strat = (not args.slide_level_split)

    # run signature + run dirs (used to find checkpoints & PCA)
    run_sig = make_run_signature_simple(
        task_type=args.task_type,
        time_col=args.time_col if args.task_type == "survival" else None,
        covariate_cols=covariate_cols,
        bag_level=args.bag_level,
    )

    run_base, split_dir, results_dir, eval_dir = resolve_run_dirs(
        runs_root=args.runs_root,
        dataset_name=args.dataset_name,
        run_sig=run_sig,
        label_frac=args.label_frac,
        exp_code=args.exp_code,
        seed=args.seed,
    )

    # models_dir: trained checkpoints live here
    models_dir = results_dir
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    # splits_dir: only required if we will load split CSVs
    need_splits = (args.split != "all")
    if need_splits:
        if args.splits_dir is not None and args.splits_dir.strip() != "":
            splits_dir = args.splits_dir
        else:
            splits_dir = split_dir
        if not os.path.isdir(splits_dir):
            raise FileNotFoundError(f"splits_dir not found: {splits_dir}")
    else:
        splits_dir = None

    # eval_dir (where we write outputs)
    save_dir = eval_dir
    os.makedirs(save_dir, exist_ok=True)

    # dataset
    dataset_csv_path = args.clinical_csv
    if dataset_csv_path is None or dataset_csv_path.strip() == "":
        raise ValueError("--clinical_csv is required.")
    if not os.path.isfile(dataset_csv_path):
        raise FileNotFoundError(f"clinical_csv not found: {dataset_csv_path}")

    # --- pre-clean survival columns for external inference (survival only) ---
    # Regression/classification CSVs have no time/event columns, so this cleaning
    # step must not run for them (it would raise on the missing columns).
    if args.task_type == "survival":
        df = pd.read_csv(dataset_csv_path)

        # keep only rows that have both time and event
        need_cols = [args.time_col, args.event_col, args.pt_id_col]
        missing_cols = [c for c in need_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in clinical CSV: {missing_cols}")

        # drop NA/inf in time/event
        df = df.replace([np.inf, -np.inf], np.nan)
        before = len(df)
        df = df.dropna(subset=[args.time_col, args.event_col])
        after = len(df)
        print(f"[clinical_csv] Dropped {before - after} rows with NA/inf in {args.time_col}/{args.event_col}")

        # coerce event to int-safe values (0/1) if it’s float-ish
        df[args.event_col] = pd.to_numeric(df[args.event_col], errors="coerce")
        df = df.dropna(subset=[args.event_col])
        df[args.event_col] = df[args.event_col].astype(int)

        # time as numeric (float ok)
        df[args.time_col] = pd.to_numeric(df[args.time_col], errors="coerce")
        df = df.dropna(subset=[args.time_col])

        # write a temp cleaned CSV and use it
        tmp_clean_csv = os.path.join(save_dir, "clinical_cleaned_for_eval.csv")
        df.to_csv(tmp_clean_csv, index=False)
        dataset_csv_path = tmp_clean_csv

    dataset = Generic_WSIFamilyDataset(
        csv_path=dataset_csv_path,
        task_type=args.task_type,
        label_col=args.label_col,
        label_dict=label_dict,
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

        # Must match training when pt filenames != slide_id
        pt_id_col=args.pt_id_col,

        # Optional institution support
        institution_col=args.institution_col,
        return_institution=args.return_institution,
    )
    dataset.bag_level = args.bag_level

    # attach feature directory (RAW pt only)
    if args.feature_dir is not None and args.feature_dir.strip() != "":
        dataset.data_dir = args.feature_dir
    else:
        if args.features_root is None or args.features_root.strip() == "":
            raise ValueError("Provide either --feature_dir OR --features_root + --feature_key.")
        if args.feature_key is None or args.feature_key.strip() == "":
            raise ValueError("Provide either --feature_dir OR --features_root + --feature_key.")
        dataset.data_dir = resolve_feature_leaf_dir(args.features_root, args.dataset_name, args.feature_key)

    if not os.path.isdir(dataset.data_dir):
        raise FileNotFoundError(f"Feature dir not found: {dataset.data_dir}")

    # Match training behavior: in patient-bag mode, drop patients with 0 embeddings
    if args.bag_level == "patient" and getattr(args, "filter_missing_embeddings", True):
        if hasattr(dataset, "filter_patients_with_missing_embeddings"):
            dataset.filter_patients_with_missing_embeddings(pt_id_col=args.pt_id_col, verbose=True)
        else:
            print("[WARN] Dataset has no filter_patients_with_missing_embeddings(); skipping.")

    # number of classes / cov dim
    args.n_classes = dataset.num_classes if args.task_type == "classification" else 1
    args.cov_dim = len(covariate_cols)

    # folds
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = list(range(start, end)) if args.fold == -1 else [args.fold]

    split_index = {"train": 0, "val": 1, "test": 2, "all": -1}

    # write settings
    settings = {
        "dataset_name": args.dataset_name,
        "clinical_csv": dataset_csv_path,
        "run_signature": run_sig,
        "runs_root": args.runs_root,
        "results_dir": results_dir,
        "models_dir": models_dir,
        "splits_dir": splits_dir,
        "eval_dir": save_dir,

        "task_type": args.task_type,
        "bag_level": args.bag_level,
        "pt_id_col": args.pt_id_col,
        "patient_strat_effective": effective_patient_strat,
        "slide_level_split": bool(args.slide_level_split),
        "patient_voting": args.patient_voting,

        # optional
        "institution_col": args.institution_col,
        "return_institution": bool(args.return_institution),
        "filter_missing_embeddings": bool(args.filter_missing_embeddings),

        "label_col": args.label_col,
        "label_map": args.label_map,
        "ignore_labels": args.ignore_labels,
        "time_col": args.time_col,
        "event_col": args.event_col,
        "target_col": args.target_col,
        "covariate_cols": covariate_cols,
        "cov_dim": args.cov_dim,

        "model_type": args.model_type,
        "model_size": args.model_size,
        "embed_dim": args.embed_dim,
        "drop_out": args.drop_out,
        "B": args.B,
        "subtyping": args.subtyping,
        "n_classes": args.n_classes,
        "alpha": args.alpha,
        "cov_fusion": args.cov_fusion,

        # feature source (RAW)
        "features_root": args.features_root,
        "feature_key": args.feature_key,
        "feature_dir_leaf": dataset.data_dir,

        # fold PCA (optional; must match training fold PCA files)
        "pca_k": args.pca_k,

        # split mode
        "split": args.split,
    }

    with open(os.path.join(save_dir, "eval_settings.json"), "w") as f:
        json.dump(settings, f, indent=2, sort_keys=True)

    print("====== EVAL SETTINGS ======")
    for k, v in settings.items():
        print(f"{k}: {v}")

    # ---------------------------------------------------------
    # run evaluation
    # ---------------------------------------------------------
    all_metrics = []

    for fold in folds:
        print(f"\n========== Evaluating fold {fold} ({args.split}) ==========")

        # load split datasets
        if split_index[args.split] < 0:
            eval_dataset = dataset
        else:
            split_csv_path = os.path.join(splits_dir, f"splits_{fold}.csv")
            if not os.path.isfile(split_csv_path):
                raise FileNotFoundError(f"Missing split CSV: {split_csv_path}")

            # Match main.py: patient-level splits store case/slide IDs (from_id=True);
            # otherwise infer (string IDs -> True, contiguous int indices -> False).
            if args.bag_level == "patient":
                from_id = True
            else:
                _col = pd.read_csv(split_csv_path).get("train")
                _vals = [str(v).strip() for v in (_col.dropna().tolist() if _col is not None else [])
                         if str(v).strip() not in ("", "nan")]
                from_id = (len(_vals) == 0) or any(not re.fullmatch(r"-?\d+", v) for v in _vals[:2000])
            print(f"[Splits] fold={fold} using from_id={from_id} (bag_level='{args.bag_level}')")

            train_split, val_split, test_split = dataset.return_splits(
                from_id=from_id,
                csv_path=split_csv_path,
                bag_level=args.bag_level
            )
            eval_dataset = [train_split, val_split, test_split][split_index[args.split]]

        # propagate bag_level where applicable
        if hasattr(eval_dataset, "bag_level"):
            eval_dataset.bag_level = args.bag_level

        # fold PCA (if enabled): load the fold PCA model and wrap eval dataset
        if args.pca_k is not None and int(args.pca_k) > 0:
            pca_dir = os.path.join(results_dir, "pca_models")
            pca_path = os.path.join(pca_dir, f"fold_{fold}_ipca_k{int(args.pca_k)}.npz")
            if not os.path.isfile(pca_path):
                raise FileNotFoundError(
                    f"Fold PCA not found: {pca_path}\n"
                    f"Did you run training with --pca_k {int(args.pca_k)} for this exp_code/seed?"
                )
            pca = load_pca_npz(pca_path)
            eval_dataset = wrap_dataset_with_pca(eval_dataset, pca)

        # checkpoint
        ckpt_path = resolve_ckpt_path(models_dir, fold)

        # allow eval_utils to tag fold for plots (KM etc.)
        args.current_fold = fold

        model, metrics, df_results = evaluate_model(eval_dataset, args, ckpt_path)
        _ = model  # unused here; keep for debugging/inspection if needed

        # per-fold outputs
        fold_dir = os.path.join(save_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        pred_path = os.path.join(fold_dir, "predictions.csv")
        df_results.to_csv(pred_path, index=False)
        print(f"[fold {fold}] saved predictions -> {pred_path}")

        row_metric = {"fold": fold, **metrics}
        all_metrics.append(row_metric)

    # summary
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(save_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n====== EVAL SUMMARY ======")
    print(summary_df)
    print(f"\nSaved summary -> {summary_path}")


###############################################################################
# CLI
###############################################################################

def build_parser():
    parser = argparse.ArgumentParser(
        description="WSI Evaluation Script (CLAMFamily run-sig paths + fold PCA) with external inference support"
    )

    # Core (must match training checkpoint location)
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Training dataset folder name where checkpoints live (e.g., colon_united / colon_tcga / colon_avant).")
    parser.add_argument("--clinical_csv", type=str, required=True,
                        help="Path to clinical master CSV for the EVAL dataset (can be external cohort).")
    parser.add_argument("--runs_root", type=str, required=True,
                        help="Root dir for runs, e.g. /gpfs/scratch/leh06/CLAMFamily/runs_final")
    parser.add_argument("--exp_code", type=str, required=True,
                        help="Experiment code used in training (determines results_dir).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label_frac", type=float, default=1.0)

    # Optional override (only used if split != all)
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Override splits dir; default uses run signature path. Not required if --split all.")

    # Features (RAW pt)
    parser.add_argument("--features_root", type=str, default=None)
    parser.add_argument("--feature_key", type=str, default=None)
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Direct path to pt files for eval cohort (recommended for external inference).")

    # Must match training when pt filenames != slide_id
    parser.add_argument("--pt_id_col", type=str, default="slide_id",
                        help="Column in clinical CSV that matches the .pt filename stem.")

    # Optional institution support
    parser.add_argument("--institution_col", type=str, default="institution")
    parser.add_argument("--return_institution", action="store_true", default=False,
                        help="If set, dataset returns institution_id as an extra field.")

    # Match training’s patient-bag robustness
    parser.add_argument("--filter_missing_embeddings", action="store_true", default=True,
                        help="If set and bag_level=patient, drop patients with 0 available slide .pt embeddings.")

    # Fold PCA (optional; loads fold PCA from training results_dir/pca_models)
    parser.add_argument("--pca_k", type=int, default=None,
                        help="If set, load fold PCA model and project eval data on-the-fly (must match training).")

    # Bagging / splitting
    parser.add_argument("--bag_level", type=str, default="slide", choices=["slide", "patient"])
    parser.add_argument("--slide_level_split", action="store_true", default=False)
    parser.add_argument("--patient_voting", type=str, default="max", choices=["max", "maj"])

    # Task type
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "survival", "regression"])
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--label_map", type=str, default="")
    parser.add_argument("--ignore_labels", type=str, default="")
    parser.add_argument("--time_col", type=str, default="time")
    parser.add_argument("--event_col", type=str, default="event")
    parser.add_argument("--target_col", type=str, default="target")
    parser.add_argument("--covariate_cols", type=str, default="")

    # Model params (must match training for proper rebuild in eval_utils)
    parser.add_argument("--model_type", type=str,
                        choices=["clam_family", "clam_sb", "clam_mb", "clam_sb_surv", "mil", "mil_mc"],
                        default="clam_family")
    parser.add_argument("--model_size", type=str, choices=["small", "big"], default="small")
    parser.add_argument("--drop_out", type=float, default=0.25)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--subtyping", action="store_true", default=False)

    # Eval controls
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k_start", type=int, default=-1)
    parser.add_argument("--k_end", type=int, default=-1)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="test",
                        help="Use --split all for external inference (no split CSVs needed).")

    # Cox / covariates
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--cov_hidden", type=int, default=16)
    parser.add_argument("--cov_use_layernorm", action="store_true", default=True)
    parser.add_argument("--cov_dropout", type=float, default=0.0)
    parser.add_argument("--cov_fusion", type=str, choices=["concat", "cox_additive"], default="concat")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)