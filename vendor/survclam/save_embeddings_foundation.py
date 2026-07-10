#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert tile embeddings stored as per-slide CSV files into per-slide .pt files
in a task-agnostic feature store layout.

Output layout:
  <features_root>/<dataset_name>/<model>_<mag>/pt_files/<slide_id>.pt

Notes:
- Only per-slide CSVs are supported.
- Valid slides are defined by clinical_csv (and optional subset_csv).
- For titan (and keep / provgigapath / musk), ONLY *_img_features.csv is used.
- No PCA is performed here (PCA will be fitted later per LOO fold).
"""

import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import torch


# -------------------------------------------------------------------------
# Embedding loader
# -------------------------------------------------------------------------

def load_slide_embeddings_from_individual_csv(csv_path, embed_dim):
    df = pd.read_csv(csv_path)

    emb = df.iloc[:, 1:embed_dim + 1].to_numpy(dtype=np.float32)

    # <slide_id>_features.csv  or  <slide_id>_img_features.csv
    slide_id = os.path.basename(csv_path).split("_")[0]
    return slide_id, emb


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def normalize_slide_id(s: str) -> str:
    s = str(s).strip()
    s = pd.Series([s]).str.replace(
        r"\.(svs|ndpi|tif|tiff|mrxs)$", "", regex=True
    ).iloc[0]
    return s


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Save per-slide CSV embeddings as per-slide .pt files (no PCA)."
    )

    # ---- dataset-level inputs
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--clinical_csv", type=str, required=True)
    parser.add_argument("--slide_id_col", type=str, default="slide_id")

    parser.add_argument("--subset_csv", type=str, default=None)
    parser.add_argument("--subset_slide_col", type=str, default=None)

    # ---- feature store
    parser.add_argument("--features_root", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mag", type=str, default="20x")

    # ---- embeddings source
    parser.add_argument("--embed_dir", type=str, required=True)
    parser.add_argument("--embed_dim", type=int, required=True)

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # 1) valid slides
    # ---------------------------------------------------------------------

    clin = pd.read_csv(args.clinical_csv)

    if args.slide_id_col not in clin.columns:
        raise ValueError(
            f"clinical_csv missing column '{args.slide_id_col}'"
        )

    valid_slides = set(
        clin[args.slide_id_col]
        .astype(str)
        .map(normalize_slide_id)
        .values
    )

    if args.subset_csv is not None:
        if args.subset_slide_col is None:
            raise ValueError(
                "--subset_slide_col is required when --subset_csv is provided"
            )

        sub = pd.read_csv(args.subset_csv)

        if args.subset_slide_col not in sub.columns:
            raise ValueError(
                f"subset_csv missing column '{args.subset_slide_col}'"
            )

        subset_slides = set(
            sub[args.subset_slide_col]
            .astype(str)
            .map(normalize_slide_id)
            .values
        )

        valid_slides = valid_slides.intersection(subset_slides)

    print(f"[INFO] Valid slides after filtering: {len(valid_slides)}")

    if len(valid_slides) == 0:
        raise RuntimeError("No valid slides after filtering.")

    # ---------------------------------------------------------------------
    # 2) collect per-slide CSVs
    # ---------------------------------------------------------------------

    # titan / keep / provgigapath / musk use *_img_features.csv
    if args.model.lower() in ["titan", "keep", "provgigapath", "musk"]:
        pattern = os.path.join(args.embed_dir, "*_img_features.csv")
    else:
        pattern = os.path.join(args.embed_dir, "*_features.csv")

    slide_csvs = sorted(glob(pattern))

    if len(slide_csvs) == 0:
        raise RuntimeError(f"No CSVs found with pattern: {pattern}")

    print(f"[INFO] Found {len(slide_csvs)} CSV files")

    # ---------------------------------------------------------------------
    # 3) output dir
    # ---------------------------------------------------------------------

    feature_key = f"{args.model}_{args.mag}"

    out_dir = os.path.join(
        args.features_root,
        args.dataset_name,
        feature_key,
        "pt_files",
    )

    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 4) save pt
    # ---------------------------------------------------------------------

    n_saved = 0

    for csv_path in slide_csvs:

        slide_id, emb = load_slide_embeddings_from_individual_csv(
            csv_path, args.embed_dim
        )

        slide_id = normalize_slide_id(slide_id)

        if slide_id not in valid_slides:
            continue

        out_path = os.path.join(out_dir, f"{slide_id}.pt")
        
        if os.path.exists(out_path):
            continue
            
        torch.save(
            torch.from_numpy(emb),
            out_path
        )

        n_saved += 1

    print(f"[DONE] Saved {n_saved} slides to:")
    print(out_dir)


if __name__ == "__main__":
    main()