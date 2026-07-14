#!/usr/bin/env python3
"""
Consistency check: does the DEPLOYED CHiPS scorer (standardized 16-fold
ensemble + fixed tertile cutpoints, pancolon/chips.py) reproduce the survival
stratification reported in the paper (which uses the out-of-fold score with
cohort-relative tertiles)?

This does NOT replace the paper's headline metrics -- those must stay on the
leak-free out-of-fold (OOF) score. This is a supplementary check that the
tool you ship behaves consistently with what the paper reports, run once on
the development cohort.

Usage:
    python scripts/check_chips_calibration.py \
        --eval_dir <full-cohort eval_dir with fold_*/predictions.csv> \
        --clinical_csv <colon_united__clinical_master.csv> \
        --time_col dfs_event_data --event_col dfs_event_ind \
        --out_dir <output dir>

Prints:
  - Tertile balance under the deployed scorer (should be ~33/33/33 by
    construction, since cutpoints are that scorer's own dev-cohort quantiles)
  - KM curves for low/intermediate/high CHiPS tertile + multivariate log-rank
  - C-index of the deployed (standardized ensemble) score, for reference
    against the paper's OOF C-index (they are NOT expected to match exactly:
    the ensemble average is not the same estimator as single-fold OOF, and
    folds saw these patients in cross-validation, unlike a true external eval)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pancolon import chips  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True,
                     help="Dir containing fold_*/predictions.csv for the full "
                          "development cohort (all patients x all folds).")
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--time_col", default="dfs_event_data")
    ap.add_argument("--event_col", default="dfs_event_ind")
    ap.add_argument("--id_col", default="case_id")
    ap.add_argument("--out_dir", default="./chips_calibration_check")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. Aggregate folds with the SAME logic as production (chips.py) ---
    fold_csvs = sorted(
        __import__("glob").glob(os.path.join(args.eval_dir, "fold_*", "predictions.csv")))
    if not fold_csvs:
        raise SystemExit(f"No fold_*/predictions.csv under {args.eval_dir}")

    fold_stats = chips.DEFAULT_FOLD_STATS
    q33, q67 = chips.DEFAULT_TERTILE_THRESHOLDS

    per_fold = []
    for f in fold_csvs:
        k = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        d = pd.read_csv(f)
        per_fold.append(d[[args.id_col, "risk"]].rename(columns={"risk": f"r{k}"}))
    merged = per_fold[0]
    for d in per_fold[1:]:
        merged = merged.merge(d, on=args.id_col, how="outer")

    risk_cols = {int(c[1:]): c for c in merged.columns if c.startswith("r") and c[1:].isdigit()}
    for k, col in risk_cols.items():
        mu, sd = fold_stats[k]
        merged[f"z{k}"] = (merged[col] - mu) / sd
    z_cols = [f"z{k}" for k in risk_cols]
    merged["chips_score"] = merged[z_cols].mean(axis=1, skipna=True)

    def tertile(v):
        if pd.isna(v):
            return np.nan
        if v < q33:
            return "Low"
        if v <= q67:
            return "Intermediate"
        return "High"

    merged["chips_tertile"] = merged["chips_score"].map(tertile)

    print("=== Tertile balance (deployed scorer, development cohort) ===")
    print((merged["chips_tertile"].value_counts(normalize=True) * 100).round(1))

    # --- 2. Merge with clinical outcomes ---
    # clinical_csv is slide-level (multiple slide_id rows can share a
    # case_id); collapse to one row per case_id before merging, or the
    # one-to-many join inflates the patient count.
    clin = pd.read_csv(args.clinical_csv)
    clin = clin.drop_duplicates(subset=[args.id_col])
    df = merged.merge(clin[[args.id_col, args.time_col, args.event_col]],
                       on=args.id_col, how="inner")
    df = df.dropna(subset=[args.time_col, args.event_col, "chips_score"])
    df[args.time_col] = pd.to_numeric(df[args.time_col], errors="coerce")
    df[args.event_col] = pd.to_numeric(df[args.event_col], errors="coerce")
    df = df.dropna(subset=[args.time_col, args.event_col])
    print(f"\nPatients with outcomes for KM/C-index: {len(df)}")

    # --- 3. C-index of the deployed score (for reference, not a claim of
    #     equivalence to the paper's OOF C-index -- see module docstring) ---
    cidx = concordance_index(df[args.time_col], -df["chips_score"], df[args.event_col])
    print(f"\nC-index of deployed (standardized-ensemble) CHiPS score "
          f"on development cohort: {cidx:.3f}")
    print("(Reference only -- folds saw these patients during training, unlike "
          "the paper's cross-validated OOF C-index; not directly comparable.)")

    # --- 4. KM by tertile + multivariate log-rank ---
    order = ["Low", "Intermediate", "High"]
    colors = {"Low": "#2166ac", "Intermediate": "gray", "High": "#d73027"}

    lr = multivariate_logrank_test(
        event_durations=df[args.time_col].astype(float),
        groups=df["chips_tertile"].astype(str),
        event_observed=df[args.event_col].astype(int),
    )
    print(f"\nMultivariate log-rank test across tertiles: p = {lr.p_value:.3g}")

    fig, ax = plt.subplots(figsize=(7, 6))
    kmf = KaplanMeierFitter()
    for g in order:
        dg = df[df["chips_tertile"] == g]
        if dg.empty:
            continue
        kmf.fit(dg[args.time_col].astype(float), dg[args.event_col].astype(int),
                label=f"{g} (n={len(dg)})")
        kmf.plot(ax=ax, ci_show=False, color=colors[g], linewidth=2.5)

    ax.text(0.1, 0.1, f"Log-rank p = {lr.p_value:.3g}",
            transform=ax.transAxes, fontsize=14, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_xlabel("Time")
    ax.set_ylabel("Disease-free survival probability")
    ax.set_title("Deployed CHiPS scorer -- development-cohort consistency check")
    ax.legend(title="CHiPS tertile")
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "chips_deployed_km_consistency_check.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_png}")

    out_csv = os.path.join(args.out_dir, "chips_deployed_scores_dev_cohort.csv")
    df[[args.id_col, "chips_score", "chips_tertile", args.time_col, args.event_col]].to_csv(
        out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
