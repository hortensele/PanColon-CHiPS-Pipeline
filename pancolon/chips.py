"""Aggregate per-fold SurvCLAM risk into the CHiPS score.

eval.py (step 7, `--split all`) writes one predictions.csv per fold under:
    {runs_root}/{dataset}/{run_sig}/eval/{exp_code}_s{seed}/fold_{k}/predictions.csv
with columns: slide_id, case_id, institution, time, event, risk.

The CHiPS score for a case is the mean of its per-fold risk (the same
leave-one-institution-out ensemble used in the paper). We also stratify each
case into a low/intermediate/high tertile using FIXED cutpoints (not a
cohort-relative rank), so a "high" label means the same thing regardless of
which other slides happen to be in this run. The cutpoints are the q33/q67 of
the colon_united training cohort's own patient-level out-of-fold risk
(folds_OOF_predictions.csv for the shipped exp_code) — see
infer.chips_tertile_thresholds in config/pipeline*.yaml.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# colon_united patient-level OOF q33/q67 for the shipped `_p4_big` checkpoints
# (n=1025). Used unless config sets infer.chips_tertile_thresholds explicitly.
DEFAULT_TERTILE_THRESHOLDS = (-1.0711, -0.2129)


def _eval_dir(cfg):
    from .steps import _run_sig
    inf = cfg.get("infer", {})
    runs_root = cfg["weights"]["survclam_runs_root"]
    return os.path.join(
        runs_root, inf.get("runs_root_dataset", "colon_united"),
        _run_sig(cfg), "eval",
        f"{inf.get('exp_code')}_s{inf.get('seed', 1)}",
    )


def aggregate_folds(cfg, out_dir):
    """Read every fold's predictions, average risk per case, write CHiPS CSV."""
    eval_dir = _eval_dir(cfg)
    inf = cfg.get("infer", {})
    id_col = "case_id" if inf.get("bag_level", "patient") == "patient" else "slide_id"

    fold_csvs = sorted(glob.glob(os.path.join(eval_dir, "fold_*", "predictions.csv")))
    want = inf.get("chips_folds", "all")
    if want != "all":
        keep = {int(x) for x in str(want).split(",") if x.strip() != ""}
        fold_csvs = [f for f in fold_csvs
                     if int(os.path.basename(os.path.dirname(f)).split("_")[1]) in keep]
    if not fold_csvs:
        raise SystemExit(f"[chips] No fold predictions found under {eval_dir}")

    per_fold = []
    for f in fold_csvs:
        k = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        d = pd.read_csv(f)
        if id_col not in d.columns or "risk" not in d.columns:
            raise SystemExit(f"[chips] {f} missing '{id_col}'/'risk' columns.")
        per_fold.append(d[[id_col, "risk"]].rename(columns={"risk": f"risk_fold{k}"}))

    merged = per_fold[0]
    for d in per_fold[1:]:
        merged = merged.merge(d, on=id_col, how="outer")

    risk_cols = [c for c in merged.columns if c.startswith("risk_fold")]
    merged["chips_score"] = merged[risk_cols].mean(axis=1, skipna=True)
    merged["n_folds"] = merged[risk_cols].notna().sum(axis=1)

    # Tertile stratification (low / intermediate / high risk) using FIXED
    # cutpoints from the training cohort, not this cohort's own quantiles —
    # a "high" tertile must mean the same thing across every run.
    q33, q67 = inf.get("chips_tertile_thresholds", DEFAULT_TERTILE_THRESHOLDS)

    def _tertile(v):
        if pd.isna(v):
            return np.nan
        if v < q33:
            return "low"
        if v <= q67:
            return "intermediate"
        return "high"

    merged["chips_tertile"] = merged["chips_score"].map(_tertile)

    merged = merged.sort_values("chips_score", ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "chips_scores.csv")
    ordered = [id_col, "chips_score", "chips_tertile", "n_folds"] + risk_cols
    merged[ordered].to_csv(out_csv, index=False)
    print(f"[chips] wrote CHiPS scores for {len(merged)} {id_col}s "
          f"(mean over {len(risk_cols)} folds) -> {out_csv}")
    return out_csv
