"""Aggregate per-fold SurvCLAM risk into the CHiPS score.

eval.py (step 7, `--split all`) writes one predictions.csv per fold under:
    {runs_root}/{dataset}/{run_sig}/eval/{exp_code}_s{seed}/fold_{k}/predictions.csv
with columns: slide_id, case_id, institution, time, event, risk.

The 16 fold models are a leave-one-institution-out ensemble, but their raw Cox
log-hazards sit on different, arbitrary offsets/scales (on the shipped model,
per-fold means range -1.06 to -0.04, spread ~1.03). A plain mean would let the
high-offset folds dominate. So each fold's risk is first STANDARDIZED against
that fold's own frozen development-cohort mean/std (DEFAULT_FOLD_STATS below),
then averaged -- giving every fold equal weight. chips_score is that
standardized ensemble mean.

We also stratify each case into a low/intermediate/high tertile using FIXED
cutpoints (not a cohort-relative rank), so a "high" label means the same thing
regardless of which other slides happen to be in this run. The cutpoints are
the q33/q67 of chips_score itself computed on the colon_united development
cohort (all 1024/1025 patients, all 16 folds) -- i.e. cutpoints and the
deployed score are on the same distribution by construction. (An earlier
version used q33/q67 of the single-fold out-of-fold risk applied to the plain
fold mean; that scale mismatch skewed tertiles to ~40/31/29 instead of
~33/33/33 -- see infer.chips_tertile_thresholds in config/pipeline*.yaml.)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frozen development-cohort calibration for the shipped `_p4_big` checkpoints
# (colon_united, n=1024 scored, task_type=survival, exp_code=
# HPL_PANCOLON_20x__dfs__lr2e4_reg1e5_do025_clip1_plateau_p4_big).
#
# DEFAULT_FOLD_STATS[k] = (mu_k, sd_k): mean/std of fold k's raw Cox
# log-hazard risk over the full development cohort (eval.py --split all
# --fold -1). Used to standardize each fold to a common scale before
# ensembling. DEFAULT_TERTILE_THRESHOLDS = (q33, q67) of the resulting
# standardized ensemble score (chips_score) on that same cohort.
#
# Regenerate both if the model is retrained, via:
#   runs_chips_calibration/compute_calibration.py <full-cohort eval_dir>
# ---------------------------------------------------------------------------
DEFAULT_FOLD_STATS = {
    0: (-0.549422, 1.105846),
    1: (-0.940757, 0.918963),
    2: (-1.047659, 1.084465),
    3: (-0.822835, 1.020946),
    4: (-0.297530, 0.668634),
    5: (-0.692043, 0.825799),
    6: (-0.763499, 1.002910),
    7: (-0.495456, 0.747983),
    8: (-0.456417, 1.072986),
    9: (-0.038732, 0.429763),
    10: (-0.446664, 0.775413),
    11: (-0.432531, 0.793312),
    12: (-0.497583, 0.694999),
    13: (-0.531619, 0.720182),
    14: (-0.783026, 0.964833),
    15: (-1.064573, 1.362966),
}
DEFAULT_TERTILE_THRESHOLDS = (-0.411285, 0.369834)


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
    """Read every fold's predictions, standardize + average risk per case,
    write CHiPS CSV."""
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

    # Per-fold standardization stats (fold index -> (mu, sd)); config can
    # override (e.g. a YAML mapping "0": [mu, sd], ...).
    fold_stats_cfg = inf.get("chips_fold_stats", DEFAULT_FOLD_STATS)
    fold_stats = {int(k): tuple(v) for k, v in fold_stats_cfg.items()}

    per_fold = []
    for f in fold_csvs:
        k = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        d = pd.read_csv(f)
        if id_col not in d.columns or "risk" not in d.columns:
            raise SystemExit(f"[chips] {f} missing '{id_col}'/'risk' columns.")
        if k not in fold_stats:
            raise SystemExit(
                f"[chips] No standardization stats for fold {k}. Set "
                f"infer.chips_fold_stats in config, or regenerate "
                f"DEFAULT_FOLD_STATS via runs_chips_calibration/compute_calibration.py."
            )
        per_fold.append(d[[id_col, "risk"]].rename(columns={"risk": f"risk_fold{k}"}))

    merged = per_fold[0]
    for d in per_fold[1:]:
        merged = merged.merge(d, on=id_col, how="outer")

    risk_cols = {int(c.replace("risk_fold", "")): c
                 for c in merged.columns if c.startswith("risk_fold")}

    # Standardize each fold's raw risk against ITS OWN frozen dev-cohort
    # mean/std, THEN average -- raw Cox log-hazards have arbitrary,
    # fold-specific offsets/scales, so this gives every fold equal weight in
    # the ensemble rather than letting high-offset folds dominate a plain mean.
    z_cols = []
    for k, col in risk_cols.items():
        mu, sd = fold_stats[k]
        zc = f"_z_fold{k}"
        merged[zc] = (merged[col] - mu) / sd
        z_cols.append(zc)

    merged["chips_score"] = merged[z_cols].mean(axis=1, skipna=True)
    merged["n_folds"] = merged[[risk_cols[k] for k in risk_cols]].notna().sum(axis=1)
    merged = merged.drop(columns=z_cols)

    # Tertile stratification (low / intermediate / high risk) using FIXED
    # cutpoints -- the q33/q67 of this SAME standardized ensemble score on the
    # development cohort, not this cohort's own quantiles -- so a "high"
    # tertile means the same thing across every run.
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
    ordered = [id_col, "chips_score", "chips_tertile", "n_folds"] + \
        [risk_cols[k] for k in sorted(risk_cols)]
    merged[ordered].to_csv(out_csv, index=False)
    print(f"[chips] wrote CHiPS scores for {len(merged)} {id_col}s "
          f"(standardized mean over {len(risk_cols)} folds) -> {out_csv}")
    return out_csv
