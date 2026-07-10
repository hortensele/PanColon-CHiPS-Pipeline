"""Step 4 helper: turn an artifact-clustering assignment into a filtered h5.

The PanColon workflow removes artifact tiles (pen marks, blur, folds, background)
BEFORE the histomorphological-phenotype clustering. It does so by:

  1. assigning the new cohort's tiles to a coarse reference clustering
     (leiden 5.0, meta_field 'cohort') — run by step_cluster_filter, and
  2. dropping tiles whose cluster is in cluster.artifact_cluster_ids, at the
     TILE level, using HPL's utilities/tile_cleaning/remove_indexes_h5.py.

This module implements the glue for (2): it reads the assignment CSV, builds the
list of tile positions to remove (a pickle, HPL-compatible), then invokes
remove_indexes_h5.py to write '<h5>_filtered.h5'. Mirrors the lab's
create_pickle_additional_pancolon_20x.py, generalized to any cohort.
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

from .config import require
from .runner import run_stage


def _res_tag(resolution) -> str:
    # HPL writes resolutions as e.g. 5.0 -> "5p0"
    return str(resolution).replace(".", "p")


def _find_assignment_csv(cfg, resolution, fold):
    """Locate the additional cohort's leiden_<res>__fold<fold> assignment CSV."""
    c = cfg.get("cluster", {})
    explicit = c.get("artifact_assignment_csv")
    if explicit:
        return explicit
    ds = cfg.get("dataset_name", "cohort")
    tag = _res_tag(resolution)
    ref_h5 = cfg.get("weights", {}).get("hpl_artifact_reference_h5", "")
    meta = c.get("artifact_meta_field", "cohort")
    # Search only specific, shallow directories — never the whole HPL install.
    roots = [
        os.path.join(os.path.dirname(ref_h5), meta, "adatas") if ref_h5 else "",
        os.path.join(cfg.get("paths", {}).get("work_dir", ""), "hpl"),
    ]
    patterns = [
        f"*{ds}*leiden_{tag}__fold{fold}.csv",
        f"*{ds}*leiden_{tag}*fold{fold}*.csv",
        f"*{ds}*leiden_{tag}*.csv",
    ]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(root, pat)))
            if hits:
                return hits[0]
    return ""


def _leiden_column(df, resolution):
    tag = str(resolution)
    lower = {c.lower(): c for c in df.columns}
    for cand in (f"leiden_{tag}", f"leiden_{_res_tag(resolution)}", tag, "leiden"):
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        if str(col).lower().startswith("leiden"):
            return col
    return None


def build_remove_pickle(cfg, out_pickle, dry_run=False):
    """Write a pickle of tile positions whose artifact cluster must be removed."""
    c = cfg.get("cluster", {})
    resolution = c.get("artifact_resolution", 5.0)
    fold = c.get("artifact_fold", 1)
    artifacts = list(c.get("artifact_cluster_ids", []) or [])

    if dry_run:
        print("[artifact_filter] DRY-RUN — would build remove-index pickle:")
        print(f"    assignment CSV (leiden {resolution}, fold {fold}): (autodetect at run time)")
        print(f"    artifact cluster ids: {artifacts}")
        print(f"    remove-index pickle -> {out_pickle}")
        return out_pickle

    src = _find_assignment_csv(cfg, resolution, fold)

    if not artifacts:
        print("[artifact_filter] cluster.artifact_cluster_ids is empty; nothing to remove.")
        # still write an empty pickle so remove_indexes_h5 is a no-op copy
    if not src or not os.path.isfile(src):
        raise SystemExit(
            "[artifact_filter] Could not locate the artifact-clustering assignment "
            "CSV from step 4a. Set cluster.artifact_assignment_csv to its path.")

    import pandas as pd
    df = pd.read_csv(src).reset_index(drop=True)
    col = _leiden_column(df, resolution)
    if col is None:
        raise SystemExit(
            f"[artifact_filter] No leiden column in {src}. Columns: {list(df.columns)}")
    mask = df[col].isin(artifacts)
    remove_idx = df.index[mask].astype(int).tolist()
    print(f"[artifact_filter] {len(remove_idx)}/{len(df)} tiles flagged as artifacts "
          f"(clusters {sorted(artifacts)}) -> {out_pickle}")
    Path(os.path.dirname(out_pickle)).mkdir(parents=True, exist_ok=True)
    with open(out_pickle, "wb") as fh:
        pickle.dump(remove_idx, fh)
    return out_pickle


def remove_artifacts(cfg, opts, projected_h5):
    """Build the remove pickle, then run HPL's remove_indexes_h5.py.

    Produces '<projected_h5 stem>_filtered.h5'.
    """
    ds = cfg.get("dataset_name", "cohort")
    work = require(cfg, "paths.work_dir")
    pickle_path = os.path.join(work, "clusters", f"{ds}_artifact_indexes.pkl")
    if not opts.dry_run:
        Path(os.path.dirname(pickle_path)).mkdir(parents=True, exist_ok=True)
    build_remove_pickle(cfg, pickle_path, dry_run=opts.dry_run)

    hpl = require(cfg, "tools.hpl_root")
    remover = os.path.join(hpl, "utilities", "tile_cleaning", "remove_indexes_h5.py")
    argv = ["python", remover,
            "--pickle_file", pickle_path,
            "--h5_file", projected_h5,
            "--override"]
    # Same activation as the tiling stage (conda env or module).
    envs = cfg.get("envs", {})
    tmod = envs.get("tiling_module", "")
    if tmod:
        env_name, mods = None, (tmod if isinstance(tmod, list) else [tmod])
    else:
        env_name, mods = envs.get("tiling"), []
    return run_stage(cfg, step="artifact_remove", env=env_name, modules=mods,
                     argv=argv, cwd=hpl,
                     dry_run=opts.dry_run, no_env_switch=opts.no_env_switch)
