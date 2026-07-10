"""Step 5 post-processing: standardize the HPC table + downstream manifest.

`run_representationsleiden_assignment.py` (step 5a, leiden 2.5 / 'cohort_cleaned')
writes a per-tile HPC label table into the HPL results tree. Artifact tiles are
already gone (removed at the tile level in step 4), so this module simply:

  1. locates that assignment CSV (config override or autodetection),
  2. standardizes its columns to (tile_id, slide_id, x, y, hpc),
  3. writes the per-tile HPC table, plus a minimal per-slide manifest and an
     external split CSV used by the .pt build and attention steps.

The column autodetection is deliberately defensive because the exact assignment
schema is HPL-version specific; override the names in config under
`cluster.columns:` if detection guesses wrong.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

# pandas is imported lazily inside build_assignment_table so that a dry-run
# (which never touches it) works even in an env with a slow/broken pandas.


# Candidate source-column names, in priority order.
_COL_CANDIDATES = {
    "tile_id": ["tile_id", "tiles", "tile", "indexes", "index"],
    "slide_id": ["slide_id", "slides", "slide", "samples", "sample", "case_id"],
    "x": ["x", "tile_x", "coord_x", "w"],
    "y": ["y", "tile_y", "coord_y", "h"],
    "hpc": ["hpc", "leiden", "cluster", "leiden_label", "leiden_2.0", "label"],
}


def _pick(df_cols, cfg_override, key):
    if cfg_override and key in cfg_override:
        return cfg_override[key]
    lower = {c.lower(): c for c in df_cols}
    for cand in _COL_CANDIDATES[key]:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # HPL prefixes every column with the set name (complete_slides, train_tiles,
    # complete_tiles, ...); match a column whose name ends with _<candidate>.
    for cand in _COL_CANDIDATES[key]:
        cl = cand.lower()
        for lc, orig in lower.items():
            if lc.endswith("_" + cl):
                return orig
    # leiden columns are often "leiden_<res>"; match by prefix for hpc
    if key == "hpc":
        for c in df_cols:
            if str(c).lower().startswith("leiden"):
                return c
    return None


def _find_assignment_csv(cfg) -> str:
    c = cfg.get("cluster", {})
    explicit = c.get("assignment_csv")
    if explicit:
        return explicit
    ds = cfg.get("dataset_name", "cohort")
    res = c.get("hpc_resolution", 2.5)
    tag = str(res).replace(".", "p")
    # Search the HPL results tree (under the HPC reference) and the work dir.
    ref_h5 = cfg.get("weights", {}).get("hpl_reference_h5", "")
    meta = c.get("hpc_meta_field", "cohort_cleaned")
    fold = c.get("hpc_fold", 1)
    # Search only specific, shallow directories — never the whole HPL install.
    search_roots = [
        os.path.join(os.path.dirname(ref_h5), meta, "adatas") if ref_h5 else "",
        os.path.join(cfg.get("paths", {}).get("work_dir", ""), "hpl"),
    ]
    patterns = [f"*{ds}*leiden_{tag}__fold{fold}.csv", f"*{ds}*leiden_{tag}*.csv",
                f"*{ds}*leiden*.csv"]
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(root, pat)))
            if hits:
                return hits[0]
    return ""


def build_assignment_table(cfg, out_csv, dry_run=False):
    """Standardize the HPC assignment CSV and emit table + manifest + split."""
    ds = cfg.get("dataset_name", "cohort")
    work = cfg.get("paths", {}).get("work_dir", "")
    clusters_dir = os.path.join(work, "clusters")
    manifest_csv = os.path.join(clusters_dir, f"{ds}_manifest.csv")
    split_csv = os.path.join(clusters_dir, f"{ds}_external_split.csv")

    if dry_run:
        print("[assign_hpc] DRY-RUN — would standardize the HPC assignment:")
        print("    source assignment CSV : (autodetect at run time)")
        print(f"    per-tile HPC table    -> {out_csv}")
        print(f"    per-slide manifest    -> {manifest_csv}")
        print(f"    external split CSV    -> {split_csv}")
        return 0

    import pandas as pd

    src = _find_assignment_csv(cfg)
    if not src or not os.path.isfile(src):
        raise SystemExit(
            "[assign_hpc] Could not locate the HPC assignment CSV produced by "
            "step assign_hpc (5a). Set cluster.assignment_csv in the config to its path."
        )
    Path(clusters_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    override = cfg.get("cluster", {}).get("columns", {})
    colmap = {}
    for key in ("tile_id", "slide_id", "x", "y", "hpc"):
        picked = _pick(df.columns, override, key)
        if picked is not None:
            colmap[picked] = key
    if "hpc" not in colmap.values() or "slide_id" not in colmap.values():
        raise SystemExit(
            f"[assign_hpc] Could not identify HPC/slide columns in {src}. "
            f"Columns present: {list(df.columns)}. Set cluster.columns in config."
        )
    std = df.rename(columns=colmap)[[v for v in colmap.values()]].copy()
    # DeepPATH names tile dirs '<slide>_files', and that suffix leaks into the
    # slide id stored in the H5. Strip it so slide_id matches the .pt filenames
    # (build_pt runs with --strip_files_suffix) and the original WSI filename.
    if "slide_id" in std.columns:
        std["slide_id"] = std["slide_id"].astype(str).str.replace(r"_files$", "", regex=True)
    std.to_csv(out_csv, index=False)
    print(f"[assign_hpc] wrote per-tile HPC table ({len(std)} tiles) -> {out_csv}")

    # Per-slide manifest with dummy survival columns (real labels optional).
    slides = sorted(std["slide_id"].astype(str).unique())
    inf = cfg.get("infer", {})
    manifest = pd.DataFrame({
        "slide_id": slides,
        "case_id": slides,
        inf.get("time_col", "dfs_event_data"): 1.0,
        inf.get("event_col", "dfs_event_ind"): 0,
    })
    manifest.to_csv(manifest_csv, index=False)
    print(f"[assign_hpc] wrote per-slide manifest ({len(slides)} slides) -> {manifest_csv}")

    # External split CSV (all slides as `test`) for the attention extractor.
    split = pd.DataFrame({"train": pd.Series(dtype=str),
                          "val": pd.Series(dtype=str),
                          "test": pd.Series(slides)})
    split.to_csv(split_csv, index=False)
    print(f"[assign_hpc] wrote external split CSV -> {split_csv}")
    return 0
