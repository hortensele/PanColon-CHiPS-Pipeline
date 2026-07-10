#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import h5py
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):  # progress bar is optional
        return iterable if iterable is not None else []


###############################################################################
# Helpers
###############################################################################

def decode_slide_ids(arr):
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def normalize_slide_id(
    s: str,
    strip_ext: bool = True,
    strip_files_suffix: bool = False,
    tcga_clip_len: int | None = None,
) -> str:
    s = str(s).strip()

    if strip_ext:
        s = pd.Series([s]).str.replace(
            r"\.(svs|ndpi|tif|tiff|mrxs)$", "", regex=True
        ).iloc[0]

    if strip_files_suffix:
        s = s.replace("_files", "")

    if tcga_clip_len is not None and s.startswith("TCGA-"):
        s = s[:tcga_clip_len]

    return s


def safe_fname(s: str) -> str:
    return str(s).replace("/", "_")


def load_existing_pt_as_numpy(pt_path: str) -> np.ndarray:
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32, copy=False)
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().astype(np.float32, copy=False)
    if isinstance(obj, dict) and "features" in obj:
        t = obj["features"]
        if torch.is_tensor(t):
            return t.detach().cpu().numpy().astype(np.float32, copy=False)
        if isinstance(t, np.ndarray):
            return t.astype(np.float32, copy=False)
    raise TypeError(f"Unsupported .pt content at {pt_path}: {type(obj)}")


def load_existing_tile_ids_csv(csv_path: str) -> list[str]:
    # expects one column "tile_id"
    df = pd.read_csv(csv_path, compression="infer")
    if "tile_id" not in df.columns:
        raise ValueError(f"tile-id csv missing 'tile_id' column: {csv_path}")
    return df["tile_id"].astype(str).tolist()


def materialize_slide_features_and_tiles(slide_id, slide_to_sources, h5_specs, chunk_size):
    """
    Load all tile embeddings AND tile_ids for a slide across all source H5s listed in slide_to_sources.

    Returns:
      feat: np.ndarray [n_tiles, D] float32
      tiles: list[str] length n_tiles, in the exact same row order as feat
    """
    feat_chunks = []
    tile_chunks = []

    for (spec_i, idxs) in slide_to_sources[slide_id]:
        spec = h5_specs[spec_i]
        with h5py.File(spec["path"], "r") as h5:
            feats_ds = h5[spec["rep_key"]]
            tiles_ds = h5[spec["tiles_key"]]

            # NOTE: tiles_ds may be bytes; decode after slicing
            for start in range(0, len(idxs), chunk_size):
                sub = idxs[start:start + chunk_size]

                feat_chunks.append(feats_ds[sub, :])

                raw_tiles = tiles_ds[sub]
                if isinstance(raw_tiles, np.ndarray):
                    tile_chunks.extend(decode_slide_ids(raw_tiles))
                else:
                    # h5py can return list-like; ensure robust decoding
                    tile_chunks.extend(decode_slide_ids(np.asarray(raw_tiles)))

    feat = np.concatenate(feat_chunks, axis=0).astype(np.float32, copy=False)
    if len(tile_chunks) != feat.shape[0]:
        raise RuntimeError(
            f"Tile-id count mismatch for slide {slide_id}: "
            f"{len(tile_chunks)} tiles vs {feat.shape[0]} embeddings"
        )

    return feat, tile_chunks


###############################################################################
# Core logic
###############################################################################

def main():
    ap = argparse.ArgumentParser(
        description="Convert HPL H5 embeddings -> per-slide .pt files + per-slide tile-id CSVs in a task-agnostic feature store layout."
    )

    # ---------- dataset-level inputs ----------
    ap.add_argument('--dataset_name', type=str, required=True,
                    help="Dataset key, e.g. colon_united / colon_tcga / colon_avant")
    ap.add_argument('--clinical_csv', type=str, required=True,
                    help="Clinical master CSV (contains slide_id column).")
    ap.add_argument('--slide_id_col', type=str, default="slide_id",
                    help="Column in clinical_csv for slide IDs (default: slide_id).")

    ap.add_argument('--subset_csv', type=str, default=None,
                    help="Optional subset CSV restricting valid slides.")
    ap.add_argument('--subset_slide_col', type=str, default=None,
                    help="Slide id column in subset_csv (required if subset_csv is set).")

    # ---------- feature-store outputs ----------
    ap.add_argument('--features_root', type=str, required=True,
                    help="Root directory for task-agnostic feature store.")
    ap.add_argument('--model', type=str, required=True,
                    help="Model name, e.g. HPL_PANCOLON / HPL_COAD")
    ap.add_argument('--mag', type=str, default='20x')
    ap.add_argument('--feature_key', type=str, default=None,
                    help="Override feature subdir name. Default: '{model}_{mag}'")

    # ---------- H5 sources (repeatable quads) ----------
    ap.add_argument('--h5', action="append", required=True,
                    help="Path to H5 embedding file (repeatable).")
    ap.add_argument('--meta_field', action="append", required=True,
                    help="Dataset name in H5 that stores slide IDs (repeatable).")
    ap.add_argument('--rep_key', action="append", required=True,
                    help="Dataset name in H5 that stores embeddings [N, D] (repeatable).")
    ap.add_argument('--tiles_key', action="append", required=True,
                    help="Dataset name in H5 that stores tile IDs (repeatable).")

    # ---------- normalization ----------
    ap.add_argument('--strip_files_suffix', action='store_true',
                    help="If set, remove '_files' from slide IDs.")
    ap.add_argument('--tcga_clip_len', type=int, default=None,
                    help="If set, clip TCGA slide IDs to first N characters.")
    ap.add_argument('--no_strip_ext', action='store_true',
                    help="Disable stripping extensions like .svs/.ndpi/etc.")

    # ---------- writing behavior ----------
    ap.add_argument('--overwrite', action='store_true',
                    help="If set, overwrite existing slide .pt and tile CSV.")
    ap.add_argument('--merge_if_exists', action='store_true',
                    help="If set and file exists, append tiles (old + new). Ignored if --overwrite.")
    ap.add_argument('--chunk_size', type=int, default=200_000,
                    help="Max #tile rows to read at a time when materializing one slide.")

    args = ap.parse_args()

    if not (len(args.h5) == len(args.meta_field) == len(args.rep_key) == len(args.tiles_key)):
        raise ValueError("Each --h5 must have matching --meta_field, --rep_key, and --tiles_key")

    # -------------------------------------------------------------------------
    # 0) Determine valid slides from clinical_csv (+ optional subset)
    # -------------------------------------------------------------------------
    clin = pd.read_csv(args.clinical_csv)
    if args.slide_id_col not in clin.columns:
        raise ValueError(f"clinical_csv missing column '{args.slide_id_col}'")

    strip_ext = (not args.no_strip_ext)

    def _norm(x):
        return normalize_slide_id(
            x,
            strip_ext=strip_ext,
            strip_files_suffix=args.strip_files_suffix,
            tcga_clip_len=args.tcga_clip_len
        )

    valid_slides = set(clin[args.slide_id_col].astype(str).map(_norm).values)

    if args.subset_csv is not None:
        if args.subset_slide_col is None:
            raise ValueError("--subset_slide_col is required when --subset_csv is provided")
        sub = pd.read_csv(args.subset_csv)
        if args.subset_slide_col not in sub.columns:
            raise ValueError(f"subset_csv missing column '{args.subset_slide_col}'")
        subset_slides = set(sub[args.subset_slide_col].astype(str).map(_norm).values)
        valid_slides = valid_slides.intersection(subset_slides)

    valid_slides = set([s for s in valid_slides if str(s).strip() != ""])
    print(f"[INFO] Valid slides after filtering: {len(valid_slides)}")
    if len(valid_slides) == 0:
        raise RuntimeError("No valid slides after filtering. Check clinical/subset CSVs.")

    # -------------------------------------------------------------------------
    # 1) Output directories
    # <features_root>/<dataset_name>/<feature_key>/
    #   pt_files/<slide>.pt
    #   tile_ids/<slide>__tile_locations.csv
    # -------------------------------------------------------------------------
    feature_key = args.feature_key if args.feature_key is not None else f"{args.model}_{args.mag}"
    base_dir = os.path.join(args.features_root, args.dataset_name, feature_key)
    pt_dir = os.path.join(base_dir, "pt_files")
    tiles_dir = os.path.join(base_dir, "tile_ids")
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2) First pass: build mapping slide -> list of tile indices (per H5 spec)
    # -------------------------------------------------------------------------
    slide_to_sources = defaultdict(list)  # sid -> list of (spec_i, idx_array)

    h5_specs = [
        dict(path=h, meta_field=m, rep_key=r, tiles_key=tk)
        for h, m, r, tk in zip(args.h5, args.meta_field, args.rep_key, args.tiles_key)
    ]

    for si, spec in enumerate(h5_specs):
        h5_path = spec["path"]
        meta_field = spec["meta_field"]
        rep_key = spec["rep_key"]
        tiles_key = spec["tiles_key"]

        if not os.path.isfile(h5_path):
            raise FileNotFoundError(h5_path)

        print(f"\n[H5 index] {si}")
        print(f"  path={h5_path}")
        print(f"  meta_field={meta_field}, rep_key={rep_key}, tiles_key={tiles_key}")

        with h5py.File(h5_path, "r") as h5:
            for k in (meta_field, rep_key, tiles_key):
                if k not in h5:
                    raise KeyError(f"H5 missing required dataset '{k}'. keys={list(h5.keys())}")

            raw_slide_ids = decode_slide_ids(h5[meta_field][:])
            slide_ids = [_norm(s) for s in raw_slide_ids]

            tmp = defaultdict(list)
            for i, sid in enumerate(slide_ids):
                if sid in valid_slides:
                    tmp[sid].append(i)

            for sid, idxs in tmp.items():
                slide_to_sources[sid].append((si, np.asarray(idxs, dtype=np.int64)))

    found_slides = sorted(slide_to_sources.keys())
    print(f"\n[INFO] Slides found in H5(s) and valid list: {len(found_slides)}")
    missing = sorted(list(valid_slides.difference(found_slides)))
    if len(missing) > 0:
        print(f"[WARN] Valid slides missing in H5(s): {len(missing)} (showing up to 20)")
        print("       ", missing[:20])

    if len(found_slides) == 0:
        raise RuntimeError("No valid slides were found in the H5(s). Nothing to write.")

    # -------------------------------------------------------------------------
    # 3) Write per-slide pt files + tile csvs (NO PCA)
    # -------------------------------------------------------------------------
    n_written = 0
    n_merged = 0
    n_skipped_existing = 0

    for sid in tqdm(found_slides, desc="Writing pt + tile CSV"):
        feat, tile_ids = materialize_slide_features_and_tiles(
            sid, slide_to_sources, h5_specs, args.chunk_size
        )

        out_pt = os.path.join(pt_dir, safe_fname(sid) + ".pt")
        out_tiles = os.path.join(tiles_dir, safe_fname(sid) + "__tile_locations.csv")

        exists = os.path.exists(out_pt) or os.path.exists(out_tiles)

        if exists and (not args.overwrite):
            if args.merge_if_exists:
                # require both to exist to merge safely
                if not (os.path.exists(out_pt) and os.path.exists(out_tiles)):
                    raise RuntimeError(
                        f"Cannot merge for slide {sid}: expected both existing files:\n"
                        f"  {out_pt}\n  {out_tiles}"
                    )
                old_feat = load_existing_pt_as_numpy(out_pt)
                old_tiles = load_existing_tile_ids_csv(out_tiles)

                merged_feat = np.concatenate([old_feat, feat], axis=0).astype(np.float32, copy=False)
                merged_tiles = old_tiles + [str(x) for x in tile_ids]

                if len(merged_tiles) != merged_feat.shape[0]:
                    raise RuntimeError(
                        f"Post-merge mismatch for slide {sid}: "
                        f"{len(merged_tiles)} tiles vs {merged_feat.shape[0]} embeddings"
                    )

                # write merged
                torch.save(torch.tensor(merged_feat, dtype=torch.float32), out_pt)
                pd.DataFrame({"tile_id": merged_tiles}).to_csv(out_tiles, index=False)

                n_merged += 1
                continue
            else:
                n_skipped_existing += 1
                continue

        # overwrite OR fresh write
        torch.save(torch.tensor(feat, dtype=torch.float32), out_pt)
        pd.DataFrame({"tile_id": [str(x) for x in tile_ids]}).to_csv(
            out_tiles, index=False)
        n_written += 1

    print(f"\n[DONE] base_dir:\n  {base_dir}")
    print(f"  embeddings (.pt): {pt_dir}")
    print(f"  tile ids (.csv): {tiles_dir}")
    print(f"\n[STATS] wrote={n_written} merged={n_merged} skipped_existing={n_skipped_existing}")


if __name__ == "__main__":
    main()
