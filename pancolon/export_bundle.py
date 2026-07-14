"""Export a self-contained, zero-install results bundle for the static viewer.

Turns the pipeline's per-slide outputs (+ the original WSIs) into a folder that a
collaborator can open with nothing but Python's stdlib http server:

    bundle/
      index.html  viewer.js  openseadragon/     (the copied static viewer)
      view.sh  view.bat  README.md
      manifest.json                             (slides + cohort summary)
      cohort_results.csv                        (copy of chips_scores.csv)
      slides/<slide>/
        he.dzi  he_files/…                      deep-zoom pyramid of the H&E
        hpc.png  attention.png                  full-extent, slide-aspect layers

All three layers share the normalized [0,1]x[0,H/W] coordinate space, so the
viewer's three panels pan/zoom in lock-step. Reuses pancolon.overlay_render for
rasterization, tile coords, HPC composition, WSI lookup and DZI writing.

Run in the survclam env (needs openslide + torch + pandas + Pillow):
    python pancolon_pipeline.py export --config config/pipeline.yaml
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

from . import overlay_render as ov


def _bundle_dir(cfg) -> str:
    exp = cfg.get("export", {}) or {}
    if exp.get("bundle_dir"):
        return exp["bundle_dir"]
    return str(Path(cfg["paths"]["work_dir"], "bundle"))


def _read_chips(work):
    path = ov.chips_csv_path(work)
    if not os.path.isfile(path):
        return {}, "slide_id", path
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    id_col = "case_id" if rows and "case_id" in rows[0] else "slide_id"
    return {str(r[id_col]): r for r in rows}, id_col, path


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_EXAMPLE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _find_example_image(examples_dir, raw_id):
    """Match a representative-tile image for one HPC id (case-insensitive
    filename stem, e.g. "HPC0.jpg" or "hpc0.png" both match id "HPC0")."""
    if not examples_dir or not os.path.isdir(examples_dir):
        return None
    target = raw_id.strip().lower()
    for name in os.listdir(examples_dir):
        stem, ext = os.path.splitext(name)
        if stem.lower() == target and ext.lower() in _EXAMPLE_EXTS:
            return os.path.join(examples_dir, name)
    return None


def _write_hpc_atlas(cfg, out_root):
    """Convert the pathologist HPC id->description CSV into hpc_atlas.json.

    Source CSV has columns `id` (e.g. "HPC0") and `description` (a pathology
    sentence). We derive a short `name` from the text before the first " - "
    (falling back to the full description) so the viewer has both a compact
    label and the full text for a tooltip/atlas panel. Returns True if written.

    Also picks up an optional representative-tile image per HPC from
    viewer/hpc_examples/ (sibling of the atlas CSV; see its README) and
    copies matched ones into the bundle so the atlas panel can show a
    thumbnail alongside the text description.
    """
    exp = cfg.get("export", {}) or {}
    src = os.path.join(cfg["repo_root"], exp.get("hpc_atlas", "viewer/hpc_atlas.csv"))
    if not os.path.isfile(src):
        print(f"[export] no HPC atlas at {src} — skipping (viewer will show bare HPC ids)")
        return False
    examples_dir = os.path.join(os.path.dirname(src), "hpc_examples")
    with open(src, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    atlas = []
    n_images = 0
    for r in rows:
        raw_id = (r.get("id") or "").strip()
        m = raw_id.upper().replace("HPC", "").strip()
        if not m.isdigit():
            continue
        hpc = int(m)
        desc = (r.get("description") or "").strip()
        name = desc.split(" - ", 1)[0].strip() if " - " in desc else desc
        risk = (r.get("risk") or "").strip().lower()
        tier = (r.get("tier") or "").strip().lower()
        entry = {
            "hpc": hpc, "id": raw_id, "name": name, "description": desc,
            "color": ov.hpc_hex(hpc),
        }
        if risk in ("high", "low"):
            entry["risk"] = risk
            entry["tier"] = tier if tier in ("primary", "secondary") else "primary"
        example = _find_example_image(examples_dir, raw_id)
        if example:
            dst_dir = os.path.join(out_root, "hpc_examples")
            os.makedirs(dst_dir, exist_ok=True)
            dst_name = f"{raw_id}{os.path.splitext(example)[1].lower()}"
            shutil.copyfile(example, os.path.join(dst_dir, dst_name))
            entry["image"] = f"hpc_examples/{dst_name}"
            n_images += 1
        atlas.append(entry)
    atlas.sort(key=lambda a: a["hpc"])
    with open(os.path.join(out_root, "hpc_atlas.json"), "w") as fh:
        json.dump(atlas, fh, indent=2)
    print(f"[export] wrote hpc_atlas.json ({len(atlas)} clusters, "
          f"{n_images} with example images) from {src}")
    return True


def export_bundle(cfg, opts=None):
    """Build the static results bundle. Honors opts.dry_run for a preview."""
    dry = bool(getattr(opts, "dry_run", False))
    work = cfg["paths"]["work_dir"]
    dataset = cfg.get("dataset_name", "cohort")
    model_key = cfg.get("build_pt", {}).get("model_key", "HPL_PANCOLON_20x")
    wsi_dir = cfg["paths"]["wsi_dir"]
    out_root = _bundle_dir(cfg)
    viewer_src = os.path.join(cfg["repo_root"], "viewer")

    tile_cfg = cfg.get("tile", {}) or {}
    tile_size_px = tile_cfg.get("tile_size", 224)
    pixel_size_um = tile_cfg.get("pixel_size", 0.504)

    slides = ov.list_slides(work, dataset, model_key)
    chips_by_id, id_col, chips_path = _read_chips(work)

    print(f"[export] dataset={dataset} slides={len(slides)} -> {out_root}")
    if not slides:
        print("[export] No per-slide attention outputs found under "
              f"{ov.attention_pt_dir(work)} — run steps 1-8 first.")
    if dry:
        print("[export] DRY-RUN — would write:")
        print(f"    viewer + manifest.json + cohort_results.csv -> {out_root}")
        for sid in slides:
            wsi = ov.find_wsi(wsi_dir, sid)
            print(f"    slides/{sid}/: "
                  f"{'he.dzi ' if wsi else '(no WSI) '}hpc.png attention.png")
        return 0

    Path(out_root, "slides").mkdir(parents=True, exist_ok=True)

    manifest_slides = []
    for sid in slides:
        slide_dir = os.path.join(out_root, "slides", sid)
        os.makedirs(slide_dir, exist_ok=True)
        entry = {"slide_id": sid}

        # per-slide CHiPS
        row = chips_by_id.get(sid, {})
        entry["chips_score"] = _num(row.get("chips_score"))
        entry["chips_tertile"] = row.get("chips_tertile") or None

        # H&E deep-zoom pyramid
        wsi = ov.find_wsi(wsi_dir, sid)
        aspect = None
        if wsi:
            dims = ov.write_dzi(wsi, slide_dir, basename="he")
            if dims:
                w0, h0 = dims
                aspect = h0 / w0
                entry["he"] = "he.dzi"
                print(f"[export] {sid}: wrote he.dzi ({w0}x{h0})")
        if "he" not in entry:
            entry["he"] = None
            print(f"[export] {sid}: no WSI found — H&E panel will be blank")

        # HPC + attention layer images (full-extent, slide aspect)
        for kind in ("hpc", "attention"):
            png, aspect_h, legend = ov.render_layer_image(
                work, dataset, model_key, wsi_dir, sid, kind,
                tile_size_px=tile_size_px, pixel_size_um=pixel_size_um)
            if png is None:
                entry[kind] = None
                continue
            with open(os.path.join(slide_dir, f"{kind}.png"), "wb") as fh:
                fh.write(png)
            entry[kind] = f"{kind}.png"
            entry[f"{kind}_legend"] = legend
            if aspect is None:
                aspect = aspect_h
        entry["aspect"] = aspect or 1.0

        # HPC composition (for the per-slide bar chart)
        entry["composition"] = ov.hpc_composition(work, dataset, sid)

        # HPC tile grid (for the hover tooltip on the HPC panel)
        grid = ov.hpc_grid(work, dataset, model_key, sid, wsi_path=wsi,
                           tile_size_px=tile_size_px, pixel_size_um=pixel_size_um)
        if grid is not None:
            with open(os.path.join(slide_dir, "hpc_grid.json"), "w") as fh:
                json.dump(grid, fh)
            entry["hpc_grid"] = "hpc_grid.json"
        else:
            entry["hpc_grid"] = None

        with open(os.path.join(slide_dir, "meta.json"), "w") as fh:
            json.dump(entry, fh, indent=2)
        manifest_slides.append(entry)

    # cohort CSV
    if os.path.isfile(chips_path):
        shutil.copyfile(chips_path, os.path.join(out_root, "cohort_results.csv"))

    # HPC atlas (id -> pathologist description), so the viewer can label
    # clusters by name instead of a bare number.
    has_atlas = _write_hpc_atlas(cfg, out_root)

    # manifest
    manifest = {
        "dataset": dataset,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "id_col": id_col,
        "n_slides": len(manifest_slides),
        "hpc_atlas": "hpc_atlas.json" if has_atlas else None,
        "slides": manifest_slides,
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    _copy_viewer(viewer_src, out_root)
    _write_launchers(out_root)
    print(f"[export] Done. View it with:\n    cd {out_root} && python -m http.server 8000"
          "\n  then open http://127.0.0.1:8000")
    return 0


def _copy_viewer(viewer_src, out_root):
    if not os.path.isdir(viewer_src):
        print(f"[export] WARNING: viewer source {viewer_src} missing; "
              "bundle will have data but no UI.")
        return
    for name in ("index.html", "viewer.js"):
        src = os.path.join(viewer_src, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_root, name))
    osd_src = os.path.join(viewer_src, "openseadragon")
    if os.path.isdir(osd_src):
        dst = os.path.join(out_root, "openseadragon")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(osd_src, dst)


def _write_launchers(out_root):
    sh = ("#!/usr/bin/env bash\n"
          "# Open the CHiPS results viewer in a browser (needs only Python).\n"
          'cd "$(dirname "$0")"\n'
          'PORT="${1:-8000}"\n'
          'echo "Serving on http://127.0.0.1:${PORT}  (Ctrl-C to stop)"\n'
          'python3 -m http.server "$PORT" 2>/dev/null || python -m http.server "$PORT"\n')
    bat = ("@echo off\r\n"
           "cd /d %~dp0\r\n"
           "set PORT=%1\r\n"
           "if \"%PORT%\"==\"\" set PORT=8000\r\n"
           "echo Serving on http://127.0.0.1:%PORT%  (Ctrl-C to stop)\r\n"
           "python -m http.server %PORT%\r\n")
    with open(os.path.join(out_root, "view.sh"), "w") as fh:
        fh.write(sh)
    os.chmod(os.path.join(out_root, "view.sh"), 0o755)
    with open(os.path.join(out_root, "view.bat"), "w") as fh:
        fh.write(bat)
    readme = (
        "# PanColon-CHiPS results\n\n"
        "Self-contained viewer for this cohort's CHiPS scores, HPC assignments,\n"
        "and SurvCLAM attention maps. No GPU, openslide, or internet needed.\n\n"
        "## Open it\n\n"
        "    ./view.sh            # macOS/Linux (or: python3 -m http.server 8000)\n"
        "    view.bat             # Windows\n\n"
        "then open http://127.0.0.1:8000 in your browser.\n\n"
        "Pick a slide to see three synchronized panels (H&E, HPC, attention) that\n"
        "pan and zoom together, plus its CHiPS score and HPC composition. The\n"
        "cohort tab shows the score table and distribution.\n")
    with open(os.path.join(out_root, "README.md"), "w") as fh:
        fh.write(readme)
