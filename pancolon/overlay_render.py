"""Per-slide result helpers for the web app.

Reads the pipeline's per-slide outputs and turns them into things the browser can
show: an attention/HPC heatmap PNG placed over the slide, an HPC composition
table for a chart, and the source-WSI lookup used by the DeepZoom viewer. The
attention/tile-coordinate logic is lifted from ``notebooks/attention_overlay.ipynb``
so there is a single source of truth.

Directory layout (under ``paths.work_dir``):
    attention/attention/{slide_id}.pt                     per-slide attention dict
    datasets/{ds}/{model_key}/tile_ids/{slide_id}__tile_locations.csv
    clusters/{ds}_hpc_assignment.csv                      per-tile HPC labels
"""
from __future__ import annotations

import glob
import io
import os
import re

import numpy as np

WSI_EXTS = ("svs", "ndpi", "tif", "tiff", "mrxs", "scn", "vms", "svslide")


# --------------------------------------------------------------------------
# Output-path helpers (kept consistent with pancolon/steps.py)
# --------------------------------------------------------------------------

def attention_pt_dir(work) -> str:
    return os.path.join(work, "attention", "attention")


def tile_ids_dir(work, dataset, model_key) -> str:
    return os.path.join(work, "datasets", dataset, model_key, "tile_ids")


def hpc_csv_path(work, dataset) -> str:
    return os.path.join(work, "clusters", f"{dataset}_hpc_assignment.csv")


def chips_csv_path(work) -> str:
    return os.path.join(work, "survclam", "chips_scores.csv")


# --------------------------------------------------------------------------
# Slide discovery
# --------------------------------------------------------------------------

def list_slides(work, dataset, model_key):
    """Slide ids that have attention output (the viewer's unit of display)."""
    ids = set()
    for p in glob.glob(os.path.join(attention_pt_dir(work), "*.pt")):
        stem = os.path.splitext(os.path.basename(p))[0]
        stem = stem[:-4] if stem.endswith("_BAG") else stem
        ids.add(stem)
    return sorted(ids)


def find_wsi(wsi_dir, slide_id):
    """Locate the source whole-slide image file for a slide id, or None."""
    for ext in WSI_EXTS:
        hits = glob.glob(os.path.join(wsi_dir, f"{slide_id}.{ext}"))
        hits += glob.glob(os.path.join(wsi_dir, f"*{slide_id}*.{ext}"))
        if hits:
            return hits[0]
    return None


# --------------------------------------------------------------------------
# Attention + tile coordinates (from the notebook)
# --------------------------------------------------------------------------

def load_attention(work, slide_id):
    """Return (tile_ids[list[str]], attention[np.ndarray]) for a slide, or (None, None)."""
    import torch
    att_dir = attention_pt_dir(work)
    cand = [os.path.join(att_dir, f"{slide_id}.pt"),
            os.path.join(att_dir, f"{slide_id}_BAG.pt")]
    cand += sorted(glob.glob(os.path.join(att_dir, f"*{slide_id}*.pt")))
    for p in cand:
        if os.path.isfile(p):
            d = torch.load(p, map_location="cpu")
            attn = d.get("attention")
            attn = attn.numpy().reshape(-1) if hasattr(attn, "numpy") \
                else np.asarray(attn).reshape(-1)
            return [str(t) for t in d.get("tile_ids", [])], attn
    return None, None


def load_tile_coords(work, dataset, model_key, slide_id):
    """Map tile_id -> (col, row) grid index. Prefers x/y columns; else parses
    the trailing '_<x>_<y>' from the tile id (matches the notebook)."""
    import pandas as pd
    tdir = tile_ids_dir(work, dataset, model_key)
    cand = glob.glob(os.path.join(tdir, f"{slide_id}__tile_locations*.csv"))
    cand += glob.glob(os.path.join(tdir, f"*{slide_id}*.csv"))
    if not cand:
        return {}
    df = pd.read_csv(cand[0])
    tcol = next((c for c in ["tile_id", "tiles", "tile"] if c in df.columns),
                df.columns[0])
    xcol = next((c for c in ["x", "tile_x", "coord_x", "w"] if c in df.columns), None)
    ycol = next((c for c in ["y", "tile_y", "coord_y", "h"] if c in df.columns), None)
    coords = {}
    for tid in df[tcol].astype(str):
        coords[tid] = None
    if xcol and ycol:
        for tid, x, y in zip(df[tcol].astype(str), df[xcol], df[ycol]):
            try:
                coords[tid] = (float(x), float(y))
            except (TypeError, ValueError):
                pass
    else:
        for tid in list(coords):
            m = re.findall(r"(\d+)", tid)
            if len(m) >= 2:
                coords[tid] = (float(m[-2]), float(m[-1]))
    return {k: v for k, v in coords.items() if v is not None}


# --------------------------------------------------------------------------
# Rasterization into a grid PNG + slide placement
# --------------------------------------------------------------------------

# tab20, id-indexed so an HPC gets the SAME colour on every slide and in both the
# map overlay and the composition chart. Identity is always carried by the HPC id
# label too, so re-use past 20 clusters is a legend match, not a data encoding.
HPC_PALETTE = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5",
]


def hpc_hex(hpc_id) -> str:
    """Deterministic colour for an HPC label (stable across slides)."""
    return HPC_PALETTE[int(round(float(hpc_id))) % len(HPC_PALETTE)]


def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _cmap_rgba(values01, cmap_name="magma"):
    from matplotlib import cm
    cmap = cm.get_cmap(cmap_name)
    return cmap(np.clip(values01, 0.0, 1.0))  # (N,4) float RGBA


def _categorical_rgba(labels):
    rgba = np.zeros((len(labels), 4))
    for i, v in enumerate(labels):
        r, g, b = _hex_to_rgb01(hpc_hex(v))
        rgba[i] = (r, g, b, 1.0)
    uniq = sorted(set(int(round(v)) for v in labels))
    return rgba, {int(c): hpc_hex(c) for c in uniq}


def _grid_image(coords, tid_to_value, categorical=False):
    """Rasterize {tile_id: value} onto the (col,row) grid.

    Returns (rgba_uint8[H,W,4], ncols, nrows, legend). ``legend`` describes the
    colour mapping: a {hpc: hex} dict for categorical, else {vmin,vmax,cmap}.
    """
    items = [(coords[t], tid_to_value[t]) for t in tid_to_value if t in coords]
    if not items:
        return None, 0, 0, None
    xs = [int(round(c[0])) for c, _ in items]
    ys = [int(round(c[1])) for c, _ in items]
    x0, y0 = min(xs), min(ys)
    ncols = max(xs) - x0 + 1
    nrows = max(ys) - y0 + 1
    vals = np.array([v for _, v in items], dtype=float)

    if categorical:
        rgba, legend = _categorical_rgba(vals)
    else:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        rgba = _cmap_rgba(norm)
        legend = {"vmin": vmin, "vmax": vmax, "cmap": "magma"}

    img = np.zeros((nrows, ncols, 4), dtype=np.uint8)
    for (col, row), rc in zip([(int(round(c[0])) - x0, int(round(c[1])) - y0)
                               for c, _ in items], rgba):
        img[row, col] = [int(rc[0] * 255), int(rc[1] * 255),
                         int(rc[2] * 255), 255]
    return img, ncols, nrows, legend


def slide_dimensions(wsi_path):
    """(width0, height0) at level 0, or None if the WSI can't be opened."""
    if not wsi_path:
        return None
    try:
        import openslide
        with openslide.OpenSlide(wsi_path) as s:
            return int(s.dimensions[0]), int(s.dimensions[1])
    except Exception:
        return None


def _layer_grid(work, dataset, model_key, slide_id, kind):
    """Rasterize one slide's attention|hpc onto its (col,row) tile grid.

    Returns (rgba_uint8[H,W,4], ncols, nrows, legend) or (None, 0, 0, None).
    """
    coords = load_tile_coords(work, dataset, model_key, slide_id)
    if not coords:
        return None, 0, 0, None
    if kind == "attention":
        tile_ids, attn = load_attention(work, slide_id)
        if tile_ids is None or attn is None or len(tile_ids) != len(attn):
            return None, 0, 0, None
        return _grid_image(coords, dict(zip(tile_ids, attn)), categorical=False)
    if kind == "hpc":
        tid_to_hpc = _slide_tile_hpc(work, dataset, slide_id)
        if not tid_to_hpc:
            return None, 0, 0, None
        return _grid_image(coords, tid_to_hpc, categorical=True)
    return None, 0, 0, None


def _slide_aspect(wsi_dir, slide_id, ncols, nrows):
    """Slide height/width (for OSD viewport height), from the WSI or the grid."""
    dims = slide_dimensions(find_wsi(wsi_dir, slide_id))
    if dims:
        w0, h0 = dims
        return h0 / w0
    return nrows / ncols if ncols else 1.0


def render_overlay_png(work, dataset, model_key, wsi_dir, slide_id, kind):
    """Build the attention|hpc heatmap PNG for a slide (webapp overlay).

    Returns (png_bytes, placement, legend) or (None, None, None). ``placement``
    is the viewport rectangle for OpenSeadragon (main image has width 1.0):
        {"x":0, "y":0, "width":1.0, "height": H0/W0}
    """
    from PIL import Image

    img, ncols, nrows, legend = _layer_grid(work, dataset, model_key, slide_id, kind)
    if img is None:
        return None, None, None
    height = _slide_aspect(wsi_dir, slide_id, ncols, nrows)
    placement = {"x": 0.0, "y": 0.0, "width": 1.0, "height": height}
    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, format="PNG")
    return buf.getvalue(), placement, legend


def render_layer_image(work, dataset, model_key, wsi_dir, slide_id, kind,
                       max_px=4096):
    """Render a full-extent, slide-aspect PNG of one layer for the static viewer.

    Unlike render_overlay_png (a raw tile-grid image, meant to be *placed* over a
    slide), this upscales the grid to the slide's own aspect ratio so all three
    layers (H&E / HPC / attention) share the normalized [0,1]x[0,H/W] coordinate
    space and pan/zoom in exact lock-step. Cells are nearest-neighbour blocks;
    background (no tile) is transparent.

    Returns (png_bytes, aspect_height, legend) or (None, None, None).
    """
    from PIL import Image

    img, ncols, nrows, legend = _layer_grid(work, dataset, model_key, slide_id, kind)
    if img is None:
        return None, None, None
    aspect_h = _slide_aspect(wsi_dir, slide_id, ncols, nrows)  # H/W
    # Target canvas: width capped at max_px, height set by the slide aspect so the
    # image maps 1:1 onto the H&E deep-zoom extent.
    out_w = min(max_px, max(ncols * 16, 512))
    out_h = max(1, int(round(out_w * aspect_h)))
    canvas = Image.fromarray(img, "RGBA").resize((out_w, out_h), Image.NEAREST)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue(), aspect_h, legend


def write_dzi(wsi_path, out_dir, basename="he", tile_size=254, overlap=1,
              fmt="jpeg", quality=80):
    """Write a static DeepZoom pyramid of a WSI to ``out_dir``.

    Produces ``{out_dir}/{basename}.dzi`` + ``{out_dir}/{basename}_files/…`` that
    OpenSeadragon can open directly over http (openslide's deepzoom_tile recipe).
    Returns (width0, height0) or None if the WSI can't be opened.
    """
    import openslide
    from openslide.deepzoom import DeepZoomGenerator

    try:
        osr = openslide.OpenSlide(wsi_path)
    except Exception:
        return None
    try:
        dz = DeepZoomGenerator(osr, tile_size=tile_size, overlap=overlap,
                               limit_bounds=True)
        tiles_dir = os.path.join(out_dir, f"{basename}_files")
        os.makedirs(tiles_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{basename}.dzi"), "w") as fh:
            fh.write(dz.get_dzi(fmt))
        for level in range(dz.level_count):
            level_dir = os.path.join(tiles_dir, str(level))
            os.makedirs(level_dir, exist_ok=True)
            cols, rows = dz.level_tiles[level]
            for col in range(cols):
                for row in range(rows):
                    tile = dz.get_tile(level, (col, row))
                    tile.save(os.path.join(level_dir, f"{col}_{row}.{fmt}"),
                              quality=quality)
        return int(osr.dimensions[0]), int(osr.dimensions[1])
    finally:
        osr.close()


# --------------------------------------------------------------------------
# HPC composition
# --------------------------------------------------------------------------

def _slide_tile_hpc(work, dataset, slide_id):
    """Return {tile_id: hpc_label} for one slide from the per-tile HPC table."""
    import pandas as pd
    path = hpc_csv_path(work, dataset)
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    if "slide_id" not in df.columns or "hpc" not in df.columns:
        return {}
    sub = df[df["slide_id"].astype(str) == str(slide_id)]
    if "tile_id" not in sub.columns:
        return {}
    return dict(zip(sub["tile_id"].astype(str), sub["hpc"]))


def hpc_composition(work, dataset, slide_id):
    """Per-HPC tile counts/fractions for a slide: [{hpc, n_tiles, frac}] sorted."""
    import pandas as pd
    path = hpc_csv_path(work, dataset)
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    if "slide_id" not in df.columns or "hpc" not in df.columns:
        return []
    sub = df[df["slide_id"].astype(str) == str(slide_id)]
    if sub.empty:
        return []
    counts = sub["hpc"].value_counts().sort_index()
    total = int(counts.sum())
    return [{"hpc": int(h), "n_tiles": int(n), "frac": n / total,
             "color": hpc_hex(h)}
            for h, n in counts.items()]
