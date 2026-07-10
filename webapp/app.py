"""Cluster web interface for PanColon-CHiPS-Pipeline.

A Flask frontend that submits the real pipeline to SLURM and lets you explore the
per-image results. It does NOT run any compute itself: it writes a per-run config,
submits the eight-step ``stage.sbatch`` dependency chain (via
``pancolon.slurm``), monitors it with ``sacct``/``squeue``, and streams each
step's SLURM log to the browser. When the run finishes it serves, per slide:

  * the CHiPS score (from survclam/chips_scores.csv),
  * the HPC composition (from clusters/<ds>_hpc_assignment.csv),
  * a deep-zoom view of the WSI (openslide) with a toggleable attention / HPC
    heatmap overlay (from attention/attention/<slide>.pt).

Because it submits to SLURM it must run on a **login node**, in the
``pancolon_survclam`` env (flask + openslide + torch + matplotlib):

    conda activate pancolon_survclam
    python webapp/app.py --config config/pipeline.yaml
    # then SSH-tunnel to http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from flask import Flask, Response, abort, jsonify, render_template, request

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pancolon import overlay_render, slurm  # noqa: E402
from pancolon.config import load_config      # noqa: E402
from pancolon.steps import STEP_ORDER        # noqa: E402

STEPS = list(STEP_ORDER)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = None  # allow large WSI uploads (streamed to disk)

BASE_CONFIG = str(REPO_ROOT / "config" / "pipeline.yaml")


# ==========================================================================
# Active run: one SLURM chain at a time.
# ==========================================================================
class Run:
    """Holds the currently-submitted (or last) SLURM run."""

    def __init__(self):
        self.run_dir = None
        self.cfg_path = None
        self.jobs = []           # list[slurm.StepJob]
        self.submitted = False
        self.dry_run = False

    def step_log(self, step):
        for j in self.jobs:
            if j.step == step:
                return j.log_path
        return None

    def states(self):
        ids = [j.job_id for j in self.jobs if j.job_id]
        by_id = slurm.query_states(ids)
        out = {s: "idle" for s in STEPS}
        for j in self.jobs:
            out[j.step] = by_id.get(j.job_id, "pending") if j.job_id else "pending"
        return out

    def overall(self, states):
        vals = [states[j.step] for j in self.jobs]
        if not vals:
            return "idle"
        if "failed" in vals:
            return "failed"
        if all(v == "done" for v in vals):
            return "done"
        return "running"


RUN = Run()
_overlay_cache = {}   # (slide, kind) -> (png_bytes, placement, legend)
_dz_cache = {}        # slide -> DeepZoomGenerator


# ==========================================================================
# Config helpers
# ==========================================================================
def load_base_config():
    with open(BASE_CONFIG) as fh:
        return yaml.safe_load(fh) or {}


def write_run_config(overrides):
    """Copy the base config, apply UI overrides, and mint a per-run directory.

    Returns (cfg_path, resolved_cfg, run_dir).
    """
    cfg = load_base_config()
    cfg.setdefault("paths", {})
    if overrides.get("wsi_dir"):
        cfg["paths"]["wsi_dir"] = overrides["wsi_dir"]
    if overrides.get("work_dir"):
        cfg["paths"]["work_dir"] = overrides["work_dir"]
    if overrides.get("dataset_name"):
        cfg["dataset_name"] = overrides["dataset_name"]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_cfg = REPO_ROOT / "webapp" / "_run_config.yaml"
    with open(run_cfg, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    resolved = load_config(str(run_cfg))
    run_dir = os.path.join(resolved["paths"]["work_dir"], "webapp_runs", run_id)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return str(run_cfg), resolved, run_dir


def active_cfg():
    """Resolved config for reading results: the last run config, else the base."""
    run_cfg = REPO_ROOT / "webapp" / "_run_config.yaml"
    path = str(run_cfg) if run_cfg.is_file() else BASE_CONFIG
    return load_config(path)


def _paths(cfg):
    work = cfg["paths"]["work_dir"]
    dataset = cfg.get("dataset_name", "cohort")
    model_key = cfg.get("build_pt", {}).get("model_key", "HPL_PANCOLON_20x")
    wsi_dir = cfg["paths"]["wsi_dir"]
    return work, dataset, model_key, wsi_dir


# ==========================================================================
# Pages
# ==========================================================================
@app.route("/")
def index():
    cfg = load_base_config()
    return render_template(
        "index.html", steps=STEPS,
        base_wsi_dir=cfg.get("paths", {}).get("wsi_dir", ""),
        base_work_dir=cfg.get("paths", {}).get("work_dir", ""),
        dataset_name=cfg.get("dataset_name", "my_cohort"))


# ==========================================================================
# Upload + submit + monitor
# ==========================================================================
@app.route("/upload", methods=["POST"])
def upload():
    """Stream one or more uploaded WSIs to disk; return the folder to tile from."""
    files = request.files.getlist("wsi")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "No files selected."}), 400
    cfg = load_base_config()
    work = request.form.get("work_dir") or cfg.get("paths", {}).get("work_dir") \
        or str(REPO_ROOT / "webapp" / "_uploads")
    work = os.path.abspath(os.path.expanduser(work))
    dest_dir = os.path.join(work, "uploaded_wsi")
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    for f in files:
        dest = os.path.join(dest_dir, os.path.basename(f.filename))
        f.save(dest)
        saved.append({"name": os.path.basename(dest),
                      "size_mb": round(os.path.getsize(dest) / 1e6, 1)})
    return jsonify({"dir": dest_dir, "files": saved, "n": len(saved)})


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json(force=True)
    if RUN.submitted and RUN.overall(RUN.states()) == "running":
        return jsonify({"error": "A run is already in progress."}), 409
    dry_run = bool(data.get("dry_run"))
    try:
        cfg_path, cfg, run_dir = write_run_config({
            "wsi_dir": data.get("wsi_dir"),
            "work_dir": data.get("work_dir"),
            "dataset_name": data.get("dataset_name"),
        })
        jobs = slurm.submit_chain(
            cfg, data.get("from_step", STEPS[0]),
            data.get("to_step", STEPS[-1]), run_dir, dry_run=dry_run)
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    if dry_run:
        # Preview only: don't touch the active run or cached results.
        return jsonify({"ok": True, "dry_run": True, "run_dir": run_dir,
                        "jobs": [{"step": j.step, "job_id": j.job_id} for j in jobs]})
    # fresh run -> clear cached results/viewers
    _overlay_cache.clear()
    _dz_cache.clear()
    RUN.run_dir = run_dir
    RUN.cfg_path = cfg_path
    RUN.jobs = jobs
    RUN.submitted = True
    RUN.dry_run = False
    return jsonify({"ok": True, "run_dir": run_dir, "dry_run": False,
                    "jobs": [{"step": j.step, "job_id": j.job_id} for j in jobs]})


@app.route("/status")
def status():
    if not RUN.jobs:
        return jsonify({"state": "idle", "steps": {s: "idle" for s in STEPS},
                        "jobs": []})
    states = RUN.states()
    return jsonify({
        "state": RUN.overall(states),
        "steps": states,
        "dry_run": RUN.dry_run,
        "jobs": [{"step": j.step, "job_id": j.job_id,
                  "state": states[j.step], "needs_gpu": j.needs_gpu}
                 for j in RUN.jobs],
    })


@app.route("/logs/<step>")
def logs(step):
    """Return SLURM log bytes for a step from ?offset= (incremental tailing)."""
    path = RUN.step_log(step)
    if not path or not os.path.isfile(path):
        return jsonify({"content": "", "offset": 0, "exists": False})
    offset = request.args.get("offset", default=0, type=int)
    size = os.path.getsize(path)
    if offset > size:   # log was truncated/rewritten
        offset = 0
    with open(path, "r", errors="replace") as fh:
        fh.seek(offset)
        content = fh.read()
    return jsonify({"content": content, "offset": size, "exists": True})


@app.route("/stop", methods=["POST"])
def stop():
    ids = [j.job_id for j in RUN.jobs if j.job_id]
    if ids:
        subprocess.run(["scancel"] + ids, capture_output=True)
    return jsonify({"ok": True})


# ==========================================================================
# Results: CHiPS + slides + HPC
# ==========================================================================
def _read_chips(work):
    path = overlay_render.chips_csv_path(work)
    if not os.path.isfile(path):
        return [], None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    id_col = "case_id" if rows and "case_id" in rows[0] else "slide_id"
    return rows, id_col


@app.route("/chips")
def chips():
    work, *_ = _paths(active_cfg())
    rows, id_col = _read_chips(work)
    return jsonify({"rows": rows[:1000], "id_col": id_col})


@app.route("/slides")
def slides():
    cfg = active_cfg()
    work, dataset, model_key, wsi_dir = _paths(cfg)
    rows, id_col = _read_chips(work)
    chips_by_id = {str(r[id_col]): r for r in rows} if id_col else {}
    out = []
    for sid in overlay_render.list_slides(work, dataset, model_key):
        c = chips_by_id.get(sid, {})
        comp = overlay_render.hpc_composition(work, dataset, sid)
        out.append({
            "slide_id": sid,
            "chips_score": c.get("chips_score"),
            "chips_percentile": c.get("chips_percentile"),
            "chips_tertile": c.get("chips_tertile"),
            "has_attention": True,
            "has_hpc": bool(comp),
            "has_wsi": overlay_render.find_wsi(wsi_dir, sid) is not None,
        })
    return jsonify({"slides": out})


@app.route("/hpc/<slide>")
def hpc(slide):
    cfg = active_cfg()
    work, dataset, *_ = _paths(cfg)
    return jsonify({"slide_id": slide,
                    "composition": overlay_render.hpc_composition(work, dataset, slide)})


# ==========================================================================
# Per-slide attention / HPC heatmap overlays
# ==========================================================================
def _overlay(slide, kind):
    key = (slide, kind)
    if key not in _overlay_cache:
        cfg = active_cfg()
        work, dataset, model_key, wsi_dir = _paths(cfg)
        _overlay_cache[key] = overlay_render.render_overlay_png(
            work, dataset, model_key, wsi_dir, slide, kind)
    return _overlay_cache[key]


@app.route("/overlay/<slide>/<kind>.png")
def overlay_png(slide, kind):
    if kind not in ("attention", "hpc"):
        abort(404)
    png, _placement, _legend = _overlay(slide, kind)
    if png is None:
        abort(404)
    return Response(png, mimetype="image/png",
                    headers={"Cache-Control": "no-cache"})


@app.route("/overlay/<slide>/<kind>.json")
def overlay_meta(slide, kind):
    if kind not in ("attention", "hpc"):
        abort(404)
    png, placement, legend = _overlay(slide, kind)
    return jsonify({"available": png is not None,
                    "placement": placement, "legend": legend})


# ==========================================================================
# DeepZoom tile server (openslide) for the interactive viewer
# ==========================================================================
def _dz(slide):
    if slide not in _dz_cache:
        cfg = active_cfg()
        _, _, _, wsi_dir = _paths(cfg)
        wsi = overlay_render.find_wsi(wsi_dir, slide)
        if not wsi:
            _dz_cache[slide] = None
        else:
            from openslide import OpenSlide
            from openslide.deepzoom import DeepZoomGenerator
            osr = OpenSlide(wsi)
            _dz_cache[slide] = DeepZoomGenerator(
                osr, tile_size=254, overlap=1, limit_bounds=True)
    return _dz_cache[slide]


@app.route("/dzi/<slide>.dzi")
def dzi(slide):
    dz = _dz(slide)
    if dz is None:
        abort(404)
    return Response(dz.get_dzi("jpeg"), mimetype="application/xml")


@app.route("/dzi/<slide>_files/<int:level>/<int:col>_<int:row>.<fmt>")
def dzi_tile(slide, level, col, row, fmt):
    dz = _dz(slide)
    if dz is None:
        abort(404)
    try:
        tile = dz.get_tile(level, (col, row))
    except (ValueError, IndexError):
        abort(404)
    buf = io.BytesIO()
    tile.save(buf, "jpeg", quality=80)
    return Response(buf.getvalue(), mimetype="image/jpeg")


# ==========================================================================
def main():
    global BASE_CONFIG
    ap = argparse.ArgumentParser(description="PanColon-CHiPS cluster web interface.")
    ap.add_argument("--config", default=BASE_CONFIG, help="Base pipeline config.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    BASE_CONFIG = os.path.abspath(args.config)
    if not os.path.isfile(BASE_CONFIG):
        sys.exit(f"Config not found: {BASE_CONFIG}\n"
                 "Copy config/pipeline.local.example.yaml to config/pipeline.yaml first.")
    print(f"[webapp] config = {BASE_CONFIG}")
    print(f"[webapp] open http://{args.host}:{args.port}  (SSH-tunnel from your laptop)")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
