# PanColon-CHiPS-Pipeline

End-to-end **inference** pipeline that takes colon-cancer whole-slide images
(WSIs) from your own cohort and produces, per slide/patient, a **CHiPS**
(Computational Histological Prognostic Score) and an attention heatmap overlay —
using the trained HPL-PanColon encoder and the SurvCLAM survival model from the
PanColon-CHiPS study. It is designed to be cloned, pointed at a folder of slides,
and run **locally on a GPU workstation** or **on a SLURM cluster**.

> This repo is inference-only. It does not train anything. Trained weights are
> downloaded separately (see [Weights](#3-download-the-trained-weights)).

```
WSIs ──▶ tile ──▶ hdf5 ──▶ HPL encoder ──▶ HPC assign + artifact filter
     ──▶ .pt store ──▶ SurvCLAM folds ──▶ CHiPS score ──▶ attention overlays
```

## The eight steps

| # | step | what it does | tool | env |
|---|------|--------------|------|-----|
| 1 | `tile` | tile WSIs at 20x | DeepPATH | tiling |
| 2 | `to_hdf5` | pack tiles into HDF5 | DeepPATH | tiling |
| 3 | `project` | project tiles through the trained HPL encoder → embeddings | HPL | tiling |
| 4 | `cluster_filter` | assign tiles to the reference HPCs (Leiden) | HPL | tiling |
| 5 | `assign_hpc` | drop artifact clusters, write the filtered HPC table | (built-in) | tiling |
| 6 | `build_pt` | build the SurvCLAM `.pt` feature store | SurvCLAM | survclam |
| 7 | `infer_survival` | score every slide with the CHiPS fold checkpoints | SurvCLAM | survclam |
| 8 | `attention_map` | per-tile attention + CHiPS-annotated overlays | SurvCLAM | survclam |

The shipped CHiPS model is the **imaging-only DFS** model (16 leave-one-institution-out
folds); the score is the mean per-fold risk. No clinical/outcome data is required to
score slides.

## Hardware & software

- **GPU** strongly recommended (steps 3, 7, 8 use it). CPU-only works for small
  cohorts but step 3 is slow.
- Linux + `conda`. OpenSlide is installed via the conda envs.
- Two conda environments (created once, below). The pipeline is split because
  DeepPATH/HPL are TensorFlow-1.x era and SurvCLAM is PyTorch — they cannot share
  one environment.

## Quickstart

### 1. Create the two environments

```bash
conda env create -f envs/env_tiling.yml      # steps 1-5 (TensorFlow)
conda env create -f envs/env_survclam.yml     # steps 6-8 (PyTorch)
```

**Already have the TF stack as a cluster module?** You can skip building
`env_tiling` and point the tiling stage at an existing environment module
instead. In the config, set `envs.tiling_module` (it takes precedence over
`envs.tiling`), e.g. on NYU BigPurple:

```yaml
envs:
  tiling_module: "condaenvs/gpu/pathgan_SSL"   # `module load`ed for steps 1-5
```

The driver then `module load`s it for steps 1–5 rather than `conda activate`.
(`module_init` can point at your `modules.sh` if `module` isn't on PATH in
non-interactive shells.)

### 2. Configure

```bash
cp config/pipeline.local.example.yaml config/pipeline.yaml
$EDITOR config/pipeline.yaml     # set paths.wsi_dir, paths.work_dir, envs.conda_sh
```

Every path and parameter lives in that one file. `--dry-run` (below) prints the
exact commands so you can sanity-check before running anything.

### 3. Download the trained weights

```bash
$EDITOR scripts/download_weights.sh    # set PUBLIC_URL (+ SHA256)
bash scripts/download_weights.sh
```

This unpacks the HPL encoder, the reference Leiden clustering, and the SurvCLAM
CHiPS fold checkpoints into `weights/`.

**Publishing the weights (maintainers):** `scripts/build_weights_bundle.sh
--config config/pipeline.yaml` assembles that bundle from the lab installs —
the encoder, the reference clustering (both reference h5s + folds pickle + the
`cohort`/`cohort_cleaned` Leiden results), and the 16 DFS CHiPS fold checkpoints
— verifies every file, writes `SHA256SUMS`, and tars
`pancolon_chips_weights.tar.gz` ready to upload to Zenodo. It prints the tar's
SHA256 to paste into `download_weights.sh`. Set `HPL_REF_DIR` (and confirm
`infer.exp_code` matches the shipped run) before running.

### 4. Run

Local (the driver activates the right env for each step):

```bash
bash scripts/run_local.sh config/pipeline.yaml
# or a sub-range:
python pancolon_pipeline.py all --config config/pipeline.yaml --from project --to infer_survival
# a single step:
python pancolon_pipeline.py infer_survival --config config/pipeline.yaml
```

SLURM (submits the whole chain with `afterok` dependencies):

```bash
bash scripts/slurm/submit_all.sh config/pipeline.yaml
```

Preview without executing (works anywhere, no envs needed):

```bash
python pancolon_pipeline.py all --config config/pipeline.yaml --dry-run --no-env-switch
```

## Sharing with collaborators (portable run + zero-install viewer)

For collaborators whose resources you don't control, the pipeline is packaged to
run three ways from one recipe — **Docker**, **Singularity/Apptainer** (HPC, no
root), or **plain conda** — and it exports a **self-contained results viewer**
that opens with nothing but Python. See **[docker/README.md](docker/README.md)**
for the full instructions; the short version:

```bash
# build once (Docker shown; Apptainer: apptainer build --fakeroot pancolon.sif docker/pancolon.def)
docker build -t pancolon-chips:latest -f docker/Dockerfile .

# run: slides in, results (incl. the viewer) out
docker run --rm --gpus all \
  -v /path/to/slides:/data -v /path/to/weights:/weights -v /path/to/out:/out \
  pancolon-chips:latest
```

This runs steps 1–8 then `pancolon_pipeline.py export`, which writes a
**results bundle** to `<work_dir>/bundle` — deep-zoom H&E, HPC, and attention
layers per slide, plus the cohort CSV and a copied static viewer. Open it
anywhere (no GPU/openslide needed):

```bash
cd <work_dir>/bundle && ./view.sh    # then open the printed http://127.0.0.1:8000
```

The viewer shows, per slide, three synchronized deep-zoom panels (original H&E,
HPC assignments, SurvCLAM attention) that pan/zoom together, plus the CHiPS score
and HPC composition; a cohort tab has the score table and distribution. You can
also run `export` on its own after a normal run:
`python pancolon_pipeline.py export --config config/pipeline.yaml`.

## Interactive app (upload slides, run on the cluster, explore results)

A web UI wraps the pipeline: point at (or upload) one or more whole-slide images,
**submit all eight steps to SLURM**, watch each step run, then explore the results
**per image** — the CHiPS score, the HPC phenotype distribution, and a zoomable
attention map.

Run it on a **login node** (it calls `sbatch`/`squeue`/`sacct`), in the survclam
env, then SSH-tunnel the port:

```bash
conda activate pancolon_survclam      # flask + openslide + torch + pyyaml
bash scripts/run_webapp.sh config/pipeline.yaml --port 5000
# from your laptop:  ssh -L 5000:127.0.0.1:5000 <login-node>
# then open http://127.0.0.1:5000
```

What it does:

- **Submit** writes a per-run config and submits the `stage.sbatch` dependency
  chain (GPU for `project`/`infer_survival`/`attention_map`, CPU otherwise); the
  step tracker follows `sacct`/`squeue` and streams each step's SLURM log. Tick
  **Dry run** to preview the exact `sbatch` commands without submitting.
- **Per-image explorer** (once the run finishes): a cohort CHiPS table plus a
  slide picker. Each slide shows its CHiPS score/percentile/tertile, an HPC
  composition chart, and an [OpenSeadragon](https://openseadragon.github.io/)
  deep-zoom view of the WSI (served from openslide) with a toggleable
  **attention ⇄ HPC** heatmap overlay and an opacity slider.

It binds to localhost and does no compute in the browser (it submits to SLURM and
reads back the pipeline's output files), so the downloaded weights must be
reachable from the cluster. OpenSeadragon is vendored under
`webapp/static/vendor/` — no CDN or internet is needed at runtime. The static
overview in `docs/pipeline_overview.html` is the non-interactive counterpart.

## Outputs

Everything lands under `paths.work_dir`:

```
work_dir/
  tiles/                       step 1
  hdf5/hdf5_<cohort>_he_complete.h5          step 2
  hpl/hdf5_<cohort>_..._filtered.h5          step 3  (img_z_latent)
  clusters/<cohort>_hpc_assignment.csv       steps 4-5 (filtered per-tile HPC)
  clusters/<cohort>_manifest.csv             per-slide manifest
  datasets/<cohort>/HPL_PANCOLON_20x/pt_files/*.pt   step 6
  survclam/chips_scores.csv                  step 7  <-- the CHiPS scores
  attention/                                 step 8  (per-tile attention + overlays)
```

`chips_scores.csv` columns: `case_id, chips_score, chips_percentile,
chips_tertile, n_folds, risk_fold0…`. For stratification use `chips_percentile`
or `chips_tertile` (cohort-relative), since the raw score is an uncentered Cox
log-hazard.

## Scoring cohorts with vs. without outcomes

- **No outcomes (pure scoring):** leave `paths.clinical_csv` blank. Step 5 writes
  a minimal manifest (dummy time/event) so the model runs; you still get CHiPS.
- **With outcomes (optional evaluation):** point `paths.clinical_csv` at a CSV
  carrying `slide_id`, `case_id`, and the `infer.time_col`/`infer.event_col`
  columns to additionally get C-index/KM from step 7.

## Attention overlays

After step 8, use the **interactive app** above for a zoomable attention/HPC
overlay per slide, or open `notebooks/attention_overlay.ipynb` (survclam env) to
render static per-slide H&E + attention + HPC panels annotated with each slide's
CHiPS score, adapted from the study's WSI overlay figure. Both read the same
per-slide outputs via `pancolon/overlay_render.py`.

## Layout

```
pancolon_pipeline.py      CLI entry point
pancolon/                 orchestration package (config, steps, CHiPS aggregation,
                          SLURM submit/monitor, per-slide overlay rendering)
webapp/                   Flask app: submit to SLURM + per-image results explorer
config/                   pipeline.yaml + filled example
envs/                     the two conda env specs
scripts/                  download_weights, run_local, run_webapp, vendor_sync, slurm/
vendor/                   code-only copies of DeepPATH / HPL / SurvCLAM (see VENDOR_MANIFEST.md)
weights/                  downloaded trained weights (gitignored)
notebooks/                attention overlay figure
```

## Configuration you must confirm

A few values in `config/pipeline.yaml` are model-specific and should match the
shipped reference clustering / checkpoints:

- `cluster.resolution` and `cluster.artifact_cluster_ids` — the Leiden resolution
  and the artifact HPC IDs of the reference clustering.
- `infer.exp_code`, `infer.time_col`, `infer.k` — identify the CHiPS checkpoints.

These ship pre-filled for the imaging-only DFS model; only change them if you
bundle a different model.

## Citation

PanColon-CHiPS study — *citation coming soon*. Upstream tools: DeepPATH,
Histomorphological-Phenotype-Learning (HPL), SurvCLAM (see `vendor/VENDOR_MANIFEST.md`).
