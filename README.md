# PanColon-CHiPS-Pipeline

Give this pipeline a folder of colon-cancer whole-slide images (WSIs) and it
produces, per slide/patient:

- a **CHiPS** score (Computational Histological Prognostic Score),
- **HPC** assignments (Histological Phenotype Clusters) painted onto the slide, and
- a **SurvCLAM attention** heatmap,

then packages them into a **self-contained viewer** you open in a browser with
nothing but Python — no GPU, no OpenSlide, no install. It uses the trained
HPL-PanColon encoder and the SurvCLAM survival model from the PanColon-CHiPS study.

> **Inference only.** This repo does not train anything. The trained weights are
> downloaded separately (see [2. Download the weights](#2-download-the-trained-weights)).

```
WSIs ──▶ tile ──▶ hdf5 ──▶ HPL encoder ──▶ HPC assign + artifact filter
     ──▶ .pt store ──▶ SurvCLAM folds ──▶ CHiPS score ──▶ attention overlays
     ──▶ export ──▶ static results viewer (open in any browser)
```

The shipped CHiPS model is the **imaging-only DFS** model (16 leave-one-institution-out
folds); the score is the mean per-fold risk. **No clinical or outcome data is
required** to score your slides.

### The eight pipeline steps

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

A 9th step, `export`, turns those outputs into the browser viewer (below).

---

## What you need

- **Linux, x86-64.** This is a Linux/x86 stack (DeepPATH/HPL are TensorFlow-1.x
  era). On Apple-Silicon/Windows you'd need Docker Desktop's x86 emulation
  (CPU-only, slow). **The results viewer has no such constraint — it runs in any
  browser on any OS.**
- **A GPU is strongly recommended** (steps 3, 7, 8 use it). CPU-only works for
  small cohorts but step 3 (encoding) is slow.
- **One way to run the compute** — pick whichever you have:
  - **Docker** (a workstation), or
  - **Singularity/Apptainer** (an HPC cluster, no root needed), or
  - **conda** (a machine with conda but no container runtime).

  The container path is recommended because it bundles both environments and all
  dependencies. Full container instructions live in
  **[docker/README.md](docker/README.md)**; the essentials are below.

---

## Quickstart

### 1. Get the code

```bash
git clone <this-repo-url> PanColon-CHiPS-Pipeline
cd PanColon-CHiPS-Pipeline
```

### 2. Download the trained weights

```bash
bash scripts/download_weights.sh
```

This fetches and unpacks the HPL encoder, the reference Leiden clustering, and the
16 SurvCLAM CHiPS fold checkpoints into `weights/` (~4 GB). The download URL is
configured inside that script; if you get a "URL is not set" error the bundle
hasn't been published yet — ask the maintainer for the Zenodo link (or see
[For maintainers](#for-maintainers) to build it).

### 3. Run — pick one path

The two conda environments exist because DeepPATH/HPL are TensorFlow-1.x and
SurvCLAM is PyTorch; they cannot share one environment. The container builds both
for you.

#### Path A — Container (recommended)

**Docker:**

```bash
docker build -t pancolon-chips:latest -f docker/Dockerfile .

docker run --rm --gpus all \
  -v /path/to/slides:/data \
  -v $PWD/weights:/weights \
  -v /path/to/output:/out \
  pancolon-chips:latest
```

**Singularity/Apptainer** (no root; typical on HPC):

```bash
apptainer build --fakeroot pancolon.sif docker/pancolon.def
apptainer run --nv \
  -B /path/to/slides:/data -B $PWD/weights:/weights -B /path/to/output:/out \
  pancolon.sif
```

Either one runs steps 1–8 **and** `export`, writing everything (including the
viewer) to `/out`. See [docker/README.md](docker/README.md) for CPU-only runs,
troubleshooting, and build notes.

#### Path B — conda (no container)

```bash
conda env create -f envs/env_tiling.yml      # steps 1-5 (TensorFlow)
conda env create -f envs/env_survclam.yml     # steps 6-8 (PyTorch)

cp config/pipeline.local.example.yaml config/pipeline.yaml
$EDITOR config/pipeline.yaml     # set paths.wsi_dir, paths.work_dir, envs.conda_sh

bash scripts/run_local.sh config/pipeline.yaml     # activates the right env per step
python pancolon_pipeline.py export --config config/pipeline.yaml
```

Every path and parameter lives in that one config file. Useful variants:

```bash
# preview the exact commands without running anything (no envs needed):
python pancolon_pipeline.py all --config config/pipeline.yaml --dry-run --no-env-switch
# run a sub-range or a single step:
python pancolon_pipeline.py all --config config/pipeline.yaml --from project --to infer_survival
python pancolon_pipeline.py infer_survival --config config/pipeline.yaml
```

### 4. Explore the results in your browser

The run produces a **results bundle** at `<work_dir>/bundle` (or `/out/bundle`
from the container). It is fully self-contained — copy or zip it to any machine
and open it with only Python:

```bash
cd bundle && ./view.sh        # then open the printed http://127.0.0.1:8000
```

Per slide you get **three synchronized deep-zoom panels** — original H&E, HPC
assignments, and SurvCLAM attention — that pan and zoom **together**, alongside
the slide's CHiPS score and HPC composition. A **Cohort** tab shows the CHiPS
score table and distribution across all your slides. No GPU or OpenSlide needed
on the viewing machine.

---

## Outputs

Everything lands under `paths.work_dir`:

```
work_dir/
  tiles/                                       step 1
  hdf5/hdf5_<cohort>_he_complete.h5            step 2
  hpl/hdf5_<cohort>_..._filtered.h5            step 3  (img_z_latent)
  clusters/<cohort>_hpc_assignment.csv         steps 4-5 (filtered per-tile HPC)
  clusters/<cohort>_manifest.csv               per-slide manifest
  datasets/<cohort>/HPL_PANCOLON_20x/pt_files/*.pt   step 6
  survclam/chips_scores.csv                    step 7  <-- the CHiPS scores
  attention/                                   step 8  (per-tile attention + overlays)
  bundle/                                       export  (the browser viewer)
```

`chips_scores.csv` columns: `case_id, chips_score, chips_percentile,
chips_tertile, n_folds, risk_fold0…`. **For stratification use `chips_percentile`
or `chips_tertile`** (cohort-relative), since the raw score is an uncentered Cox
log-hazard.

### Scoring with vs. without outcomes

- **No outcomes (pure scoring):** leave `paths.clinical_csv` blank. Step 5 writes
  a minimal manifest (dummy time/event) so the model runs; you still get CHiPS.
- **With outcomes (optional evaluation):** point `paths.clinical_csv` at a CSV
  carrying `slide_id`, `case_id`, and the `infer.time_col`/`infer.event_col`
  columns to additionally get C-index / KM from step 7.

### Static figure alternative

Prefer a static figure to the interactive viewer? Open
`notebooks/attention_overlay.ipynb` (survclam env) to render per-slide
H&E + attention + HPC panels annotated with each slide's CHiPS score. It reads the
same per-slide outputs via `pancolon/overlay_render.py`.

---

## Configuration reference

A few values in `config/pipeline.yaml` identify the shipped model and reference
clustering. They come **pre-filled for the imaging-only DFS model** — only change
them if you deliberately bundle a different model:

- `cluster.resolution`, `cluster.artifact_cluster_ids` — the Leiden resolution and
  artifact HPC IDs of the reference clustering.
- `infer.exp_code`, `infer.time_col`, `infer.k` — identify the CHiPS checkpoints.
- `paths.wsi_dir`, `paths.work_dir`, `envs.conda_sh` — **you set these** to your
  slides folder, an output folder, and your conda's `conda.sh` (conda path only).

## Layout

```
pancolon_pipeline.py      CLI entry point
pancolon/                 orchestration package (config, steps, CHiPS aggregation,
                          export, per-slide overlay rendering)
viewer/                   the static browser viewer copied into every results bundle
docker/                   Dockerfile + Apptainer def + container README
config/                   container config + local-example template (copy to pipeline.yaml, gitignored)
envs/                     the two conda env specs
scripts/                  download_weights, run_local, build_weights_bundle, …
vendor/                   code-only copies of DeepPATH / HPL / SurvCLAM (see VENDOR_MANIFEST.md)
weights/                  downloaded trained weights (gitignored)
webapp/                   optional SLURM web UI (see below)
notebooks/                attention overlay figure
```

---

## Optional: SLURM web UI (if you run your own cluster)

If you already operate a **SLURM cluster**, `webapp/` is a browser UI that submits
the eight steps to SLURM, tracks each step, and then explores results per image.
It is an alternative to the command-line run above — not required, and unrelated to
the portable container path.

```bash
conda activate pancolon_survclam
bash scripts/run_webapp.sh config/pipeline.yaml --port 5000
# from your laptop:  ssh -L 5000:127.0.0.1:5000 <login-node>   →  http://127.0.0.1:5000
```

It binds to localhost, submits an `sbatch` dependency chain (GPU for
`project`/`infer_survival`/`attention_map`), streams each step's SLURM log, and
serves a per-image explorer with a live OpenSeadragon deep-zoom view of each WSI
and a toggleable attention ⇄ HPC overlay. OpenSeadragon is vendored under
`webapp/static/vendor/` — no internet needed at runtime. The downloaded weights
must be reachable from the cluster.

---

## For maintainers

Content below is for whoever **publishes** the weights or refreshes the vendored
tools — collaborators can ignore it.

**Building the weights bundle for Zenodo.** `scripts/build_weights_bundle.sh
--config config/pipeline.yaml` assembles the publishable bundle from the source
installs — the HPL encoder, the reference clustering (both anchor h5s + the
`cohort`/`cohort_cleaned` fold-1 Leiden adatas + folds pickle), and the 16 DFS
CHiPS fold checkpoints — into the `weights/` layout, verifies every file, writes
`SHA256SUMS`, and tars `pancolon_chips_weights.tar.gz` ready to upload. It prints
the tar's SHA256. Then set `PUBLIC_URL` and `EXPECTED_SHA256` in
`scripts/download_weights.sh` so collaborators' `download_weights.sh` works. The
source paths default to the lab installs and are overridable via env vars
(`HPL_INSTALL`, `HPL_REF_DIR`, `SURVCLAM_RUNS_SRC`, …) at the top of the script.

**Zenodo upload checklist.** Once `pancolon_chips_weights.tar.gz` is built:

1. **New record** at https://zenodo.org/uploads → drag in
   `pancolon_chips_weights.tar.gz` (2.3 GB; Zenodo allows up to 50 GB/record).
2. **Upload type:** Dataset (or Software/Model if you prefer). **Title:**
   `PanColon-CHiPS trained weights (HPL encoder + reference clustering + SurvCLAM DFS folds)`.
3. **Authors / Creators:** you + co-authors, with ORCID and affiliation.
4. **Description:** what the bundle contains and that it pairs with this repo —
   e.g. "Trained weights for the PanColon-CHiPS inference pipeline
   (github.com/hortensele/PanColon-CHiPS-Pipeline): HPL BarlowTwins_3 encoder, the
   colon reference Leiden clustering (fold-1 adatas + anchor h5s + folds pickle),
   and the 16 leave-one-institution-out SurvCLAM DFS CHiPS fold checkpoints.
   Imaging-only; no clinical data. Unpack with scripts/download_weights.sh."
5. **License:** pick one that matches the repo (e.g. MIT / CC-BY-4.0). Note the
   weights derive from CLAM/DeepPATH/HPL-based training — keep it compatible with
   those upstreams.
6. **Version:** `v1.0.0` (use Zenodo's versioning for future re-uploads so the DOI
   resolves to the latest while old versions stay pinned).
7. **Related identifiers:** add the GitHub repo URL as "is supplement to".
8. **Publish**, then copy the **file download URL** (the
   `…/records/<id>/files/pancolon_chips_weights.tar.gz` link, *not* the record page)
   into `PUBLIC_URL` in `scripts/download_weights.sh`. `EXPECTED_SHA256` is already
   set to the built tar's hash — leave it. Commit + push that one edit.
9. **Round-trip test:** in a clean checkout, run `bash scripts/download_weights.sh`
   and confirm it downloads, passes the checksum, and unpacks into `weights/`.

**Using an existing TF module instead of building `env_tiling`.** On a cluster
that already provides the TensorFlow stack as a module, set `envs.tiling_module`
in the config (it takes precedence over `envs.tiling`); the driver `module load`s
it for steps 1–5 rather than `conda activate`. `envs.module_init` can point at a
`modules.sh` if `module` isn't on PATH in non-interactive shells.

**Refreshing the vendored tools.** `scripts/vendor_sync.sh` re-pulls the
code-only copies of DeepPATH / HPL / SurvCLAM; see `vendor/VENDOR_MANIFEST.md`.

## Citation

PanColon-CHiPS study — *citation coming soon*.

**SurvCLAM** is our own survival-analysis engine (developed as part of this study),
built on top of **CLAM** (Lu et al., Mahmood Lab —
https://github.com/mahmoodlab/CLAM); please cite CLAM if you use it. It is
orchestrated alongside two third-party upstream tools: **DeepPATH** (Coudray et
al.) and **Histomorphological-Phenotype-Learning / HPL** (Claudio Quiros et al.).
Each vendored tool keeps its own upstream license; see
`vendor/VENDOR_MANIFEST.md`.
