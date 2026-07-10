# Running the PanColon-CHiPS pipeline

Three ways to run the same pipeline, from one recipe. Pick whichever matches the
machine you have. **However you run the compute, the results viewer is separate
and needs nothing** (see [the viewer](#the-results-viewer)).

You always mount/point at three things:

| what | where | contents |
|------|-------|----------|
| **slides** | `/data` | your `.svs` / `.ndpi` / `.tif` whole-slide images |
| **weights** | `/weights` | the model bundle from Zenodo (`scripts/download_weights.sh`) |
| **outputs** | `/out` | all results, incl. `/out/bundle` (the viewer) |

---

## 1. Docker (workstation with Docker)

```bash
# build once
docker build -t pancolon-chips:latest -f docker/Dockerfile .

# run the whole pipeline + export the results bundle
docker run --rm --gpus all \
  -v /path/to/slides:/data \
  -v /path/to/weights:/weights \
  -v /path/to/out:/out \
  pancolon-chips:latest
```

Drop `--gpus all` to run CPU-only (slower; fine for a few slides). Run a single
step instead of the whole thing by passing arguments, e.g.
`docker run … pancolon-chips:latest list` or `… export`.

## 2. Singularity / Apptainer (HPC, no root, no Docker)

Docker needs root and is banned on most clusters — use Apptainer/Singularity.
Build a `.sif` without root:

```bash
apptainer build --fakeroot pancolon.sif docker/pancolon.def

apptainer run --nv \
  -B /path/to/slides:/data \
  -B /path/to/weights:/weights \
  -B /path/to/out:/out \
  pancolon.sif
```

Drop `--nv` for CPU-only. If you already built the Docker image on a build box,
you can convert it instead of rebuilding:
`apptainer build pancolon.sif docker-daemon://pancolon-chips:latest`.

## 3. Conda envs (no container runtime at all)

If the machine has `conda` but no Docker/Apptainer:

```bash
conda env create -f envs/env_tiling.yml      # steps 1-5 (TensorFlow)
conda env create -f envs/env_survclam.yml    # steps 6-8 (PyTorch)

cp config/pipeline.local.example.yaml config/pipeline.yaml
$EDITOR config/pipeline.yaml    # set paths.wsi_dir, paths.work_dir, envs.conda_sh,
                                # and weights.* (from scripts/download_weights.sh)

conda activate pancolon_survclam
bash scripts/run_local.sh config/pipeline.yaml     # runs steps 1-8
python pancolon_pipeline.py export --config config/pipeline.yaml   # results bundle
```

The driver activates the right env per step, so you drive from the survclam env.
Preview commands first with `--dry-run`.

---

## The results viewer

Every run writes a **self-contained results bundle** to `/out/bundle` (or
`<work_dir>/bundle`). It needs no GPU, no openslide, no internet — just Python:

```bash
cd /out/bundle
./view.sh            # macOS/Linux   (Windows: view.bat)
# open the printed http://127.0.0.1:8000
```

Per slide you get three synchronized deep-zoom panels — original H&E, HPC
assignments, SurvCLAM attention — that pan/zoom together, plus the CHiPS score
and HPC composition; the cohort tab has the score table and distribution.

---

## Notes & limitations

- **This is a Linux/x86 stack.** The container runs on any Linux host with a
  container runtime; on Apple-Silicon/Windows use Docker Desktop (x86 emulation,
  CPU-only, slow), and note the TensorFlow-1 tiling env may be unhappy on ARM.
  The **viewer** has no such limits and runs on any OS.
- **GPU is optional but recommended.** Steps 3/7/8 use it; CPU works for small
  cohorts.
- **The build's hardest part is the TF1 tiling env** (`env_tiling.yml`:
  TensorFlow 1.15 / CUDA 10). If the image fails to build there, that env is the
  place to adjust pins for your base image / driver.
- **Weights are not in the image.** Get them once with
  `scripts/download_weights.sh` (see `scripts/build_weights_bundle.sh` for what
  the bundle contains) and mount the folder at `/weights`.
