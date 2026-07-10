# Vendored components

This directory holds **code-only** copies of the three upstream tools the
pipeline orchestrates. No trained weights, datasets, runs, logs, notebooks, or
reference data blobs are vendored — weights ship separately via
`scripts/download_weights.sh`. Refresh these trees with
`bash scripts/vendor_sync.sh` (edit the `*_SRC` paths for your clones).

Each upstream project keeps its own license; consult the upstream repositories.

---

## `vendor/deeppath/`  — DeepPATH

- **Upstream:** https://github.com/ncoudray/DeepPATH
- **Used for:** steps 1–2 (tiling + JPEG→HDF5).
- **Vendored:** `DeepPATH_code/00_preprocessing/` (tiling + hdf5 conversion),
  plus top-level `README.md` / `requirements.txt`.
- **Entry points the pipeline calls:**
  - `00_preprocessing/0b_tileLoop_deepzoom4.py` — DeepZoom tiling at target magnification.
  - `00_preprocessing/0e_jpgtoHDF.py` — pack tiles into an HDF5 store.
- **Excluded:** `example_*`, `archive/`, training/TFRecord code, all images.

## `vendor/hpl/`  — Histomorphological-Phenotype-Learning (HPL)

- **Upstream:** https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning
- **Used for:** steps 3–5 (encoder projection, Leiden clustering + assignment).
- **Vendored:** root `run_representations*.py`, plus the `models/`,
  `data_manipulation/`, and `utilities/` packages they import.
- **Entry points the pipeline calls:**
  - `run_representationspathology_projection_dataset.py` — project tiles through
    the trained BarlowTwins_3 encoder → `img_z_latent` embeddings.
  - `run_representationsleiden_assignment.py` — assign new tiles to the shipped
    reference Leiden clustering (HPCs).
  - `run_representationsleiden.py` — kept for reference (fits clustering; not run
    in inference).
- **Excluded:** `data_model_output/`, `results/`, `utilities/files/` (reference
  data), `*-Copy*.py` per-cohort variants, notebooks, logs.
- **Note:** the SLURM/`sb_*` wrapper scripts are intentionally NOT vendored; the
  pipeline calls the canonical `run_*.py` scripts directly with resolved args.

## `vendor/survclam/`  — SurvCLAM (our survival engine, built on CLAM)

- **Origin:** SurvCLAM is our own survival-analysis engine, developed for the
  PanColon-CHiPS study on top of **CLAM** (Lu et al., Mahmood Lab —
  https://github.com/mahmoodlab/CLAM). It is not a third-party dependency like
  DeepPATH/HPL; please cite CLAM if you use it.
- **Source:** https://github.com/tsirigoslab/SurvCLAM
- **Used for:** steps 6–8 (.pt feature store, survival inference, attention/CHiPS).
- **Vendored:** root `save_embeddings_hpl.py`, `eval.py`, `main.py`,
  `extract_last_layer_and_attention.py`, `create_splits_*.py`, the `README_*.md`
  docs, and the `utils/`, `models/`, `dataset_modules/` packages.
- **Entry points the pipeline calls:**
  - `save_embeddings_hpl.py` — HPL embeddings HDF5 → SurvCLAM `.pt` feature store.
  - `eval.py` — external inference (`--split all`) across the CHiPS fold
    checkpoints → per-fold `predictions.csv` (`risk`).
  - `extract_last_layer_and_attention.py` — per-tile attention + risk for maps.
- **Excluded:** `runs*/`, `datasets/`, `dataset_csv/`, `pca_models*/`,
  `eval_results/`, logs.

---

## Refresh procedure

```bash
# from clones (defaults point at the lab's local paths):
DEEPPATH_SRC=/path/to/DeepPATH \
HPL_SRC=/path/to/Histomorphological-Phenotype-Learning \
SURVCLAM_SRC=/path/to/SurvCLAM \
bash scripts/vendor_sync.sh
```

If an upstream refactor moves an entry point, update both this manifest and the
corresponding `step_*` function in `pancolon/steps.py`.
