# CLAMFamily: Unified Pipeline for Classification, Survival, and Regression on Whole-Slide Images

This repository extends and generalizes the [CLAM (Lu et al., Nature Biomedical Engineering, 2021)](https://www.nature.com/articles/s41551-020-00682-w) framework to support **multiple downstream tasks** (classification, survival prediction, and regression), while maintaining patient-level consistency, covariate integration, and flexible stratified splitting.

---

## ✨ Key Features

- **Three Task Types**
  - 🩸 **Classification** — multi-class MIL with attention pooling
  - ⏳ **Survival Analysis** — Cox proportional hazards loss (CLAM-Survival extension)
  - 📈 **Regression** — continuous outcome prediction with MSE/Huber loss and R²/RMSE evaluation

- **Patient- and Slide-Level Bag Control**
  - `--bag_level patient` → one bag per patient (aggregating all slides)
  - `--bag_level slide` → one bag per slide (classic CLAM setup)

- **Improved Split Generation**
  - Stratified Monte-Carlo sampling in fold creation for all task types
  - Quantile-based **regression bins** (`--reg_bins`)
  - **Fixed test set** across all folds for consistent evaluation
  - Safe `--test_frac 0` support (CV without test, for an external/separate test set)

- **Covariate Fusion**
  - Joint modeling of clinical variables (e.g. `Age`, `Sex`)
  - Supports `--cov_fusion concat` or `--cov_fusion cox_additive` (for survival)

- **Early Stopping**
  - Optional (`--early_stopping`) on the task's validation metric

---

## 🚀 What to run (pick your task)

Every task follows the same **three steps** — `split → train → eval` — driven by three
entrypoints (`create_splits_seq.py`, `main.py`, `eval.py`). Copy-paste-ready commands
live in the per-task guides:

| Task | Guide | Loss / metric |
|------|-------|---------------|
| Classification | [`README_classification.md`](README_classification.md) | cross-entropy / AUC |
| Survival | [`README_survival.md`](README_survival.md) | Cox partial likelihood / C-index |
| Regression | [`README_regression.md`](README_regression.md) | Huber (or MSE) / R², RMSE |

### Prerequisites

Activate the conda environment before running anything (provides the correct
`python`, torch, numpy, pandas):

```bash
source /gpfs/scratch/leh06/07_CLAM/code/CLAM/clam_latest.sh   # conda env: clam_latest (py3.10)
```

Inputs you supply:
- **`--clinical_csv`** — one row per slide, with `slide_id` + the label/time/target column(s).
- **`--feature_dir`** — the leaf directory holding per-slide `*.pt` embedding files.

### Output layout (run signature)

All three steps agree on one on-disk layout keyed by a **run signature** derived from
the task, target, covariates, and bag level:

```
{runs_root}/{dataset_name}/{run_sig}/
  splits/label_frac_100/splits_{0..k-1}.csv     # from create_splits_seq.py
  results/{exp_code}_s{seed}/                    # checkpoints + per-fold metrics (main.py)
  eval/{exp_code}_s{seed}/                       # per-fold predictions + summary.csv (eval.py)
```

Example run signatures:
- Classification, patient bags, no covariates: `classification__classification__no_covariates__patient_level`
- Regression, patient bags, no covariates: `regression__regression__no_covariates__patient_level`
- Survival (`time_col=os_event_data`), patient bags, `Age,Sex` covariates: `survival__os_event_data__age_sex_covariates__patient_level`

The split directory is resolved from `runs_root/dataset_name/run_sig` and is **independent
of `--exp_code`**, so multiple experiments (e.g. different losses) can share one fixed test set.

### SLURM

Self-contained SBATCH wrappers and orchestrators live in
`/gpfs/data/tsirigoslab/home/leh06/colon_project/CLAMFamily/`
(e.g. `sbatch run_clam_surv_pipeline_TCGA_LUAD_classification.sh`). Each per-task guide
points to the matching wrapper.
