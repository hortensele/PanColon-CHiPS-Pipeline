# SURVIVAL (Cox Proportional Hazards, with optional covariates)

Cox partial-likelihood survival modeling with `clam_family`, optionally fusing clinical
covariates (e.g. `Age`, `Sex`). Model selection maximizes the validation **C-index**.

The pipeline uses the unified run-signature layout shared with classification and regression:

```
{runs_root}/{dataset_name}/{run_sig}/
  splits/label_frac_100/splits_{0..k-1}.csv
  results/{exp_code}_s{seed}/
  eval/{exp_code}_s{seed}/
```

The survival `run_sig` encodes the event-time column and covariates. For
`--time_col os_event_data`, `--covariate_cols "Age,Sex"`, patient bags:
`run_sig = survival__os_event_data__age_sex_covariates__patient_level`
(with no covariates it becomes `survival__os_event_data__no_covariates__patient_level`).

> **Before you start**, activate the environment:
> ```bash
> source /gpfs/scratch/leh06/07_CLAM/code/CLAM/clam_latest.sh
> ```
> The clinical CSV must contain a **time** column (`--time_col`, continuous) and an
> **event** column (`--event_col`, 1=event/death, 0=censored).

---

### 1. Create stratified splits

Use `--test_frac 0` to run cross-validation **without** an internal test set (when you
hold out a separate external cohort); use `--test_frac 0.1` for a fixed internal test set.

```bash
RUNS_ROOT="/gpfs/scratch/leh06/CLAMFamily/runs"
DATASET_NAME="colon_united_os_main_HPL_PANCOLON_20x"
CLINICAL_CSV="/gpfs/scratch/leh06/CLAMFamily/dataset_csv/colon_united_os_main_HPL_PANCOLON_20x.csv"

python create_splits_seq.py \
  --dataset_name "$DATASET_NAME" \
  --clinical_csv "$CLINICAL_CSV" \
  --runs_root "$RUNS_ROOT" \
  --task_type survival \
  --time_col os_event_data \
  --event_col os_event_ind \
  --covariate_cols "Age,Sex" \
  --bag_level patient \
  --pt_id_col slide_id \
  --k 10 \
  --val_frac 0.1 \
  --test_frac 0 \
  --seed 1
```

### 2. Train across folds

```bash
RUNS_ROOT="/gpfs/scratch/leh06/CLAMFamily/runs"
FEATURES_ROOT="/gpfs/scratch/leh06/CLAMFamily/datasets"
DATASET_NAME="colon_united_os_main_HPL_PANCOLON_20x"
CLINICAL_CSV="/gpfs/scratch/leh06/CLAMFamily/dataset_csv/colon_united_os_main_HPL_PANCOLON_20x.csv"
FEATURE_DIR="$FEATURES_ROOT/colon_united_os_main_HPL_PANCOLON_features_20x/pt_files"
EMBED_DIM=128                 # HPL features
EXP_CODE="hpl20x_clam_surv"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_name "$DATASET_NAME" \
  --clinical_csv "$CLINICAL_CSV" \
  --runs_root "$RUNS_ROOT" \
  --task_type survival \
  --time_col os_event_data \
  --event_col os_event_ind \
  --covariate_cols "Age,Sex" \
  --cov_fusion concat \
  --bag_level patient \
  --pt_id_col slide_id \
  --features_root "$FEATURES_ROOT" \
  --feature_key hpl_20x \
  --feature_dir "$FEATURE_DIR" \
  --model_type clam_family \
  --model_size small \
  --embed_dim $EMBED_DIM \
  --drop_out 0.25 \
  --max_epochs 200 \
  --lr 2e-4 \
  --opt adam \
  --reg 1e-4 \
  --k 10 \
  --exp_code "$EXP_CODE" \
  --seed 1 \
  --early_stopping
```

Notes:
- `--cov_fusion concat` fuses covariates into the bag embedding before the risk head.
  Use `--cov_fusion cox_additive` to add an independent linear covariate risk term to the
  bag risk (classic Cox-style additive form). Omit `--covariate_cols` for imaging-only.
- Model selection (early stopping + LR scheduler) maximizes validation **C-index**.

### 3. Evaluate

```bash
RUNS_ROOT="/gpfs/scratch/leh06/CLAMFamily/runs"
DATASET_NAME="colon_united_os_main_HPL_PANCOLON_20x"
CLINICAL_CSV="/gpfs/scratch/leh06/CLAMFamily/dataset_csv/colon_united_os_main_HPL_PANCOLON_20x.csv"
FEATURE_DIR="/gpfs/scratch/leh06/CLAMFamily/datasets/colon_united_os_main_HPL_PANCOLON_features_20x/pt_files"
EXP_CODE="hpl20x_clam_surv"

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --dataset_name "$DATASET_NAME" \
  --clinical_csv "$CLINICAL_CSV" \
  --runs_root "$RUNS_ROOT" \
  --exp_code "$EXP_CODE" \
  --seed 1 \
  --task_type survival \
  --time_col os_event_data \
  --event_col os_event_ind \
  --covariate_cols "Age,Sex" \
  --cov_fusion concat \
  --bag_level patient \
  --pt_id_col slide_id \
  --feature_dir "$FEATURE_DIR" \
  --model_type clam_family \
  --model_size small \
  --embed_dim 128 \
  --drop_out 0.25 \
  --k 10 \
  --split val          # train | val | test | all  (use 'val' when test_frac=0)
```

Writes per-fold C-index + `summary.csv` under
`{runs_root}/{dataset_name}/{run_sig}/eval/{exp_code}_s{seed}/`. To score a **separate
external cohort**, point `--dataset_name` / `--clinical_csv` / `--feature_dir` at that cohort
(built with matching `--time_col`/`--event_col`) and use `--split all`.

---

### SLURM pipeline

```bash
cd /gpfs/data/tsirigoslab/home/leh06/colon_project/CLAMFamily
sbatch run_clam_surv_pipeline_COLON_UNITED_os.sh
```

Submits split → train → eval chained with `afterok` dependencies.
