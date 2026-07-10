# REGRESSION (Continuous Outcome Prediction)

Predicts a continuous target (e.g., biomarker score, risk index) with `clam_family`.
The tuned default uses a **log1p** target transform + **Huber** loss and selects the
checkpoint on validation **RMSE**; evaluation reports **R²**, RMSE, MAE.

The pipeline uses the unified run-signature layout shared with classification and
survival:

```
{runs_root}/{dataset_name}/{run_sig}/
  splits/label_frac_100/splits_{0..k-1}.csv
  results/{exp_code}_s{seed}/
  eval/{exp_code}_s{seed}/
```

For regression at patient level with no covariates:
`run_sig = regression__regression__no_covariates__patient_level`

> **Before you start**, activate the environment:
> ```bash
> source /gpfs/scratch/leh06/07_CLAM/code/CLAM/clam_latest.sh
> ```

---

### 1. Create regression splits (10-fold CV, fixed test set)

`create_splits_seq.py` writes `splits_{0..k-1}.csv` into the run-sig layout above.
With `--k > 1` it produces **stratified k-fold CV with a single fixed test set held
out across all folds** (regression folds are stratified by quantile bins of the
target via `--reg_bins`). `--k 1` produces a single final split (no test set).

```bash
RUNS_ROOT="/gpfs/scratch/leh06/CLAMFamily/runs"
DATASET_NAME="tcga_luad_til_regional_fraction"
CLINICAL_CSV="/gpfs/scratch/leh06/CLAMFamily/dataset_csv/tcga_luad_til_regional_fraction_titan_20x.csv"

python create_splits_seq.py \
  --dataset_name "$DATASET_NAME" \
  --clinical_csv "$CLINICAL_CSV" \
  --runs_root "$RUNS_ROOT" \
  --task_type regression \
  --target_col score \
  --bag_level patient \
  --pt_id_col slide_id \
  --reg_bins 4 \
  --k 10 \
  --val_frac 0.1 \
  --test_frac 0.1 \
  --seed 1
```

### 2. Train across folds

The flags below are the tuned configuration (log1p + Huber, RMSE-based selection).
The target is z-scored per fold in log1p space; the stats are baked into the model
buffers so predictions come back in **raw target units** automatically.

```bash
RUNS_ROOT="/gpfs/scratch/leh06/CLAMFamily/runs"
FEATURES_ROOT="/gpfs/scratch/leh06/CLAMFamily/datasets"
DATASET_NAME="tcga_luad_til_regional_fraction"
CLINICAL_CSV="/gpfs/scratch/leh06/CLAMFamily/dataset_csv/tcga_luad_til_regional_fraction_titan_20x.csv"
FEATURE_DIR="$FEATURES_ROOT/tcga_luad_til_regional_fraction_titan_features_20x/pt_files"
EMBED_DIM=768                 # raw TITAN features (no PCA)
EXP_CODE="titan20x_clam_reg"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_name "$DATASET_NAME" \
  --clinical_csv "$CLINICAL_CSV" \
  --runs_root "$RUNS_ROOT" \
  --task_type regression \
  --target_col score \
  --bag_level patient \
  --pt_id_col slide_id \
  --features_root "$FEATURES_ROOT" \
  --feature_key titan_20x \
  --feature_dir "$FEATURE_DIR" \
  --model_type clam_family \
  --model_size small \
  --embed_dim $EMBED_DIM \
  --drop_out 0.25 \
  --max_epochs 200 \
  --lr 2e-4 \
  --opt adam \
  --reg 1e-4 \
  --standardize_target \
  --target_transform log1p \
  --reg_loss huber \
  --huber_delta 1.0 \
  --lambda_pred_l2 0.0 \
  --es_patience 25 \
  --es_min_epoch 40 \
  --plateau_patience 6 \
  --k 10 \
  --exp_code "$EXP_CODE" \
  --seed 1 \
  --early_stopping
```

Notes:
- Model selection (early stopping + LR scheduler) is on validation **RMSE** (minimized),
  which is less noisy than R² on small (~40-patient) validation folds.
- `--reg_loss huber` (SmoothL1, `--huber_delta 1.0`) down-weights the right-skewed
  outlier tail; drop it (or use `--reg_loss mse`) for a plain MSE baseline.
- `--target_transform none` disables the log1p compression.
- For regression there is no alpha sweep, so each fold trains once.

### 3. Evaluate

`eval.py` resolves checkpoints from the same `run_sig` + `exp_code` + `seed`, loops
folds `0..k-1`, and writes per-fold predictions + `summary.csv` (columns
`fold,rmse,mae,mse,r2`) under
`{runs_root}/{dataset_name}/{run_sig}/eval/{exp_code}_s{seed}/`. Predictions are in raw
`score` units (the log1p/standardization is inverted inside the model's forward pass).

```bash
RUNS_ROOT="/gpfs/scratch/leh06/CLAMFamily/runs"
DATASET_NAME="tcga_luad_til_regional_fraction"
CLINICAL_CSV="/gpfs/scratch/leh06/CLAMFamily/dataset_csv/tcga_luad_til_regional_fraction_titan_20x.csv"
FEATURE_DIR="/gpfs/scratch/leh06/CLAMFamily/datasets/tcga_luad_til_regional_fraction_titan_features_20x/pt_files"
EXP_CODE="titan20x_clam_reg"

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --dataset_name "$DATASET_NAME" \
  --clinical_csv "$CLINICAL_CSV" \
  --runs_root "$RUNS_ROOT" \
  --exp_code "$EXP_CODE" \
  --seed 1 \
  --task_type regression \
  --target_col score \
  --bag_level patient \
  --pt_id_col slide_id \
  --feature_dir "$FEATURE_DIR" \
  --model_type clam_family \
  --model_size small \
  --embed_dim 768 \
  --drop_out 0.25 \
  --k 10 \
  --split test          # train | val | test | all
```

---

### SLURM pipeline

The three steps are wrapped as self-contained SBATCH scripts in
`colon_project/CLAMFamily/` and chained by one orchestrator:

```bash
cd /gpfs/data/tsirigoslab/home/leh06/colon_project/CLAMFamily
sbatch run_clam_surv_pipeline_TCGA_LUAD_regression.sh
```

This submits `01_sb_split_regression.py` → `02_train_regression.py`
→ `03_eval_regression.py` with `afterok` dependencies.
