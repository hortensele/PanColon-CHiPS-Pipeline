# CLASSIFICATION (Patient-Level or Slide-Level MIL)

Multi-class MIL classification with attention pooling using `clam_family`.
Cross-entropy loss; model selection maximizes validation **AUC**.

The pipeline uses the unified run-signature layout shared with survival and regression:

```
{runs_root}/{dataset_name}/{run_sig}/
  splits/label_frac_100/splits_{0..k-1}.csv
  results/{exp_code}_s{seed}/
  eval/{exp_code}_s{seed}/
```

For classification at patient level with no covariates:
`run_sig = classification__classification__no_covariates__patient_level`

> **Before you start**, activate the environment:
> ```bash
> source /gpfs/scratch/leh06/07_CLAM/code/CLAM/clam_latest.sh
> ```
> `n_classes` is inferred from `--label_map` (e.g. `"low:0,high:1"` → 2 classes).

---

### 1. Create stratified splits (10-fold CV, fixed test set)

```bash
RUNS_ROOT="/gpfs/scratch/leh06/CLAMFamily/runs"
DATASET_NAME="tcga_luad_til_regional_fraction_binary"
CLINICAL_CSV="/gpfs/scratch/leh06/CLAMFamily/dataset_csv/tcga_luad_til_regional_fraction_binary_titan_20x.csv"

python create_splits_seq.py \
  --dataset_name "$DATASET_NAME" \
  --clinical_csv "$CLINICAL_CSV" \
  --runs_root "$RUNS_ROOT" \
  --task_type classification \
  --label_col label \
  --label_map "low:0,high:1" \
  --bag_level patient \
  --pt_id_col slide_id \
  --k 10 \
  --val_frac 0.1 \
  --test_frac 0.1 \
  --seed 1
```

### 2. Train across folds

```bash
RUNS_ROOT="/gpfs/scratch/leh06/CLAMFamily/runs"
FEATURES_ROOT="/gpfs/scratch/leh06/CLAMFamily/datasets"
DATASET_NAME="tcga_luad_til_regional_fraction_binary"
CLINICAL_CSV="/gpfs/scratch/leh06/CLAMFamily/dataset_csv/tcga_luad_til_regional_fraction_binary_titan_20x.csv"
FEATURE_DIR="$FEATURES_ROOT/tcga_luad_til_regional_fraction_binary_titan_features_20x/pt_files"
EMBED_DIM=768                 # raw TITAN features (no PCA)
EXP_CODE="titan20x_clam_cls"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_name "$DATASET_NAME" \
  --clinical_csv "$CLINICAL_CSV" \
  --runs_root "$RUNS_ROOT" \
  --task_type classification \
  --label_col label \
  --label_map "low:0,high:1" \
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
  --reg 1e-5 \
  --k 10 \
  --exp_code "$EXP_CODE" \
  --seed 1 \
  --early_stopping
```

### 3. Evaluate

`eval.py` resolves checkpoints from the same `run_sig` + `exp_code` + `seed`, loops folds
`0..k-1`, and writes per-fold predictions + `summary.csv` under
`{runs_root}/{dataset_name}/{run_sig}/eval/{exp_code}_s{seed}/`.

```bash
RUNS_ROOT="/gpfs/scratch/leh06/CLAMFamily/runs"
DATASET_NAME="tcga_luad_til_regional_fraction_binary"
CLINICAL_CSV="/gpfs/scratch/leh06/CLAMFamily/dataset_csv/tcga_luad_til_regional_fraction_binary_titan_20x.csv"
FEATURE_DIR="/gpfs/scratch/leh06/CLAMFamily/datasets/tcga_luad_til_regional_fraction_binary_titan_features_20x/pt_files"
EXP_CODE="titan20x_clam_cls"

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --dataset_name "$DATASET_NAME" \
  --clinical_csv "$CLINICAL_CSV" \
  --runs_root "$RUNS_ROOT" \
  --exp_code "$EXP_CODE" \
  --seed 1 \
  --task_type classification \
  --label_col label \
  --label_map "low:0,high:1" \
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

```bash
cd /gpfs/data/tsirigoslab/home/leh06/colon_project/CLAMFamily
sbatch run_clam_surv_pipeline_TCGA_LUAD_classification.sh
```

Submits `01_sb_split_classification.py` → `02_train_classification.py`
→ `03_eval_classification.py` chained with `afterok` dependencies.
