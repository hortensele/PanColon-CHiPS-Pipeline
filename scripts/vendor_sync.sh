#!/usr/bin/env bash
# ==========================================================================
# Populate vendor/ with CODE-ONLY copies of the three upstream tools.
# Data, trained weights, runs, logs, notebooks and per-cohort script variants
# are excluded (weights ship separately via scripts/download_weights.sh).
#
#   bash scripts/vendor_sync.sh
#
# Override the source locations if your clones live elsewhere:
#   DEEPPATH_SRC=... HPL_SRC=... SURVCLAM_SRC=... bash scripts/vendor_sync.sh
#
# Re-run to refresh vendored code after pulling upstream changes.
# ==========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEEPPATH_SRC="${DEEPPATH_SRC:-/gpfs/data/tsirigoslab/home/leh06/PathGAN/DeepPATH}"
HPL_SRC="${HPL_SRC:-/gpfs/data/tsirigoslab/home/leh06/Histomorphological-Phenotype-Learning}"
SURVCLAM_SRC="${SURVCLAM_SRC:-/gpfs/scratch/leh06/CLAMFamily}"

COMMON_EXCLUDES=(
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc'
  --exclude '.ipynb_checkpoints' --exclude '*.ipynb'
  --exclude '*.out' --exclude '*.err' --exclude 'logs'
  --exclude '*.h5' --exclude '*.h5ad' --exclude '*.pt' --exclude '*.pkl'
  --exclude '*.csv' --exclude '*.npz' --exclude '*.png' --exclude '*.jpg'
  --exclude '*.pdf' --exclude '*.gz' --exclude '*.tar' --exclude '*.zip'
  # HPL: 'utilities/files' holds reference data blobs, not code
  --exclude 'files' --exclude 'data'
)

echo "[vendor_sync] DeepPATH  <- $DEEPPATH_SRC"
rsync -a "${COMMON_EXCLUDES[@]}" \
  --exclude 'example_*' --exclude 'archive' \
  "$DEEPPATH_SRC/DeepPATH_code/00_preprocessing" \
  "$DEEPPATH_SRC/README.md" "$DEEPPATH_SRC/requirements.txt" \
  "$REPO_ROOT/vendor/deeppath/DeepPATH_code/" 2>/dev/null || \
rsync -a "${COMMON_EXCLUDES[@]}" --exclude 'example_*' --exclude 'archive' \
  "$DEEPPATH_SRC/DeepPATH_code/00_preprocessing" \
  "$REPO_ROOT/vendor/deeppath/DeepPATH_code/"

echo "[vendor_sync] HPL       <- $HPL_SRC"
rsync -a "${COMMON_EXCLUDES[@]}" \
  --exclude 'data_model_output' --exclude 'results' --exclude 'datasets/*_h5' \
  --exclude '*-Copy*.py' --exclude 'flatten_nodes_analysis' \
  --include 'run_*.py' \
  "$HPL_SRC/data_manipulation" "$HPL_SRC/models" "$HPL_SRC/utilities" \
  "$REPO_ROOT/vendor/hpl/" 2>/dev/null || true
# root-level run_*.py scripts the pipeline calls
rsync -a "${COMMON_EXCLUDES[@]}" \
  "$HPL_SRC"/run_representationspathology_projection_dataset.py \
  "$HPL_SRC"/run_representationsleiden.py \
  "$HPL_SRC"/run_representationsleiden_assignment.py \
  "$REPO_ROOT/vendor/hpl/" 2>/dev/null || true

echo "[vendor_sync] SurvCLAM  <- $SURVCLAM_SRC"
rsync -a "${COMMON_EXCLUDES[@]}" \
  --exclude 'runs' --exclude 'runs_final' --exclude 'datasets' \
  --exclude 'dataset_csv' --exclude 'pca_models*' --exclude 'eval_results' \
  --exclude 'umap_cohort_comparison' \
  "$SURVCLAM_SRC/utils" "$SURVCLAM_SRC/models" "$SURVCLAM_SRC/dataset_modules" \
  "$REPO_ROOT/vendor/survclam/" 2>/dev/null || true
rsync -a "${COMMON_EXCLUDES[@]}" \
  "$SURVCLAM_SRC"/save_embeddings_hpl.py \
  "$SURVCLAM_SRC"/save_embeddings_foundation.py \
  "$SURVCLAM_SRC"/eval.py "$SURVCLAM_SRC"/main.py \
  "$SURVCLAM_SRC"/extract_last_layer_and_attention.py \
  "$SURVCLAM_SRC"/create_splits_seq.py "$SURVCLAM_SRC"/create_splits_loo.py \
  "$SURVCLAM_SRC"/README*.md \
  "$REPO_ROOT/vendor/survclam/" 2>/dev/null || true

echo "[vendor_sync] Done. Vendored trees:"
du -sh "$REPO_ROOT"/vendor/* 2>/dev/null
