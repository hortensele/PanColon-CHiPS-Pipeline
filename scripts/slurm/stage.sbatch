#!/usr/bin/env bash
# ==========================================================================
# Generic SLURM stage runner: executes ONE pipeline step.
# Resources (partition, gres, mem, time) are supplied by submit_all.sh via
# sbatch CLI flags, which override any directive here — so this file carries
# only conservative defaults and no site-specific paths.
#
# Required environment variables (exported by submit_all.sh):
#   PANCOLON_STEP    e.g. project
#   PANCOLON_CONFIG  path to the pipeline YAML
#   PANCOLON_REPO    repo root
# Optional:
#   PANCOLON_PYTHON  python used to launch the driver (default: python)
# ==========================================================================
#SBATCH --job-name=pancolon
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00

set -euo pipefail

: "${PANCOLON_STEP:?set PANCOLON_STEP}"
: "${PANCOLON_CONFIG:?set PANCOLON_CONFIG}"
: "${PANCOLON_REPO:?set PANCOLON_REPO}"
PYTHON="${PANCOLON_PYTHON:-python}"

cd "$PANCOLON_REPO"
mkdir -p logs
echo "[slurm] step=${PANCOLON_STEP} config=${PANCOLON_CONFIG} host=$(hostname)"

# The driver activates the correct conda env for the step itself.
"$PYTHON" pancolon_pipeline.py "$PANCOLON_STEP" --config "$PANCOLON_CONFIG"
