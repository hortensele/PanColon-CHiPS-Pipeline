#!/usr/bin/env bash
# ==========================================================================
# Run the full pipeline locally (single GPU workstation, no scheduler).
#
#   bash scripts/run_local.sh [config/pipeline.yaml] [--dry-run] [extra args...]
#
# The pipeline driver activates the correct conda env per step, so you only
# need `conda` on PATH and the two envs created (see envs/). Extra args are
# passed straight through to `pancolon_pipeline.py all` (e.g. --from project).
# ==========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${REPO_ROOT}/config/pipeline.yaml}"
shift || true

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  echo "Copy config/pipeline.local.example.yaml to config/pipeline.yaml and edit it."
  exit 1
fi

# Use the survclam env's python to drive (it has pyyaml); the driver switches
# envs itself for each step.
PYTHON="${PANCOLON_PYTHON:-python}"

cd "$REPO_ROOT"
echo "[run_local] config=$CONFIG"
"$PYTHON" pancolon_pipeline.py all --config "$CONFIG" "$@"
