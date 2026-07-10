#!/usr/bin/env bash
# ==========================================================================
# Launch the CHiPS web interface.
#
#   conda activate pancolon_survclam      # flask + openslide + torch + pyyaml
#   bash scripts/run_webapp.sh [config/pipeline.yaml] [--port 5000]
#
# The interface SUBMITS the pipeline to SLURM (sbatch) and monitors it, so run
# it on a LOGIN NODE where sbatch/squeue/sacct are available and the downloaded
# weights are reachable. It binds to localhost only; reach it by SSH-tunnelling
# the port, e.g.:  ssh -L 5000:127.0.0.1:5000 <login-node>
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

PYTHON="${PANCOLON_PYTHON:-python}"
cd "$REPO_ROOT"
exec "$PYTHON" webapp/app.py --config "$CONFIG" "$@"
