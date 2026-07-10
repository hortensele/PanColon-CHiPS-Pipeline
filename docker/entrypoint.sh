#!/usr/bin/env bash
# ==========================================================================
# Container entrypoint. With no arguments it runs the whole pipeline and then
# exports the static results bundle. With arguments it forwards them to
# pancolon_pipeline.py (e.g. `list`, `export`, `all --from project`).
#
# The driver is launched from the survclam env (it has pyyaml); it activates the
# correct stage env per step itself via envs.conda_sh in the config.
# ==========================================================================
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate pancolon_survclam

CONFIG="${PANCOLON_CONFIG:-/opt/pancolon/config/pipeline.container.yaml}"
cd /opt/pancolon

if [[ "$#" -eq 0 ]]; then
  echo "[pancolon] running full pipeline -> config=$CONFIG"
  python pancolon_pipeline.py all    --config "$CONFIG"
  echo "[pancolon] exporting results bundle"
  python pancolon_pipeline.py export --config "$CONFIG"
  echo "[pancolon] done. Results bundle is in your mounted /out (see /out/bundle)."
else
  # If the caller didn't pass --config, use the container config.
  if [[ "$*" != *"--config"* ]]; then
    exec python pancolon_pipeline.py "$@" --config "$CONFIG"
  fi
  exec python pancolon_pipeline.py "$@"
fi
