#!/usr/bin/env bash
# ==========================================================================
# Submit the whole pipeline to SLURM as a dependency chain (each step starts
# only after the previous one succeeds). Same steps as the local driver.
#
#   bash scripts/slurm/submit_all.sh config/pipeline.yaml
#
# Resource choices below mirror config/pipeline.yaml's `slurm:` section; edit
# them (or this file) for your cluster. GPU steps: project, infer_survival,
# attention_map. Everything else runs CPU-only.
# ==========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-${REPO_ROOT}/config/pipeline.yaml}"
STAGE="${REPO_ROOT}/scripts/slurm/stage.sbatch"

# ---- Cluster resources (edit for your site) ------------------------------
GPU_PARTITION="${GPU_PARTITION:-gpu4_short}"
CPU_PARTITION="${CPU_PARTITION:-cpu_short}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GPU_MEM="${GPU_MEM:-100G}"
CPU_MEM="${CPU_MEM:-60G}"
TIME_GPU="${TIME_GPU:-12:00:00}"
TIME_CPU="${TIME_CPU:-12:00:00}"
ACCOUNT="${ACCOUNT:-}"   # set to add --account

# steps and whether each needs a GPU
STEPS=(tile to_hdf5 project cluster_filter assign_hpc build_pt infer_survival attention_map)
declare -A NEEDS_GPU=( [project]=1 [infer_survival]=1 [attention_map]=1 )

export PANCOLON_CONFIG="$CONFIG"
export PANCOLON_REPO="$REPO_ROOT"

acct_flag=(); [[ -n "$ACCOUNT" ]] && acct_flag=(--account "$ACCOUNT")

prev_jid=""
for step in "${STEPS[@]}"; do
  if [[ -n "${NEEDS_GPU[$step]:-}" ]]; then
    res=(--partition "$GPU_PARTITION" --gres "$GPU_GRES" --mem "$GPU_MEM" --time "$TIME_GPU")
  else
    res=(--partition "$CPU_PARTITION" --mem "$CPU_MEM" --time "$TIME_CPU")
  fi
  dep=(); [[ -n "$prev_jid" ]] && dep=(--dependency "afterok:${prev_jid}")

  jid=$(sbatch --parsable \
        --job-name "pancolon_${step}" \
        "${acct_flag[@]}" "${res[@]}" "${dep[@]}" \
        --export=ALL,PANCOLON_STEP="$step" \
        "$STAGE")
  echo "submitted ${step} as job ${jid}${prev_jid:+ (after ${prev_jid})}"
  prev_jid="$jid"
done

echo "Chain submitted. Track with: squeue -u \$USER"
