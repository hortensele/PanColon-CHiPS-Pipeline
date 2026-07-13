#!/usr/bin/env bash
# ==========================================================================
# Assemble the PanColon-CHiPS weights bundle for publishing (e.g. Zenodo).
#
#   bash scripts/build_weights_bundle.sh --config config/pipeline.yaml
#
# It copies the trained encoder, the HPL reference clustering (the development
# cohort's Leiden assignments), and the 16 SurvCLAM CHiPS fold checkpoints from
# your lab installs into the `weights/` layout that scripts/download_weights.sh
# expects, verifies every file is present, writes SHA256SUMS + a size manifest,
# and tars the result. If a source path can't be resolved it stops with a clear
# error — nothing half-built ships.
#
# WHY THE ADATAS: HPL's run_representationsleiden_assignment.py
# (assign_additional_only) loads the reference cohort's per-fold Leiden AnnData
# `<ref_basename>_leiden_<res>__fold<i>.h5ad` and assigns *new* tiles into those
# existing clusters. Those adatas ARE the development-cohort HPC/artifact
# assignments and MUST ship. The multi-GB complete `.h5` files are only used as a
# path anchor (never read), but they're small here (~1.2 GB) so we ship them too.
#
# NAMING: HPL derives the adata filename from the complete-h5 basename in the
# config (weights.hpl_reference_h5 / hpl_artifact_reference_h5). The source files
# are named after the reference dataset (colon_cancer_20x_250K), so this script
# RENAMES them on copy to the config basenames. Target names are derived from the
# live config, so they always match what the pipeline will look for at run time.
# ==========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/config/pipeline.yaml"
DEST="${REPO_ROOT}/weights_bundle"
PYTHON="${PANCOLON_PYTHON:-python}"

# ---- EDIT ME: source locations of the real trained files -----------------
HPL_INSTALL="${HPL_INSTALL:-/path/to/Histomorphological-Phenotype-Learning}"
SURVCLAM_RUNS_SRC="${SURVCLAM_RUNS_SRC:-/path/to/CLAMFamily/runs}"
# The reference dataset whose clustering + encoder we ship (its name appears in
# the source filenames):
REF_DATASET="${REF_DATASET:-colon_cancer_20x_250K}"
REF_TREE="${REF_TREE:-${HPL_INSTALL}/results/BarlowTwins_3/${REF_DATASET}/h224_w224_n3_zdim128}"
# The encoder checkpoint stem dir (…/checkpoints/BarlowTwins_3.ckt):
HPL_CKPT_DIR="${HPL_CKPT_DIR:-${HPL_INSTALL}/data_model_output/BarlowTwins_3/${REF_DATASET}/h224_w224_n3_zdim128/checkpoints}"
# The dir holding the two reference h5s + the cohort / cohort_cleaned adatas trees:
HPL_REF_DIR="${HPL_REF_DIR:-${REF_TREE}}"
# The reference folds pickle (lives under the HPL utilities tree, not REF_DIR):
HPL_FOLDS_PKL="${HPL_FOLDS_PKL:-${HPL_INSTALL}/utilities/files/COLON/colon_cohort.pkl}"
# --------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --dest)   DEST="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

fail() { printf 'ERROR: %b\n' "$*" >&2; exit 1; }
[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"

# ---- Derive run identity + reference target names from the live config ----
# Emits (space separated): RUN_SIG EXP_CODE RUNS_DATASET SEED K \
#   ART_STEM HPC_STEM ART_RESP HPC_RESP FOLD ART_H5_BASE HPC_H5_BASE PKL_BASE
read -r RUN_SIG EXP_CODE_CFG RUNS_DATASET SEED K \
        ART_STEM HPC_STEM ART_RESP HPC_RESP FOLD \
        ART_H5_BASE HPC_H5_BASE PKL_BASE < <("$PYTHON" - "$CONFIG" <<'PY'
import os, sys
sys.path.insert(0, ".")
from pancolon.config import load_config
from pancolon.steps import _run_sig
cfg = load_config(sys.argv[1])
inf = cfg.get("infer", {}); cl = cfg.get("cluster", {}); w = cfg.get("weights", {})

def h5_base(p):      return os.path.basename(p)               # hdf5_..._filtered.h5
def stem(p):         return os.path.basename(p).split("hdf5_", 1)[1].rsplit(".h5", 1)[0]
def resp(r):         return str(r).replace(".", "p")

art_h5 = w.get("hpl_artifact_reference_h5", "")
hpc_h5 = w.get("hpl_reference_h5", "")
print(
    _run_sig(cfg), inf.get("exp_code"), inf.get("runs_root_dataset", "colon_united"),
    inf.get("seed", 1), inf.get("k", 16),
    stem(art_h5), stem(hpc_h5),
    resp(cl.get("artifact_resolution", 5.0)), resp(cl.get("hpc_resolution", 2.5)),
    cl.get("hpc_fold", 1),
    h5_base(art_h5), h5_base(hpc_h5),
    os.path.basename(w.get("hpl_folds_pickle", "colon_reference_folds.pkl")),
)
PY
)
EXP_CODE="${EXP_CODE:-$EXP_CODE_CFG}"
echo "[bundle] run signature : $RUN_SIG"
echo "[bundle] exp_code       : $EXP_CODE  (seed $SEED, $K folds)"
echo "[bundle] reference      : dataset=$REF_DATASET fold=$FOLD  artifact=${ART_RESP} hpc=${HPC_RESP}"

STAGE="${DEST}"
rm -rf "$STAGE"; mkdir -p "$STAGE/hpl/reference" "$STAGE/survclam/runs"

copy_in() {  # copy_in <src> <dest>   (errors if src missing; renames on copy)
  local src="$1" dst="$2"
  [[ -e "$src" ]] || fail "missing source: $src"
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
  echo "  + $(basename "$src")  ->  ${dst#$STAGE/}"
}

# ---- 1. HPL encoder -------------------------------------------------------
echo "[bundle] HPL encoder (BarlowTwins_3, ${REF_DATASET})"
for f in BarlowTwins_3.ckt.data-00000-of-00001 BarlowTwins_3.ckt.index \
         BarlowTwins_3.ckt.meta checkpoint; do
  copy_in "${HPL_CKPT_DIR}/${f}" "${STAGE}/hpl/${f}"
done

# ---- 2. HPL reference clustering -----------------------------------------
echo "[bundle] HPL reference clustering (fold ${FOLD} adatas + anchor h5s)"
[[ -d "$HPL_REF_DIR" ]] || fail \
  "HPL_REF_DIR is not a directory:\n  $HPL_REF_DIR\n\
  Point it at the reference tree holding hdf5_${REF_DATASET}_he_complete_cohort{,_filtered}.h5\n\
  and the cohort/ + cohort_cleaned/ adatas subdirs."

# Source (dataset-named) files:
SRC_ART_H5="${HPL_REF_DIR}/hdf5_${REF_DATASET}_he_complete_cohort.h5"
SRC_HPC_H5="${HPL_REF_DIR}/hdf5_${REF_DATASET}_he_complete_cohort_filtered.h5"
SRC_ART_ADATA="${HPL_REF_DIR}/cohort/adatas/${REF_DATASET}_he_complete_cohort_leiden_${ART_RESP}__fold${FOLD}.h5ad"
SRC_HPC_ADATA="${HPL_REF_DIR}/cohort_cleaned/adatas/${REF_DATASET}_he_complete_cohort_filtered_leiden_${HPC_RESP}__fold${FOLD}.h5ad"

# Anchor h5s (renamed to the config basenames):
copy_in "$SRC_ART_H5" "${STAGE}/hpl/reference/${ART_H5_BASE}"
copy_in "$SRC_HPC_H5" "${STAGE}/hpl/reference/${HPC_H5_BASE}"
# Folds pickle (renamed to the config basename):
copy_in "$HPL_FOLDS_PKL" "${STAGE}/hpl/reference/${PKL_BASE}"
# The reference adatas, RENAMED so their stem matches the anchor-h5 basename that
# HPL's assign_additional_only derives (<stem>_leiden_<res>__fold<i>.h5ad):
copy_in "$SRC_ART_ADATA" "${STAGE}/hpl/reference/cohort/adatas/${ART_STEM}_leiden_${ART_RESP}__fold${FOLD}.h5ad"
copy_in "$SRC_HPC_ADATA" "${STAGE}/hpl/reference/cohort_cleaned/adatas/${HPC_STEM}_leiden_${HPC_RESP}__fold${FOLD}.h5ad"

# ---- 3. SurvCLAM CHiPS fold checkpoints -----------------------------------
echo "[bundle] SurvCLAM CHiPS folds (s_0..s_$((K-1)))"
SRC_RUN="${SURVCLAM_RUNS_SRC}/${RUNS_DATASET}/${RUN_SIG}/results/${EXP_CODE}_s${SEED}"
[[ -d "$SRC_RUN" ]] || fail \
  "SurvCLAM run dir not found:\n  $SRC_RUN\n\
  The config's infer.exp_code does not resolve on disk. List candidates with:\n\
    ls ${SURVCLAM_RUNS_SRC}/${RUNS_DATASET}/${RUN_SIG}/results/\n\
  then set EXP_CODE=<the real dir minus _s${SEED}> and re-run, or fix\n\
  infer.exp_code in $CONFIG."
DST_RUN="${STAGE}/survclam/runs/${RUNS_DATASET}/${RUN_SIG}/results/${EXP_CODE}_s${SEED}"
mkdir -p "$DST_RUN"
missing=()
for k in $(seq 0 $((K-1))); do
  f="${SRC_RUN}/s_${k}_checkpoint.pt"
  if [[ -f "$f" ]]; then cp -a "$f" "${DST_RUN}/"; else missing+=("s_${k}_checkpoint.pt"); fi
done
[[ ${#missing[@]} -eq 0 ]] || fail "missing fold checkpoints in $SRC_RUN: ${missing[*]}"
echo "  + ${K} fold checkpoints"

# ---- 4. checksums, manifest, tarball -------------------------------------
echo "[bundle] writing SHA256SUMS + size manifest"
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
  | xargs -0 sha256sum > SHA256SUMS )
du -ah "$STAGE" | sort -k2 > "${STAGE}/FILE_SIZES.txt"
TOTAL="$(du -sh "$STAGE" | cut -f1)"

TARBALL="${REPO_ROOT}/pancolon_chips_weights.tar.gz"
echo "[bundle] taring -> ${TARBALL}  (total ${TOTAL})"
tar -czf "$TARBALL" -C "$STAGE" .
BUNDLE_SHA="$(sha256sum "$TARBALL" | awk '{print $1}')"

cat <<EOF

[bundle] DONE.
  staged files : $STAGE
  tarball      : $TARBALL  (${TOTAL})
  tar SHA256   : $BUNDLE_SHA

Next:
  1. Upload $TARBALL to Zenodo (or your host).
  2. In scripts/download_weights.sh set:
       PUBLIC_URL       = <the Zenodo file URL>
       EXPECTED_SHA256  = $BUNDLE_SHA
  3. Collaborators then run: bash scripts/download_weights.sh
EOF
