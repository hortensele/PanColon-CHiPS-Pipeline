#!/usr/bin/env bash
# ==========================================================================
# Download and unpack the trained PanColon-CHiPS weights bundle.
#
#   bash scripts/download_weights.sh [--url URL] [--dest weights]
#
# The bundle unpacks into weights/ with this layout:
#   weights/hpl/BarlowTwins_3.ckt.{data-00000-of-00001,index,meta}
#   weights/hpl/checkpoint
#   weights/hpl/reference/hdf5_colon_reference_he_complete_filtered.h5
#   weights/hpl/reference/colon_reference_folds.pkl
#   weights/survclam/runs/colon_united/survival__dfs_event_data__no_covariates__patient_level/
#       results/HPL_PANCOLON_20x__dfs__lr2e4_reg1e5_do025_clip1_plateau_p4_big_s1/s_{0..15}_checkpoint.pt
#
# Fill in PUBLIC_URL (and SHA256) once the bundle is hosted on Zenodo/Drive/S3.
# Maintainers: build that bundle with scripts/build_weights_bundle.sh (it prints
# the SHA256 to paste below).
# ==========================================================================
set -euo pipefail

# ---- EDIT ME: where the weights bundle is hosted -------------------------
PUBLIC_URL="CHANGE_ME_https://zenodo.org/record/XXXXXXX/files/pancolon_chips_weights.tar.gz"
EXPECTED_SHA256="ba75493401ecb259a732d18e0856e1e7ba931e690c686916868eec4f02deb380"   # sha256 of pancolon_chips_weights.tar.gz
# --------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/weights"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)  PUBLIC_URL="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "$PUBLIC_URL" == CHANGE_ME* ]]; then
  echo "ERROR: PUBLIC_URL is not set. Edit scripts/download_weights.sh (or pass --url)."
  echo "       Point it at the hosted pancolon_chips_weights.tar.gz."
  exit 1
fi

mkdir -p "$DEST"
BUNDLE="${DEST}/pancolon_chips_weights.tar.gz"

echo "[download_weights] Fetching bundle -> ${BUNDLE}"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "$BUNDLE" "$PUBLIC_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$BUNDLE" "$PUBLIC_URL"
else
  echo "ERROR: need curl or wget." >&2; exit 1
fi

if [[ "$EXPECTED_SHA256" != "CHANGE_ME" ]]; then
  echo "[download_weights] Verifying SHA256..."
  ACTUAL="$(sha256sum "$BUNDLE" | awk '{print $1}')"
  if [[ "$ACTUAL" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: checksum mismatch."
    echo "  expected: $EXPECTED_SHA256"
    echo "  actual:   $ACTUAL"
    exit 1
  fi
  echo "[download_weights] Checksum OK."
fi

echo "[download_weights] Unpacking into ${DEST}"
tar -xzf "$BUNDLE" -C "$DEST"
rm -f "$BUNDLE"

echo "[download_weights] Done. Verify with:"
echo "  ls ${DEST}/hpl ${DEST}/survclam/runs"
