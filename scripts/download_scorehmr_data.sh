#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOREHMR_DIR="${REPO_ROOT}/third_party/ScoreHMR"

if [ ! -f "${SCOREHMR_DIR}/download_data.sh" ]; then
  echo "Missing ${SCOREHMR_DIR}/download_data.sh. Run scripts/setup_scorehmr.sh first." >&2
  exit 1
fi

cd "${SCOREHMR_DIR}"
bash download_data.sh

cat <<MSG

Downloaded ScoreHMR data under ${SCOREHMR_DIR}/data.
You still need the licensed SMPL neutral model at:
  ${SCOREHMR_DIR}/data/smpl/SMPL_NEUTRAL.pkl
See docs/MODEL_ASSETS.md.
MSG
