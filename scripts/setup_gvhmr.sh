#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
GVHMR_DIR="${THIRD_PARTY}/GVHMR"
GVHMR_REPO="${GVHMR_REPO:-https://github.com/zju3dv/GVHMR.git}"
GVHMR_REF="${GVHMR_REF:-main}"

mkdir -p "${THIRD_PARTY}"
if [ ! -d "${GVHMR_DIR}/.git" ]; then
  git clone --recursive "${GVHMR_REPO}" "${GVHMR_DIR}"
fi

git -C "${GVHMR_DIR}" fetch --all --tags
git -C "${GVHMR_DIR}" checkout "${GVHMR_REF}"
git -C "${GVHMR_DIR}" submodule update --init --recursive

mkdir -p "${GVHMR_DIR}/inputs/checkpoints/body_models/smpl" \
         "${GVHMR_DIR}/inputs/checkpoints/body_models/smplx" \
         "${GVHMR_DIR}/inputs/checkpoints/gvhmr" \
         "${GVHMR_DIR}/inputs/checkpoints/hmr2" \
         "${GVHMR_DIR}/inputs/checkpoints/vitpose" \
         "${GVHMR_DIR}/inputs/checkpoints/yolo"

echo "GVHMR source is ready at ${GVHMR_DIR}"
echo "Next: run scripts/download_gvhmr_assets.sh, then place licensed SMPL/SMPLX files as described in docs/MODEL_ASSETS.md."
