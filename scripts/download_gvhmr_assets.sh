#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GVHMR_DIR="${REPO_ROOT}/third_party/GVHMR"
CKPT_DIR="${GVHMR_DIR}/inputs/checkpoints"

if [ ! -d "${GVHMR_DIR}" ]; then
  echo "Missing ${GVHMR_DIR}. Run scripts/setup_gvhmr.sh first." >&2
  exit 1
fi

_downloader() {
  local url="$1"
  local dir="$2"
  local out="$3"
  mkdir -p "${dir}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "${url}" -d "${dir}" -o "${out}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${dir}/${out}"
  elif command -v wget >/dev/null 2>&1; then
    wget "${url}" -O "${dir}/${out}"
  else
    echo "Need aria2c, curl, or wget to download ${url}" >&2
    exit 1
  fi
}

_downloader "https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt" "${CKPT_DIR}/gvhmr" "gvhmr_siga24_release.ckpt"
_downloader "https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt" "${CKPT_DIR}/hmr2" "epoch=10-step=25000.ckpt"
_downloader "https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth" "${CKPT_DIR}/vitpose" "vitpose-h-multi-coco.pth"
_downloader "https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt" "${CKPT_DIR}/yolo" "yolov8x.pt"

cat <<MSG

Downloaded public GVHMR checkpoints under ${CKPT_DIR}.
SMPL and SMPLX are license-gated and must be downloaded separately from the official websites.
See docs/MODEL_ASSETS.md for the exact filenames and locations.
MSG
