#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/assets/scorehmr/data"
DST="${REPO_ROOT}/third_party/ScoreHMR/data"

if [ ! -d "${SRC}" ]; then
  echo "Missing tracked ScoreHMR small assets: ${SRC}" >&2
  exit 1
fi
if [ ! -d "${REPO_ROOT}/third_party/ScoreHMR" ]; then
  echo "Missing ScoreHMR checkout: ${REPO_ROOT}/third_party/ScoreHMR" >&2
  echo "Run scripts/setup_scorehmr.sh first." >&2
  exit 1
fi

mkdir -p "${DST}"
cp -a "${SRC}/." "${DST}/"
echo "Synced ScoreHMR small runtime assets to ${DST}"
