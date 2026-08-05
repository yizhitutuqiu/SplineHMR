#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
SCOREHMR_DIR="${THIRD_PARTY}/ScoreHMR"
BASE_COMMIT="${SCOREHMR_BASE_COMMIT:-f623ed1}"
PATCH_FILE="${REPO_ROOT}/patches/scorehmr_spline_diff.patch"

mkdir -p "${THIRD_PARTY}"
if [ ! -d "${SCOREHMR_DIR}/.git" ]; then
  git clone --recursive https://github.com/statho/ScoreHMR.git "${SCOREHMR_DIR}"
fi

if git -C "${SCOREHMR_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Spline-Diff patch is already applied at ${SCOREHMR_DIR}."
  exit 0
fi

if [ -n "$(git -C "${SCOREHMR_DIR}" status --porcelain)" ]; then
  echo "ScoreHMR checkout has local changes: ${SCOREHMR_DIR}" >&2
  echo "Please commit/stash them or remove this local checkout before rerunning setup." >&2
  exit 1
fi

git -C "${SCOREHMR_DIR}" fetch --all --tags
git -C "${SCOREHMR_DIR}" checkout "${BASE_COMMIT}"
if git -C "${SCOREHMR_DIR}" apply --check "${PATCH_FILE}"; then
  git -C "${SCOREHMR_DIR}" apply "${PATCH_FILE}"
else
  echo "Patch cannot be applied cleanly to ${SCOREHMR_DIR}." >&2
  exit 1
fi

echo "ScoreHMR source is ready at ${SCOREHMR_DIR}"
echo "Next: install dependencies/checkpoints following ScoreHMR README, or run scripts/install_scorehmr_deps.sh as a starting point."
