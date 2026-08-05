#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-splinehmr}"
cd "${REPO_ROOT}"
conda run -n "${CONDA_ENV}" python -m splinehmr.demo --method spline-diff "$@"
