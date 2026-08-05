from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def assets_root() -> Path:
    return (repo_root() / "assets").resolve()


def body_model_root() -> Path:
    return (assets_root() / "body_models").resolve()


def local_body_model_utils_root() -> Path:
    return (repo_root() / "multi_view_smpl_optimizer" / "utils" / "body_model").resolve()


def find_gvhmr_root() -> Path:
    # Optional only. Demo inference starts from precomputed hmr4d_results.pt.
    return (repo_root() / "third_party" / "GVHMR").resolve()


def find_scorehmr_root() -> Path:
    return (repo_root() / "third_party" / "ScoreHMR").resolve()


def require_body_models() -> Path:
    root = body_model_root()
    required = [
        root / "smpl" / "SMPL_NEUTRAL.pkl",
        root / "smplx" / "SMPLX_NEUTRAL.npz",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        rel = "\n".join(f"  - {p.relative_to(repo_root())}" for p in missing)
        raise FileNotFoundError(
            "SplineHMR demo needs licensed SMPL/SMPLX body models inside this repository:\n"
            f"{rel}\n"
            "See docs/MODEL_ASSETS.md. Full GVHMR source/checkpoints are not required for Spline-Opt."
        )
    return root


def require_scorehmr_root() -> Path:
    path = find_scorehmr_root()
    if not (path / "score_hmr").exists():
        raise FileNotFoundError(
            f"ScoreHMR is required inside this repository at {path}. Run scripts/setup_scorehmr.sh."
        )
    return path


def add_runtime_paths() -> Path:
    root = repo_root()
    sp = str(root)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return root
