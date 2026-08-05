from __future__ import annotations

from pathlib import Path


def get_gvhmr_root() -> Path:
    """Deprecated compatibility shim.

    SplineHMR demo inference is self-contained and does not require a GVHMR checkout.
    Code paths that need regressors should use local files under
    multi_view_smpl_optimizer/utils/body_model instead.
    """
    return Path(__file__).resolve().parents[2]
