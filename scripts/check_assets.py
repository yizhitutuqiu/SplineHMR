from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD_DISPLAY = {"spline-opt": "Spline-Opt", "spline-diff": "Spline-Diff", "all": "SplineHMR"}

SPLINE_OPT_REQUIRED = [
    ("SMPL neutral", ROOT / "assets/body_models/smpl/SMPL_NEUTRAL.pkl"),
    ("SMPLX neutral", ROOT / "assets/body_models/smplx/SMPLX_NEUTRAL.npz"),
    ("SMPLX->SMPL sparse regressor", ROOT / "multi_view_smpl_optimizer/utils/body_model/smplx2smpl_sparse.pt"),
    ("SMPL COCO17 regressor", ROOT / "multi_view_smpl_optimizer/utils/body_model/smpl_coco17_J_regressor.pt"),
    ("SMPL neutral J regressor", ROOT / "multi_view_smpl_optimizer/utils/body_model/smpl_neutral_J_regressor.pt"),
]

SPLINE_DIFF_REQUIRED = [
    ("ScoreHMR source", ROOT / "third_party/ScoreHMR/score_hmr/__init__.py"),
    ("ScoreHMR model", ROOT / "third_party/ScoreHMR/data/model_weights/score_hmr/model-100.pt"),
    ("ScoreHMR PARE model", ROOT / "third_party/ScoreHMR/data/model_weights/pare/pare_checkpoint.ckpt"),
    ("ScoreHMR PARE config", ROOT / "third_party/ScoreHMR/data/model_weights/pare/pare_config.yaml"),
    ("ScoreHMR SMPL neutral", ROOT / "third_party/ScoreHMR/data/smpl/SMPL_NEUTRAL.pkl"),
    ("ScoreHMR SMPL-to-J19 regressor", ROOT / "third_party/ScoreHMR/data/SMPL_to_J19.pkl"),
    ("ScoreHMR SMPL mean params", ROOT / "third_party/ScoreHMR/data/smpl_mean_params.npz"),
    ("ScoreHMR betas stats", ROOT / "third_party/ScoreHMR/data/stats/betas_stats_eft_fits.npz"),
    ("ScoreHMR PARE feat stats", ROOT / "third_party/ScoreHMR/data/stats/pare_feat_stats.npz"),
    ("ScoreHMR ProHMR feat stats", ROOT / "third_party/ScoreHMR/data/stats/prohmr_feat_stats.npz"),
]


def _check(items: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    missing = []
    for name, path in items:
        ok = path.exists()
        print(f"[{'OK' if ok else 'MISS'}] {name}: {path.relative_to(ROOT)}")
        if not ok:
            missing.append((name, path))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SplineHMR demo assets.")
    parser.add_argument("--method", choices=["spline-opt", "spline-diff", "all"], default="spline-opt")
    args = parser.parse_args()

    required = list(SPLINE_OPT_REQUIRED)
    if args.method in {"spline-diff", "all"}:
        required += SPLINE_DIFF_REQUIRED

    missing = _check(required)
    if missing:
        print("\nMissing assets. See docs/MODEL_ASSETS.md for download/setup instructions.")
        raise SystemExit(1)
    print(f"\nAll required assets for {METHOD_DISPLAY[args.method]} are present.")


if __name__ == "__main__":
    main()
