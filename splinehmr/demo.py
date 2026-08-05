from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .io import find_input_video, infer_length, load_demo_input, make_refined_hmr_pack, slice_tensor_dict
from .paths import add_runtime_paths, body_model_root, find_scorehmr_root, repo_root, require_body_models, require_scorehmr_root
from .render import render_outputs

METHOD_DISPLAY = {"spline-opt": "Spline-Opt", "spline-diff": "Spline-Diff"}


def _parse_args() -> argparse.Namespace:
    root = repo_root()
    p = argparse.ArgumentParser(description="Run SplineHMR demo inference from GVHMR outputs.")
    p.add_argument("--method", choices=["spline-opt", "spline-diff"], required=True)
    p.add_argument("--input_dir", type=Path, default=root / "inputs" / "climbing_3mb")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--m_per_t", type=str, default=None, help="Optional override. By default Spline-Opt reads configs/spline_opt.yaml; Spline-Diff reads ScoreHMR custom/configs/bspline.yaml.")
    p.add_argument("--max_iter", type=int, default=None, help="Optional Spline-Opt LBFGS iteration override.")
    p.add_argument("--spline_opt_config", type=Path, default=root / "configs" / "spline_opt.yaml")
    p.add_argument("--scorehmr_optim_iters", type=int, default=2)
    p.add_argument("--scorehmr_blend_weight", type=float, default=None)
    p.add_argument("--scorehmr_use_pare_cond", action="store_true")
    p.add_argument("--scorehmr_use_tanh", choices=["true", "false"], default=None)
    p.add_argument("--scorehmr_tanh_amp", type=float, default=None)
    p.add_argument("--skip_render", action="store_true")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--no_fast_render", dest="fast_render", action="store_false")
    p.set_defaults(fast_render=True)
    return p.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_m_per_t(value: Any, fps: float | None) -> int:
    if isinstance(value, int):
        return max(1, int(value))
    s = str(value).strip().lower()
    if s.startswith("fps_div_"):
        div = max(1, int(s.split("fps_div_", 1)[1]))
        return max(1, int(int(round(fps or 30.0)) / div))
    if s == "fps":
        return max(1, int(round(fps or 30.0)))
    if s.startswith("every_"):
        return max(1, int(s.rsplit("_", 1)[-1]))
    return max(1, int(float(s)))


def _extract_static_conf_logits(hmr: dict[str, Any]) -> torch.Tensor | None:
    net = hmr.get("net_outputs", None)
    if not isinstance(net, dict):
        return None
    x = net.get("static_conf_logits", None)
    if x is None and isinstance(net.get("model_output", None), dict):
        x = net["model_output"].get("static_conf_logits", None)
    if not torch.is_tensor(x):
        return None
    if x.ndim == 3 and int(x.shape[0]) == 1:
        return x[0]
    return x


def _run_spline_opt(data: dict[str, Any], T: int, args: argparse.Namespace, fps: float | None) -> dict[str, Any]:
    from multi_view_smpl_optimizer.utils.refiner_main import BsplineRefineConfig, refine_body_pose_bspline_lbfgs

    optim_cfg = _load_yaml(args.spline_opt_config)
    cfg_cfg = dict(optim_cfg.get("cfg", {}) or {})
    if args.m_per_t is not None:
        cfg_cfg["m_per_t"] = args.m_per_t
    if args.max_iter is not None:
        cfg_cfg["max_iter"] = int(args.max_iter)

    bs_cfg = BsplineRefineConfig(
        degree=int(cfg_cfg.get("degree", 3)),
        m_per_t=_resolve_m_per_t(cfg_cfg.get("m_per_t", "fps_div_2"), fps),
        conf_thr=float(cfg_cfg.get("conf_thr", 0.3)),
        amp_body_pose=float(cfg_cfg.get("amp_body_pose", 1.0)),
        amp_global_orient=float(cfg_cfg.get("amp_global_orient", 1.0)),
        amp_transl=float(cfg_cfg.get("amp_transl", 1.0)),
        prior_w_body_pose=float(cfg_cfg.get("prior_w_body_pose", 0.1)),
        prior_w_global_orient=float(cfg_cfg.get("prior_w_global_orient", 0.2)),
        prior_w_transl=float(cfg_cfg.get("prior_w_transl", 0.2)),
        mv_consistency_w=float(cfg_cfg.get("mv_consistency_w", 10.0)),
        ankle_ground_align_w=float(cfg_cfg.get("ankle_ground_align_w", 10.0)),
        static_motion_w=float(cfg_cfg.get("static_motion_w", 2.0)),
        static_joint_w=tuple(cfg_cfg.get("static_joint_w", (1.0, 1.0, 1.0, 1.0, 0.1, 0.1))),
        static_softmax_tau=float(cfg_cfg.get("static_softmax_tau", 1.5)),
        static_use_smpl24=bool(cfg_cfg.get("static_use_smpl24", True)),
        smooth_w=float(cfg_cfg.get("smooth_w", 0.02)),
        max_iter=int(cfg_cfg.get("max_iter", 60)),
        lr=float(cfg_cfg.get("lr", 1.0)),
        line_search_fn=cfg_cfg.get("line_search_fn", "strong_wolfe"),
        verbose=bool(cfg_cfg.get("verbose", True)),
        learn_knots=bool(cfg_cfg.get("learn_knots", False)),
        knot_min_gap=float(cfg_cfg.get("knot_min_gap", 1e-3)),
        knot_pos_w=float(cfg_cfg.get("knot_pos_w", 1.0)),
        knot_gap_w=float(cfg_cfg.get("knot_gap_w", 0.2)),
        knot_smooth_w=float(cfg_cfg.get("knot_smooth_w", 0.0)),
        optimize_pose_in_rot6d=bool(cfg_cfg.get("optimize_pose_in_rot6d", True)),
    ).resolve()

    hmr = data["hmr"]
    params = hmr["smpl_params_incam"]
    out = refine_body_pose_bspline_lbfgs(
        body_pose=params["body_pose"][:T],
        betas=params["betas"][:T],
        global_orient=params["global_orient"][:T],
        transl=params["transl"][:T],
        K_fullimg=hmr["K_fullimg"][:T],
        bbx_xys=data["bbx_xys"][:T],
        coco17=data["vitpose"][:T],
        cfg=bs_cfg,
        device=args.device,
        optimize_body_pose=True,
        optimize_global_orient=bool(optim_cfg.get("optimize_global_orient", True)),
        optimize_transl=bool(optim_cfg.get("optimize_transl", True)),
        pose_limit_in_loss=bool(optim_cfg.get("pose_limit_in_loss", False)),
        static_conf_logits=_extract_static_conf_logits(hmr),
    )
    out.setdefault("stats", {})
    out["stats"]["spline_opt_config"] = str(args.spline_opt_config)
    out["stats"]["spline_opt_config_defaults"] = optim_cfg
    return out


def _run_spline_diff(data: dict[str, Any], T: int, args: argparse.Namespace, fps: float | None) -> dict[str, Any]:
    scorehmr_root = require_scorehmr_root()

    from .scorehmr_wrapper import ScoreHMRRefineCfg, refine_smpl_to_2d_with_scorehmr

    hmr = data["hmr"]
    params = hmr["smpl_params_incam"]
    body_pose = params["body_pose"][:T].detach().cpu().float()
    D = int(body_pose.shape[1])
    if D < 69:
        body_pose_69 = torch.cat([body_pose, torch.zeros((T, 69 - D), dtype=body_pose.dtype)], dim=1)
    else:
        body_pose_69 = body_pose[:, :69]

    tanh_override = None
    if args.scorehmr_use_tanh is not None:
        tanh_override = str(args.scorehmr_use_tanh).lower() == "true"

    cfg = ScoreHMRRefineCfg(
        num_samples=1,
        optim_iters=int(args.scorehmr_optim_iters),
        early_stopping=True,
        use_pare_cond=bool(args.scorehmr_use_pare_cond),
        use_bspline_smooth_noise=True,
        bspline_fps=float(fps or 30.0),
        bspline_m_per_t=args.m_per_t,
        bspline_order=None,
        bspline_use_tanh=tanh_override,
        bspline_tanh_amp=args.scorehmr_tanh_amp,
        bspline_blend_weight=args.scorehmr_blend_weight,
    )
    out = refine_smpl_to_2d_with_scorehmr(
        global_orient_aa=params["global_orient"][:T],
        body_pose_aa=body_pose_69,
        betas=params["betas"][:T],
        init_cam_t=params["transl"][:T],
        keypoints_2d=data["vitpose"][:T],
        image_size_hw=_video_hw(args.input_dir),
        K_3x3=hmr["K_fullimg"][:T],
        device=args.device,
        cfg=cfg,
        fps=float(fps or 30.0),
    )
    bp_ref = out["body_pose_aa_refined"].detach().cpu().float().view(T, 69)
    if D < 69:
        bp_ref = bp_ref[:, :D]
    return {
        "body_pose_refined": bp_ref,
        "global_orient_refined": out["global_orient_aa_refined"].detach().cpu().float().view(T, 3),
        "transl_refined": out["cam_t_refined"].detach().cpu().float().view(T, 3),
        "stats": {
            "optimizer": "Spline-Diff",
            "T": int(T),
            "D": int(D),
            "scorehmr_root": str(scorehmr_root),
            "render_model": "smpl_direct",
        },
    }


def _video_hw(input_dir: Path) -> tuple[int, int]:
    import cv2

    video_path = find_input_video(input_dir)
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    return int(frame.shape[0]), int(frame.shape[1])


def _video_fps(video_path: Path) -> float:
    import imageio.v3 as iio

    try:
        meta = iio.immeta(str(video_path), plugin="pyav")
        fps = meta.get("fps", None)
        return float(fps) if fps else 30.0
    except Exception:
        return 30.0


def main() -> None:
    args = _parse_args()
    root = add_runtime_paths()
    require_body_models()
    input_dir = args.input_dir.resolve()
    video_path = find_input_video(input_dir)
    data = load_demo_input(input_dir)
    T = infer_length(data["hmr"], data["vitpose"], data["bbx_xys"], args.max_frames)
    fps = _video_fps(video_path)

    method = str(args.method)
    method_display = METHOD_DISPLAY[method]
    out_dir = args.output_dir or (root / "outputs" / input_dir.name / method)
    out_dir.mkdir(parents=True, exist_ok=True)

    if method == "spline-opt":
        refined = _run_spline_opt(data, T, args, fps)
        render_model = "smplx2smpl"
    else:
        try:
            refined = _run_spline_diff(data, T, args, fps)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Spline-Diff needs ScoreHMR dependencies in the active environment; missing module: {exc.name}. "
                "Install them with scripts/install_scorehmr_deps.sh after setting up ScoreHMR."
            ) from exc
        render_model = "smpl_direct"

    before_pack = slice_tensor_dict(data["hmr"], T)
    after_pack = make_refined_hmr_pack(data["hmr"], refined, T, method)
    before_path = out_dir / "hmr4d_results_before.pt"
    after_path = out_dir / "hmr4d_results.pt"
    torch.save(before_pack, before_path)
    torch.save(after_pack, after_path)

    render_paths = {}
    if not bool(args.skip_render):
        render_paths = render_outputs(
            video_path=video_path,
            before_pack=before_pack,
            after_pack=after_pack,
            out_dir=out_dir,
            T=T,
            device=args.device,
            crf=int(args.crf),
            fast_render=bool(args.fast_render),
            render_model=render_model,
        )

    report = {
        "method": method,
        "method_display": method_display,
        "input_dir": str(input_dir),
        "video_path": str(video_path),
        "body_model_root": str(body_model_root()),
        "scorehmr_root": str(find_scorehmr_root()),
        "num_frames": int(T),
        "fps": float(fps),
        "before_params": str(before_path),
        "after_params": str(after_path),
        "renders": {k: str(v) for k, v in render_paths.items()},
        "render_model": render_model,
        "stats": refined.get("stats", {}),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        import sys

        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
