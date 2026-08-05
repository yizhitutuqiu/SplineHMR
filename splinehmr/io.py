from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


def find_input_video(input_dir: Path) -> Path:
    preferred = input_dir / "0_input_video.mp4"
    if preferred.exists():
        return preferred
    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 video found in {input_dir}")
    if len(videos) > 1:
        names = ", ".join(p.name for p in videos)
        raise RuntimeError(f"Multiple videos found in {input_dir}: {names}")
    return videos[0]


def load_demo_input(input_dir: Path) -> dict[str, Any]:
    hmr_path = input_dir / "hmr4d_results.pt"
    vitpose_path = input_dir / "preprocess" / "vitpose.pt"
    bbx_path = input_dir / "preprocess" / "bbx.pt"
    if not hmr_path.exists():
        raise FileNotFoundError(hmr_path)
    if not vitpose_path.exists():
        raise FileNotFoundError(vitpose_path)
    if not bbx_path.exists():
        raise FileNotFoundError(bbx_path)

    hmr = torch.load(str(hmr_path), map_location="cpu")
    vitpose = torch.load(str(vitpose_path), map_location="cpu")
    bbx = torch.load(str(bbx_path), map_location="cpu")
    if "smpl_params_incam" not in hmr or "K_fullimg" not in hmr:
        raise KeyError("hmr4d_results.pt must contain smpl_params_incam and K_fullimg")
    if not isinstance(bbx, dict) or "bbx_xys" not in bbx:
        raise KeyError("preprocess/bbx.pt must contain key bbx_xys")
    return {"hmr": hmr, "vitpose": vitpose, "bbx_xys": bbx["bbx_xys"]}


def infer_length(hmr: dict[str, Any], vitpose: torch.Tensor, bbx_xys: torch.Tensor, max_frames: int | None) -> int:
    params = hmr["smpl_params_incam"]
    lengths = [
        int(hmr["K_fullimg"].shape[0]),
        int(vitpose.shape[0]),
        int(bbx_xys.shape[0]),
        int(params["body_pose"].shape[0]),
        int(params["global_orient"].shape[0]),
        int(params["transl"].shape[0]),
    ]
    T = min(lengths)
    if max_frames is not None:
        T = min(T, int(max_frames))
    if T <= 0:
        raise ValueError("No valid frames in demo input")
    return T


def slice_tensor_dict(obj: Any, T: int) -> Any:
    if torch.is_tensor(obj):
        return obj[:T].detach().cpu() if obj.ndim > 0 and int(obj.shape[0]) >= T else obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: slice_tensor_dict(v, T) for k, v in obj.items()}
    return deepcopy(obj)


def make_refined_hmr_pack(hmr: dict[str, Any], refined: dict[str, torch.Tensor], T: int, method: str) -> dict[str, Any]:
    out = slice_tensor_dict(hmr, T)
    params = out["smpl_params_incam"]
    body_pose = refined["body_pose_refined"].detach().cpu().float()
    params["body_pose"] = body_pose[:, : int(params["body_pose"].shape[1])]
    params["global_orient"] = refined["global_orient_refined"].detach().cpu().float().view(T, 3)
    params["transl"] = refined["transl_refined"].detach().cpu().float().view(T, 3)
    out.setdefault("splinehmr_meta", {})
    method_display = {"spline-opt": "Spline-Opt", "spline-diff": "Spline-Diff"}.get(method, method)
    out["splinehmr_meta"].update({"method": method, "method_display": method_display, "num_frames": int(T), "render_model": ("smpl_direct" if method == "spline-diff" else "smplx2smpl")})
    return out
