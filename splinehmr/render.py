from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .paths import add_runtime_paths, local_body_model_utils_root, require_body_models

RenderModel = Literal["smplx2smpl", "smpl_direct"]


def _smplx_verts_to_smpl_verts(verts_smplx: torch.Tensor, smplx2smpl: torch.Tensor) -> torch.Tensor:
    T = int(verts_smplx.shape[0])
    Vx = int(verts_smplx.shape[1])
    x = verts_smplx.permute(1, 0, 2).reshape(Vx, T * 3)
    y = torch.sparse.mm(smplx2smpl, x) if bool(getattr(smplx2smpl, "is_sparse", False)) else smplx2smpl @ x
    return y.reshape(int(y.shape[0]), T, 3).permute(1, 0, 2).contiguous()


def _normalize_smpl_body_pose(body_pose: torch.Tensor, T: int) -> torch.Tensor:
    body_pose = body_pose[:T].float()
    if body_pose.ndim != 2:
        body_pose = body_pose.view(int(body_pose.shape[0]), -1)
    D = int(body_pose.shape[1])
    if D < 69:
        body_pose = torch.cat([body_pose, torch.zeros((T, 69 - D), device=body_pose.device, dtype=body_pose.dtype)], dim=1)
    elif D > 69:
        body_pose = body_pose[:, :69]
    return body_pose


@torch.no_grad()
def smpl_verts_from_hmr_pack(
    hmr_pack: dict[str, Any],
    T: int,
    device: str | torch.device,
    *,
    render_model: RenderModel = "smplx2smpl",
) -> tuple[torch.Tensor, np.ndarray]:
    add_runtime_paths()
    require_body_models()
    from multi_view_smpl_optimizer.utils.smplx_utils import make_smplx

    dev = torch.device(device)
    params = hmr_pack["smpl_params_incam"]
    smpl = make_smplx("smpl").to(dev).eval()
    faces = smpl.faces.detach().cpu().numpy() if torch.is_tensor(smpl.faces) else np.asarray(smpl.faces)

    if render_model == "smpl_direct":
        body_pose = _normalize_smpl_body_pose(params["body_pose"].to(dev), T)
        betas = params["betas"][:T].to(device=dev, dtype=torch.float32)
        if betas.ndim == 1:
            betas = betas[None].expand(T, -1)
        betas = betas[:, :10]
        global_orient = params["global_orient"][:T].to(device=dev, dtype=torch.float32).view(T, 3)
        transl = params["transl"][:T].to(device=dev, dtype=torch.float32).view(T, 3)
        out = smpl(body_pose=body_pose, betas=betas, global_orient=global_orient, transl=transl)
        verts = out.vertices if hasattr(out, "vertices") else out
        return verts.detach().cpu().float(), faces.astype(np.int64)

    smplx = make_smplx("supermotion", use_pca=False, flat_hand_mean=True).to(dev).eval()
    smplx2smpl_path = local_body_model_utils_root() / "smplx2smpl_sparse.pt"
    if not smplx2smpl_path.exists():
        raise FileNotFoundError(f"Missing local SMPLX->SMPL regressor: {smplx2smpl_path}")
    smplx2smpl = torch.load(str(smplx2smpl_path), map_location=dev)

    kwargs = {}
    for key in ("body_pose", "betas", "global_orient", "transl"):
        val = params[key]
        if torch.is_tensor(val):
            kwargs[key] = val[:T].to(device=dev, dtype=torch.float32)
        else:
            kwargs[key] = val
    out = smplx(**kwargs)
    verts_smplx = out.vertices if hasattr(out, "vertices") else out["vertices"]
    verts_smpl = _smplx_verts_to_smpl_verts(verts_smplx, smplx2smpl)
    return verts_smpl.detach().cpu(), faces.astype(np.int64)


def _draw_overlay_legend(image: np.ndarray) -> np.ndarray:
    """Draw an RGB legend: red = before optimization, green = after optimization."""
    try:
        import cv2
    except Exception:
        return image
    out = image.copy()
    x0, y0 = 18, 18
    line_h = 28
    box_w, box_h = 300, 82
    panel = out[y0 : y0 + box_h, x0 : x0 + box_w].copy()
    panel[:] = np.array([0, 0, 0], dtype=np.uint8)
    out[y0 : y0 + box_h, x0 : x0 + box_w] = (0.45 * out[y0 : y0 + box_h, x0 : x0 + box_w] + 0.55 * panel).astype(np.uint8)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)
    cv2.rectangle(out, (x0 + 15, y0 + 18), (x0 + 39, y0 + 38), (255, 0, 0), -1)
    cv2.putText(out, "Before optimization", (x0 + 50, y0 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (x0 + 15, y0 + 18 + line_h), (x0 + 39, y0 + 38 + line_h), (0, 255, 0), -1)
    cv2.putText(out, "After optimization", (x0 + 50, y0 + 36 + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_outputs(
    *,
    video_path: Path,
    before_pack: dict[str, Any],
    after_pack: dict[str, Any],
    out_dir: Path,
    T: int,
    device: str | torch.device,
    crf: int = 23,
    fast_render: bool = True,
    render_model: RenderModel = "smplx2smpl",
) -> dict[str, Path]:
    add_runtime_paths()
    from .gvhmr_compat.video_io import get_video_lwh, get_video_reader, get_writer
    from .gvhmr_compat.renderer import Renderer

    out_dir.mkdir(parents=True, exist_ok=True)
    verts_before, faces = smpl_verts_from_hmr_pack(before_pack, T=T, device=device, render_model=render_model)
    verts_after, _ = smpl_verts_from_hmr_pack(after_pack, T=T, device=device, render_model=render_model)

    length, width, height = get_video_lwh(str(video_path))
    try:
        import imageio.v3 as iio

        meta = iio.immeta(str(video_path), plugin="pyav")
        fps = int(round(float(meta.get("fps", 30.0) or 30.0)))
    except Exception:
        fps = 30
    n_render = min(int(length), int(T))
    K0 = before_pack["K_fullimg"][0]
    if not torch.is_tensor(K0):
        K0 = torch.tensor(K0, dtype=torch.float32)

    dev = torch.device(device)
    renderer = Renderer(width, height, device=str(dev), faces=faces, K=K0.to(dev))
    if fast_render:
        renderer.update_bbox = lambda *args, **kwargs: None
        renderer.reset_bbox = lambda *args, **kwargs: None

    paths = {
        "before_video": out_dir / "render_before.mp4",
        "after_video": out_dir / "render_after.mp4",
        "compare_video": out_dir / "render_compare.mp4",
        "overlay_compare_video": out_dir / "render_overlay_compare.mp4",
    }
    writers = {
        "before_video": get_writer(str(paths["before_video"]), fps=fps, crf=int(crf)),
        "after_video": get_writer(str(paths["after_video"]), fps=fps, crf=int(crf)),
        "compare_video": get_writer(str(paths["compare_video"]), fps=fps, crf=int(crf)),
        "overlay_compare_video": get_writer(str(paths["overlay_compare_video"]), fps=fps, crf=int(crf)),
    }
    reader = get_video_reader(str(video_path))
    try:
        for i, img_raw in enumerate(reader):
            if i >= n_render:
                break
            img_before = renderer.render_mesh(verts_before[i].to(dev), img_raw, colors=[0.8, 0.8, 0.8], VI=50)
            img_after = renderer.render_mesh(verts_after[i].to(dev), img_raw, colors=[0.8, 0.8, 0.8], VI=50)
            img_overlay = renderer.render_mesh(verts_before[i].to(dev), img_raw, colors=[255, 0, 0], VI=50, alpha=0.58)
            img_overlay = renderer.render_mesh(verts_after[i].to(dev), img_overlay, colors=[0, 255, 0], VI=50, alpha=0.58)
            img_overlay = _draw_overlay_legend(img_overlay)
            writers["before_video"].write_frame(img_before)
            writers["after_video"].write_frame(img_after)
            writers["compare_video"].write_frame(np.concatenate([img_before, img_after], axis=1))
            writers["overlay_compare_video"].write_frame(img_overlay)
    finally:
        for writer in writers.values():
            writer.close()
        try:
            reader.close()
        except Exception:
            pass
    return paths
