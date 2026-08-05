"""
GVHMR native static-joint postprocess, copied into multi_view_smpl_optimizer.

This module is intentionally thin: the full logic lives in `utils/static_utils/*`
copied from GVHMR, and we just expose a stable one-stop API here.
"""

from __future__ import annotations

from pathlib import Path
import torch

from .static_utils.endecoder_lite import EnDecoderLite
from .static_utils.postprocess_native import pp_static_joint, pp_static_joint_cam, process_ik


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_smplx_npz_dir() -> str:
    # Reuse the existing GVHMR model files (code is decoupled; weights are shared on disk).
    from multi_view_smpl_optimizer.utils.gvhmr_utils import get_gvhmr_root
    gvhmr_root = get_gvhmr_root()
    return str(gvhmr_root / "inputs" / "checkpoints" / "body_models" / "smplx")


@torch.no_grad()
def apply_gvhmr_static_postprocess(
    *,
    body_pose: torch.Tensor,  # (T,63) or (T,69) (only first 63 used by GVHMR IK)
    betas: torch.Tensor,  # (10,) or (T,10)
    global_orient: torch.Tensor,  # (T,3)
    transl: torch.Tensor,  # (T,3)
    static_conf_logits: torch.Tensor,  # (T,6) or (T-1,6)
    static_cam: bool = False,
    smplx_model_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Exactly run GVHMR native postprocess chain:
      - pp_static_joint / pp_static_joint_cam (transl correction)
      - process_ik (static-confidence CCD IK)

    Returns updated global SMPL params:
      - body_pose (T,63) or (T,69) with tail preserved if provided
      - global_orient (T,3) unchanged
      - transl (T,3) updated
      - betas unchanged
    """
    T = int(min(body_pose.shape[0], global_orient.shape[0], transl.shape[0]))
    body_pose_t = body_pose[:T]
    global_orient_t = global_orient[:T]
    transl_t = transl[:T]

    # EnDecoderLite expects batched inputs (B,L,...)
    endecoder = EnDecoderLite(smplx_model_path=smplx_model_path or _default_smplx_npz_dir()).to(body_pose.device)

    # Make betas (B,L,10) to match GVHMR code path
    if betas.ndim == 1:
        betas_bl = betas.view(1, 1, -1).expand(1, T, -1)
    elif betas.ndim == 2:
        betas_bl = betas[:T].unsqueeze(0)
    else:
        raise ValueError(f"betas must be (10,) or (T,10), got {tuple(betas.shape)}")

    pred_global = {
        "body_pose": body_pose_t[:, :63].unsqueeze(0),
        "betas": betas_bl,
        "global_orient": global_orient_t.unsqueeze(0),
        "transl": transl_t.unsqueeze(0),
    }
    pred_incam = pred_global.copy()

    # static logits to (B,L,6)
    if static_conf_logits.ndim == 2:
        static_bl = static_conf_logits[:T].unsqueeze(0)
    elif static_conf_logits.ndim == 3:
        static_bl = static_conf_logits[:, :T]
    else:
        raise ValueError(f"static_conf_logits must be (T,6) or (B,T,6), got {tuple(static_conf_logits.shape)}")

    outputs = {
        "pred_smpl_params_incam": pred_incam,
        "pred_smpl_params_global": pred_global,
        "static_conf_logits": static_bl,
    }

    if static_cam:
        outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(outputs, endecoder)
    else:
        outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, endecoder)

    body_pose_post = process_ik(outputs, endecoder)  # (B,L,63)
    outputs["pred_smpl_params_global"]["body_pose"] = body_pose_post

    out_body63 = body_pose_post[0]
    out_transl = outputs["pred_smpl_params_global"]["transl"][0]

    # Preserve 69D tail if input was 69D
    if int(body_pose.shape[1]) == 69:
        out_body = torch.cat([out_body63, body_pose_t[:, 63:69]], dim=-1)
    else:
        out_body = out_body63

    return {
        "body_pose": out_body,
        "global_orient": global_orient_t,
        "transl": out_transl,
        "betas": betas,
    }

