"""
ScoreHMR 封装（不改官方 ScoreHMR 代码）：

提供一个“初始化 SMPL 参数 + 2D 关键点 → 利用 ScoreHMR 的 keypoint guidance 做迭代式重投影贴合”的入口。

注意：ScoreHMR 的“优化”方式不是 SMPLify-X 那种显式 LBFGS/Adam 对 SMPL 参数做优化循环，
而是 diffusion sampling 过程中用 2D 重投影 loss 的梯度做 score guidance，同时用 Adam 只优化 camera translation。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = Path(_THIS_DIR).resolve().parents[0]
_SCOREHMR_ROOT = str(_REPO_ROOT / "third_party" / "ScoreHMR")

# Make ScoreHMR importable (it uses top-level imports like `from constants import ...`)
if _SCOREHMR_ROOT not in sys.path:
    sys.path.insert(0, _SCOREHMR_ROOT)


def _to_torch(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(x), device=device, dtype=dtype)


def _coco17_to_openpose25_tj3(kp_coco17_tj3: torch.Tensor) -> torch.Tensor:
    """
    COCO17 (T,17,3) -> OpenPose25 (T,25,3).
    17 shared joints are re-indexed losslessly; missing joints conf=0.
    """
    x = kp_coco17_tj3
    if x.ndim != 3 or x.shape[-1] != 3 or x.shape[1] != 17:
        raise ValueError(f"Expected (T,17,3) coco17, got {tuple(x.shape)}")
    T = int(x.shape[0])
    out = torch.zeros((T, 25, 3), device=x.device, dtype=x.dtype)

    # COCO17 order:
    # 0 nose,1 leye,2 reye,3 lear,4 rear,5 lsho,6 rsho,7 lelb,8 relb,9 lwri,10 rwri,
    # 11 lhip,12 rhip,13 lkne,14 rkne,15 lank,16 rank
    # OpenPose BODY25 order indices we fill:
    # 0 nose,2 rsho,3 relb,4 rwri,5 lsho,6 lelb,7 lwri,9 rhip,10 rkne,11 rank,12 lhip,13 lkne,14 lank,
    # 15 reye,16 leye,17 rear,18 lear
    mapping = {
        0: 0,
        1: 16,  # leye
        2: 15,  # reye
        3: 18,  # lear
        4: 17,  # rear
        5: 5,  # lsho
        6: 2,  # rsho
        7: 6,  # lelb
        8: 3,  # relb
        9: 7,  # lwri
        10: 4,  # rwri
        11: 12,  # lhip
        12: 9,  # rhip
        13: 13,  # lkne
        14: 10,  # rkne
        15: 14,  # lank
        16: 11,  # rank
    }
    for c_i, op_i in mapping.items():
        out[:, op_i, :] = x[:, c_i, :]

    # Derive neck (1) and mid-hip (8) when available
    lsho = out[:, 5, :2]
    rsho = out[:, 2, :2]
    lsho_c = out[:, 5, 2:3]
    rsho_c = out[:, 2, 2:3]
    neck_xy = 0.5 * (lsho + rsho)
    neck_c = torch.minimum(lsho_c, rsho_c)
    out[:, 1, :2] = neck_xy
    out[:, 1, 2:3] = neck_c

    lhip = out[:, 12, :2]
    rhip = out[:, 9, :2]
    lhip_c = out[:, 12, 2:3]
    rhip_c = out[:, 9, 2:3]
    mid_xy = 0.5 * (lhip + rhip)
    mid_c = torch.minimum(lhip_c, rhip_c)
    out[:, 8, :2] = mid_xy
    out[:, 8, 2:3] = mid_c
    return out


def _ensure_scorehmr_joints_tj3(kp_tj3: torch.Tensor) -> torch.Tensor:
    """
    ScoreHMR 的 keypoint guidance 使用 44 维关键点定义（25 OpenPose + 19 extra）。
    为了与 `model_joints` 维度对齐，我们把输入扩展到 (T, 44, 3)，多出来的 joints conf=0。

    输入允许：
    - (T, 17, 3) COCO17
    - (T, 25, 3) OpenPose25
    - (T, J, 2) 或 (T, J, 3) 其他：若 J<44 则 padding
    """
    if kp_tj3.ndim != 3 or int(kp_tj3.shape[-1]) not in (2, 3):
        raise ValueError(f"Expected (T,J,2/3), got {tuple(kp_tj3.shape)}")
    T, J = int(kp_tj3.shape[0]), int(kp_tj3.shape[1])
    x = kp_tj3
    if int(x.shape[-1]) == 2:
        conf = torch.ones((T, J, 1), device=x.device, dtype=x.dtype)
        x = torch.cat([x, conf], dim=-1)

    # Sanitize NaN/Inf and conf range.
    try:
        bad = ~torch.isfinite(x)
        if bool(bad.any()):
            x = x.clone()
            x[bad] = 0.0
    except Exception:
        pass
    try:
        xy_bad = ~torch.isfinite(x[:, :, :2])
        if bool(xy_bad.any()):
            x = x.clone()
            x[xy_bad.any(dim=-1), 2] = 0.0
    except Exception:
        pass
    try:
        x[:, :, 2] = torch.clamp(x[:, :, 2], 0.0, 1.0)
    except Exception:
        pass
    if J == 17:
        x = _coco17_to_openpose25_tj3(x)
        J = 25
    if J < 44:
        pad = torch.zeros((T, 44 - J, 3), device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    return x


def _aa24_to_rotmat24(aa_24x3: torch.Tensor) -> torch.Tensor:
    """
    axis-angle (B,24,3) -> rotmat (B,24,3,3) using ScoreHMR's implementation.
    """
    from score_hmr.utils.geometry import aa_to_rotmat  # type: ignore

    B = int(aa_24x3.shape[0])
    aa = aa_24x3.reshape(B * 24, 3)
    R = aa_to_rotmat(aa).reshape(B, 24, 3, 3)
    return R


def _rotmat_to_aa(rotmat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    rotmat (...,3,3) -> axis-angle (...,3)
    Minimal local implementation (avoid extra deps).
    """
    R = rotmat
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {tuple(R.shape)}")
    # trace = 1 + 2 cos(theta)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (tr - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    r = torch.stack([rx, ry, rz], dim=-1)
    denom = 2.0 * torch.sin(theta).unsqueeze(-1)
    axis = r / torch.clamp(denom, min=1e-8)
    aa = axis * theta.unsqueeze(-1)
    aa = torch.nan_to_num(aa, nan=0.0, posinf=0.0, neginf=0.0)
    return aa


@dataclass
class ScoreHMRRefineCfg:
    """
    仅包含 wrapper 需要的关键参数。
    """

    ckpt_name: str = "score_hmr"
    milestone: int = 100
    use_default_ckpt: bool = True
    # sampling / guidance
    num_samples: int = 1
    optim_iters: int = 2
    early_stopping: bool = True
    temporal_guidance: bool = False
    # if cond_feats not provided, use zeros
    allow_dummy_cond: bool = True
    # optionally compute cond_feats via PARE from images (closer to official demo/eval)
    use_pare_cond: bool = False
    # print small logs indicating PARE/cond path is used
    log_pare: bool = False
    # B-spline temporal smoothing of pred_noise (default off). When True, reads custom/configs/bspline.yaml.
    use_bspline_smooth_noise: bool = False
    # FPS for m_per_t resolution (e.g. "fps_div_2" -> M = fps//2). Fallback when batch has no "fps".
    bspline_fps: Optional[float] = None
    bspline_m_per_t: Optional[str] = None
    bspline_order: Optional[int] = None
    bspline_use_tanh: Optional[bool] = None
    bspline_tanh_amp: Optional[float] = None
    # Elastic soft blend weight (0~1). None = use bspline.yaml blend_weight. 0.3~0.5: balance smooth vs 2D fit.
    bspline_blend_weight: Optional[float] = None


_CACHED = {}


def _patch_scorehmr_path_constants() -> None:
    """
    ScoreHMR uses relative paths in `constants.py` and also re-exports them into other modules
    via `from constants import ...`. When imported from outside the ScoreHMR repo root,
    these relative paths break (e.g. cannot find `data/model_weights/...`).

    We patch both:
    - `ScoreHMR/constants.py` module variables
    - `score_hmr.models.model_utils` copied variables
    """
    try:
        import constants as c  # type: ignore
    except Exception:
        return

    ckpt_dir = os.path.join(_SCOREHMR_ROOT, "data", "model_weights")
    results_dir = os.path.join(_SCOREHMR_ROOT, "logs")
    pare_ckpt = os.path.join(ckpt_dir, "pare", "pare_checkpoint.ckpt")

    try:
        c.CHECKPOINT_DIR = ckpt_dir  # type: ignore[attr-defined]
        c.RESULTS_DIR = results_dir  # type: ignore[attr-defined]
        c.PARE_CHECKPOINT = pare_ckpt  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        from score_hmr.models import model_utils as mu  # type: ignore

        mu.CHECKPOINT_DIR = ckpt_dir  # type: ignore[attr-defined]
        mu.RESULTS_DIR = results_dir  # type: ignore[attr-defined]
        mu.PARE_CHECKPOINT = pare_ckpt  # type: ignore[attr-defined]
    except Exception:
        pass


def _get_scorehmr_model(*, device: torch.device, cfg: ScoreHMRRefineCfg):
    """
    Lazy-load ScoreHMR diffusion model (EMA weights).
    """
    key = (
        str(device),
        cfg.ckpt_name,
        int(cfg.milestone),
        bool(cfg.use_default_ckpt),
        bool(getattr(cfg, "temporal_guidance", False)),
        bool(getattr(cfg, "use_bspline_smooth_noise", False)),
        getattr(cfg, "bspline_m_per_t", None),
        getattr(cfg, "bspline_order", None),
        getattr(cfg, "bspline_use_tanh", None),
        getattr(cfg, "bspline_tanh_amp", None),
        getattr(cfg, "bspline_blend_weight", None),
    )
    if key in _CACHED:
        return _CACHED[key]

    _patch_scorehmr_path_constants()

    from score_hmr.configs import model_config  # type: ignore
    from score_hmr.models.model_utils import load_diffusion_model  # type: ignore

    mcfg = model_config()
    mcfg = mcfg.clone()
    mcfg.defrost()
    # Patch relative asset paths to absolute ones under ScoreHMR repo.
    try:
        mcfg.SMPL.MODEL_PATH = os.path.join(_SCOREHMR_ROOT, "data", "smpl")
    except Exception:
        pass
    try:
        mcfg.SMPL.JOINT_REGRESSOR_EXTRA = os.path.join(_SCOREHMR_ROOT, "data", "SMPL_to_J19.pkl")
    except Exception:
        pass
    try:
        mcfg.SMPL.MEAN_PARAMS = os.path.join(_SCOREHMR_ROOT, "data", "smpl_mean_params.npz")
    except Exception:
        pass
    try:
        mcfg.MODEL.BETAS_STATS = os.path.join(_SCOREHMR_ROOT, "data", "stats", "betas_stats_eft_fits.npz")
    except Exception:
        pass
    mcfg.freeze()

    kwargs = dict(
        name=cfg.ckpt_name,
        milestone=int(cfg.milestone),
        use_default_ckpt=bool(cfg.use_default_ckpt),
        device=device,
        keypoint_guidance=True,
        temporal_guidance=bool(getattr(cfg, "temporal_guidance", False)),
        early_stopping=bool(cfg.early_stopping),
        optim_iters=int(cfg.optim_iters),
    )
    if getattr(cfg, "use_bspline_smooth_noise", False):
        kwargs["use_bspline_smooth_noise"] = True
        if getattr(cfg, "bspline_fps", None) is not None:
            kwargs["bspline_fps"] = float(cfg.bspline_fps)
        if getattr(cfg, "bspline_m_per_t", None) is not None:
            kwargs["bspline_m_per_t"] = str(cfg.bspline_m_per_t)
        if getattr(cfg, "bspline_order", None) is not None:
            kwargs["bspline_order"] = int(cfg.bspline_order)
        if getattr(cfg, "bspline_use_tanh", None) is not None:
            kwargs["bspline_use_tanh"] = bool(cfg.bspline_use_tanh)
        if getattr(cfg, "bspline_tanh_amp", None) is not None:
            kwargs["bspline_tanh_amp"] = float(cfg.bspline_tanh_amp)
        if getattr(cfg, "bspline_blend_weight", None) is not None:
            kwargs["bspline_blend_weight"] = float(cfg.bspline_blend_weight)
    model = load_diffusion_model(mcfg, **kwargs)
    _CACHED[key] = model
    return model


def _load_pare_feat_stats(*, device: torch.device, use_betas: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load feature standardization stats for PARE features from ScoreHMR repo.
    This avoids relying on ScoreHMR's `StandarizeImageFeatures` which uses relative paths.
    """
    stats_path = os.path.join(_SCOREHMR_ROOT, "data", "stats", "pare_feat_stats.npz")
    obj = np.load(stats_path)
    pose_feat_mean = obj["pose_feats_mean"].reshape(-1).astype(np.float32)
    pose_feat_std = obj["pose_feats_std"].reshape(-1).astype(np.float32)
    if use_betas:
        cam_shape_feat_mean = obj["cam_shape_feats_mean"].reshape(-1).astype(np.float32)
        cam_shape_feat_std = obj["cam_shape_feats_std"].reshape(-1).astype(np.float32)
        feat_mean = np.concatenate([pose_feat_mean, cam_shape_feat_mean], axis=0)
        feat_std = np.concatenate([pose_feat_std, cam_shape_feat_std], axis=0)
    else:
        feat_mean = pose_feat_mean
        feat_std = pose_feat_std
    mean_t = torch.from_numpy(feat_mean).to(device=device, dtype=torch.float32)
    std_t = torch.from_numpy(feat_std).to(device=device, dtype=torch.float32)
    return mean_t, std_t


def _get_pare_and_standardizer(*, model: Any, device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """
    Lazy-load PARE model + feature standardization stats.
    """
    key = (str(device), "pare", "use_betas=False")
    if key in _CACHED:
        return _CACHED[key]

    _patch_scorehmr_path_constants()

    from score_hmr.models.model_utils import load_pare  # type: ignore

    # PARE code inside ScoreHMR uses relative paths like "data/smpl_mean_params.npz".
    # Make it robust by temporarily switching CWD to the ScoreHMR repo root.
    _cwd0 = os.getcwd()
    try:
        os.chdir(_SCOREHMR_ROOT)
        pare = load_pare(model.cfg.SMPL).to(device)  # type: ignore[attr-defined]
    finally:
        try:
            os.chdir(_cwd0)
        except Exception:
            pass
    pare.eval()
    mean_t, std_t = _load_pare_feat_stats(device=device, use_betas=False)

    _CACHED[key] = (pare, mean_t, std_t)
    return pare, mean_t, std_t


def _build_pare_image_patches_from_paths(
    *,
    image_paths: list[str],
    bbx_xys: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Build ScoreHMR/PARE input images as in ScoreHMR datasets:
    - crop to bbox center/size
    - resize to DEFAULT_IMG_SIZE (224)
    - convert BGR->RGB, HWC->CHW
    - normalize by DEFAULT_MEAN/STD (in 0..255 scale)

    Returns: (T,3,224,224) float32 tensor on `device`.
    """
    import cv2

    from constants import DEFAULT_IMG_SIZE, DEFAULT_MEAN, DEFAULT_STD  # type: ignore
    from score_hmr.datasets.utils import convert_cvimg_to_tensor, generate_image_patch  # type: ignore

    T = int(len(image_paths))
    out = torch.zeros((T, 3, int(DEFAULT_IMG_SIZE), int(DEFAULT_IMG_SIZE)), dtype=torch.float32, device=device)

    mean = np.asarray(DEFAULT_MEAN, dtype=np.float32).reshape(3)
    std = np.asarray(DEFAULT_STD, dtype=np.float32).reshape(3)

    for i, p in enumerate(image_paths):
        im_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise RuntimeError(f"Failed to read image: {p}")
        H0, W0 = int(im_bgr.shape[0]), int(im_bgr.shape[1])
        if bbx_xys is not None and torch.is_tensor(bbx_xys) and bbx_xys.ndim == 2 and int(bbx_xys.shape[1]) == 3:
            cx = float(bbx_xys[i, 0].item())
            cy = float(bbx_xys[i, 1].item())
            s = float(bbx_xys[i, 2].item())
            bbox_size = float(s) * 200.0  # HMR-style scale -> pixels
        else:
            cx = 0.5 * float(W0)
            cy = 0.5 * float(H0)
            bbox_size = float(min(H0, W0))

        img_patch_cv, _trans = generate_image_patch(
            im_bgr,
            cx,
            cy,
            bbox_size,
            bbox_size,
            float(DEFAULT_IMG_SIZE),
            float(DEFAULT_IMG_SIZE),
            False,  # do_flip
            1.0,  # scale
            0.0,  # rot
            load_image=True,
        )
        # Convert to RGB as ScoreHMR does
        img_rgb = img_patch_cv[:, :, ::-1].copy()
        img_chw = convert_cvimg_to_tensor(img_rgb)  # (3,224,224) float32 0..255
        # Normalize per-channel
        for c in range(3):
            img_chw[c, :, :] = (img_chw[c, :, :] - mean[c]) / std[c]
        out[i] = torch.from_numpy(img_chw).to(device=device, dtype=torch.float32)

    return out


@torch.no_grad()
def refine_smpl_to_2d_with_scorehmr(
    *,
    global_orient_aa: torch.Tensor,
    body_pose_aa: torch.Tensor,
    betas: torch.Tensor,
    init_cam_t: torch.Tensor,
    keypoints_2d: torch.Tensor,
    image_size_hw: Tuple[int, int],
    K_3x3: Optional[torch.Tensor] = None,
    cond_feats: Optional[torch.Tensor] = None,
    image_paths: Optional[list[str]] = None,
    bbx_xys: Optional[torch.Tensor] = None,
    device: str | torch.device = "cuda",
    cfg: Optional[ScoreHMRRefineCfg] = None,
    fps: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        global_orient_aa: (T,3) axis-angle
        body_pose_aa: (T,69) axis-angle for 23 joints, or (T,23,3)
        betas: (10,) or (T,10)
        init_cam_t: (T,3) camera translation used by ScoreHMR (will be optimized by Adam internally)
        keypoints_2d: (T,J,3) in pixel coords; supports COCO17/OpenPose25; will be padded to >=40 joints
        image_size_hw: (H,W)
        K_3x3: optional (3,3) or (T,3,3) intrinsics; if None use ScoreHMR default focal (cfg.EXTRA.FOCAL_LENGTH) and center=0.5*img
        cond_feats: optional (T,C) image features; if None and allow_dummy_cond, uses zeros
    Returns:
        dict with refined params:
        - global_orient_aa_refined: (T,3)
        - body_pose_aa_refined: (T,69)
        - betas_refined: (T,10) (unchanged unless model uses betas)
        - cam_t_refined: (T,3)
        - pred_pose_rotmat: (T,24,3,3) (refined)
    """
    if cfg is None:
        cfg = ScoreHMRRefineCfg()

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    model = _get_scorehmr_model(device=dev, cfg=cfg)
    model = model.to(dev)
    model.eval()

    dtype = torch.float32
    T = int(global_orient_aa.shape[0])

    go = _to_torch(global_orient_aa, device=dev, dtype=dtype).view(T, 3)
    bp = _to_torch(body_pose_aa, device=dev, dtype=dtype)
    if bp.ndim == 2 and int(bp.shape[1]) == 69:
        bp_23x3 = bp.view(T, 23, 3)
    elif bp.ndim == 3 and int(bp.shape[1]) == 23 and int(bp.shape[2]) == 3:
        bp_23x3 = bp
    else:
        raise ValueError(f"body_pose_aa must be (T,69) or (T,23,3), got {tuple(bp.shape)}")

    aa24 = torch.cat([go.view(T, 1, 3), bp_23x3], dim=1)  # (T,24,3)
    pred_pose = _aa24_to_rotmat24(aa24)  # (T,24,3,3)

    b0 = _to_torch(betas, device=dev, dtype=dtype)
    if b0.ndim == 1:
        b0 = b0.view(1, -1).repeat(T, 1)
    b0 = b0.view(T, 10)

    cam_t0 = _to_torch(init_cam_t, device=dev, dtype=dtype).view(T, 3)

    kp = _to_torch(keypoints_2d, device=dev, dtype=dtype)
    kp = _ensure_scorehmr_joints_tj3(kp)  # (T,>=40,3)
    joints_2d = kp[:, :, :2].contiguous()
    joints_conf = kp[:, :, 2:3].contiguous()

    H, W = int(image_size_hw[0]), int(image_size_hw[1])
    img_size = torch.tensor([H, W], device=dev, dtype=dtype).view(1, 2).repeat(T, 1)
    camera_center = 0.5 * img_size[:, [1, 0]]  # (T,2) in (cx,cy) order; img_size stored (H,W)

    # focal_length is (T,2) in pixels
    if K_3x3 is not None:
        Kt = _to_torch(K_3x3, device=dev, dtype=dtype)
        if Kt.ndim == 2:
            Kt = Kt.view(1, 3, 3).repeat(T, 1, 1)
        focal = torch.stack([Kt[:, 0, 0], Kt[:, 1, 1]], dim=-1)
        camera_center = torch.stack([Kt[:, 0, 2], Kt[:, 1, 2]], dim=-1)
    else:
        # fallback to config focal
        try:
            focal_val = float(model.cfg.EXTRA.FOCAL_LENGTH)  # type: ignore[attr-defined]
        except Exception:
            focal_val = 5000.0
        focal = torch.full((T, 2), focal_val, device=dev, dtype=dtype)

    # cond feats: prefer explicit cond_feats; else optionally compute via PARE from images; else dummy zeros
    cond: torch.Tensor
    if cond_feats is not None:
        cond = _to_torch(cond_feats, device=dev, dtype=dtype)
        if cond.ndim != 2 or int(cond.shape[0]) != T:
            raise ValueError(f"cond_feats must be (T,C), got {tuple(cond.shape)}")
    else:
        use_pare_cond = bool(getattr(cfg, "use_pare_cond", False))
        if use_pare_cond and image_paths is not None and len(image_paths) == T:
            try:
                import time

                t_pare0 = time.time()
                pare, mean_t, std_t = _get_pare_and_standardizer(model=model, device=dev)
                imgs = _build_pare_image_patches_from_paths(image_paths=list(image_paths), bbx_xys=bbx_xys, device=dev)
                pare_out = pare(imgs, get_feats=True)
                pose_feats = pare_out["pose_feats"].reshape(T, -1).to(dtype=dtype)
                cond = (pose_feats - mean_t) / std_t
                if bool(getattr(cfg, "log_pare", False)):
                    dt = float(time.time() - t_pare0)
                    # keep logs concise but informative
                    try:
                        m = float(cond.mean().item())
                        s = float(cond.std(unbiased=False).item())
                    except Exception:
                        m, s = 0.0, 0.0
                    print(
                        f"[Spline-Diff][PARE] ok T={T} imgs={tuple(imgs.shape)} pose_feats={tuple(pose_feats.shape)} "
                        f"cond={tuple(cond.shape)} cond_mean={m:.4f} cond_std={s:.4f} dt={dt:.3f}s"
                    )
            except Exception as e:
                # fallback to dummy cond for robustness
                use_pare_cond = False
                cond = torch.empty((0, 0), device=dev, dtype=dtype)
                if bool(getattr(cfg, "log_pare", False)):
                    import traceback

                    print("[Spline-Diff][PARE] failed, fallback to dummy cond")
                    print("[Spline-Diff][PARE] err:", repr(e))
                    print(traceback.format_exc())
        else:
            cond = torch.empty((0, 0), device=dev, dtype=dtype)

        if cond.numel() == 0:
            if not bool(cfg.allow_dummy_cond):
                raise ValueError("cond_feats is required when allow_dummy_cond=False")
            # Use zeros with correct dimension expected by denoising model
            img_feats = str(model.cfg.MODEL.DENOISING_MODEL.IMG_FEATS)  # type: ignore[attr-defined]
            from score_hmr.models.denoising_model import PREDICTORS  # type: ignore

            C = int(PREDICTORS[img_feats]["thetas_emb_dim"])
            cond = torch.zeros((T, C), device=dev, dtype=dtype)
            if bool(getattr(cfg, "log_pare", False)):
                print(f"[Spline-Diff][Cond] using dummy zeros cond: T={T} C={C}")

    # If num_samples>1, repeat cond to match batch_size (=T*num_samples)
    ns = int(getattr(cfg, "num_samples", 1))
    if ns > 1 and int(cond.shape[0]) == T:
        cond = cond.repeat_interleave(ns, dim=0)

    # Build batch dict as ScoreHMR expects
    batch: Dict[str, Any] = {
        # used for real_batch_size
        "keypoints_2d": kp,
        # regression estimates
        "pred_pose": pred_pose,
        "pred_betas": b0,
        # keypoint guidance inputs
        "joints_2d": joints_2d,
        "joints_conf": joints_conf,
        "camera_center": camera_center,
        "focal_length": focal,
        "img_size": img_size,
        "init_cam_t": cam_t0,
    }
    if fps is not None:
        batch["fps"] = float(fps)
    elif getattr(cfg, "bspline_fps", None) is not None:
        batch["fps"] = float(cfg.bspline_fps)

    if getattr(cfg, "use_bspline_smooth_noise", False):
        w = getattr(cfg, "bspline_blend_weight", None)
        w_str = f"blend_weight={w}" if w is not None else "blend_weight=config"
        print(
            "[Spline-Diff] B-spline temporal smoothing enabled (elastic soft blend): %s; T=%d, num_samples=%d"
            % (w_str, T, int(cfg.num_samples))
        )

    out = model.sample(batch, cond, batch_size=T * int(cfg.num_samples))
    x0 = out["x_0"]

    # Convert x_0 to SMPL params (rotmat)
    from score_hmr.utils.utils import prepare_smpl_params  # type: ignore

    pred_params = prepare_smpl_params(
        x0,
        num_samples=int(cfg.num_samples),
        use_betas=bool(model.use_betas),
        betas_min=getattr(model, "betas_min", None),
        betas_max=getattr(model, "betas_max", None),
        pred_betas=b0,
    )
    # For num_samples==1, shapes are:
    # - global_orient: (T,1,3,3)
    # - body_pose: (T,23,3,3)
    Rg = pred_params["global_orient"].reshape(T, 3, 3)
    Rb = pred_params["body_pose"].reshape(T, 23, 3, 3)
    R24 = torch.cat([Rg[:, None], Rb], dim=1)

    go_aa = _rotmat_to_aa(Rg).view(T, 3)
    bp_aa = _rotmat_to_aa(Rb.reshape(T * 23, 3, 3)).reshape(T, 23, 3).reshape(T, 69)

    cam_t_ref = out.get("camera_translation", None)
    if cam_t_ref is None:
        cam_t_ref = cam_t0
    else:
        cam_t_ref = cam_t_ref.reshape(T, 3)

    bet_ref = pred_params["betas"]
    if bet_ref.ndim == 2:
        bet_ref = bet_ref.reshape(T, 10)
    elif bet_ref.ndim == 3:
        bet_ref = bet_ref[:, 0, :].reshape(T, 10)

    return {
        "global_orient_aa_refined": go_aa.detach(),
        "body_pose_aa_refined": bp_aa.detach(),
        "betas_refined": bet_ref.detach(),
        "cam_t_refined": cam_t_ref.detach(),
        "pred_pose_rotmat": R24.detach(),
    }


def main(argv: Optional[list[str]] = None) -> None:
    """
    快速冒烟测试（无需图片特征）：
    - 随机生成一段初始化 SMPL 参数 + 2D 关键点序列
    - 调用 `refine_smpl_to_2d_with_scorehmr` 跑一次 keypoint guidance
    - 打印输出 tensor shapes 与耗时，验证 ScoreHMR wrapper 跑通

    Quick test（对齐 smplify-x_wrapper.py 的 quick_test 输出）：
    - 读取 3DPW/smpl_outputs/courtyard_arguing_00 的第 0 帧初始化 SMPL 参数 + vitpose 2D
    - 调用本文件的 `refine_smpl_to_2d_with_scorehmr` 做一次优化
    - 渲染 mesh_before/mesh_after 以及 gt_vs_proj_before/after
    """
    import argparse
    import time

    p = argparse.ArgumentParser()
    p.add_argument(
        "--quick_test",
        action="store_true",
        help="Run one ScoreHMR keypoint-guidance quick test on a user-provided dataset directory.",
    )
    p.add_argument("--quick_test_dataset_dir", type=str, default="")
    p.add_argument("--quick_test_view", type=int, default=0)
    p.add_argument("--quick_test_frame", type=int, default=0)
    p.add_argument("--quick_test_max_kb", type=int, default=400)
    p.add_argument("--quick_test_render_device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument(
        "--quick_test_out_dir",
        type=str,
        default="",
        help="默认：bss-smplify/outputs/scorehmr_wrapper_test/<seq>_viewX_tXXXXX",
    )

    p.add_argument("--T", type=int, default=8, help="sequence length")
    p.add_argument("--H", type=int, default=1080)
    p.add_argument("--W", type=int, default=1920)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--use_pare_cond", action="store_true", default=False, help="quick_test 时用 PARE 从图片提取 cond_feats（更接近官方）"
    )
    p.add_argument("--log_pare", action="store_true", default=False, help="打印 PARE/cond_feats 路径关键日志（仅用于调试）")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--optim_iters", type=int, default=2)
    p.add_argument("--early_stopping", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args(argv)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    if bool(getattr(args, "quick_test", False)):
        import json

        import cv2

        # Use the SplineHMR-vendored GVHMR-like renderer when pytorch3d is available.
        repo_root = Path(__file__).resolve().parents[1]
        Renderer = None
        _renderer_backend = "none"
        try:
            from splinehmr.gvhmr_compat.renderer import Renderer as _Renderer

            Renderer = _Renderer
            _renderer_backend = "pytorch3d"
        except Exception:
            # In some envs (e.g. mismatched pytorch3d build), importing the renderer fails.
            # We will fall back to a lightweight OpenCV wireframe overlay.
            Renderer = None
            _renderer_backend = "opencv_wireframe"

        def _write_jpg_under_kb(path: Path, image_bgr: np.ndarray, *, max_kb: int = 400) -> None:
            img = image_bgr
            for _scale_try in range(6):
                for q in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]:
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
                    if not ok:
                        continue
                    if int(buf.nbytes) <= int(max_kb) * 1024:
                        path.write_bytes(buf.tobytes())
                        return
                h, w = int(img.shape[0]), int(img.shape[1])
                nh, nw = max(64, int(h * 0.85)), max(64, int(w * 0.85))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(30)])
            if ok:
                path.write_bytes(buf.tobytes())
            else:
                cv2.imwrite(str(path.with_suffix(".png")), img)

        def _draw_gt_vs_proj(
            image_bgr: np.ndarray, *, gt_xyc: np.ndarray, proj_xy: np.ndarray, conf_thr: float = 0.05
        ) -> np.ndarray:
            im = image_bgr.copy()
            gt = np.asarray(gt_xyc, dtype=np.float32)
            pr = np.asarray(proj_xy, dtype=np.float32)
            if gt.shape[0] != pr.shape[0]:
                j = min(int(gt.shape[0]), int(pr.shape[0]))
                gt = gt[:j]
                pr = pr[:j]
            for i in range(int(gt.shape[0])):
                c = float(gt[i, 2]) if gt.shape[1] >= 3 else 1.0
                if c < conf_thr:
                    continue
                gx, gy = float(gt[i, 0]), float(gt[i, 1])
                px, py = float(pr[i, 0]), float(pr[i, 1])
                cv2.circle(im, (int(round(gx)), int(round(gy))), 4, (0, 255, 0), 2)
                cv2.circle(im, (int(round(px)), int(round(py))), 4, (0, 0, 255), 2)
                cv2.line(im, (int(round(gx)), int(round(gy))), (int(round(px)), int(round(py))), (0, 255, 255), 1)
            return im

        def _project_points(K: np.ndarray, X: np.ndarray) -> np.ndarray:
            z = X[:, 2:3]
            xy = X[:, :2] / z
            uv = (K[:2, :2] @ xy.T).T + K[:2, 2][None, :]
            return uv

        def _render_mesh_overlay(
            *,
            verts_cam: torch.Tensor,
            faces_np: np.ndarray,
            K_np: np.ndarray,
            image_bgr: np.ndarray,
            render_device: str,
        ) -> np.ndarray:
            if Renderer is not None:
                # pytorch3d-based renderer (preferred)
                r = Renderer(
                    int(image_bgr.shape[1]),
                    int(image_bgr.shape[0]),
                    device=render_device,
                    faces=faces_np,
                    K=torch.tensor(K_np),
                    bin_size=0,
                )
                return r.render_mesh(
                    verts_cam.detach().cpu().to(render_device),
                    background=image_bgr,
                    colors=[0.8, 0.8, 0.8],
                    VI=50,
                    alpha=0.80,
                )

            # Fallback: OpenCV wireframe overlay (fast, dependency-light).
            im = image_bgr.copy()
            V = verts_cam.detach().cpu().numpy().astype(np.float32)
            z = V[:, 2]
            valid = np.isfinite(V).all(axis=1) & (z > 1e-6)
            if int(valid.sum()) <= 0:
                return im
            uv = _project_points(K_np, V)
            uv = uv.astype(np.float32)
            h, w = im.shape[0], im.shape[1]

            # Subsample faces to keep it lightweight
            step = 25
            faces_sub = faces_np[::step] if faces_np.ndim == 2 else np.asarray(faces_np).reshape(-1, 3)[::step]
            overlay = im.copy()
            col = (200, 200, 200)  # BGR gray-ish
            for f in faces_sub:
                i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
                if not (valid[i0] and valid[i1] and valid[i2]):
                    continue
                pts = np.array([uv[i0], uv[i1], uv[i2]], dtype=np.float32)
                if not np.isfinite(pts).all():
                    continue
                # clip / skip if completely offscreen
                if (pts[:, 0].max() < 0) or (pts[:, 0].min() > w) or (pts[:, 1].max() < 0) or (pts[:, 1].min() > h):
                    continue
                pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
                try:
                    cv2.polylines(overlay, [pts_i], isClosed=True, color=col, thickness=1, lineType=cv2.LINE_AA)
                except Exception:
                    continue
            alpha = 0.55
            try:
                im = cv2.addWeighted(overlay, alpha, im, 1.0 - alpha, 0.0)
            except Exception:
                im = overlay
            return im

        ds = Path(str(args.quick_test_dataset_dir))
        view = int(args.quick_test_view)
        t = int(args.quick_test_frame)

        pred_path = ds / "smpl_results" / f"{view}.pt"
        vit_path = ds / "preprocess" / str(view) / "vitpose.pt"
        bbx_path = ds / "preprocess" / str(view) / "bbx.pt"
        rgb_dir = ds / str(view) / "rgb"
        img_path = rgb_dir / f"image_{t:05d}.jpg"
        if not img_path.exists():
            img_path = rgb_dir / f"image_{t:05d}.png"

        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        if not vit_path.exists():
            raise FileNotFoundError(vit_path)
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        pred = torch.load(pred_path, map_location="cpu")
        vit = torch.load(vit_path, map_location="cpu")  # (T,17,3)
        bbx_xys = None
        try:
            if bbx_path.exists():
                bbx_obj = torch.load(bbx_path, map_location="cpu")
                if isinstance(bbx_obj, dict) and ("bbx_xys" in bbx_obj) and torch.is_tensor(bbx_obj["bbx_xys"]):
                    bbx_xys = bbx_obj["bbx_xys"].detach().cpu().float()
        except Exception:
            bbx_xys = None
        params = pred["smpl_params_incam"]
        K_full = pred["K_fullimg"]
        K_3x3 = K_full[t] if (torch.is_tensor(K_full) and K_full.ndim == 3) else K_full
        K_np = (
            (K_3x3.detach().cpu().numpy() if torch.is_tensor(K_3x3) else np.asarray(K_3x3, dtype=np.float32))
            .reshape(3, 3)
            .astype(np.float32)
        )

        im0 = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im0 is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        H0, W0 = int(im0.shape[0]), int(im0.shape[1])

        # Init params from GVHMR outputs
        go0 = params["global_orient"][t : t + 1].detach().cpu().float()
        bp0 = params["body_pose"][t : t + 1].detach().cpu().float()
        if bp0.ndim == 2 and int(bp0.shape[1]) == 63:
            # SMPLX-style 21 joints -> pad to SMPL 23 joints (69)
            bp0 = torch.cat([bp0, torch.zeros((1, 6), dtype=bp0.dtype)], dim=1)
        bet0 = params["betas"][t : t + 1].detach().cpu().float()
        cam_t0 = params["transl"][t : t + 1].detach().cpu().float()

        # Keypoints: COCO17 -> OP25 -> pad to 44 (ScoreHMR)
        kp_coco = vit[t : t + 1].detach().cpu().float()  # (1,17,3)
        kp44 = _ensure_scorehmr_joints_tj3(kp_coco)  # (1,44,3) after mapping/pad inside
        gt_xyc = kp44[0].detach().cpu().numpy().astype(np.float32)

        dev = torch.device(str(args.device))
        cfg = ScoreHMRRefineCfg(
            num_samples=int(args.num_samples),
            optim_iters=int(args.optim_iters),
            early_stopping=bool(args.early_stopping),
        )
        if bool(getattr(args, "use_pare_cond", False)):
            cfg.use_pare_cond = True
            # if user didn't pass --log_pare explicitly, still show minimal evidence by default
            cfg.log_pare = True
        if bool(getattr(args, "log_pare", False)):
            cfg.log_pare = True

        # Run refinement (T=1)
        t0 = time.time()
        out = refine_smpl_to_2d_with_scorehmr(
            global_orient_aa=go0.to(dev),
            body_pose_aa=bp0.to(dev),
            betas=bet0.to(dev),
            init_cam_t=cam_t0.to(dev),
            keypoints_2d=kp44.to(dev),  # already (1,44,3)
            image_size_hw=(H0, W0),
            K_3x3=torch.tensor(K_np, dtype=torch.float32, device=dev),
            cond_feats=None,
            image_paths=[str(img_path)],
            bbx_xys=(
                bbx_xys[t : t + 1] if (bbx_xys is not None and torch.is_tensor(bbx_xys) and bbx_xys.ndim == 2) else None
            ),
            device=dev,
            cfg=cfg,
        )
        dt = time.time() - t0

        go1 = out["global_orient_aa_refined"].view(1, 3).detach().cpu().float()
        bp1 = out["body_pose_aa_refined"].view(1, 69).detach().cpu().float()
        bet1 = out["betas_refined"].view(1, 10).detach().cpu().float()
        cam_t1 = out["cam_t_refined"].view(1, 3).detach().cpu().float()

        # Use ScoreHMR SMPL model to get vertices/joints (44 joints) for projection visualization
        model = _get_scorehmr_model(device=dev, cfg=cfg)
        smpl = model.smpl
        smpl = smpl.to(dev)
        smpl.eval()

        def _verts_joints_cam(go: torch.Tensor, bp: torch.Tensor, bet: torch.Tensor, cam_t: torch.Tensor):
            # Be robust across smplx versions: explicitly convert axis-angle to rotation matrices
            # and feed pose2rot=False path.
            from score_hmr.utils.geometry import aa_to_rotmat  # type: ignore

            go = go.view(-1, 3)
            bp = bp.view(-1, 69)
            bet = bet.view(-1, 10)
            B = int(go.shape[0])
            Rg = aa_to_rotmat(go).view(B, 3, 3)
            Rb = aa_to_rotmat(bp.view(B * 23, 3)).view(B, 23, 3, 3)
            out0 = smpl(global_orient=Rg, body_pose=Rb, betas=bet, pose2rot=False)
            v = out0.vertices[0]  # (V,3)
            j = out0.joints[0]  # (44,3)
            tvec = cam_t.view(1, 3)
            return (v + tvec), (j + tvec)

        v0_cam, j0_cam = _verts_joints_cam(go0.to(dev), bp0.to(dev), bet0.to(dev), cam_t0.to(dev))
        v1_cam, j1_cam = _verts_joints_cam(go1.to(dev), bp1.to(dev), bet1.to(dev), cam_t1.to(dev))

        proj0 = _project_points(K_np, j0_cam.detach().cpu().numpy().astype(np.float32))
        proj1 = _project_points(K_np, j1_cam.detach().cpu().numpy().astype(np.float32))

        # Render device
        rd = str(args.quick_test_render_device).strip().lower()
        if rd == "auto":
            rd = "cuda" if torch.cuda.is_available() else "cpu"
        if rd == "cuda" and (not torch.cuda.is_available()):
            rd = "cpu"

        faces = smpl.faces
        faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
        mesh_before = _render_mesh_overlay(
            verts_cam=v0_cam, faces_np=faces_np, K_np=K_np, image_bgr=im0, render_device=rd
        )
        mesh_after = _render_mesh_overlay(
            verts_cam=v1_cam, faces_np=faces_np, K_np=K_np, image_bgr=im0, render_device=rd
        )
        gt_vs_proj_before = _draw_gt_vs_proj(im0, gt_xyc=gt_xyc, proj_xy=proj0, conf_thr=0.05)
        gt_vs_proj_after = _draw_gt_vs_proj(im0, gt_xyc=gt_xyc, proj_xy=proj1, conf_thr=0.05)

        out_dir = (
            Path(str(args.quick_test_out_dir))
            if str(args.quick_test_out_dir).strip()
            else (repo_root / "bss-smplify" / "outputs" / "scorehmr_wrapper_test" / f"{ds.name}_view{view}_t{t:05d}")
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Dump a renderer payload for debugging across environments.
        payload_path = out_dir / "gvhmr_render_payload.pt"
        # NOTE: keep payload pickle-robust across conda envs:
        # do NOT store numpy arrays (can trigger `numpy._core` unpickle issues).
        payload = {
            "image_path": str(img_path),
            "K_3x3": K_np.astype(np.float32).tolist(),
            "faces": faces_np.astype(np.int64).tolist(),
            "verts_before_cam": v0_cam.detach().cpu().float(),
            "verts_after_cam": v1_cam.detach().cpu().float(),
            "gt_xyc": gt_xyc.astype(np.float32).tolist(),
            "proj_before": proj0.astype(np.float32).tolist(),
            "proj_after": proj1.astype(np.float32).tolist(),
        }
        torch.save(payload, payload_path)
        # test
        _write_jpg_under_kb(out_dir / "mesh_before.jpg", mesh_before, max_kb=int(args.quick_test_max_kb))
        _write_jpg_under_kb(out_dir / "mesh_after.jpg", mesh_after, max_kb=int(args.quick_test_max_kb))
        _write_jpg_under_kb(out_dir / "gt_vs_proj_before.jpg", gt_vs_proj_before, max_kb=int(args.quick_test_max_kb))
        _write_jpg_under_kb(out_dir / "gt_vs_proj_after.jpg", gt_vs_proj_after, max_kb=int(args.quick_test_max_kb))

        (out_dir / "report.json").write_text(
            json.dumps(
                {
                    "dataset_dir": str(ds),
                    "view": int(view),
                    "frame": int(t),
                    "pred_path": str(pred_path),
                    "vit_path": str(vit_path),
                    "img_path": str(img_path),
                    "out_dir": str(out_dir),
                    "device": str(dev),
                    "render_device": str(rd),
                    "renderer_backend": str(_renderer_backend),
                    "gvhmr_render_payload": str(payload_path),
                    "cfg": {
                        "num_samples": int(cfg.num_samples),
                        "optim_iters": int(cfg.optim_iters),
                        "early_stopping": bool(cfg.early_stopping),
                        "ckpt_name": str(cfg.ckpt_name),
                        "milestone": int(cfg.milestone),
                        "use_default_ckpt": bool(cfg.use_default_ckpt),
                    },
                    "timing_sec": float(dt),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("[Spline-Diff quick_test] out_dir:", out_dir)
        print("[Spline-Diff quick_test] Renderer payload saved for debugging:")
        print(
            "  conda run -n splinehmr python tools/render_scorehmr_payload.py "
            f"--payload {str(payload_path)} --out_dir {str(out_dir)}"
        )
        return

    T = int(args.T)
    H = int(args.H)
    W = int(args.W)
    dev = torch.device(str(args.device))

    # Random init SMPL (axis-angle)
    global_orient = (0.05 * torch.randn(T, 3)).float()
    body_pose = (0.05 * torch.randn(T, 69)).float()
    betas = torch.zeros(10).float()
    init_cam_t = torch.zeros(T, 3).float()
    init_cam_t[:, 2] = 2.5

    # Random keypoints: default COCO17 (T,17,3)
    kp = torch.zeros(T, 17, 3).float()
    kp[:, :, 0] = torch.rand(T, 17) * float(W)
    kp[:, :, 1] = torch.rand(T, 17) * float(H)
    kp[:, :, 2] = 1.0

    cfg = ScoreHMRRefineCfg(
        num_samples=int(args.num_samples),
        optim_iters=int(args.optim_iters),
        early_stopping=bool(args.early_stopping),
    )

    t0 = time.time()
    out = refine_smpl_to_2d_with_scorehmr(
        global_orient_aa=global_orient.to(dev),
        body_pose_aa=body_pose.to(dev),
        betas=betas.to(dev),
        init_cam_t=init_cam_t.to(dev),
        keypoints_2d=kp.to(dev),
        image_size_hw=(H, W),
        K_3x3=None,
        cond_feats=None,
        device=dev,
        cfg=cfg,
    )
    dt = time.time() - t0

    def _sh(x: Any) -> str:
        if torch.is_tensor(x):
            return f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
        return type(x).__name__

    print(f"[Spline-Diff smoke] done in {dt:.3f}s T={T} device={dev}")
    for k in [
        "global_orient_aa_refined",
        "body_pose_aa_refined",
        "betas_refined",
        "cam_t_refined",
        "pred_pose_rotmat",
    ]:
        print(f"[Spline-Diff smoke] out[{k}] {_sh(out.get(k))}")


if __name__ == "__main__":
    main()
