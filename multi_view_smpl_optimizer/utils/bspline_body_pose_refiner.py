"""
用 B-spline + LBFGS 拟合一个关于时间 t 的 body_pose 修正函数，使 SMPL/SMPL-X 投影 COCO17 更贴近输入 COCO17 序列。

核心思想（按用户需求）：
- 只优化 body_pose 的 69 维（SMPL: 23 joints * 3 axis-angle = 69）
- 修正项 δ(t) 由 M 个三次 B-spline 基函数加权求和：δ_k(t) = Σ_j c_{k,j} B_j(t)
- 软约束：c_{k,j} = 0.5 * tanh(u_{k,j})，优化变量是 u
- 损失：归一化平均重投影误差（像素域，除以 bbox size s）
        - 支持对 2D 置信度做加权（类似 SMPLify-X：置信度越高，误差项权重越大）
        - 同时保留 conf_thr 作为低置信度剔除阈值
      + 先验项（限制修正幅度）

依赖：需要本仓库的 SMPL/SMPLX 相关依赖可用（make_smplx / project_p2d）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import math
import time
import torch
import torch.nn.functional as F

_POSE_LIMITER_CACHE: dict[str, torch.nn.Module] = {}


class _SMPLCoco17Model:
    """
    纯 SMPL + COCO17 J regressor，与 supermotion_coco17 相同调用约定：
    __call__(body_pose, betas, global_orient, transl) -> (T, 17, 3)。
    body_pose 63 维时自动 pad 到 69。
    """

    def __init__(self, device: torch.device | str):
        from .smplx_utils import make_smplx

        dev = torch.device(device) if isinstance(device, str) else device
        J_path = Path(__file__).resolve().parent / "body_model" / "smpl_coco17_J_regressor.pt"
        if not J_path.exists():
            raise FileNotFoundError(f"use_smpl=True 需要 COCO17 regressor: {J_path}")
        self.smpl = make_smplx("smpl").to(dev).eval()
        J = torch.load(str(J_path), map_location="cpu", weights_only=False)
        if getattr(J, "is_sparse", False):
            J = J.to_dense()
        self.J_reg = J.float().to(dev)
        self._dev = dev

    def __call__(
        self,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
    ) -> torch.Tensor:
        T = int(body_pose.shape[0])
        bp = body_pose.to(self._dev).float()
        if bp.ndim != 2:
            bp = bp.view(T, -1)
        D = int(bp.shape[1])
        if D < 69:
            bp = torch.cat(
                [bp, torch.zeros((T, 69 - D), device=bp.device, dtype=bp.dtype)],
                dim=1,
            )
        elif D > 69:
            bp = bp[:, :69]
        betas = betas.to(self._dev).float()
        global_orient = global_orient.to(self._dev).float()
        transl = transl.to(self._dev).float()
        if betas.ndim == 1:
            betas = betas[None, :].expand(T, -1)
        out = self.smpl(body_pose=bp, betas=betas, global_orient=global_orient, transl=transl)
        verts = out.get("vertices", out.get("v")) if isinstance(out, dict) else getattr(out, "vertices", out)
        if not torch.is_tensor(verts):
            verts = torch.as_tensor(verts, device=self._dev, dtype=torch.float32)
        joints3d = torch.einsum("jv,fvc->fjc", self.J_reg, verts)
        return joints3d


def _make_coco17_model(cfg: BsplineRefineConfig, device: torch.device | str):
    """2D 模式下使用的 COCO17 前向模型；use_smpl 时用纯 SMPL+J_reg，否则用 supermotion_coco17。"""
    if bool(getattr(cfg, "use_smpl", False)):
        return _SMPLCoco17Model(device)
    from .smplx_utils import make_smplx

    return make_smplx("supermotion_coco17").to(device).eval()


def _get_pose_limiter(device: torch.device) -> torch.nn.Module:
    """
    Lazily load GVHMR pose limiter. This is differentiable (piecewise) but may introduce
    zero-grad regions due to clamp. Use only if you really want the optimization objective
    to match the visualization-time pose limit.
    """
    key = str(device)
    if key in _POSE_LIMITER_CACHE:
        return _POSE_LIMITER_CACHE[key]
    try:
        from smpl_limit_utils.smpl_knee_limit_util import SMPLPoseLimiterViaSKEL
    except Exception as e:
        raise ImportError(
            "Failed to import GVHMR pose limiter. Make sure GVHMR is on PYTHONPATH "
            "(e.g., add <repo>/GVHMR to sys.path and chdir to proj root like demo.py)."
        ) from e
    limiter = SMPLPoseLimiterViaSKEL(device=device)
    limiter.eval()
    _POSE_LIMITER_CACHE[key] = limiter
    return limiter


def _maybe_pose_limit(x: torch.Tensor, *, enabled: bool, device: torch.device) -> torch.Tensor:
    if not enabled:
        return x
    if int(x.shape[-1]) not in (63, 69):
        raise ValueError(f"pose_limit_in_loss expects body_pose last dim in (63,69), got {tuple(x.shape)}")
    limiter = _get_pose_limiter(device)
    # IMPORTANT:
    # The limiter implementation uses in-place updates (e.g. flat[:, j] = ...),
    # which breaks autograd if we try to backprop through it.
    # We therefore run limiter in no_grad on a detached tensor, and use a
    # straight-through estimator (STE) so gradients flow as identity:
    #   y = limiter(x_detached)  (forward)
    #   dy/dx ≈ I               (backward)
    x0 = x.detach()
    with torch.no_grad():
        y0 = limiter(x0)
    return y0 + (x - x0)


def _aa_to_rot6d_flat(x_aa: torch.Tensor) -> torch.Tensor:
    """
    Axis-angle -> rot6d (first two columns of R), flattened.
    Input:  (T, D) where D % 3 == 0
    Output: (T, (D/3)*6)

    NOTE: The 6D ordering matches `rot6d_to_rotmat` in `utils/aa_to_6d/geometry.py`,
    i.e. first 3 numbers are the first column of R, next 3 numbers are the second column.
    """
    if x_aa.ndim != 2:
        raise ValueError(f"_aa_to_rot6d_flat expects (T,D), got {tuple(x_aa.shape)}")
    T, D = int(x_aa.shape[0]), int(x_aa.shape[1])
    if D % 3 != 0:
        raise ValueError(f"_aa_to_rot6d_flat expects D%3==0, got D={D}")
    J = D // 3
    from .aa_to_6d.geometry import aa_to_rotmat

    R = aa_to_rotmat(x_aa.reshape(-1, 3)).reshape(T, J, 3, 3)
    # (T,J,3,2) -> (T,J,2,3) to get [col1(3), col2(3)]
    rot6d = R[..., :2].permute(0, 1, 3, 2).reshape(T, J * 6)
    return rot6d


def _rot6d_flat_to_aa(x_6d: torch.Tensor) -> torch.Tensor:
    """
    rot6d (flattened) -> axis-angle.
    Input:  (T, D6) where D6 % 6 == 0
    Output: (T, (D6/6)*3)
    """
    if x_6d.ndim != 2:
        raise ValueError(f"_rot6d_flat_to_aa expects (T,D6), got {tuple(x_6d.shape)}")
    T, D6 = int(x_6d.shape[0]), int(x_6d.shape[1])
    if D6 % 6 != 0:
        raise ValueError(f"_rot6d_flat_to_aa expects D6%6==0, got D6={D6}")
    J = D6 // 6
    from .aa_to_6d.geometry import rot6d_to_rotmat
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle

    R = rot6d_to_rotmat(x_6d.reshape(-1, 6)).reshape(T * J, 3, 3)
    aa = quaternion_to_axis_angle(matrix_to_quaternion(R)).reshape(T, J * 3)
    return aa


def _u_to_c(u: torch.Tensor, *, amp: float, use_tanh: bool) -> torch.Tensor:
    a = float(amp)
    if bool(use_tanh):
        return a * torch.tanh(u)
    return a * u


def _c_to_u(c: torch.Tensor, *, amp: float, use_tanh: bool, eps: float = 1e-6) -> torch.Tensor:
    a = float(amp)
    if bool(use_tanh):
        cc = c.clamp(-a, a)
        return torch.atanh((cc / a).clamp(-1.0 + float(eps), 1.0 - float(eps)))
    return c / a


def _temporal_vel_acc_loss(x: torch.Tensor, *, vel_w: float, acc_w: float) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected (T,D), got {tuple(x.shape)}")
    T = int(x.shape[0])
    loss = torch.zeros((), device=x.device, dtype=x.dtype)
    if float(vel_w) > 0.0 and T >= 2:
        dx = x[1:] - x[:-1]
        loss = loss + float(vel_w) * (dx * dx).mean()
    if float(acc_w) > 0.0 and T >= 3:
        ddx = x[2:] - 2.0 * x[1:-1] + x[:-2]
        loss = loss + float(acc_w) * (ddx * ddx).mean()
    return loss


@dataclass
class BsplineRefineConfig:
    degree: int = 3  # cubic
    m_per_t: int = 10  # M ~= ceil(T / m_per_t)
    # amplitude constraints (soft): c = amp * tanh(u)
    amp: float = 0.5  # backward-compat alias for amp_body_pose
    amp_body_pose: float | None = None
    amp_global_orient: float = 0.5
    amp_transl: float = 0.5
    conf_thr: float = 0.3  # for coco17 with conf
    # confidence weighting (SMPLify-X style): weights *= conf^power (after thresholding)
    use_conf_weight: bool = True
    conf_power: float = 1.0
    # prior weights to prevent over-adjust
    prior_w: float = 0.05  # backward-compat alias for prior_w_body_pose
    prior_w_body_pose: float | None = None
    prior_w_global_orient: float = 15
    prior_w_transl: float = 15
    mv_consistency_w: float = 10.0  # multi-view body_pose consistency weight (only used when multi_view_joint=True)
    ankle_ground_align_w: float = 10.0  # optional ankle-ground alignment loss weight (only when ankle_static_prob is provided)
    # static-motion loss（默认关闭：该模块易出 bug，需显式设 static_motion_w>0 且传入 static_conf_logits 才启用）
    static_motion_w: float = 0.0
    static_joint_w: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    static_softmax_tau: float = 1.0  # temperature for softmax(logits / tau). tau<1 sharper; tau>1 smoother
    static_use_smpl24: bool = True  # if body_pose is 63D, use SMPL-24 joints (includes feet). Otherwise fallback to COCO17 subset.
    smooth_w: float = 0.0  # optional: 2nd-diff smoothness on control points
    # optionally learn knot (time) positions (shared across all params/views)
    learn_knots: bool = False
    knot_min_gap: float = 1e-3  # minimum normalized gap in (0,1) to avoid degenerate knots
    knot_pos_w: float = 3.0  # prior: keep internal knots close to uniform positions
    knot_gap_w: float = 0.3  # prior: keep knot gaps close to uniform
    knot_smooth_w: float = 0.0  # prior: smoothness on internal knot positions
    max_iter: int = 60
    lr: float = 1.0
    line_search_fn: str | None = "strong_wolfe"
    verbose: bool = True
    grad_clip_norm: float = 0.0
    # 2D 重投影用纯 SMPL（否则用 supermotion_coco17，即 SMPL-X 系）。默认 False，与原有行为一致。
    use_smpl: bool = False
    # PoseLimit（关节角度约束）：默认关闭；需在 optim_cfg 或 cfg 中显式开启。
    pose_limit: bool = False
    pose_limit_in_loss: bool = False
    # 是否在 rot6d 空间优化（默认关闭，保持历史行为：axis-angle 直接相加）。
    # 开启后：AA -> rot6d，在 rot6d 上加 bspline delta，再 rot6d -> AA 送入 SMPL。
    optimize_pose_in_rot6d: bool = False
    optimize_root_as_constant_delta: bool = False
    use_tanh: bool = True
    temporal_vel_w: float = 0.0
    temporal_acc_w: float = 0.0

    # ===== bspline_plus 两阶段（仅当 refine_body_pose_bspline_lbfgs_plus 时使用） =====
    refiner_plus_stage1_m_per_t: int | str = "fps"  # Stage1 段长，同 m_per_t 规则："fps" / "fps_div_N" / int
    refiner_plus_tau: float = 0.1  # 区间平均误差 > tau 才参与 Stage2 加密
    refiner_plus_top_frac: float = 0.5  # 高误差区间中取 Top 比例
    refiner_plus_insert_knots: int = 2  # 每个选中区间内插入的节点数

    # ===== Optional 3D keypoint loss (default OFF; keep legacy behavior) =====
    # If k3d_gt is provided (shape (T,24,3) by default), the optimizer will ignore 2D reprojection loss
    # and use ONLY this 3D loss (+ existing regularizers like priors/smoothness).
    #
    # IMPORTANT:
    # - This file does NOT implement "SMPL params -> 3D keypoints" conversion for the 3D loss.
    # - You must provide `k3d_pred_fn`, a callable that returns predicted 3D keypoints in camera coordinates.
    #
    # Default joint set is "smpl24" (24 joints) with pelvis alignment using hips indices (1,2),
    # consistent with GVHMR `compute_camcoord_metrics(pelvis_idxs=[1,2])`.
    k3d_gt: torch.Tensor | None = None  # (T,J,3) in camera coords; default expects J=24 (SMPL24)
    k3d_kind: str = "smpl24"  # label only; caller decides how to interpret in k3d_pred_fn
    k3d_pred_fn: Any | None = None  # callable(body_pose, betas, global_orient, transl, **kw)->(T,J,3)
    k3d_pelvis_idxs: tuple[int, int] = (1, 2)  # align by mean of these joints per-frame
    k3d_align_by_pelvis: bool = True
    k3d_foot_prior_k: float = 1.0
    k3d_max_frame_mm: float = 200.0  # drop frames where per-frame mpjpe(mm) > this value
    k3d_mm_scale: float = 1000.0  # convert meters->mm (matches GVHMR eval conventions)
    k3d_reproj_w: float = 0.0
    k3d_reproj_cam: dict[str, Any] | None = None

    def resolve(self) -> "BsplineRefineConfig":
        """
        Fill None fields from backward-compat aliases.
        """
        if self.amp_body_pose is None:
            self.amp_body_pose = float(self.amp)
        if self.prior_w_body_pose is None:
            self.prior_w_body_pose = float(self.prior_w)
        return self


# 关节先验增强表（写死在这里，默认全为 1.0），按 (关节名, 权重) 形式。
# 用途：在 body_pose 的先验损失里，每个关节的三个维度 (axis-angle xyz) 的平方惩罚项乘以对应权重。
# - 权重 > 1：更强约束，期望这个关节改动更小
# - 权重 < 1：更弱约束，允许这个关节改动更大
#
# 注意：当 body_pose 是 63D（21 joints）时，会自动截断使用前 21 个关节权重。
BODY_POSE_PRIOR_MULT_23: tuple[tuple[str, float], ...] = (
    ("Pelvis", 1.0),
    ("L_Hip", 1.0),
    ("R_Hip", 1.0),
    ("Spine1", 1.0),
    ("L_Knee", 1.0),
    ("R_Knee", 1.0),
    ("Spine2", 1.0),
    ("L_Ankle", 10.0),
    ("R_Ankle", 10.0),
    ("Spine3", 1.0),
    ("L_Foot", 10.0),
    ("R_Foot", 10.0),
    ("Neck", 10),
    ("L_Collar", 1.0),
    ("R_Collar", 1.0),
    ("Head", 10),
    ("L_Shoulder", 1.0),
    ("R_Shoulder", 1.0),
    ("L_Elbow", 1.0),
    ("R_Elbow", 1.0),
    ("L_Wrist", 1.0),
    ("R_Wrist", 1.0),
    ("L_Hand", 1.0),
)


def _body_pose_prior_weights_per_dim(D: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build per-dimension weights for body_pose prior.
    D must be divisible by 3. For 69D -> 23 joints; for 63D -> 21 joints.
    Return: (1, D) tensor for broadcasting.
    """
    if D % 3 != 0:
        raise ValueError(f"body_pose last dim must be multiple of 3, got D={D}")
    J = D // 3
    if J > 23:
        raise ValueError(f"Unexpected joint count J={J} from D={D} (max supported 23)")
    wj = torch.tensor([w for _, w in BODY_POSE_PRIOR_MULT_23[:J]], device=device, dtype=dtype)  # (J,)
    wd = wj.repeat_interleave(3)  # (D,)
    return wd.view(1, D)


def _body_pose_prior_weights_per_dim_rot6d(D6: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build per-dimension weights for body_pose prior when optimizing in rot6d space.
    D6 must be divisible by 6. For 69D-AA -> 23 joints -> 138D-6D; for 63D-AA -> 21 joints -> 126D-6D.
    Return: (1, D6) tensor for broadcasting.
    """
    if D6 % 6 != 0:
        raise ValueError(f"D6 must be divisible by 6, got D6={D6}")
    J = D6 // 6
    if J > 23:
        raise ValueError(f"Unexpected joint count J={J} from D6={D6} (max supported 23)")
    wj = torch.tensor([w for _, w in BODY_POSE_PRIOR_MULT_23[:J]], device=device, dtype=dtype).view(J, 1)  # (J,1)
    wd = wj.repeat(1, 6).reshape(1, D6)  # (1,D6)
    return wd


def _maybe_scale_foot_prior_weights(w: torch.Tensor, *, k: float, use_rot6d: bool) -> torch.Tensor:
    if (not torch.is_tensor(w)) or w.ndim != 2 or int(w.shape[0]) != 1:
        raise ValueError(f"Expected w to be (1,D), got {tuple(getattr(w, 'shape', None))}")
    kk = float(k)
    if not (kk > 0.0) or kk == 1.0:
        return w
    stride = 6 if bool(use_rot6d) else 3
    D = int(w.shape[1])
    if D % stride != 0:
        return w
    J = D // stride
    left_foot_j = 10
    right_foot_j = 11
    if left_foot_j < J:
        s = left_foot_j * stride
        w[:, s : s + stride] = w[:, s : s + stride] * kk
    if right_foot_j < J:
        s = right_foot_j * stride
        w[:, s : s + stride] = w[:, s : s + stride] * kk
    return w


# GVHMR static_conf_dim=6 corresponds to these SMPL24 joint indices:
#   [L_Ankle, L_Foot, R_Ankle, R_Foot, L_Wrist, R_Wrist]
_STATIC_JOINT_IDS_SMPL24: tuple[int, int, int, int, int, int] = (7, 10, 8, 11, 20, 21)
# COCO17 indices available in our reprojection model (SmplxLiteCoco17):
_COCO17_L_WRIST = 9
_COCO17_R_WRIST = 10
_COCO17_L_ANKLE = 15
_COCO17_R_ANKLE = 16


def _align_static_logits(static_logits: torch.Tensor, *, T: int) -> torch.Tensor:
    """
    Align static logits to length (T-1, 6) to match per-frame velocities.
    Accepts:
      - (T,6): uses [:T-1]
      - (T-1,6): uses as-is
      - (1,T,6): squeeze and use [:T-1]
    """
    x = static_logits
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2 or int(x.shape[1]) != 6:
        raise ValueError(f"static_conf_logits must be (T,6) or (T-1,6), got {tuple(static_logits.shape)}")
    if int(x.shape[0]) == int(T):
        x = x[: max(int(T) - 1, 0)]
    elif int(x.shape[0]) == int(T - 1):
        pass
    else:
        # best-effort prefix align
        x = x[: max(min(int(x.shape[0]), int(T) - 1), 0)]
    return x


def _static_motion_loss(
    *,
    joints17: torch.Tensor,  # (T,17,3) in camera coords
    static_logits: torch.Tensor | None,  # (T,6) or (T-1,6)
    cfg: BsplineRefineConfig,
    smpl24_model,  # optional callable returning (T,24,3) or None
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    transl: torch.Tensor,
) -> torch.Tensor:
    """
    Static-motion loss (default enabled when static_logits is provided):
      mean_t sum_j softmax(logits_t)[j] * w_j * ||joint_j(t+1)-joint_j(t)|| / scale

    - If cfg.static_use_smpl24 and body_pose is 63D: uses SMPL24 joints to include feet.
    - Otherwise uses COCO17 subset (ankle+wrists) and ignores foot terms.
    """
    if static_logits is None:
        return torch.zeros((), device=joints17.device, dtype=torch.float32)
    if float(cfg.static_motion_w) <= 0:
        return torch.zeros((), device=joints17.device, dtype=torch.float32)

    T = int(joints17.shape[0])
    if T <= 1:
        return torch.zeros((), device=joints17.device, dtype=torch.float32)

    x = _align_static_logits(static_logits.to(joints17.device).float(), T=T)  # (T-1,6)
    if int(x.shape[0]) <= 0:
        return torch.zeros((), device=joints17.device, dtype=torch.float32)

    tau = float(cfg.static_softmax_tau)
    if not (tau > 0):
        raise ValueError(f"static_softmax_tau must be > 0, got {cfg.static_softmax_tau}")
    prob6 = torch.softmax(x / tau, dim=-1)  # (T-1,6)
    w6 = torch.tensor(cfg.static_joint_w, device=joints17.device, dtype=torch.float32).view(1, 6)

    use_smpl24 = bool(cfg.static_use_smpl24) and (smpl24_model is not None) and (int(body_pose.shape[1]) == 63)
    if use_smpl24:
        j24 = smpl24_model(body_pose=body_pose, betas=betas, global_orient=global_orient, transl=transl)  # (T,24,3)
        j_sel = j24[:, list(_STATIC_JOINT_IDS_SMPL24)]  # (T,6,3)
        disp = j_sel[1:] - j_sel[:-1]  # (T-1,6,3)
        speed = torch.linalg.norm(disp, dim=-1)  # (T-1,6)
        # normalize by body scale (pelvis->neck distance)
        scale = torch.linalg.norm(j24[:, 12] - j24[:, 0], dim=-1).mean().clamp_min(1e-6)
        speed_n = speed / scale
        term = (prob6 * w6) * speed_n
        return term.sum(dim=-1).mean()

    # fallback: use COCO17 subset (ankles+wrists). COCO17 does not have feet,
    # so we approximate foot speed with the corresponding ankle speed.
    j_sel = joints17[:, [_COCO17_L_ANKLE, _COCO17_R_ANKLE, _COCO17_L_WRIST, _COCO17_R_WRIST]]  # (T,4,3)
    disp = j_sel[1:] - j_sel[:-1]  # (T-1,4,3)
    speed4 = torch.linalg.norm(disp, dim=-1)  # (T-1,4) => [L_ankle, R_ankle, L_wrist, R_wrist]
    # normalize by shoulder width
    scale = torch.linalg.norm(joints17[:, 5] - joints17[:, 6], dim=-1).mean().clamp_min(1e-6)
    speed4n = speed4 / scale

    # expand to 6 dims: [L_ankle, L_foot≈L_ankle, R_ankle, R_foot≈R_ankle, L_wrist, R_wrist]
    speed6n = torch.stack(
        [
            speed4n[:, 0],
            speed4n[:, 0],
            speed4n[:, 1],
            speed4n[:, 1],
            speed4n[:, 2],
            speed4n[:, 3],
        ],
        dim=-1,
    )  # (T-1,6)
    term = (prob6 * w6) * speed6n
    return term.sum(dim=-1).mean()


def _refine_bspline_two_view_lbfgs(
    *,
    body_pose_v0: torch.Tensor,
    betas_v0: torch.Tensor,
    global_orient_v0: torch.Tensor,
    transl_v0: torch.Tensor,
    K_fullimg_v0: torch.Tensor,
    bbx_xys_v0: torch.Tensor,
    coco17_v0: torch.Tensor,
    body_pose_v1: torch.Tensor,
    betas_v1: torch.Tensor,
    global_orient_v1: torch.Tensor,
    transl_v1: torch.Tensor,
    K_fullimg_v1: torch.Tensor,
    bbx_xys_v1: torch.Tensor,
    coco17_v1: torch.Tensor,
    cfg: BsplineRefineConfig,
    device: str | torch.device,
    optimize_body_pose: bool,
    optimize_global_orient: bool,
    optimize_transl: bool,
    pose_limit_in_loss: bool,
    # optional ankle static prob + ground normal (camera coords)
    ankle_static_prob_v0: torch.Tensor | None = None,  # (T,2) in [0,1]
    ground_normal_cam_v0: torch.Tensor | None = None,  # (3,) or (T,3)
    ankle_static_prob_v1: torch.Tensor | None = None,
    ground_normal_cam_v1: torch.Tensor | None = None,
    ankle_ground_align_w: float | None = None,
    # optional static conf logits (T,6) for static-motion loss
    static_conf_logits_v0: torch.Tensor | None = None,
    static_conf_logits_v1: torch.Tensor | None = None,
) -> dict:
    """
    Two-view joint optimization:
    - Optimize both views' deltas in ONE LBFGS run.
    - Add multi-view consistency loss on body_pose only:
        mv_consistency_w * mean( (body0_ref - body1_ref)^2 * joint_prior_w )
    - global_orient/transl are per-camera; no cross-view term (but can still be optimized per-view).
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device

    # Normalize shapes & choose common T
    bp0 = body_pose_v0.detach().cpu()
    bp1 = body_pose_v1.detach().cpu()
    if bp0.ndim != 2 or bp1.ndim != 2:
        raise ValueError(f"body_pose must be (T,D). got v0={tuple(bp0.shape)} v1={tuple(bp1.shape)}")
    D0 = int(bp0.shape[1])
    D1 = int(bp1.shape[1])
    if D0 != D1:
        raise ValueError(f"body_pose dim mismatch across views: D0={D0} D1={D1}")
    D = D0
    if D <= 0:
        raise ValueError(f"Invalid body_pose dim: {D}")

    T = min(
        int(bp0.shape[0]),
        int(bp1.shape[0]),
        int(global_orient_v0.shape[0]),
        int(global_orient_v1.shape[0]),
        int(transl_v0.shape[0]),
        int(transl_v1.shape[0]),
        int(coco17_v0.shape[0]),
        int(coco17_v1.shape[0]),
        int(bbx_xys_v0.shape[0]),
        int(bbx_xys_v1.shape[0]),
    )
    if torch.is_tensor(static_conf_logits_v0) and static_conf_logits_v0.ndim >= 2:
        T = min(int(T), int(static_conf_logits_v0.shape[-2]))
    if torch.is_tensor(static_conf_logits_v1) and static_conf_logits_v1.ndim >= 2:
        T = min(int(T), int(static_conf_logits_v1.shape[-2]))

    # choose number of basis functions
    M = int(math.ceil(T / float(cfg.m_per_t)))
    M = max(M, cfg.degree + 2)  # ensure m > degree
    t = torch.linspace(0.0, 1.0, T, dtype=torch.float32)
    B: torch.Tensor | None = None
    B_dev: torch.Tensor | None = None
    # learnable knot gaps (shared across all params/views). Only used if cfg.learn_knots=True.
    u_knot_gaps: torch.Tensor | None = None
    t_dev: torch.Tensor | None = None
    internal_knots_uni: torch.Tensor | None = None
    gaps_uni: torch.Tensor | None = None

    if not bool(cfg.learn_knots):
        B = _bspline_basis_matrix(t, m=M, degree=int(cfg.degree)).float()  # (T,M) on CPU
        B_dev = B.to(dev)
    else:
        # K = M-degree-1 internal knots => (K+1) gaps => length = M-degree
        t_dev = torch.linspace(0.0, 1.0, T, dtype=torch.float32, device=dev)
        u_knot_gaps = torch.zeros((int(M - int(cfg.degree)),), device=dev, dtype=torch.float32, requires_grad=True)
        internal_knots_uni = _uniform_internal_knots(int(M), int(cfg.degree), device=dev, dtype=torch.float32)
        if int(u_knot_gaps.numel()) > 0:
            gaps_uni = torch.full(
                (int(u_knot_gaps.numel()),),
                1.0 / float(int(u_knot_gaps.numel())),
                device=dev,
                dtype=torch.float32,
            )

    # Move tensors
    body0 = bp0[:T].to(dev).float()
    body1 = bp1[:T].to(dev).float()
    bet0 = (betas_v0[:T] if (torch.is_tensor(betas_v0) and betas_v0.ndim == 2) else betas_v0).to(dev)
    bet1 = (betas_v1[:T] if (torch.is_tensor(betas_v1) and betas_v1.ndim == 2) else betas_v1).to(dev)
    glob0 = global_orient_v0[:T].to(dev).float()
    glob1 = global_orient_v1[:T].to(dev).float()
    tr0 = transl_v0[:T].to(dev).float()
    tr1 = transl_v1[:T].to(dev).float()
    K0 = (K_fullimg_v0[:T] if (torch.is_tensor(K_fullimg_v0) and K_fullimg_v0.ndim == 3) else K_fullimg_v0).to(dev)
    K1 = (K_fullimg_v1[:T] if (torch.is_tensor(K_fullimg_v1) and K_fullimg_v1.ndim == 3) else K_fullimg_v1).to(dev)
    bb0 = bbx_xys_v0[:T].to(dev)
    bb1 = bbx_xys_v1[:T].to(dev)
    co0 = coco17_v0[:T].to(dev)
    co1 = coco17_v1[:T].to(dev)

    # SMPL model once (2D reproj: use_smpl 时用纯 SMPL+J_reg，否则 supermotion_coco17)
    from .smplx_utils import make_smplx

    smpl_coco17 = _make_coco17_model(cfg, dev)
    smpl_smpl24 = None
    if bool(cfg.static_use_smpl24) and int(D) == 63:
        try:
            smpl_smpl24 = make_smplx("supermotion_smpl24").to(dev).eval()
        except Exception:
            smpl_smpl24 = None

    use_rot6d = bool(cfg.optimize_pose_in_rot6d)
    D_body_opt = int((D // 3) * 6) if use_rot6d else int(D)
    D_glob_opt = 6 if use_rot6d else 3

    # body prior weights:
    # - opt-space weights: for delta regularization (matches d_bodies shape)
    # - aa-space weights: for multi-view consistency on axis-angle (matches bodies_m shape)
    body_prior_w_aa = _body_pose_prior_weights_per_dim(D, device=dev, dtype=torch.float32)  # (1,D)
    body_prior_w_opt = (
        _body_pose_prior_weights_per_dim_rot6d(D_body_opt, device=dev, dtype=torch.float32)
        if use_rot6d
        else body_prior_w_aa
    )  # (1, D_body_opt) or (1, D)
    # ankle alignment weight
    w_ank = float(cfg.ankle_ground_align_w if ankle_ground_align_w is None else ankle_ground_align_w)

    # Vars
    params_to_opt: list[torch.Tensor] = []
    u_body0 = u_body1 = None
    u_glob0 = u_glob1 = None
    u_tr0 = u_tr1 = None

    if bool(cfg.learn_knots):
        if u_knot_gaps is None:
            raise RuntimeError("cfg.learn_knots=True but u_knot_gaps is None")
        params_to_opt.append(u_knot_gaps)

    if optimize_body_pose:
        u_body0 = torch.zeros((D_body_opt, M), device=dev, dtype=torch.float32, requires_grad=True)
        u_body1 = torch.zeros((D_body_opt, M), device=dev, dtype=torch.float32, requires_grad=True)
        params_to_opt += [u_body0, u_body1]
    if optimize_global_orient:
        u_glob0 = torch.zeros((D_glob_opt, M), device=dev, dtype=torch.float32, requires_grad=True)
        u_glob1 = torch.zeros((D_glob_opt, M), device=dev, dtype=torch.float32, requires_grad=True)
        params_to_opt += [u_glob0, u_glob1]
    if optimize_transl:
        u_tr0 = torch.zeros((3, M), device=dev, dtype=torch.float32, requires_grad=True)
        u_tr1 = torch.zeros((3, M), device=dev, dtype=torch.float32, requires_grad=True)
        params_to_opt += [u_tr0, u_tr1]
    if not params_to_opt:
        raise ValueError("At least one optimize_* must be True.")

    # baseline stats
    with torch.no_grad():
        smpl2d0 = _smpl_body_pose_to_coco17_2d(
            body_pose=body0, betas=bet0, global_orient=glob0, transl=tr0, K_fullimg=K0, device=dev, smpl_model=smpl_coco17
        )
        smpl2d1 = _smpl_body_pose_to_coco17_2d(
            body_pose=body1, betas=bet1, global_orient=glob1, transl=tr1, K_fullimg=K1, device=dev, smpl_model=smpl_coco17
        )
        base0 = float(_mean_normed_reproj_err(smpl2d0, co0, bb0, conf_thr=float(cfg.conf_thr)).item())
        base1 = float(_mean_normed_reproj_err(smpl2d1, co1, bb1, conf_thr=float(cfg.conf_thr)).item())

    opt = torch.optim.LBFGS(params_to_opt, lr=float(cfg.lr), max_iter=int(cfg.max_iter), line_search_fn=cfg.line_search_fn)
    last = {"loss": None}

    def _get_B_and_knots():
        if not bool(cfg.learn_knots):
            if B_dev is None:
                raise RuntimeError("Expected precomputed B_dev when learn_knots=False")
            return B_dev, None, None, None
        if (u_knot_gaps is None) or (t_dev is None):
            raise RuntimeError("Expected u_knot_gaps/t_dev when learn_knots=True")
        U, internal, gaps = _knots_from_u_gaps(
            u_knot_gaps,
            m=int(M),
            degree=int(cfg.degree),
            min_gap=float(cfg.knot_min_gap),
        )
        Bk = _bspline_basis_matrix(t_dev, m=int(M), degree=int(cfg.degree), knots=U).float()
        return Bk, U, internal, gaps

    def _build_view(body_base, glob_base, tr_base, u_body, u_glob, u_tr, *, B_use: torch.Tensor):
        # body pose (AA or rot6d space)
        if u_body is not None:
            c_body = _u_to_c(u_body, amp=float(cfg.amp_body_pose), use_tanh=bool(cfg.use_tanh))
            d_body = (c_body @ B_use.T).T  # (T, D_body_opt) if rot6d else (T, D)
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(body_base)  # (T, D_body_opt)
                body = _rot6d_flat_to_aa(base6d + d_body)  # (T, D)
            else:
                body = body_base + d_body
        else:
            c_body = None
            if use_rot6d:
                d_body = torch.zeros((int(body_base.shape[0]), int(D_body_opt)), device=body_base.device, dtype=body_base.dtype)
            else:
                d_body = torch.zeros_like(body_base)
            body = body_base

        # global orient (AA or rot6d space)
        if u_glob is not None:
            c_glob = _u_to_c(u_glob, amp=float(cfg.amp_global_orient), use_tanh=bool(cfg.use_tanh))
            d_glob = (c_glob @ B_use.T).T  # (T, D_glob_opt) if rot6d else (T, 3)
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(glob_base)  # (T, 6)
                glob = _rot6d_flat_to_aa(base6d + d_glob)  # (T, 3)
            else:
                glob = glob_base + d_glob
        else:
            c_glob = None
            if use_rot6d:
                d_glob = torch.zeros((int(glob_base.shape[0]), int(D_glob_opt)), device=glob_base.device, dtype=glob_base.dtype)
            else:
                d_glob = torch.zeros_like(glob_base)
            glob = glob_base

        if u_tr is not None:
            c_tr = _u_to_c(u_tr, amp=float(cfg.amp_transl), use_tanh=bool(cfg.use_tanh))
            d_tr = (c_tr @ B_use.T).T
            tr = tr_base + d_tr
        else:
            c_tr = None
            d_tr = torch.zeros_like(tr_base)
            tr = tr_base
        return body, glob, tr, d_body, d_glob, d_tr, c_body, c_glob, c_tr

    def _closure():
        opt.zero_grad(set_to_none=True)
        B_use, U_use, internal_use, gaps_use = _get_B_and_knots()
        b0, g0, t0, db0, dg0, dt0, cb0, cg0, ct0 = _build_view(body0, glob0, tr0, u_body0, u_glob0, u_tr0, B_use=B_use)
        b1, g1, t1, db1, dg1, dt1, cb1, cg1, ct1 = _build_view(body1, glob1, tr1, u_body1, u_glob1, u_tr1, B_use=B_use)

        # Optional pose limiter inside objective
        b0m = _maybe_pose_limit(b0, enabled=pose_limit_in_loss, device=dev)
        b1m = _maybe_pose_limit(b1, enabled=pose_limit_in_loss, device=dev)

        # Forward SMPL to get joints3d for optional ankle loss + joints2d for reproj
        from .geo_transform import project_p2d
        j3d0 = smpl_coco17(body_pose=b0m, betas=bet0, global_orient=g0, transl=t0)  # (T,17,3)
        j3d1 = smpl_coco17(body_pose=b1m, betas=bet1, global_orient=g1, transl=t1)  # (T,17,3)
        smpl2d0 = project_p2d(j3d0, K=K0)
        smpl2d1 = project_p2d(j3d1, K=K1)

        loss = _mean_normed_reproj_err(smpl2d0, co0, bb0, conf_thr=float(cfg.conf_thr)) + _mean_normed_reproj_err(
            smpl2d1, co1, bb1, conf_thr=float(cfg.conf_thr)
        )

        # optional static-motion loss (always enabled when logits provided)
        if float(cfg.static_motion_w) > 0.0:
            ls0 = _static_motion_loss(
                joints17=j3d0,
                static_logits=static_conf_logits_v0,
                cfg=cfg,
                smpl24_model=smpl_smpl24,
                body_pose=b0m,
                betas=bet0,
                global_orient=g0,
                transl=t0,
            )
            ls1 = _static_motion_loss(
                joints17=j3d1,
                static_logits=static_conf_logits_v1,
                cfg=cfg,
                smpl24_model=smpl_smpl24,
                body_pose=b1m,
                betas=bet1,
                global_orient=g1,
                transl=t1,
            )
            loss = loss + float(cfg.static_motion_w) * (ls0 + ls1)

        # priors
        if u_body0 is not None:
            loss = loss + float(cfg.prior_w_body_pose) * ((db0 * db0) * body_prior_w_opt).mean()
            loss = loss + float(cfg.prior_w_body_pose) * ((db1 * db1) * body_prior_w_opt).mean()
        if u_glob0 is not None:
            loss = loss + float(cfg.prior_w_global_orient) * (dg0 * dg0).mean()
            loss = loss + float(cfg.prior_w_global_orient) * (dg1 * dg1).mean()
        if u_tr0 is not None:
            loss = loss + float(cfg.prior_w_transl) * (dt0 * dt0).mean()
            loss = loss + float(cfg.prior_w_transl) * (dt1 * dt1).mean()

        # optional temporal loss on deltas (velocity + acceleration)
        if float(cfg.temporal_vel_w) > 0.0 or float(cfg.temporal_acc_w) > 0.0:
            if u_body0 is not None:
                loss = loss + _temporal_vel_acc_loss(
                    db0, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
                loss = loss + _temporal_vel_acc_loss(
                    db1, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
            if u_glob0 is not None:
                loss = loss + _temporal_vel_acc_loss(
                    dg0, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
                loss = loss + _temporal_vel_acc_loss(
                    dg1, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
            if u_tr0 is not None:
                loss = loss + _temporal_vel_acc_loss(
                    dt0, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
                loss = loss + _temporal_vel_acc_loss(
                    dt1, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )

        # multi-view consistency on body_pose only
        loss = loss + float(cfg.mv_consistency_w) * (((b0m - b1m) * (b0m - b1m)) * body_prior_w_aa).mean()

        # Optional ankle-ground alignment (only if probabilities are provided)
        def _ankle_term(j3d: torch.Tensor, p: torch.Tensor | None, n: torch.Tensor | None) -> torch.Tensor:
            if (p is None) or (n is None) or (w_ank <= 0):
                return torch.zeros((), device=j3d.device, dtype=j3d.dtype)
            # p: (T,2) left/right ankle static prob in [0,1]
            pp = p.to(j3d.device).float()
            if pp.ndim == 3 and int(pp.shape[0]) == 1:
                pp = pp[0]
            pp = pp[: j3d.shape[0], :2]
            pp = torch.clamp(pp, 0.0, 1.0)
            p_pair = 0.5 * (pp[:, 0] + pp[:, 1])  # (T,)

            nn = n.to(j3d.device).float()
            if nn.ndim == 1:
                nn = nn[None].expand(j3d.shape[0], -1)
            nn = nn[: j3d.shape[0]]
            nn = nn / (nn.norm(dim=-1, keepdim=True) + 1e-8)

            # COCO17: left_ankle=15, right_ankle=16
            v = j3d[:, 15] - j3d[:, 16]  # (T,3)
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
            dot = (v * nn).sum(dim=-1)  # (T,)
            return float(w_ank) * (p_pair * dot * dot).mean()

        loss = loss + _ankle_term(j3d0, ankle_static_prob_v0, ground_normal_cam_v0)
        loss = loss + _ankle_term(j3d1, ankle_static_prob_v1, ground_normal_cam_v1)

        # optional smoothness on control points
        if float(cfg.smooth_w) > 0 and M >= 3:
            for c in (cb0, cb1, cg0, cg1, ct0, ct1):
                if c is None:
                    continue
                d2 = c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]
                loss = loss + float(cfg.smooth_w) * (d2 * d2).mean()

        # optional knot priors (only when learn_knots=True)
        if bool(cfg.learn_knots):
            if (internal_knots_uni is None) or (gaps_uni is None):
                raise RuntimeError("Expected internal_knots_uni/gaps_uni when learn_knots=True")
            if (internal_use is None) or (gaps_use is None):
                raise RuntimeError("Expected internal_use/gaps_use when learn_knots=True")
            if int(internal_use.numel()) > 0 and float(cfg.knot_pos_w) > 0.0:
                loss = loss + float(cfg.knot_pos_w) * ((internal_use - internal_knots_uni) ** 2).mean()
            if int(gaps_use.numel()) > 0 and float(cfg.knot_gap_w) > 0.0:
                loss = loss + float(cfg.knot_gap_w) * ((gaps_use - gaps_uni) ** 2).mean()
            if int(internal_use.numel()) >= 3 and float(cfg.knot_smooth_w) > 0.0:
                d2k = internal_use[2:] - 2.0 * internal_use[1:-1] + internal_use[:-2]
                loss = loss + float(cfg.knot_smooth_w) * (d2k * d2k).mean()

        loss.backward()
        last["loss"] = float(loss.detach().item())
        return loss

    if cfg.verbose:
        print(
            f"[Spline-Opt] (two_view_joint) T={T} M={M} degree={cfg.degree} conf_thr={cfg.conf_thr} "
            f"opt(body,glob,tr)=({optimize_body_pose},{optimize_global_orient},{optimize_transl}) "
            f"mv_w={cfg.mv_consistency_w} learn_knots={bool(cfg.learn_knots)}"
        )
        print(f"[Spline-Opt] baseline mean(||SMPL-GT||/s) view0={base0:.6f} view1={base1:.6f}")

    t_start = time.perf_counter()
    opt.step(_closure)
    t_end = time.perf_counter()
    if cfg.verbose:
        print(f"[Spline-Opt] optimize_time_sec={t_end - t_start:.3f} (two_view_joint)")

    with torch.no_grad():
        B_use, U_use, internal_use, gaps_use = _get_B_and_knots()
        b0, g0, t0, db0, dg0, dt0, _, _, _ = _build_view(body0, glob0, tr0, u_body0, u_glob0, u_tr0, B_use=B_use)
        b1, g1, t1, db1, dg1, dt1, _, _, _ = _build_view(body1, glob1, tr1, u_body1, u_glob1, u_tr1, B_use=B_use)
        # For API compatibility: always report deltas in axis-angle space.
        db0_aa = b0 - body0
        db1_aa = b1 - body1
        dg0_aa = g0 - glob0
        dg1_aa = g1 - glob1
        b0m = _maybe_pose_limit(b0, enabled=pose_limit_in_loss, device=dev)
        b1m = _maybe_pose_limit(b1, enabled=pose_limit_in_loss, device=dev)
        from .geo_transform import project_p2d
        j3d0 = smpl_coco17(body_pose=b0m, betas=bet0, global_orient=g0, transl=t0)
        j3d1 = smpl_coco17(body_pose=b1m, betas=bet1, global_orient=g1, transl=t1)
        smpl2d0 = project_p2d(j3d0, K=K0)
        smpl2d1 = project_p2d(j3d1, K=K1)
        fin0 = float(_mean_normed_reproj_err(smpl2d0, co0, bb0, conf_thr=float(cfg.conf_thr)).item())
        fin1 = float(_mean_normed_reproj_err(smpl2d1, co1, bb1, conf_thr=float(cfg.conf_thr)).item())

    if cfg.verbose:
        print(f"[Spline-Opt] final mean(||SMPL-GT||/s) view0={fin0:.6f} view1={fin1:.6f} (loss={last['loss']})")

    # Store final basis (and knots if learned)
    B_out = B.detach().cpu() if B is not None else None
    knots_out = None
    internal_out = None
    gaps_out = None
    if bool(cfg.learn_knots):
        if U_use is None:
            raise RuntimeError("Expected U_use when learn_knots=True")
        B_out = _bspline_basis_matrix(t, m=int(M), degree=int(cfg.degree), knots=U_use.detach().cpu()).float().detach().cpu()
        knots_out = U_use.detach().cpu()
        internal_out = internal_use.detach().cpu() if internal_use is not None else None
        gaps_out = gaps_use.detach().cpu() if gaps_use is not None else None
        if cfg.verbose and (internal_out is not None) and int(internal_out.numel()) > 0:
            k8 = internal_out[:8].tolist()
            print(f"[Spline-Opt] learned_knot_internal_first8={k8}")

    out = {
        "view0": {
            "body_pose_refined": b0.detach().cpu(),
            "global_orient_refined": g0.detach().cpu(),
            "transl_refined": t0.detach().cpu(),
            "delta_body_pose": db0_aa.detach().cpu(),
            "delta_global_orient": dg0_aa.detach().cpu(),
            "delta_transl": dt0.detach().cpu(),
        },
        "view1": {
            "body_pose_refined": b1.detach().cpu(),
            "global_orient_refined": g1.detach().cpu(),
            "transl_refined": t1.detach().cpu(),
            "delta_body_pose": db1_aa.detach().cpu(),
            "delta_global_orient": dg1_aa.detach().cpu(),
            "delta_transl": dt1.detach().cpu(),
        },
        "weights_B": B_out if B_out is not None else torch.empty((0, 0)),
        "stats": {
            "baseline_err_view0": base0,
            "baseline_err_view1": base1,
            "final_err_view0": fin0,
            "final_err_view1": fin1,
            "T": int(T),
            "M": int(M),
            "degree": int(cfg.degree),
            "D": int(D),
            "multi_view_joint": True,
            "mv_consistency_w": float(cfg.mv_consistency_w),
            "learn_knots": bool(cfg.learn_knots),
            "knot_min_gap": float(cfg.knot_min_gap),
            "knot_pos_w": float(cfg.knot_pos_w),
            "knot_gap_w": float(cfg.knot_gap_w),
            "knot_smooth_w": float(cfg.knot_smooth_w),
        },
    }
    if knots_out is not None:
        out["knots"] = knots_out
    if internal_out is not None:
        out["knot_internal"] = internal_out
    if gaps_out is not None:
        out["knot_gaps"] = gaps_out
    return out


def _refine_bspline_multi_view_lbfgs(
    *,
    views: list[dict[str, Any]],
    cfg: BsplineRefineConfig,
    device: str | torch.device,
    optimize_body_pose: bool,
    optimize_global_orient: bool,
    optimize_transl: bool,
    pose_limit_in_loss: bool,
    ankle_ground_align_w: float | None = None,
) -> dict:
    """
    Multi-view joint optimization for N>=2 views.

    Input `views` is a list of dicts, each dict must contain:
      - body_pose: (T,D) (D in {63,69} or any multiple of 3)
      - betas: (T,10) or (10,)
      - global_orient: (T,3)
      - transl: (T,3)
      - K_fullimg: (T,3,3) or (3,3)
      - bbx_xys: (T,3)
      - coco17: (T,17,3)
    Optional per-view keys:
      - ankle_static_prob: (T,2) in [0,1]
      - ground_normal_cam: (3,) or (T,3)

    Consistency loss (body_pose only) generalizes the 2-view case:
      mv_consistency_w * mean_{i<j} mean( (b_i - b_j)^2 * body_prior_w )
    For N=2, this reduces EXACTLY to the original term.
    """
    if len(views) < 2:
        raise ValueError(f"multi_view_joint expects >=2 views, got {len(views)}")

    # Exact backward-compat: for N=2, delegate to the original implementation
    if len(views) == 2:
        v0 = views[0]
        v1 = views[1]
        return _refine_bspline_two_view_lbfgs(
            body_pose_v0=v0["body_pose"],
            betas_v0=v0["betas"],
            global_orient_v0=v0["global_orient"],
            transl_v0=v0["transl"],
            K_fullimg_v0=v0["K_fullimg"],
            bbx_xys_v0=v0["bbx_xys"],
            coco17_v0=v0["coco17"],
            body_pose_v1=v1["body_pose"],
            betas_v1=v1["betas"],
            global_orient_v1=v1["global_orient"],
            transl_v1=v1["transl"],
            K_fullimg_v1=v1["K_fullimg"],
            bbx_xys_v1=v1["bbx_xys"],
            coco17_v1=v1["coco17"],
            cfg=cfg,
            device=device,
            optimize_body_pose=optimize_body_pose,
            optimize_global_orient=optimize_global_orient,
            optimize_transl=optimize_transl,
            pose_limit_in_loss=pose_limit_in_loss,
            ankle_static_prob_v0=v0.get("ankle_static_prob", None),
            ground_normal_cam_v0=v0.get("ground_normal_cam", None),
            ankle_static_prob_v1=v1.get("ankle_static_prob", None),
            ground_normal_cam_v1=v1.get("ground_normal_cam", None),
            static_conf_logits_v0=v0.get("static_conf_logits", None),
            static_conf_logits_v1=v1.get("static_conf_logits", None),
            ankle_ground_align_w=ankle_ground_align_w,
        )

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    use_k3d = torch.is_tensor(getattr(cfg, "k3d_gt", None))

    # Validate and normalize shapes; compute common T and D
    bp_list = []
    D = None
    for i, v in enumerate(views):
        bp = v["body_pose"]
        if (not torch.is_tensor(bp)) or bp.ndim != 2:
            raise ValueError(f"views[{i}].body_pose must be (T,D), got {getattr(bp, 'shape', None)}")
        d = int(bp.shape[1])
        if D is None:
            D = d
        elif d != D:
            raise ValueError(f"body_pose dim mismatch across views: expected D={D}, got {d} at view {i}")
        bp_list.append(bp.detach().cpu())
    assert D is not None
    if D <= 0:
        raise ValueError(f"Invalid body_pose dim: {D}")

    # common T (min across all sequences used in loss)
    T = None
    for i, v in enumerate(views):
        cand = [
            int(v["body_pose"].shape[0]),
            int(v["global_orient"].shape[0]),
            int(v["transl"].shape[0]),
            int(v["coco17"].shape[0]),
            int(v["bbx_xys"].shape[0]),
        ]
        # K_fullimg may be (3,3)
        K = v["K_fullimg"]
        if torch.is_tensor(K) and K.ndim == 3:
            cand.append(int(K.shape[0]))
        if torch.is_tensor(v.get("betas", None)) and v["betas"].ndim == 2:
            cand.append(int(v["betas"].shape[0]))
        if torch.is_tensor(v.get("ankle_static_prob", None)) and v["ankle_static_prob"].ndim >= 2:
            cand.append(int(v["ankle_static_prob"].shape[0]))
        if torch.is_tensor(v.get("ground_normal_cam", None)) and v["ground_normal_cam"].ndim == 2:
            cand.append(int(v["ground_normal_cam"].shape[0]))
        if torch.is_tensor(v.get("static_conf_logits", None)) and v["static_conf_logits"].ndim >= 2:
            cand.append(int(v["static_conf_logits"].shape[-2]))
        t_i = min(cand)
        T = t_i if T is None else min(int(T), int(t_i))
    assert T is not None
    if T <= 0:
        raise ValueError(f"Invalid T={T} after alignment")

    # choose number of basis functions
    M = int(math.ceil(T / float(cfg.m_per_t)))
    M = max(M, cfg.degree + 2)
    t = torch.linspace(0.0, 1.0, T, dtype=torch.float32)
    B: torch.Tensor | None = None
    B_dev: torch.Tensor | None = None
    u_knot_gaps: torch.Tensor | None = None
    t_dev: torch.Tensor | None = None
    internal_knots_uni: torch.Tensor | None = None
    gaps_uni: torch.Tensor | None = None
    if not bool(cfg.learn_knots):
        B = _bspline_basis_matrix(t, m=M, degree=int(cfg.degree)).float()  # (T,M) on CPU
        B_dev = B.to(dev)
    else:
        t_dev = torch.linspace(0.0, 1.0, T, dtype=torch.float32, device=dev)
        u_knot_gaps = torch.zeros((int(M - int(cfg.degree)),), device=dev, dtype=torch.float32, requires_grad=True)
        internal_knots_uni = _uniform_internal_knots(int(M), int(cfg.degree), device=dev, dtype=torch.float32)
        if int(u_knot_gaps.numel()) > 0:
            gaps_uni = torch.full(
                (int(u_knot_gaps.numel()),),
                1.0 / float(int(u_knot_gaps.numel())),
                device=dev,
                dtype=torch.float32,
            )

    # SMPL model once (2D reproj: use_smpl 时用纯 SMPL+J_reg，否则 supermotion_coco17)
    from .smplx_utils import make_smplx

    smpl_coco17 = _make_coco17_model(cfg, dev)
    smpl_smpl24 = None
    if bool(cfg.static_use_smpl24) and int(D) == 63:
        try:
            smpl_smpl24 = make_smplx("supermotion_smpl24").to(dev).eval()
        except Exception:
            smpl_smpl24 = None

    use_rot6d = bool(cfg.optimize_pose_in_rot6d)
    D_body_opt = int((D // 3) * 6) if use_rot6d else int(D)
    D_glob_opt = 6 if use_rot6d else 3

    # body prior weights:
    # - opt-space weights: for delta regularization (matches d_bodies shape)
    # - aa-space weights: for multi-view consistency on axis-angle (matches bodies_m shape)
    body_prior_w_aa = _body_pose_prior_weights_per_dim(D, device=dev, dtype=torch.float32)  # (1,D)
    body_prior_w_opt = (
        _body_pose_prior_weights_per_dim_rot6d(D_body_opt, device=dev, dtype=torch.float32)
        if use_rot6d
        else body_prior_w_aa
    )  # (1, D_body_opt) or (1, D)
    if bool(use_k3d):
        body_prior_w_opt = _maybe_scale_foot_prior_weights(
            body_prior_w_opt,
            k=float(getattr(cfg, "k3d_foot_prior_k", 1.0)),
            use_rot6d=bool(use_rot6d),
        )
        body_prior_w_aa = _maybe_scale_foot_prior_weights(
            body_prior_w_aa,
            k=float(getattr(cfg, "k3d_foot_prior_k", 1.0)),
            use_rot6d=False,
        )
    # ankle alignment weight
    w_ank = float(cfg.ankle_ground_align_w if ankle_ground_align_w is None else ankle_ground_align_w)

    # Move per-view tensors to device and align prefix length T
    bodies0 = []
    betas0 = []
    globs0 = []
    trs0 = []
    Ks0 = []
    bbxs0 = []
    cocos0 = []
    ankle_ps = []
    ground_ns = []
    for v in views:
        bodies0.append(v["body_pose"].detach().cpu()[:T].to(dev).float())
        bet = v["betas"]
        bet = (bet[:T] if (torch.is_tensor(bet) and bet.ndim == 2) else bet).to(dev).float()
        betas0.append(bet)
        globs0.append(v["global_orient"][:T].to(dev).float())
        trs0.append(v["transl"][:T].to(dev).float())
        K = v["K_fullimg"]
        K = (K[:T] if (torch.is_tensor(K) and K.ndim == 3) else K).to(dev).float()
        Ks0.append(K)
        bbxs0.append(v["bbx_xys"][:T].to(dev).float())
        cocos0.append(v["coco17"][:T].to(dev).float())
        ankle_ps.append(v.get("ankle_static_prob", None))
        ground_ns.append(v.get("ground_normal_cam", None))

    # Vars
    params_to_opt: list[torch.Tensor] = []
    if bool(cfg.learn_knots):
        if u_knot_gaps is None:
            raise RuntimeError("cfg.learn_knots=True but u_knot_gaps is None")
        params_to_opt.append(u_knot_gaps)
    u_bodies: list[torch.Tensor | None] = [None] * len(views)
    u_globs: list[torch.Tensor | None] = [None] * len(views)
    u_trs: list[torch.Tensor | None] = [None] * len(views)
    if optimize_body_pose:
        for i in range(len(views)):
            u = torch.zeros((D_body_opt, M), device=dev, dtype=torch.float32, requires_grad=True)
            u_bodies[i] = u
            params_to_opt.append(u)
    if optimize_global_orient:
        for i in range(len(views)):
            u = torch.zeros((D_glob_opt, M), device=dev, dtype=torch.float32, requires_grad=True)
            u_globs[i] = u
            params_to_opt.append(u)
    if optimize_transl:
        for i in range(len(views)):
            u = torch.zeros((3, M), device=dev, dtype=torch.float32, requires_grad=True)
            u_trs[i] = u
            params_to_opt.append(u)
    if not params_to_opt:
        raise ValueError("At least one optimize_* must be True.")

    # baseline stats
    base_errs: list[float] = []
    with torch.no_grad():
        from .geo_transform import project_p2d

        for i in range(len(views)):
            j3d = smpl_coco17(body_pose=bodies0[i], betas=betas0[i], global_orient=globs0[i], transl=trs0[i])
            s2d = project_p2d(j3d, K=Ks0[i])
            base_errs.append(float(_mean_normed_reproj_err(s2d, cocos0[i], bbxs0[i], conf_thr=float(cfg.conf_thr)).item()))

    opt = torch.optim.LBFGS(params_to_opt, lr=float(cfg.lr), max_iter=int(cfg.max_iter), line_search_fn=cfg.line_search_fn)
    last = {"loss": None}

    def _get_B_and_knots():
        if not bool(cfg.learn_knots):
            if B_dev is None:
                raise RuntimeError("Expected precomputed B_dev when learn_knots=False")
            return B_dev, None, None, None
        if (u_knot_gaps is None) or (t_dev is None):
            raise RuntimeError("Expected u_knot_gaps/t_dev when learn_knots=True")
        U, internal, gaps = _knots_from_u_gaps(
            u_knot_gaps,
            m=int(M),
            degree=int(cfg.degree),
            min_gap=float(cfg.knot_min_gap),
        )
        Bk = _bspline_basis_matrix(t_dev, m=int(M), degree=int(cfg.degree), knots=U).float()
        return Bk, U, internal, gaps

    def _build_view(i: int, *, B_use: torch.Tensor):
        body_base = bodies0[i]
        glob_base = globs0[i]
        tr_base = trs0[i]
        u_body = u_bodies[i]
        u_glob = u_globs[i]
        u_tr = u_trs[i]

        # body pose (AA or rot6d space)
        if u_body is not None:
            c_body = _u_to_c(u_body, amp=float(cfg.amp_body_pose), use_tanh=bool(cfg.use_tanh))
            d_body = (c_body @ B_use.T).T  # (T, D_body_opt) if rot6d else (T, D)
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(body_base)  # (T, D_body_opt)
                body = _rot6d_flat_to_aa(base6d + d_body)  # (T, D)
            else:
                body = body_base + d_body
        else:
            c_body = None
            if use_rot6d:
                d_body = torch.zeros((int(body_base.shape[0]), int(D_body_opt)), device=body_base.device, dtype=body_base.dtype)
            else:
                d_body = torch.zeros_like(body_base)
            body = body_base

        # global orient (AA or rot6d space)
        if u_glob is not None:
            c_glob = _u_to_c(u_glob, amp=float(cfg.amp_global_orient), use_tanh=bool(cfg.use_tanh))
            d_glob = (c_glob @ B_use.T).T  # (T, D_glob_opt) if rot6d else (T, 3)
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(glob_base)  # (T, 6)
                glob = _rot6d_flat_to_aa(base6d + d_glob)  # (T, 3)
            else:
                glob = glob_base + d_glob
        else:
            c_glob = None
            if use_rot6d:
                d_glob = torch.zeros((int(glob_base.shape[0]), int(D_glob_opt)), device=glob_base.device, dtype=glob_base.dtype)
            else:
                d_glob = torch.zeros_like(glob_base)
            glob = glob_base

        if u_tr is not None:
            c_tr = _u_to_c(u_tr, amp=float(cfg.amp_transl), use_tanh=bool(cfg.use_tanh))
            d_tr = (c_tr @ B_use.T).T
            tr = tr_base + d_tr
        else:
            c_tr = None
            d_tr = torch.zeros_like(tr_base)
            tr = tr_base

        return body, glob, tr, d_body, d_glob, d_tr, c_body, c_glob, c_tr

    def _ankle_term(j3d: torch.Tensor, p: torch.Tensor | None, n: torch.Tensor | None) -> torch.Tensor:
        if (p is None) or (n is None) or (w_ank <= 0):
            return torch.zeros((), device=j3d.device, dtype=j3d.dtype)
        pp = p.to(j3d.device).float()
        if pp.ndim == 3 and int(pp.shape[0]) == 1:
            pp = pp[0]
        pp = pp[: j3d.shape[0], :2]
        pp = torch.clamp(pp, 0.0, 1.0)
        p_pair = 0.5 * (pp[:, 0] + pp[:, 1])

        nn = n.to(j3d.device).float()
        if nn.ndim == 1:
            nn = nn[None].expand(j3d.shape[0], -1)
        nn = nn[: j3d.shape[0]]
        nn = nn / (nn.norm(dim=-1, keepdim=True) + 1e-8)

        v = j3d[:, 15] - j3d[:, 16]
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        dot = (v * nn).sum(dim=-1)
        return float(w_ank) * (p_pair * dot * dot).mean()

    def _closure():
        opt.zero_grad(set_to_none=True)

        # build views
        B_use, U_use, internal_use, gaps_use = _get_B_and_knots()
        built = [_build_view(i, B_use=B_use) for i in range(len(views))]
        bodies = [x[0] for x in built]
        globs = [x[1] for x in built]
        trs = [x[2] for x in built]
        d_bodies = [x[3] for x in built]
        d_globs = [x[4] for x in built]
        d_trs = [x[5] for x in built]
        c_bodies = [x[6] for x in built]
        c_globs = [x[7] for x in built]
        c_trs = [x[8] for x in built]

        bodies_m = [_maybe_pose_limit(b, enabled=pose_limit_in_loss, device=dev) for b in bodies]

        # forward SMPL to get joints3d (ankle) + joints2d (reproj)
        from .geo_transform import project_p2d

        j3ds = [smpl_coco17(body_pose=bodies_m[i], betas=betas0[i], global_orient=globs[i], transl=trs[i]) for i in range(len(views))]
        smpl2ds = [project_p2d(j3ds[i], K=Ks0[i]) for i in range(len(views))]

        loss = torch.zeros((), device=dev, dtype=torch.float32)
        # reproj sum
        for i in range(len(views)):
            loss = loss + _mean_normed_reproj_err(smpl2ds[i], cocos0[i], bbxs0[i], conf_thr=float(cfg.conf_thr))

        # optional static-motion loss per view (always enabled when logits provided)
        if float(cfg.static_motion_w) > 0.0:
            for i in range(len(views)):
                static_logits = views[i].get("static_conf_logits", None)
                ls = _static_motion_loss(
                    joints17=j3ds[i],
                    static_logits=static_logits,
                    cfg=cfg,
                    smpl24_model=smpl_smpl24,
                    body_pose=bodies_m[i],
                    betas=betas0[i],
                    global_orient=globs[i],
                    transl=trs[i],
                )
                loss = loss + float(cfg.static_motion_w) * ls

        # priors
        for i in range(len(views)):
            if u_bodies[i] is not None:
                loss = loss + float(cfg.prior_w_body_pose) * ((d_bodies[i] * d_bodies[i]) * body_prior_w_opt).mean()
            if u_globs[i] is not None:
                loss = loss + float(cfg.prior_w_global_orient) * (d_globs[i] * d_globs[i]).mean()
            if u_trs[i] is not None:
                loss = loss + float(cfg.prior_w_transl) * (d_trs[i] * d_trs[i]).mean()

        # optional temporal loss on deltas (velocity + acceleration)
        if float(cfg.temporal_vel_w) > 0.0 or float(cfg.temporal_acc_w) > 0.0:
            for i in range(len(views)):
                if u_bodies[i] is not None:
                    loss = loss + _temporal_vel_acc_loss(
                        d_bodies[i], vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                    )
                if u_globs[i] is not None:
                    loss = loss + _temporal_vel_acc_loss(
                        d_globs[i], vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                    )
                if u_trs[i] is not None:
                    loss = loss + _temporal_vel_acc_loss(
                        d_trs[i], vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                    )

        # multi-view consistency on body_pose only: average over pairs
        pair_cnt = 0
        cons = torch.zeros((), device=dev, dtype=torch.float32)
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                diff = bodies_m[i] - bodies_m[j]
                cons = cons + ((diff * diff) * body_prior_w_aa).mean()
                pair_cnt += 1
        if pair_cnt > 0:
            cons = cons / float(pair_cnt)
        loss = loss + float(cfg.mv_consistency_w) * cons

        # Optional ankle-ground alignment
        for i in range(len(views)):
            loss = loss + _ankle_term(j3ds[i], ankle_ps[i], ground_ns[i])

        # optional smoothness on control points
        if float(cfg.smooth_w) > 0 and M >= 3:
            for c in c_bodies + c_globs + c_trs:
                if c is None:
                    continue
                d2 = c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]
                loss = loss + float(cfg.smooth_w) * (d2 * d2).mean()

        # optional knot priors (only when learn_knots=True)
        if bool(cfg.learn_knots):
            if (internal_knots_uni is None) or (gaps_uni is None):
                raise RuntimeError("Expected internal_knots_uni/gaps_uni when learn_knots=True")
            if (internal_use is None) or (gaps_use is None):
                raise RuntimeError("Expected internal_use/gaps_use when learn_knots=True")
            if int(internal_use.numel()) > 0 and float(cfg.knot_pos_w) > 0.0:
                loss = loss + float(cfg.knot_pos_w) * ((internal_use - internal_knots_uni) ** 2).mean()
            if int(gaps_use.numel()) > 0 and float(cfg.knot_gap_w) > 0.0:
                loss = loss + float(cfg.knot_gap_w) * ((gaps_use - gaps_uni) ** 2).mean()
            if int(internal_use.numel()) >= 3 and float(cfg.knot_smooth_w) > 0.0:
                d2k = internal_use[2:] - 2.0 * internal_use[1:-1] + internal_use[:-2]
                loss = loss + float(cfg.knot_smooth_w) * (d2k * d2k).mean()

        loss.backward()
        last["loss"] = float(loss.detach().item())
        return loss

    if cfg.verbose:
        print(
            f"[Spline-Opt] (multi_view_joint) V={len(views)} T={T} M={M} degree={cfg.degree} conf_thr={cfg.conf_thr} "
            f"opt(body,glob,tr)=({optimize_body_pose},{optimize_global_orient},{optimize_transl}) "
            f"mv_w={cfg.mv_consistency_w} learn_knots={bool(cfg.learn_knots)}"
        )
        for i, e in enumerate(base_errs):
            print(f"[Spline-Opt] baseline mean(||SMPL-GT||/s) view{i}={e:.6f}")

    t_start = time.perf_counter()
    opt.step(_closure)
    t_end = time.perf_counter()
    if cfg.verbose:
        print(f"[Spline-Opt] optimize_time_sec={t_end - t_start:.3f} (multi_view_joint)")

    # final stats & outputs
    B_use, U_use, internal_use, gaps_use = _get_B_and_knots()
    B_out = B.detach().cpu() if B is not None else None
    knots_out = None
    internal_out = None
    gaps_out = None
    if bool(cfg.learn_knots):
        if U_use is None:
            raise RuntimeError("Expected U_use when learn_knots=True")
        B_out = _bspline_basis_matrix(t, m=int(M), degree=int(cfg.degree), knots=U_use.detach().cpu()).float().detach().cpu()
        knots_out = U_use.detach().cpu()
        internal_out = internal_use.detach().cpu() if internal_use is not None else None
        gaps_out = gaps_use.detach().cpu() if gaps_use is not None else None
        if cfg.verbose and (internal_out is not None) and int(internal_out.numel()) > 0:
            k8 = internal_out[:8].tolist()
            print(f"[Spline-Opt] learned_knot_internal_first8={k8}")

    out: dict[str, Any] = {
        "weights_B": B_out if B_out is not None else torch.empty((0, 0)),
        "stats": {
            "T": int(T),
            "M": int(M),
            "degree": int(cfg.degree),
            "D": int(D),
            "multi_view_joint": True,
            "mv_consistency_w": float(cfg.mv_consistency_w),
            "learn_knots": bool(cfg.learn_knots),
            "knot_min_gap": float(cfg.knot_min_gap),
            "knot_pos_w": float(cfg.knot_pos_w),
            "knot_gap_w": float(cfg.knot_gap_w),
            "knot_smooth_w": float(cfg.knot_smooth_w),
        },
    }
    with torch.no_grad():
        from .geo_transform import project_p2d

        fin_errs: list[float] = []
        for i in range(len(views)):
            b, g, tr, db, dg, dtr, _, _, _ = _build_view(i, B_use=B_use)
            # For API compatibility: always report deltas in axis-angle space.
            db_aa = b - bodies0[i]
            dg_aa = g - globs0[i]
            bm = _maybe_pose_limit(b, enabled=pose_limit_in_loss, device=dev)
            j3d = smpl_coco17(body_pose=bm, betas=betas0[i], global_orient=g, transl=tr)
            s2d = project_p2d(j3d, K=Ks0[i])
            fin_errs.append(float(_mean_normed_reproj_err(s2d, cocos0[i], bbxs0[i], conf_thr=float(cfg.conf_thr)).item()))
            out[f"view{i}"] = {
                "body_pose_refined": b.detach().cpu(),
                "global_orient_refined": g.detach().cpu(),
                "transl_refined": tr.detach().cpu(),
                "delta_body_pose": db_aa.detach().cpu(),
                "delta_global_orient": dg_aa.detach().cpu(),
                "delta_transl": dtr.detach().cpu(),
            }

        for i, e in enumerate(base_errs):
            out["stats"][f"baseline_err_view{i}"] = e
        for i, e in enumerate(fin_errs):
            out["stats"][f"final_err_view{i}"] = e
        out["stats"]["loss"] = last["loss"]

    if knots_out is not None:
        out["knots"] = knots_out
    if internal_out is not None:
        out["knot_internal"] = internal_out
    if gaps_out is not None:
        out["knot_gaps"] = gaps_out

    if cfg.verbose:
        for i, e in enumerate(fin_errs):
            print(f"[Spline-Opt] final mean(||SMPL-GT||/s) view{i}={e:.6f} (loss={last['loss']})")

    return out


def _open_uniform_knot_vector(m: int, degree: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Open-uniform clamped knot vector on [0,1].
    length = m + degree + 1
    """
    p = int(degree)
    if m <= p:
        raise ValueError(f"Need m > degree, got m={m} degree={p}")
    n_knots = m + p + 1
    # first p+1 are 0, last p+1 are 1
    n_internal = n_knots - 2 * (p + 1)
    if n_internal < 0:
        n_internal = 0
    if n_internal == 0:
        internal = torch.empty((0,), device=device, dtype=dtype)
    else:
        internal = torch.linspace(0.0, 1.0, n_internal + 2, device=device, dtype=dtype)[1:-1]
    knots = torch.cat(
        [
            torch.zeros((p + 1,), device=device, dtype=dtype),
            internal,
            torch.ones((p + 1,), device=device, dtype=dtype),
        ],
        dim=0,
    )
    assert int(knots.numel()) == n_knots
    return knots


def _uniform_internal_knots(m: int, degree: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Return uniform internal knot positions in (0,1) for clamped B-spline.
    Count = m - degree - 1.
    """
    p = int(degree)
    k = int(m - p - 1)
    if k <= 0:
        return torch.empty((0,), device=device, dtype=dtype)
    # positions i/(k+1), i=1..k
    return torch.linspace(0.0, 1.0, k + 2, device=device, dtype=dtype)[1:-1]


def _knots_from_u_gaps(
    u_gaps: torch.Tensor,
    *,
    m: int,
    degree: int,
    min_gap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a clamped knot vector U from learnable gap variables.

    Let K = m - degree - 1 be number of internal knots.
    We parameterize K+1 positive gaps (including the last gap to 1.0) via:
      gaps_raw = softplus(u_gaps) + min_gap
      gaps = gaps_raw / gaps_raw.sum()
      internal_knots = cumsum(gaps)[:-1]   # length K, in (0,1)
    Then:
      U = [0]*(p+1) + internal_knots + [1]*(p+1)

    Returns: (U, internal_knots, gaps)
      - U: (m+degree+1,)
      - internal_knots: (K,)
      - gaps: (K+1,) normalized to sum=1
    """
    if u_gaps.ndim != 1:
        raise ValueError(f"u_gaps must be 1D, got {tuple(u_gaps.shape)}")
    p = int(degree)
    K = int(m - p - 1)
    if K < 0:
        raise ValueError(f"Invalid knot counts from (m,degree)=({m},{p})")
    if int(u_gaps.numel()) != int(K + 1):
        raise ValueError(f"u_gaps length mismatch: expected {K+1} (K+1 gaps), got {int(u_gaps.numel())}")
    gaps_raw = F.softplus(u_gaps) + float(min_gap)
    gaps = gaps_raw / (gaps_raw.sum() + 1e-12)
    if K == 0:
        internal = torch.empty((0,), device=u_gaps.device, dtype=u_gaps.dtype)
    else:
        internal = torch.cumsum(gaps, dim=0)[:-1]  # (K,)
    U = torch.cat(
        [
            torch.zeros((p + 1,), device=u_gaps.device, dtype=u_gaps.dtype),
            internal,
            torch.ones((p + 1,), device=u_gaps.device, dtype=u_gaps.dtype),
        ],
        dim=0,
    )
    if int(U.numel()) != int(m + p + 1):
        raise RuntimeError(f"Unexpected knot vector length: got {int(U.numel())}, expected {m+p+1}")
    return U, internal, gaps


def _bspline_basis_matrix(t: torch.Tensor, m: int, degree: int, *, knots: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute B-spline basis matrix B of shape (T, m) for times t in [0,1].
    Uses Cox-de Boor recursion (vectorized over t, loop over degree).

    Note:
    - If knots is None, uses open-uniform clamped knots on [0,1].
    - If knots is provided, it must be a 1D tensor of length (m + degree + 1).
    """
    if t.ndim != 1:
        raise ValueError("t must be 1D (T,)")
    T = int(t.shape[0])
    p = int(degree)
    device = t.device
    dtype = t.dtype

    # Knot vector U length K = m + p + 1
    if knots is None:
        U = _open_uniform_knot_vector(m, p, device=device, dtype=dtype)  # (K,)
    else:
        U = knots.to(device=device, dtype=dtype)
        if U.ndim != 1:
            raise ValueError(f"knots must be 1D, got {tuple(U.shape)}")
        if int(U.numel()) != int(m + p + 1):
            raise ValueError(f"knots length mismatch: got {int(U.numel())}, expected {m+p+1}")
    K = int(U.numel())

    # Degree-0 basis: there are K-1 functions
    tt = t[:, None]  # (T,1)
    u0 = U[:-1][None, :]  # (1,K-1)
    u1 = U[1:][None, :]  # (1,K-1)
    N = ((tt >= u0) & (tt < u1)).to(dtype)  # (T,K-1)
    # include t==1 in the last interval
    N = torch.where((tt == 1.0) & (u1 == 1.0), torch.ones_like(N), N)

    # Cox–de Boor recursion; after k steps, number of basis is (K-1-k)
    eps = torch.tensor(1e-12, device=device, dtype=dtype)
    for k in range(1, p + 1):
        n_prev = int(N.shape[1])  # = K-1-(k-1)
        n_new = n_prev - 1  # = K-1-k
        if n_new <= 0:
            break
        N_new = torch.zeros((T, n_new), device=device, dtype=dtype)

        # i = 0..n_new-1
        for i in range(n_new):
            # denominators
            denom1 = U[i + k] - U[i]  # scalar tensor
            denom2 = U[i + k + 1] - U[i + 1]

            # For clamped knots, denom can be exactly 0 due to repeated end knots.
            # Use masked safe division to keep gradients where denom>0.
            m1 = (denom1 > 0).to(dtype)
            m2 = (denom2 > 0).to(dtype)
            denom1_safe = torch.where(denom1 > 0, denom1, torch.ones_like(denom1))
            denom2_safe = torch.where(denom2 > 0, denom2, torch.ones_like(denom2))
            term1 = m1 * ((t - U[i]) / (denom1_safe + eps)) * N[:, i]
            term2 = m2 * ((U[i + k + 1] - t) / (denom2_safe + eps)) * N[:, i + 1]
            N_new[:, i] = term1 + term2

        N = N_new

    # After recursion, N has shape (T, K-1-p) == (T, m)
    if int(N.shape[1]) != m:
        raise RuntimeError(f"B-spline basis shape mismatch: got {tuple(N.shape)} expected (T,{m}) with T={T} K={K} p={p}")
    return N


def _mean_normed_reproj_err(
    smpl2d: torch.Tensor,
    coco17: torch.Tensor,
    bbx_xys: torch.Tensor,
    *,
    conf_thr: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    smpl2d: (T,17,2) pixel
    coco17: (T,17,3) pixel+conf
    bbx_xys: (T,3) (cx,cy,s) where s is bbox size
    return scalar tensor:
      - if use_conf_weight: weighted mean over valid points of ||p_smpl - p_coco|| / s
        where weight = conf^conf_power (and conf >= conf_thr)
      - else: simple mean over valid points (mask by conf_thr only)
    """
    T = min(int(smpl2d.shape[0]), int(coco17.shape[0]), int(bbx_xys.shape[0]))
    sm = smpl2d[:T].float()
    gt = coco17[:T].float()
    bb = bbx_xys[:T].float()

    s = bb[:, 2].clamp_min(eps)  # (T,)
    conf = gt[:, :, 2].clamp(0.0, 1.0)  # (T,17)
    valid = conf >= float(conf_thr)

    dx = sm[:, :, 0] - gt[:, :, 0]
    dy = sm[:, :, 1] - gt[:, :, 1]
    dist = torch.sqrt(dx * dx + dy * dy)  # (T,17)
    dist_n = dist / s[:, None]  # (T,17)

    if not valid.any():
        # no valid points: return 0 to avoid NaN gradients
        return torch.zeros((), device=smpl2d.device, dtype=torch.float32)

    # Backward-compat: mask-only mean
    if not bool(getattr(_mean_normed_reproj_err, "_use_conf_weight", False)):  # type: ignore[attr-defined]
        return dist_n[valid].mean()

    # Confidence-weighted mean (SMPLify-X style)
    conf_p = conf.clamp_min(0.0)
    w = torch.where(valid, conf_p, torch.zeros_like(conf_p))  # (T,17)
    # allow adjusting sharpness via power
    p = float(getattr(_mean_normed_reproj_err, "_conf_power", 1.0))  # type: ignore[attr-defined]
    if p != 1.0:
        w = torch.pow(w, p)
    w_sum = w.sum().clamp_min(eps)
    return (dist_n * w).sum() / w_sum


def _per_frame_normed_reproj_err(
    smpl2d: torch.Tensor,
    coco17: torch.Tensor,
    bbx_xys: torch.Tensor,
    *,
    conf_thr: float,
    use_max: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Per-frame normalized reprojection error.
    Returns (T,) tensor: for each frame, mean (or max if use_max) over valid joints of ||p_smpl - p_coco||/s.
    Frames with no valid joints get 0.
    """
    T = min(int(smpl2d.shape[0]), int(coco17.shape[0]), int(bbx_xys.shape[0]))
    sm = smpl2d[:T].float()
    gt = coco17[:T].float()
    bb = bbx_xys[:T].float()
    s = bb[:, 2].clamp_min(eps)
    conf = gt[:, :, 2].clamp(0.0, 1.0)
    valid = conf >= float(conf_thr)
    dx = sm[:, :, 0] - gt[:, :, 0]
    dy = sm[:, :, 1] - gt[:, :, 1]
    dist = torch.sqrt(dx * dx + dy * dy)
    dist_n = dist / s[:, None]
    dist_n_masked = torch.where(valid, dist_n, torch.full_like(dist_n, float("nan")))
    if use_max:
        out = torch.nanmax(dist_n_masked, dim=1)
    else:
        out = torch.nanmean(dist_n_masked, dim=1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _set_conf_weighting_for_reproj_loss(*, use_conf_weight: bool, conf_power: float) -> None:
    """
    Internal helper: configure confidence weighting for `_mean_normed_reproj_err`
    without threading extra args through all call sites.
    """
    setattr(_mean_normed_reproj_err, "_use_conf_weight", bool(use_conf_weight))
    setattr(_mean_normed_reproj_err, "_conf_power", float(conf_power))


def _k3d_mpjpe_mm_like_eval(
    *,
    pred_j3d: torch.Tensor,  # (T,J,3) camera coords
    gt_j3d: torch.Tensor,  # (T,J,3) camera coords
    pelvis_idxs: tuple[int, int] = (1, 2),
    align_by_pelvis: bool = True,
    max_frame_mm: float = 200.0,
    mm_scale: float = 1000.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Differentiable MPJPE-style loss that matches GVHMR eval conventions:
    - Align both pred/gt by pelvis (mean of pelvis_idxs) per frame
    - Per-frame mpjpe(mm): mean_j ||pred-gt||_2 * mm_scale
    - Drop frames where per-frame mpjpe is non-finite or > max_frame_mm
    - Return mean over remaining frames; if none remain, return a large finite loss.

    This mirrors the *per-frame dropping* logic we use in `bss-smplify/validation/eval_3dpw.py`.
    """
    # normalize shapes
    pj = pred_j3d
    gj = gt_j3d
    if pj.ndim != 3 or int(pj.shape[-1]) != 3:
        raise ValueError(f"pred_j3d must be (T,J,3), got {tuple(pj.shape)}")
    if gj.ndim != 3 or int(gj.shape[-1]) != 3:
        raise ValueError(f"gt_j3d must be (T,J,3), got {tuple(gj.shape)}")
    T = min(int(pj.shape[0]), int(gj.shape[0]))
    J = min(int(pj.shape[1]), int(gj.shape[1]))
    if T <= 0 or J <= 0:
        raise ValueError(f"Invalid pred/gt shapes: pred={tuple(pj.shape)} gt={tuple(gj.shape)}")
    pj = pj[:T, :J]
    gj = gj[:T, :J]

    if bool(align_by_pelvis):
        i0, i1 = int(pelvis_idxs[0]), int(pelvis_idxs[1])
        if not (0 <= i0 < J and 0 <= i1 < J):
            raise IndexError(f"pelvis_idxs={pelvis_idxs} out of range for J={J}")
        pred_pelvis = pj[:, [i0, i1]].mean(dim=1, keepdim=True)
        gt_pelvis = gj[:, [i0, i1]].mean(dim=1, keepdim=True)
        pj0 = pj - pred_pelvis
        gj0 = gj - gt_pelvis
    else:
        pj0 = pj
        gj0 = gj

    # Per-frame mpjpe in mm (use float64 internally to avoid overflow -> inf)
    d = (pj0 - gj0).double()
    err_j = torch.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)  # (T,J) float64
    err_f = err_j.mean(dim=1) * float(mm_scale)  # (T,) mm

    keep = torch.isfinite(err_f) & (err_f <= float(max_frame_mm))
    n_total = int(err_f.numel())
    n_keep = int(keep.sum().item())
    # if no frames remain, return a large finite loss (avoid NaN / inf)
    if n_keep <= 0:
        loss = err_f.new_tensor(1e6).float()
    else:
        loss = err_f[keep].mean().float()

    stats = {
        "k3d_T": int(T),
        "k3d_J": int(J),
        "k3d_mm_scale": float(mm_scale),
        "k3d_max_frame_mm": float(max_frame_mm),
        "k3d_n_frames_total": int(n_total),
        "k3d_n_frames_kept": int(n_keep),
        "k3d_n_frames_dropped": int(n_total - n_keep),
    }
    return loss, stats


def _k3d_mpjpe_mm_like_eval_masked(
    *,
    pred_j3d: torch.Tensor,  # (T,J,3)
    gt_j3d: torch.Tensor,  # (T,J,3)
    valid_mask: torch.Tensor,  # (T,J) bool
    pelvis_idxs: tuple[int, int] = (1, 2),
    align_by_pelvis: bool = True,
    max_frame_mm: float = 200.0,
    mm_scale: float = 1000.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    pj = pred_j3d
    gj = gt_j3d
    vm = valid_mask
    if pj.ndim != 3 or int(pj.shape[-1]) != 3:
        raise ValueError(f"pred_j3d must be (T,J,3), got {tuple(pj.shape)}")
    if gj.ndim != 3 or int(gj.shape[-1]) != 3:
        raise ValueError(f"gt_j3d must be (T,J,3), got {tuple(gj.shape)}")
    if vm.ndim != 2:
        raise ValueError(f"valid_mask must be (T,J), got {tuple(vm.shape)}")
    T = min(int(pj.shape[0]), int(gj.shape[0]), int(vm.shape[0]))
    J = min(int(pj.shape[1]), int(gj.shape[1]), int(vm.shape[1]))
    if T <= 0 or J <= 0:
        raise ValueError(f"Invalid pred/gt/mask shapes: pred={tuple(pj.shape)} gt={tuple(gj.shape)} mask={tuple(vm.shape)}")
    pj = pj[:T, :J]
    gj = gj[:T, :J]
    vm = vm[:T, :J].bool()

    finite = torch.isfinite(pj).all(dim=-1) & torch.isfinite(gj).all(dim=-1)
    m = vm & finite

    if bool(align_by_pelvis):
        i0, i1 = int(pelvis_idxs[0]), int(pelvis_idxs[1])
        if not (0 <= i0 < J and 0 <= i1 < J):
            raise IndexError(f"pelvis_idxs={pelvis_idxs} out of range for J={J}")
        pelvis_ok = m[:, i0] & m[:, i1]
        if not bool(pelvis_ok.any()):
            err_f = pj.new_full((T,), float("inf")).double()
            keep = torch.isfinite(err_f) & (err_f <= float(max_frame_mm))
            n_total = int(err_f.numel())
            n_keep = int(keep.sum().item())
            loss = err_f.new_tensor(1e6).float()
            stats = {
                "k3d_T": int(T),
                "k3d_J": int(J),
                "k3d_mm_scale": float(mm_scale),
                "k3d_max_frame_mm": float(max_frame_mm),
                "k3d_n_frames_total": int(n_total),
                "k3d_n_frames_kept": int(n_keep),
                "k3d_n_frames_dropped": int(n_total - n_keep),
                "k3d_masked": True,
                "k3d_pelvis_frames_ok": 0,
            }
            return loss, stats
        pred_pelvis = pj[:, [i0, i1]].mean(dim=1, keepdim=True)
        gt_pelvis = gj[:, [i0, i1]].mean(dim=1, keepdim=True)
        pj0 = pj - pred_pelvis
        gj0 = gj - gt_pelvis
        m = m & pelvis_ok[:, None]
    else:
        pj0 = pj
        gj0 = gj

    d = (pj0 - gj0).double()
    err_j = torch.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)  # (T,J)

    w = m.double()
    denom = w.sum(dim=1)  # (T,)
    sum_err = (err_j * w).sum(dim=1)
    err_f = sum_err / torch.clamp(denom, min=1.0)
    err_f = torch.where(denom > 0.0, err_f, err_f.new_full(err_f.shape, float("inf")))
    err_f = err_f * float(mm_scale)

    keep = torch.isfinite(err_f) & (err_f <= float(max_frame_mm))
    n_total = int(err_f.numel())
    n_keep = int(keep.sum().item())
    if n_keep <= 0:
        loss = err_f.new_tensor(1e6).float()
    else:
        loss = err_f[keep].mean().float()

    stats = {
        "k3d_T": int(T),
        "k3d_J": int(J),
        "k3d_mm_scale": float(mm_scale),
        "k3d_max_frame_mm": float(max_frame_mm),
        "k3d_n_frames_total": int(n_total),
        "k3d_n_frames_kept": int(n_keep),
        "k3d_n_frames_dropped": int(n_total - n_keep),
        "k3d_masked": True,
    }
    if bool(align_by_pelvis):
        stats["k3d_pelvis_frames_ok"] = int((m[:, int(pelvis_idxs[0])] & m[:, int(pelvis_idxs[1])]).sum().item())
    return loss, stats


def _smpl_body_pose_to_coco17_2d(
    *,
    body_pose: torch.Tensor,  # (T,D)
    betas: torch.Tensor,  # (T,10) or (10,)
    global_orient: torch.Tensor,  # (T,3)
    transl: torch.Tensor,  # (T,3)
    K_fullimg: torch.Tensor,  # (T,3,3) or (3,3)
    device: torch.device,
    smpl_model=None,
) -> torch.Tensor:
    """
    Return (T,17,2) projected COCO17 keypoints.
    """
    from .geo_transform import project_p2d

    T = int(body_pose.shape[0])
    body_pose = body_pose.to(device).float()
    betas = betas.to(device).float()
    global_orient = global_orient.to(device).float()
    transl = transl.to(device).float()
    if K_fullimg.ndim == 2:
        K = K_fullimg[None].repeat(T, 1, 1).to(device).float()
    else:
        K = K_fullimg.to(device).float()

    if smpl_model is None:
        from .smplx_utils import make_smplx

        smpl_model = make_smplx("supermotion_coco17").to(device).eval()

    joints3d = smpl_model(body_pose=body_pose, betas=betas, global_orient=global_orient, transl=transl)  # (T,17,3)
    joints2d = project_p2d(joints3d, K=K)  # (T,17,2)
    return joints2d


def refine_body_pose_bspline_lbfgs(
    *,
    body_pose: torch.Tensor,  # (T,D) axis-angle in rad (commonly D in {63,69})
    betas: torch.Tensor,  # (T,10) or (10,)
    global_orient: torch.Tensor,  # (T,3)
    transl: torch.Tensor,  # (T,3)
    K_fullimg: torch.Tensor,  # (T,3,3) or (3,3)
    bbx_xys: torch.Tensor,  # (T,3) cx,cy,s (for normalization)
    coco17: torch.Tensor,  # (T,17,3) x,y,conf
    multi_view_joint: bool = False,
    # view1 inputs (required when multi_view_joint=True)
    body_pose_v1: torch.Tensor | None = None,
    betas_v1: torch.Tensor | None = None,
    global_orient_v1: torch.Tensor | None = None,
    transl_v1: torch.Tensor | None = None,
    K_fullimg_v1: torch.Tensor | None = None,
    bbx_xys_v1: torch.Tensor | None = None,
    coco17_v1: torch.Tensor | None = None,
    cfg: BsplineRefineConfig | None = None,
    device: str | torch.device = "cuda",
    optimize_body_pose: bool = True,
    optimize_global_orient: bool = True,
    optimize_transl: bool = True,
    pose_limit_in_loss: bool = False,
    # optional ankle static probability (left/right ankle) and ground normal for extra loss
    ankle_static_prob: torch.Tensor | None = None,  # (T,2) in [0,1]
    ground_normal_cam: torch.Tensor | None = None,  # (3,) or (T,3), camera coords
    ankle_static_prob_v1: torch.Tensor | None = None,
    ground_normal_cam_v1: torch.Tensor | None = None,
    # optional static conf logits (T,6) for static-motion loss
    static_conf_logits: torch.Tensor | None = None,
    static_conf_logits_v1: torch.Tensor | None = None,
    ankle_ground_align_w: float | None = None,
    # new: multi-view inputs for N-view joint optimization (>=2)
    multi_view_data: list[dict[str, Any]] | None = None,
) -> dict:
    """
    返回 dict，包含：
    - body_pose_refined: (T,D) 修正后的 body_pose
    - delta_body_pose: (T,D) 拟合出的修正项 δ(t)
    - u: (D,M) 优化变量
    - c: (D,M) 控制点权重（已软约束到 [-amp, amp]）
    - weights_B: (T,M) B-spline basis matrix
    - stats: 训练前后误差等

    兼容扩展（不改变函数签名）：
    - 若在 cfg 中提供 `k3d_gt`（默认期望 SMPL24: (T,24,3) 相机坐标），则进入 **3D-only** 模式：
      - 忽略 2D 重投影项（coco17/bbx_xys/K_fullimg 仅作为占位输入，不参与损失）
      - 3D 误差使用与 `bss-smplify/validation/eval_3dpw.py` 一致的 MPJPE(mm) 口径：
        pelvis 对齐（默认 hips idx=(1,2)）+ per-frame 丢帧（>cfg.k3d_max_frame_mm）+ 均值
      - 需要 cfg.k3d_pred_fn 提供 “SMPL params -> 3D keypoints” 的可调用对象（本文件不实现该转换）
    """
    if cfg is None:
        cfg = BsplineRefineConfig()
    cfg = cfg.resolve()
    _set_conf_weighting_for_reproj_loss(use_conf_weight=bool(cfg.use_conf_weight), conf_power=float(cfg.conf_power))

    # 3D-only mode is enabled by providing cfg.k3d_gt (no signature change).
    # For minimal-intrusion, we currently support 3D-only optimization ONLY in single-view mode.
    if torch.is_tensor(getattr(cfg, "k3d_gt", None)) and bool(multi_view_joint):
        raise NotImplementedError("cfg.k3d_gt (3D-only mode) is currently supported only when multi_view_joint=False")

    # ====== multi-view joint mode (stereo synchronized) ======
    if multi_view_joint:
        # New N-view path: if provided, use generalized multi-view optimizer.
        if multi_view_data is not None:
            if len(multi_view_data) < 2:
                raise ValueError(f"multi_view_data expects >=2 views, got {len(multi_view_data)}")
            # Ensure required keys exist per view
            req = ["body_pose", "betas", "global_orient", "transl", "K_fullimg", "bbx_xys", "coco17"]
            for i, v in enumerate(multi_view_data):
                missing_k = [k for k in req if k not in v]
                if missing_k:
                    raise ValueError(f"multi_view_data[{i}] missing keys: {missing_k}")
            return _refine_bspline_multi_view_lbfgs(
                views=multi_view_data,
                cfg=cfg,
                device=device,
                optimize_body_pose=optimize_body_pose,
                optimize_global_orient=optimize_global_orient,
                optimize_transl=optimize_transl,
                pose_limit_in_loss=pose_limit_in_loss,
                ankle_ground_align_w=ankle_ground_align_w,
            )

        missing = []
        for name, v in [
            ("body_pose_v1", body_pose_v1),
            ("betas_v1", betas_v1),
            ("global_orient_v1", global_orient_v1),
            ("transl_v1", transl_v1),
            ("K_fullimg_v1", K_fullimg_v1),
            ("bbx_xys_v1", bbx_xys_v1),
            ("coco17_v1", coco17_v1),
        ]:
            if v is None:
                missing.append(name)
        if missing:
            raise ValueError(f"multi_view_joint=True requires: {', '.join(missing)}")

        # Delegate to a dedicated 2-view optimizer to keep single-view code unchanged
        return _refine_bspline_two_view_lbfgs(
            body_pose_v0=body_pose,
            betas_v0=betas,
            global_orient_v0=global_orient,
            transl_v0=transl,
            K_fullimg_v0=K_fullimg,
            bbx_xys_v0=bbx_xys,
            coco17_v0=coco17,
            body_pose_v1=body_pose_v1,
            betas_v1=betas_v1,
            global_orient_v1=global_orient_v1,
            transl_v1=transl_v1,
            K_fullimg_v1=K_fullimg_v1,
            bbx_xys_v1=bbx_xys_v1,
            coco17_v1=coco17_v1,
            cfg=cfg,
            device=device,
            optimize_body_pose=optimize_body_pose,
            optimize_global_orient=optimize_global_orient,
            optimize_transl=optimize_transl,
            pose_limit_in_loss=pose_limit_in_loss,
            ankle_static_prob_v0=ankle_static_prob,
            ground_normal_cam_v0=ground_normal_cam,
            ankle_static_prob_v1=ankle_static_prob_v1,
            ground_normal_cam_v1=ground_normal_cam_v1,
            static_conf_logits_v0=static_conf_logits,
            static_conf_logits_v1=static_conf_logits_v1,
            ankle_ground_align_w=ankle_ground_align_w,
        )

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    body_pose = body_pose.detach().cpu()
    T = int(body_pose.shape[0])
    if body_pose.ndim != 2:
        raise ValueError(f"body_pose must be (T,D), got {tuple(body_pose.shape)}")
    D = int(body_pose.shape[1])
    if D <= 0:
        raise ValueError(f"Invalid body_pose last dim: {D}")

    # Optional 3D-only mode: if cfg.k3d_gt is provided, ignore 2D reprojection loss completely.
    use_k3d = torch.is_tensor(getattr(cfg, "k3d_gt", None))
    k3d_gt = getattr(cfg, "k3d_gt", None)
    k3d_pred_fn = getattr(cfg, "k3d_pred_fn", None)
    if bool(use_k3d):
        if not torch.is_tensor(k3d_gt):
            raise TypeError("cfg.k3d_gt must be a torch.Tensor when provided")
        if k3d_pred_fn is None:
            # Minimal native support: if k3d_kind == '3dpw14', build joints14 predictor consistent with MetricMocap.
            if str(getattr(cfg, "k3d_kind", "")).lower() in ("3dpw14", "threedpw14", "pw3d14"):
                try:
                    from pathlib import Path
                    import torch as _torch
                    body_model_utils_root = Path(__file__).resolve().parent / "body_model"
                    J_path = body_model_utils_root / "smpl_3dpw14_J_regressor_sparse.pt"
                    X2S_path = body_model_utils_root / "smplx2smpl_sparse.pt"
                    if not (J_path.exists() and X2S_path.exists()):
                        raise FileNotFoundError("Missing 3DPW14 regressor assets under multi_view_smpl_optimizer/utils/body_model")

                    from .smplx_utils import make_smplx

                    smplx = make_smplx("supermotion_EVAL3DPW").to(dev).eval()
                    smplx2smpl = _torch.load(str(X2S_path), map_location="cpu").to(dev)
                    J_reg = _torch.load(str(J_path), map_location="cpu").to_dense().float().to(dev)

                    def _smplx_verts_to_smpl_verts(verts_smplx: _torch.Tensor) -> _torch.Tensor:
                        F = int(verts_smplx.shape[0])
                        Vx = int(verts_smplx.shape[1])
                        X = verts_smplx.permute(1, 0, 2).reshape(Vx, F * 3)
                        if getattr(smplx2smpl, "is_sparse", False):
                            Y = _torch.sparse.mm(smplx2smpl, X)
                        else:
                            Y = smplx2smpl @ X
                        Vs = int(Y.shape[0])
                        return Y.reshape(Vs, F, 3).permute(1, 0, 2).contiguous()

                    def k3d_pred_fn(*, body_pose, betas, global_orient, transl, **_kw):
                        out = smplx(body_pose=body_pose, betas=betas, global_orient=global_orient, transl=transl)
                        verts_smpl = _smplx_verts_to_smpl_verts(out.vertices)
                        return _torch.einsum("jv,fvc->fjc", J_reg, verts_smpl)

                    cfg.k3d_pred_fn = k3d_pred_fn  # type: ignore[attr-defined]
                    k3d_pred_fn = k3d_pred_fn
                    # MetricMocap pelvis_idxs=[2,3] for 3DPW14
                    if tuple(getattr(cfg, "k3d_pelvis_idxs", (1, 2))) == (1, 2):
                        cfg.k3d_pelvis_idxs = (2, 3)  # type: ignore[attr-defined]
                except Exception as e:
                    raise ValueError(
                        "3D keypoint mode enabled (cfg.k3d_gt provided) but cfg.k3d_pred_fn is None, "
                        f"and native 3dpw14 predictor failed: {repr(e)}"
                    ) from e
            else:
                raise ValueError("3D keypoint mode enabled (cfg.k3d_gt provided) but cfg.k3d_pred_fn is None")
        # Align T to GT length
        T = min(int(T), int(k3d_gt.shape[0]))

    # choose number of basis functions
    M = int(math.ceil(T / float(cfg.m_per_t)))
    M = max(M, cfg.degree + 2)  # ensure m > degree

    # time samples in [0,1]
    t = torch.linspace(0.0, 1.0, T, dtype=torch.float32)
    B: torch.Tensor | None = None
    B_dev: torch.Tensor | None = None
    u_knot_gaps: torch.Tensor | None = None
    t_dev: torch.Tensor | None = None
    internal_knots_uni: torch.Tensor | None = None
    gaps_uni: torch.Tensor | None = None
    if not bool(cfg.learn_knots):
        B = _bspline_basis_matrix(t, m=M, degree=int(cfg.degree)).float()  # (T,M) on CPU
        B_dev = B.to(dev)
    else:
        t_dev = torch.linspace(0.0, 1.0, T, dtype=torch.float32, device=dev)
        u_knot_gaps = torch.zeros((int(M - int(cfg.degree)),), device=dev, dtype=torch.float32, requires_grad=True)
        internal_knots_uni = _uniform_internal_knots(int(M), int(cfg.degree), device=dev, dtype=torch.float32)
        if int(u_knot_gaps.numel()) > 0:
            gaps_uni = torch.full((int(u_knot_gaps.numel()),), 1.0 / float(int(u_knot_gaps.numel())), device=dev, dtype=torch.float32)
    body0 = body_pose[:T].to(dev).float()
    betas = (betas[:T] if (torch.is_tensor(betas) and betas.ndim == 2) else betas).to(dev)
    glob0 = global_orient[:T].to(dev).float()
    transl0 = transl[:T].to(dev).float()
    if not bool(use_k3d):
        K_fullimg = (K_fullimg[:T] if (torch.is_tensor(K_fullimg) and K_fullimg.ndim == 3) else K_fullimg).to(dev)
        bbx_xys = bbx_xys[:T].to(dev)
        coco17 = coco17[:T].to(dev)
    else:
        # keep locals defined for type checkers; they won't be used
        K_fullimg = K_fullimg
        bbx_xys = bbx_xys
        coco17 = coco17

    params_to_opt = []
    if bool(cfg.learn_knots):
        if u_knot_gaps is None:
            raise RuntimeError("cfg.learn_knots=True but u_knot_gaps is None")
        params_to_opt.append(u_knot_gaps)
    # init u=0 => c=0 => delta=0
    u_body = None
    u_glob = None
    u_tr = None
    u_glob_const = None
    u_tr_const = None
    use_rot6d = bool(cfg.optimize_pose_in_rot6d)
    D_body_opt = int((D // 3) * 6) if use_rot6d else int(D)
    D_glob_opt = 6 if use_rot6d else 3
    if optimize_body_pose:
        u_body = torch.zeros((D_body_opt, M), device=dev, dtype=torch.float32, requires_grad=True)
        params_to_opt.append(u_body)
    if optimize_global_orient:
        if bool(cfg.optimize_root_as_constant_delta):
            u_glob_const = torch.zeros((D_glob_opt,), device=dev, dtype=torch.float32, requires_grad=True)
            params_to_opt.append(u_glob_const)
        else:
            u_glob = torch.zeros((D_glob_opt, M), device=dev, dtype=torch.float32, requires_grad=True)
            params_to_opt.append(u_glob)
    if optimize_transl:
        if bool(cfg.optimize_root_as_constant_delta):
            u_tr_const = torch.zeros((3,), device=dev, dtype=torch.float32, requires_grad=True)
            params_to_opt.append(u_tr_const)
        else:
            u_tr = torch.zeros((3, M), device=dev, dtype=torch.float32, requires_grad=True)
            params_to_opt.append(u_tr)

    if len(params_to_opt) == 0:
        raise ValueError("At least one of optimize_body_pose/optimize_global_orient/optimize_transl must be True.")

    # create SMPL coco17 regressor model once (important for speed) (2D mode only)
    smpl_coco17 = None
    smpl_smpl24 = None
    if not bool(use_k3d):
        from .smplx_utils import make_smplx

        smpl_coco17 = _make_coco17_model(cfg, dev)
        if bool(cfg.static_use_smpl24) and int(D) == 63:
            try:
                smpl_smpl24 = make_smplx("supermotion_smpl24").to(dev).eval()
            except Exception:
                smpl_smpl24 = None

    # per-dimension weights for body_pose prior (joint-wise) (match optimization space)
    body_prior_w = (
        _body_pose_prior_weights_per_dim_rot6d(D_body_opt, device=dev, dtype=torch.float32)
        if use_rot6d
        else _body_pose_prior_weights_per_dim(D, device=dev, dtype=torch.float32)
    )  # (1, D_body_opt) or (1, D)

    # baseline error (no refine)
    with torch.no_grad():
        body0_eval = _maybe_pose_limit(body0, enabled=pose_limit_in_loss, device=dev)
        if bool(use_k3d):
            pred = k3d_pred_fn(
                body_pose=body0_eval,
                betas=betas,
                global_orient=glob0,
                transl=transl0,
                kind=str(getattr(cfg, "k3d_kind", "smpl24")),
                view_id=0,
            )
            if not torch.is_tensor(pred):
                raise TypeError("cfg.k3d_pred_fn must return a torch.Tensor")
            vm = getattr(cfg, "k3d_valid_mask", None)
            if torch.is_tensor(vm):
                loss0, _st0 = _k3d_mpjpe_mm_like_eval_masked(
                    pred_j3d=pred,
                    gt_j3d=k3d_gt[:T].to(dev).float(),
                    valid_mask=vm[:T].to(dev),
                    pelvis_idxs=tuple(getattr(cfg, "k3d_pelvis_idxs", (1, 2))),
                    align_by_pelvis=bool(getattr(cfg, "k3d_align_by_pelvis", True)),
                    max_frame_mm=float(getattr(cfg, "k3d_max_frame_mm", 200.0)),
                    mm_scale=float(getattr(cfg, "k3d_mm_scale", 1000.0)),
                )
            else:
                loss0, _st0 = _k3d_mpjpe_mm_like_eval(
                    pred_j3d=pred,
                    gt_j3d=k3d_gt[:T].to(dev).float(),
                    pelvis_idxs=tuple(getattr(cfg, "k3d_pelvis_idxs", (1, 2))),
                    align_by_pelvis=bool(getattr(cfg, "k3d_align_by_pelvis", True)),
                    max_frame_mm=float(getattr(cfg, "k3d_max_frame_mm", 200.0)),
                    mm_scale=float(getattr(cfg, "k3d_mm_scale", 1000.0)),
                )
            base_err = float(loss0.detach().cpu().item())
            base_k3d_stats = dict(_st0)
        else:
            if smpl_coco17 is None:
                raise RuntimeError("smpl_coco17 is None in 2D mode")
            from .geo_transform import project_p2d

            j3d0 = smpl_coco17(body_pose=body0_eval, betas=betas, global_orient=glob0, transl=transl0)
            smpl2d0 = project_p2d(j3d0, K=K_fullimg)
            base_err = float(_mean_normed_reproj_err(smpl2d0, coco17, bbx_xys, conf_thr=float(cfg.conf_thr)).item())
            base_k3d_stats = None

    opt = torch.optim.LBFGS(params_to_opt, lr=float(cfg.lr), max_iter=int(cfg.max_iter), line_search_fn=cfg.line_search_fn)

    last = {"loss": None}

    def _get_B_and_knots():
        if not bool(cfg.learn_knots):
            if B_dev is None:
                raise RuntimeError("Expected precomputed B_dev when learn_knots=False")
            return B_dev, None, None, None
        if (u_knot_gaps is None) or (t_dev is None):
            raise RuntimeError("Expected u_knot_gaps/t_dev when learn_knots=True")
        U, internal, gaps = _knots_from_u_gaps(
            u_knot_gaps,
            m=int(M),
            degree=int(cfg.degree),
            min_gap=float(cfg.knot_min_gap),
        )
        Bk = _bspline_basis_matrix(t_dev, m=int(M), degree=int(cfg.degree), knots=U).float()
        return Bk, U, internal, gaps

    def _closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        B_use, U_use, internal_use, gaps_use = _get_B_and_knots()
        # body pose (AA or rot6d space)
        if u_body is not None:
            c_body = _u_to_c(u_body, amp=float(cfg.amp_body_pose), use_tanh=bool(cfg.use_tanh))  # (D_body_opt,M) if rot6d else (D,M)
            delta_body = (c_body @ B_use.T).T  # (T,D_body_opt) if rot6d else (T,D)
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(body0)  # (T,D_body_opt)
                body = _rot6d_flat_to_aa(base6d + delta_body)  # (T,D)
            else:
                body = body0 + delta_body
        else:
            c_body = None
            if use_rot6d:
                delta_body = torch.zeros((int(body0.shape[0]), int(D_body_opt)), device=body0.device, dtype=body0.dtype)
            else:
                delta_body = torch.zeros_like(body0)
            body = body0

        # global orient (AA or rot6d space)
        if u_glob is not None:
            c_glob = _u_to_c(u_glob, amp=float(cfg.amp_global_orient), use_tanh=bool(cfg.use_tanh))  # (D_glob_opt,M) if rot6d else (3,M)
            delta_glob = (c_glob @ B_use.T).T  # (T,D_glob_opt) if rot6d else (T,3)
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(glob0)  # (T,6)
                glob = _rot6d_flat_to_aa(base6d + delta_glob)  # (T,3)
            else:
                glob = glob0 + delta_glob
        elif u_glob_const is not None:
            c_glob = None
            d0 = _u_to_c(u_glob_const.view(int(D_glob_opt), 1), amp=float(cfg.amp_global_orient), use_tanh=bool(cfg.use_tanh)).view(1, int(D_glob_opt))
            delta_glob = d0.expand(int(glob0.shape[0]), -1)
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(glob0)  # (T,6)
                glob = _rot6d_flat_to_aa(base6d + delta_glob)  # (T,3)
            else:
                glob = glob0 + delta_glob
        else:
            c_glob = None
            if use_rot6d:
                delta_glob = torch.zeros((int(glob0.shape[0]), int(D_glob_opt)), device=glob0.device, dtype=glob0.dtype)
            else:
                delta_glob = torch.zeros_like(glob0)
            glob = glob0

        # translation
        if u_tr is not None:
            c_tr = _u_to_c(u_tr, amp=float(cfg.amp_transl), use_tanh=bool(cfg.use_tanh))  # (3,M)
            delta_tr = (c_tr @ B_use.T).T  # (T,3)
            tr = transl0 + delta_tr
        elif u_tr_const is not None:
            c_tr = None
            d0 = _u_to_c(u_tr_const.view(3, 1), amp=float(cfg.amp_transl), use_tanh=bool(cfg.use_tanh)).view(1, 3)
            delta_tr = d0.expand(int(transl0.shape[0]), -1)
            tr = transl0 + delta_tr
        else:
            c_tr = None
            delta_tr = torch.zeros_like(transl0)
            tr = transl0

        # reprojection loss (2D) OR k3d loss (3D-only)
        # Optional pose limiter inside objective
        bodym = _maybe_pose_limit(body, enabled=pose_limit_in_loss, device=dev)
        if bool(use_k3d):
            pred = k3d_pred_fn(
                body_pose=bodym,
                betas=betas,
                global_orient=glob,
                transl=tr,
                kind=str(getattr(cfg, "k3d_kind", "smpl24")),
                view_id=0,
            )
            if not torch.is_tensor(pred):
                raise TypeError("cfg.k3d_pred_fn must return a torch.Tensor")
            vm = getattr(cfg, "k3d_valid_mask", None)
            if torch.is_tensor(vm):
                k3d_loss, _st = _k3d_mpjpe_mm_like_eval_masked(
                    pred_j3d=pred,
                    gt_j3d=k3d_gt[:T].to(dev).float(),
                    valid_mask=vm[:T].to(dev),
                    pelvis_idxs=tuple(getattr(cfg, "k3d_pelvis_idxs", (1, 2))),
                    align_by_pelvis=bool(getattr(cfg, "k3d_align_by_pelvis", True)),
                    max_frame_mm=float(getattr(cfg, "k3d_max_frame_mm", 200.0)),
                    mm_scale=float(getattr(cfg, "k3d_mm_scale", 1000.0)),
                )
            else:
                k3d_loss, _st = _k3d_mpjpe_mm_like_eval(
                    pred_j3d=pred,
                    gt_j3d=k3d_gt[:T].to(dev).float(),
                    pelvis_idxs=tuple(getattr(cfg, "k3d_pelvis_idxs", (1, 2))),
                    align_by_pelvis=bool(getattr(cfg, "k3d_align_by_pelvis", True)),
                    max_frame_mm=float(getattr(cfg, "k3d_max_frame_mm", 200.0)),
                    mm_scale=float(getattr(cfg, "k3d_mm_scale", 1000.0)),
                )
            loss = k3d_loss
            cam = getattr(cfg, "k3d_reproj_cam", None)
            w_reproj = float(getattr(cfg, "k3d_reproj_w", 0.0))
            if w_reproj > 0.0 and isinstance(cam, dict):
                K0 = cam.get("K0", None)
                K1 = cam.get("K1", None)
                R10 = cam.get("R10", None)
                t10 = cam.get("t10", None)
                if torch.is_tensor(K0) and torch.is_tensor(K1) and torch.is_tensor(R10) and torch.is_tensor(t10):
                    K0 = K0.to(dev).float()
                    K1 = K1.to(dev).float()
                    R10 = R10.to(dev).float()
                    t10 = t10.to(dev).float().view(1, 1, 3)
                    gt = k3d_gt[:T].to(dev).float()

                    def _proj_pinhole_torch(X: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                        Z = X[..., 2]
                        okz = Z.abs() > 1e-6
                        denom = torch.where(okz, Z, torch.ones_like(Z))
                        x = X[..., 0] / denom
                        y = X[..., 1] / denom
                        u = K[0, 0] * x + K[0, 2]
                        v = K[1, 1] * y + K[1, 2]
                        uv = torch.stack([u, v], dim=-1)
                        ok = okz & torch.isfinite(uv).all(dim=-1)
                        return uv, ok

                    def _to_cam1(Xw: torch.Tensor) -> torch.Tensor:
                        return (Xw @ R10.T) + t10

                    uvp0, okp0 = _proj_pinhole_torch(pred, K0)
                    uvg0, okg0 = _proj_pinhole_torch(gt, K0)
                    uvp1, okp1 = _proj_pinhole_torch(_to_cam1(pred), K1)
                    uvg1, okg1 = _proj_pinhole_torch(_to_cam1(gt), K1)

                    def _mean_l2(uv_a: torch.Tensor, uv_b: torch.Tensor, ok_a: torch.Tensor, ok_b: torch.Tensor) -> torch.Tensor:
                        ok = ok_a & ok_b
                        if not bool(ok.any()):
                            return torch.zeros((), device=uv_a.device, dtype=uv_a.dtype)
                        e = torch.linalg.norm(uv_a - uv_b, dim=-1)
                        return e[ok].mean()

                    reproj0 = _mean_l2(uvp0, uvg0, okp0, okg0)
                    reproj1 = _mean_l2(uvp1, uvg1, okp1, okg1)
                    reproj = 0.5 * (reproj0 + reproj1)
                    loss = loss + w_reproj * reproj
            if float(cfg.static_motion_w) > 0 and static_conf_logits is not None:
                ls = _static_motion_loss(
                    joints17=pred,
                    static_logits=static_conf_logits[:T],
                    cfg=cfg,
                    smpl24_model=None,
                    body_pose=bodym,
                    betas=betas,
                    global_orient=glob,
                    transl=tr,
                )
                loss = loss + float(cfg.static_motion_w) * ls
        else:
            if smpl_coco17 is None:
                raise RuntimeError("smpl_coco17 is None in 2D mode")
            # Forward SMPL to get joints3d for optional ankle loss + joints2d for reproj
            from .geo_transform import project_p2d

            joints3d = smpl_coco17(body_pose=bodym, betas=betas, global_orient=glob, transl=tr)  # (T,17,3)
            smpl2d = project_p2d(joints3d, K=K_fullimg)
            reproj = _mean_normed_reproj_err(smpl2d, coco17, bbx_xys, conf_thr=float(cfg.conf_thr))
            # static-motion loss (enabled when static_conf_logits is provided)
            loss = reproj
            if float(cfg.static_motion_w) > 0.0:
                ls = _static_motion_loss(
                    joints17=joints3d,
                    static_logits=static_conf_logits,
                    cfg=cfg,
                    smpl24_model=smpl_smpl24,
                    body_pose=bodym,
                    betas=betas,
                    global_orient=glob,
                    transl=tr,
                )
                loss = loss + float(cfg.static_motion_w) * ls

        # prior to prevent over-adjust
        if u_body is not None:
            # joint-weighted prior: mean( w * delta^2 )
            loss = loss + float(cfg.prior_w_body_pose) * ((delta_body * delta_body) * body_prior_w).mean()
        if (u_glob is not None) or (u_glob_const is not None):
            loss = loss + float(cfg.prior_w_global_orient) * (delta_glob * delta_glob).mean()
        if (u_tr is not None) or (u_tr_const is not None):
            loss = loss + float(cfg.prior_w_transl) * (delta_tr * delta_tr).mean()

        # optional temporal loss on deltas (velocity + acceleration)
        if float(cfg.temporal_vel_w) > 0.0 or float(cfg.temporal_acc_w) > 0.0:
            if u_body is not None:
                loss = loss + _temporal_vel_acc_loss(
                    delta_body, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
            if (u_glob is not None) or (u_glob_const is not None):
                loss = loss + _temporal_vel_acc_loss(
                    delta_glob, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
            if (u_tr is not None) or (u_tr_const is not None):
                loss = loss + _temporal_vel_acc_loss(
                    delta_tr, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )

        # Optional ankle-ground alignment loss (only when ankle_static_prob is provided) (2D mode only)
        if (not bool(use_k3d)) and (ankle_static_prob is not None):
            if ground_normal_cam is None:
                raise ValueError("ankle_static_prob is provided but ground_normal_cam is None")
            w_ank = float(cfg.ankle_ground_align_w if ankle_ground_align_w is None else ankle_ground_align_w)
            if w_ank > 0:
                pp = ankle_static_prob.to(dev).float()
                if pp.ndim == 3 and int(pp.shape[0]) == 1:
                    pp = pp[0]
                pp = pp[: joints3d.shape[0], :2]
                pp = torch.clamp(pp, 0.0, 1.0)
                p_pair = 0.5 * (pp[:, 0] + pp[:, 1])  # (T,)

                nn = ground_normal_cam.to(dev).float()
                if nn.ndim == 1:
                    nn = nn[None].expand(joints3d.shape[0], -1)
                nn = nn[: joints3d.shape[0]]
                nn = nn / (nn.norm(dim=-1, keepdim=True) + 1e-8)

                v = joints3d[:, 15] - joints3d[:, 16]  # (T,3) left_ankle - right_ankle
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
                dot = (v * nn).sum(dim=-1)
                loss = loss + float(w_ank) * (p_pair * dot * dot).mean()
        if float(cfg.smooth_w) > 0:
            # smoothness on control points (2nd difference over j)
            if M >= 3:
                if c_body is not None:
                    d2 = c_body[:, 2:] - 2.0 * c_body[:, 1:-1] + c_body[:, :-2]
                    loss = loss + float(cfg.smooth_w) * (d2 * d2).mean()
                if c_glob is not None:
                    d2 = c_glob[:, 2:] - 2.0 * c_glob[:, 1:-1] + c_glob[:, :-2]
                    loss = loss + float(cfg.smooth_w) * (d2 * d2).mean()
                if c_tr is not None:
                    d2 = c_tr[:, 2:] - 2.0 * c_tr[:, 1:-1] + c_tr[:, :-2]
                    loss = loss + float(cfg.smooth_w) * (d2 * d2).mean()

        # optional knot priors (only when learn_knots=True)
        if bool(cfg.learn_knots):
            if (internal_knots_uni is None) or (gaps_uni is None):
                raise RuntimeError("Expected internal_knots_uni/gaps_uni when learn_knots=True")
            if (internal_use is None) or (gaps_use is None):
                raise RuntimeError("Expected internal_use/gaps_use when learn_knots=True")
            if int(internal_use.numel()) > 0 and float(cfg.knot_pos_w) > 0.0:
                loss = loss + float(cfg.knot_pos_w) * ((internal_use - internal_knots_uni) ** 2).mean()
            if int(gaps_use.numel()) > 0 and float(cfg.knot_gap_w) > 0.0:
                loss = loss + float(cfg.knot_gap_w) * ((gaps_use - gaps_uni) ** 2).mean()
            if int(internal_use.numel()) >= 3 and float(cfg.knot_smooth_w) > 0.0:
                d2k = internal_use[2:] - 2.0 * internal_use[1:-1] + internal_use[:-2]
                loss = loss + float(cfg.knot_smooth_w) * (d2k * d2k).mean()
        loss.backward()
        last_grad_norm = None
        try:
            gcn = float(getattr(cfg, "grad_clip_norm", 0.0))
            if gcn > 0.0:
                import torch.nn.utils as _tnu

                total_norm = _tnu.clip_grad_norm_(params_to_opt, max_norm=gcn, error_if_nonfinite=False)
                if torch.is_tensor(total_norm):
                    last_grad_norm = float(total_norm.detach().cpu().item())
        except Exception:
            last_grad_norm = None
        last["loss"] = float(loss.detach().item())
        if last_grad_norm is not None:
            last["grad_norm"] = last_grad_norm
        return loss

    if cfg.verbose:
        print(
            f"[Spline-Opt] T={T} M={M} degree={cfg.degree} conf_thr={cfg.conf_thr} "
            f"opt(body,glob,tr)=({optimize_body_pose},{optimize_global_orient},{optimize_transl}) "
            f"learn_knots={bool(cfg.learn_knots)}"
        )
        if bool(use_k3d):
            print(f"[Spline-Opt] baseline mpjpe_mm={base_err:.6f}  k3d={getattr(cfg, 'k3d_kind', 'smpl24')}")
            try:
                if isinstance(base_k3d_stats, dict):
                    nk = int(base_k3d_stats.get("k3d_n_frames_kept", 0))
                    nt = int(base_k3d_stats.get("k3d_n_frames_total", 0))
                    print(f"[Spline-Opt] baseline k3d_frames_kept={nk}/{nt} max_frame_mm={float(getattr(cfg,'k3d_max_frame_mm',200.0))}")
            except Exception:
                pass
        else:
            print(f"[Spline-Opt] baseline mean(||SMPL-GT||/s)={base_err:.6f}")

    t_start = time.perf_counter()
    opt.step(_closure)
    t_end = time.perf_counter()
    if cfg.verbose:
        print(f"[Spline-Opt] optimize_time_sec={t_end - t_start:.3f} (single_view)")

    with torch.no_grad():
        B_use, U_use, internal_use, gaps_use = _get_B_and_knots()
        if u_body is not None:
            c_body = _u_to_c(u_body, amp=float(cfg.amp_body_pose), use_tanh=bool(cfg.use_tanh))
            delta_body = (c_body @ B_use.T).T
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(body0)
                body = _rot6d_flat_to_aa(base6d + delta_body)
            else:
                body = body0 + delta_body
        else:
            c_body = None
            if use_rot6d:
                delta_body = torch.zeros((int(body0.shape[0]), int(D_body_opt)), device=body0.device, dtype=body0.dtype)
            else:
                delta_body = torch.zeros_like(body0)
            body = body0

        if u_glob is not None:
            c_glob = _u_to_c(u_glob, amp=float(cfg.amp_global_orient), use_tanh=bool(cfg.use_tanh))
            delta_glob = (c_glob @ B_use.T).T
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(glob0)
                glob = _rot6d_flat_to_aa(base6d + delta_glob)
            else:
                glob = glob0 + delta_glob
        elif u_glob_const is not None:
            c_glob = None
            d0 = _u_to_c(u_glob_const.view(int(D_glob_opt), 1), amp=float(cfg.amp_global_orient), use_tanh=bool(cfg.use_tanh)).view(1, int(D_glob_opt))
            delta_glob = d0.expand(int(glob0.shape[0]), -1)
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(glob0)
                glob = _rot6d_flat_to_aa(base6d + delta_glob)
            else:
                glob = glob0 + delta_glob
        else:
            c_glob = None
            if use_rot6d:
                delta_glob = torch.zeros((int(glob0.shape[0]), int(D_glob_opt)), device=glob0.device, dtype=glob0.dtype)
            else:
                delta_glob = torch.zeros_like(glob0)
            glob = glob0

        if u_tr is not None:
            c_tr = _u_to_c(u_tr, amp=float(cfg.amp_transl), use_tanh=bool(cfg.use_tanh))
            delta_tr = (c_tr @ B_use.T).T
            tr = transl0 + delta_tr
        elif u_tr_const is not None:
            c_tr = None
            d0 = _u_to_c(u_tr_const.view(3, 1), amp=float(cfg.amp_transl), use_tanh=bool(cfg.use_tanh)).view(1, 3)
            delta_tr = d0.expand(int(transl0.shape[0]), -1)
            tr = transl0 + delta_tr
        else:
            c_tr = None
            delta_tr = torch.zeros_like(transl0)
            tr = transl0

        if bool(use_k3d):
            body_eval = _maybe_pose_limit(body, enabled=pose_limit_in_loss, device=dev)
            pred = k3d_pred_fn(
                body_pose=body_eval,
                betas=betas,
                global_orient=glob,
                transl=tr,
                kind=str(getattr(cfg, "k3d_kind", "smpl24")),
                view_id=0,
            )
            if not torch.is_tensor(pred):
                raise TypeError("cfg.k3d_pred_fn must return a torch.Tensor")
            k3d_loss, final_k3d_stats = _k3d_mpjpe_mm_like_eval(
                pred_j3d=pred,
                gt_j3d=k3d_gt[:T].to(dev).float(),
                pelvis_idxs=tuple(getattr(cfg, "k3d_pelvis_idxs", (1, 2))),
                align_by_pelvis=bool(getattr(cfg, "k3d_align_by_pelvis", True)),
                max_frame_mm=float(getattr(cfg, "k3d_max_frame_mm", 200.0)),
                mm_scale=float(getattr(cfg, "k3d_mm_scale", 1000.0)),
            )
            final_err = float(k3d_loss.detach().cpu().item())
        else:
            if smpl_coco17 is None:
                raise RuntimeError("smpl_coco17 is None in 2D mode")
            from .geo_transform import project_p2d

            body_eval = _maybe_pose_limit(body, enabled=pose_limit_in_loss, device=dev)
            j3d = smpl_coco17(body_pose=body_eval, betas=betas, global_orient=glob, transl=tr)
            smpl2d = project_p2d(j3d, K=K_fullimg)
            final_err = float(_mean_normed_reproj_err(smpl2d, coco17, bbx_xys, conf_thr=float(cfg.conf_thr)).item())
            final_k3d_stats = None

        # Store final basis (and knots if learned)
        B_out = B.detach().cpu() if B is not None else None
        knots_out = None
        internal_out = None
        gaps_out = None
        if bool(cfg.learn_knots):
            if U_use is None:
                raise RuntimeError("Expected U_use when learn_knots=True")
            B_out = _bspline_basis_matrix(t, m=int(M), degree=int(cfg.degree), knots=U_use.detach().cpu()).float().detach().cpu()
            knots_out = U_use.detach().cpu()
            internal_out = internal_use.detach().cpu() if internal_use is not None else None
            gaps_out = gaps_use.detach().cpu() if gaps_use is not None else None
            if cfg.verbose and (internal_out is not None) and int(internal_out.numel()) > 0:
                k8 = internal_out[:8].tolist()
                print(f"[Spline-Opt] learned_knot_internal_first8={k8}")

    if cfg.verbose:
        if bool(use_k3d):
            print(f"[Spline-Opt] final mpjpe_mm={final_err:.6f}  (loss={last['loss']})")
            try:
                if isinstance(final_k3d_stats, dict):
                    nk = int(final_k3d_stats.get("k3d_n_frames_kept", 0))
                    nt = int(final_k3d_stats.get("k3d_n_frames_total", 0))
                    print(f"[Spline-Opt] final k3d_frames_kept={nk}/{nt} max_frame_mm={float(getattr(cfg,'k3d_max_frame_mm',200.0))}")
            except Exception:
                pass
        else:
            print(f"[Spline-Opt] final mean(||SMPL-GT||/s)={final_err:.6f}  (loss={last['loss']})")
        try:
            gcn = float(getattr(cfg, "grad_clip_norm", 0.0))
            if gcn > 0.0:
                lg = last.get("grad_norm", None)
                if isinstance(lg, (int, float)):
                    print(f"[Spline-Opt] grad_clip_norm={gcn} last_grad_norm={float(lg):.6f}")
                else:
                    print(f"[Spline-Opt] grad_clip_norm={gcn} last_grad_norm=None")
        except Exception:
            pass

    # For API compatibility: always report deltas in axis-angle space.
    delta_body_pose_aa = body - body0
    delta_global_orient_aa = glob - glob0
    out = {
        "body_pose_refined": body.detach().cpu(),
        "global_orient_refined": glob.detach().cpu(),
        "transl_refined": tr.detach().cpu(),
        "delta_body_pose": delta_body_pose_aa.detach().cpu(),
        "delta_global_orient": delta_global_orient_aa.detach().cpu(),
        "delta_transl": delta_tr.detach().cpu(),
        "u_body": (u_body.detach().cpu() if u_body is not None else None),
        "u_global_orient": (u_glob.detach().cpu() if u_glob is not None else None),
        "u_transl": (u_tr.detach().cpu() if u_tr is not None else None),
        "u_global_orient_const": (u_glob_const.detach().cpu() if u_glob_const is not None else None),
        "u_transl_const": (u_tr_const.detach().cpu() if u_tr_const is not None else None),
        "c_body": (c_body.detach().cpu() if c_body is not None else None),
        "c_global_orient": (c_glob.detach().cpu() if c_glob is not None else None),
        "c_transl": (c_tr.detach().cpu() if c_tr is not None else None),
        "weights_B": B_out if B_out is not None else torch.empty((0, 0)),
        "stats": {
            "baseline_err": float(base_err),
            "final_err": float(final_err),
            "err_mode": ("k3d_mpjpe_mm" if bool(use_k3d) else "reproj_normed_px"),
            "T": int(T),
            "M": int(M),
            "degree": int(cfg.degree),
            "amp_body_pose": float(cfg.amp_body_pose),
            "amp_global_orient": float(cfg.amp_global_orient),
            "amp_transl": float(cfg.amp_transl),
            "optimize_root_as_constant_delta": bool(cfg.optimize_root_as_constant_delta),
            "grad_clip_norm": float(getattr(cfg, "grad_clip_norm", 0.0)),
            "last_grad_norm": (float(last.get("grad_norm")) if isinstance(last.get("grad_norm", None), (int, float)) else None),
            "D": int(D),
            "learn_knots": bool(cfg.learn_knots),
            "knot_min_gap": float(cfg.knot_min_gap),
            "knot_pos_w": float(cfg.knot_pos_w),
            "knot_gap_w": float(cfg.knot_gap_w),
            "knot_smooth_w": float(cfg.knot_smooth_w),
            "k3d_kind": (str(getattr(cfg, "k3d_kind", "smpl24")) if bool(use_k3d) else None),
            "k3d_base_stats": (base_k3d_stats if bool(use_k3d) else None),
            "k3d_final_stats": (final_k3d_stats if bool(use_k3d) else None),
        },
    }
    if bool(cfg.learn_knots):
        if knots_out is not None:
            out["knots"] = knots_out
        if internal_out is not None:
            out["knot_internal"] = internal_out
        if gaps_out is not None:
            out["knot_gaps"] = gaps_out
    return out


def _refine_bspline_single_view_stage2(
    *,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    K_fullimg: torch.Tensor,
    bbx_xys: torch.Tensor,
    coco17: torch.Tensor,
    cfg: BsplineRefineConfig,
    device: torch.device,
    stage1_out: dict,
    selected_interval_indices: list[int],
    n_intervals: int,
    m_per_t: int,
    insert_knots_per_interval: int = 2,
    optimize_body_pose: bool = True,
    optimize_global_orient: bool = True,
    optimize_transl: bool = True,
    pose_limit_in_loss: bool = False,
    ankle_static_prob: torch.Tensor | None = None,
    ground_normal_cam: torch.Tensor | None = None,
    static_conf_logits: torch.Tensor | None = None,
) -> dict:
    """
    Stage 2: re-optimize with non-uniform knots (insert_knots_per_interval knots per selected interval),
    initializing from Stage 1 trajectory. Single-view 2D only. Used by refine_body_pose_bspline_lbfgs_plus.
    """
    p = int(cfg.degree)
    T = int(body_pose.shape[0])
    D = int(body_pose.shape[1])
    M1 = int(stage1_out["stats"]["M"])
    K1 = M1 - p - 1
    if K1 < 0:
        raise ValueError(f"Stage1 M1={M1} degree={p} -> K1={K1} invalid")
    internal_orig = torch.linspace(0.0, 1.0, K1 + 2, device=device, dtype=torch.float32)[1:-1]
    k_ins = max(0, int(insert_knots_per_interval))
    to_insert: list[float] = []
    for i in selected_interval_indices:
        if i < 0 or i >= n_intervals:
            continue
        n_intervals_safe = max(1, n_intervals)
        for j in range(1, k_ins + 1):
            u = (i + float(j) / (k_ins + 1)) / n_intervals_safe
            to_insert.append(u)
    internal_new = torch.tensor(sorted(list(internal_orig.cpu().tolist()) + to_insert), device=device, dtype=torch.float32)
    U_new = torch.cat(
        [
            torch.zeros((p + 1,), device=device, dtype=torch.float32),
            internal_new,
            torch.ones((p + 1,), device=device, dtype=torch.float32),
        ],
        dim=0,
    )
    M2 = int(internal_new.numel()) + p + 1
    t = torch.linspace(0.0, 1.0, T, device=device, dtype=torch.float32)
    B2 = _bspline_basis_matrix(t, m=M2, degree=p, knots=U_new).float()
    body0 = body_pose[:T].to(device).float()
    betas_d = (betas[:T] if (torch.is_tensor(betas) and betas.ndim == 2) else betas).to(device)
    glob0 = global_orient[:T].to(device).float()
    transl0 = transl[:T].to(device).float()
    K_fullimg = (K_fullimg[:T] if (torch.is_tensor(K_fullimg) and K_fullimg.ndim == 3) else K_fullimg).to(device)
    bbx_xys = bbx_xys[:T].to(device)
    coco17 = coco17[:T].to(device)
    _set_conf_weighting_for_reproj_loss(use_conf_weight=bool(cfg.use_conf_weight), conf_power=float(cfg.conf_power))
    amp_b = float(cfg.amp_body_pose)
    amp_g = float(cfg.amp_global_orient)
    amp_t = float(cfg.amp_transl)
    use_rot6d = bool(cfg.optimize_pose_in_rot6d)
    D = int(body0.shape[1])
    D_body_opt = int((D // 3) * 6) if use_rot6d else int(D)
    D_glob_opt = 6 if use_rot6d else 3

    if use_rot6d:
        body0_6d = _aa_to_rot6d_flat(body0)
        glob0_6d = _aa_to_rot6d_flat(glob0)
        body_s1_6d = _aa_to_rot6d_flat(stage1_out["body_pose_refined"][:T].to(device).float())
        glob_s1_6d = _aa_to_rot6d_flat(stage1_out["global_orient_refined"][:T].to(device).float())
        delta_body_s1 = body_s1_6d - body0_6d
        delta_glob_s1 = glob_s1_6d - glob0_6d
    else:
        delta_body_s1 = (stage1_out["body_pose_refined"][:T] - body_pose[:T]).to(device).float()
        delta_glob_s1 = (stage1_out["global_orient_refined"][:T] - global_orient[:T]).to(device).float()
    delta_transl_s1 = (stage1_out["transl_refined"] - transl[:T]).to(device).float()
    eps_u = 1e-6

    def _fit_initial(delta: torch.Tensor, amp: float) -> torch.Tensor:
        # B2 (T, M2), delta (T, D) => solve B2 @ x = delta, x (M2, D) => c = x.T (D, M2)
        sol = torch.linalg.lstsq(B2, delta, rcond=None).solution
        c = sol.T
        return _c_to_u(c, amp=float(amp), use_tanh=bool(cfg.use_tanh), eps=float(eps_u))

    u_body = _fit_initial(delta_body_s1, amp_b).requires_grad_(True) if optimize_body_pose else None
    u_glob = _fit_initial(delta_glob_s1, amp_g).requires_grad_(True) if optimize_global_orient else None
    u_tr = _fit_initial(delta_transl_s1, amp_t).requires_grad_(True) if optimize_transl else None
    params_to_opt = [x for x in (u_body, u_glob, u_tr) if x is not None]
    if not params_to_opt:
        return stage1_out
    smpl_coco17 = _make_coco17_model(cfg, device)
    body_prior_w = (
        _body_pose_prior_weights_per_dim_rot6d(D_body_opt, device=device, dtype=torch.float32)
        if use_rot6d
        else _body_pose_prior_weights_per_dim(D, device=device, dtype=torch.float32)
    )
    opt = torch.optim.LBFGS(params_to_opt, lr=float(cfg.lr), max_iter=int(cfg.max_iter), line_search_fn=cfg.line_search_fn)
    last = {"loss": None}

    def _closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        B_use = B2
        if u_body is not None:
            c_body = _u_to_c(u_body, amp=float(amp_b), use_tanh=bool(cfg.use_tanh))
            delta_body = (c_body @ B_use.T).T
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(body0)
                body = _rot6d_flat_to_aa(base6d + delta_body)
            else:
                body = body0 + delta_body
        else:
            if use_rot6d:
                delta_body = torch.zeros((int(body0.shape[0]), int(D_body_opt)), device=body0.device, dtype=body0.dtype)
            else:
                delta_body = torch.zeros_like(body0)
            body = body0
        if u_glob is not None:
            c_glob = _u_to_c(u_glob, amp=float(amp_g), use_tanh=bool(cfg.use_tanh))
            delta_glob = (c_glob @ B_use.T).T
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(glob0)
                glob = _rot6d_flat_to_aa(base6d + delta_glob)
            else:
                glob = glob0 + delta_glob
        else:
            if use_rot6d:
                delta_glob = torch.zeros((int(glob0.shape[0]), int(D_glob_opt)), device=glob0.device, dtype=glob0.dtype)
            else:
                delta_glob = torch.zeros_like(glob0)
            glob = glob0
        if u_tr is not None:
            c_tr = _u_to_c(u_tr, amp=float(amp_t), use_tanh=bool(cfg.use_tanh))
            delta_tr = (c_tr @ B_use.T).T
            tr = transl0 + delta_tr
        else:
            delta_tr = torch.zeros_like(transl0)
            tr = transl0
        bodym = _maybe_pose_limit(body, enabled=pose_limit_in_loss, device=device)
        from .geo_transform import project_p2d
        joints3d = smpl_coco17(body_pose=bodym, betas=betas_d, global_orient=glob, transl=tr)
        smpl2d = project_p2d(joints3d, K=K_fullimg)
        reproj = _mean_normed_reproj_err(smpl2d, coco17, bbx_xys, conf_thr=float(cfg.conf_thr))
        loss = reproj
        if u_body is not None:
            loss = loss + float(cfg.prior_w_body_pose) * ((delta_body * delta_body) * body_prior_w).mean()
        if u_glob is not None:
            loss = loss + float(cfg.prior_w_global_orient) * (delta_glob * delta_glob).mean()
        if u_tr is not None:
            loss = loss + float(cfg.prior_w_transl) * (delta_tr * delta_tr).mean()
        if float(cfg.temporal_vel_w) > 0.0 or float(cfg.temporal_acc_w) > 0.0:
            if u_body is not None:
                loss = loss + _temporal_vel_acc_loss(
                    delta_body, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
            if u_glob is not None:
                loss = loss + _temporal_vel_acc_loss(
                    delta_glob, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
            if u_tr is not None:
                loss = loss + _temporal_vel_acc_loss(
                    delta_tr, vel_w=float(cfg.temporal_vel_w), acc_w=float(cfg.temporal_acc_w)
                )
        if float(cfg.smooth_w) > 0 and M2 >= 3:
            if u_body is not None:
                c_body = _u_to_c(u_body, amp=float(amp_b), use_tanh=bool(cfg.use_tanh))
                d2 = c_body[:, 2:] - 2.0 * c_body[:, 1:-1] + c_body[:, :-2]
                loss = loss + float(cfg.smooth_w) * (d2 * d2).mean()
            if u_glob is not None:
                c_glob = _u_to_c(u_glob, amp=float(amp_g), use_tanh=bool(cfg.use_tanh))
                d2 = c_glob[:, 2:] - 2.0 * c_glob[:, 1:-1] + c_glob[:, :-2]
                loss = loss + float(cfg.smooth_w) * (d2 * d2).mean()
            if u_tr is not None:
                c_tr = _u_to_c(u_tr, amp=float(amp_t), use_tanh=bool(cfg.use_tanh))
                d2 = c_tr[:, 2:] - 2.0 * c_tr[:, 1:-1] + c_tr[:, :-2]
                loss = loss + float(cfg.smooth_w) * (d2 * d2).mean()
        if ankle_static_prob is not None and ground_normal_cam is not None:
            w_ank = float(cfg.ankle_ground_align_w)
            if w_ank > 0:
                pp = ankle_static_prob.to(device).float()[:T, :2].clamp(0.0, 1.0)
                p_pair = 0.5 * (pp[:, 0] + pp[:, 1])
                nn = ground_normal_cam.to(device).float()
                if nn.ndim == 1:
                    nn = nn[None].expand(T, -1)
                nn = nn[:T] / (nn.norm(dim=-1, keepdim=True) + 1e-8)
                v = joints3d[:, 15] - joints3d[:, 16]
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
                loss = loss + w_ank * (p_pair * (v * nn).sum(dim=-1).pow(2)).mean()
        if float(cfg.static_motion_w) > 0 and static_conf_logits is not None:
            ls = _static_motion_loss(
                joints17=joints3d,
                static_logits=static_conf_logits[:T],
                cfg=cfg,
                smpl24_model=None,
                body_pose=bodym,
                betas=betas_d,
                global_orient=glob,
                transl=tr,
            )
            loss = loss + float(cfg.static_motion_w) * ls
        loss.backward()
        last["loss"] = float(loss.detach().item())
        return loss

    t_start = time.perf_counter()
    opt.step(_closure)
    t_end = time.perf_counter()
    with torch.no_grad():
        if u_body is not None:
            c_body = _u_to_c(u_body, amp=float(amp_b), use_tanh=bool(cfg.use_tanh))
            delta_body = (c_body @ B2.T).T
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(body0)
                body = _rot6d_flat_to_aa(base6d + delta_body)
            else:
                body = body0 + delta_body
        else:
            body = body0
            if use_rot6d:
                delta_body = torch.zeros((int(body0.shape[0]), int(D_body_opt)), device=body0.device, dtype=body0.dtype)
            else:
                delta_body = torch.zeros_like(body0)
        if u_glob is not None:
            c_glob = _u_to_c(u_glob, amp=float(amp_g), use_tanh=bool(cfg.use_tanh))
            delta_glob = (c_glob @ B2.T).T
            if use_rot6d:
                base6d = _aa_to_rot6d_flat(glob0)
                glob = _rot6d_flat_to_aa(base6d + delta_glob)
            else:
                glob = glob0 + delta_glob
        else:
            glob = glob0
            if use_rot6d:
                delta_glob = torch.zeros((int(glob0.shape[0]), int(D_glob_opt)), device=glob0.device, dtype=glob0.dtype)
            else:
                delta_glob = torch.zeros_like(glob0)
        if u_tr is not None:
            c_tr = _u_to_c(u_tr, amp=float(amp_t), use_tanh=bool(cfg.use_tanh))
            delta_tr = (c_tr @ B2.T).T
            tr = transl0 + delta_tr
        else:
            tr = transl0
            delta_tr = torch.zeros_like(transl0)
        body_eval = _maybe_pose_limit(body, enabled=pose_limit_in_loss, device=device)
        from .geo_transform import project_p2d
        j3d = smpl_coco17(body_pose=body_eval, betas=betas_d, global_orient=glob, transl=tr)
        smpl2d = project_p2d(j3d, K=K_fullimg)
        final_err = float(_mean_normed_reproj_err(smpl2d, coco17, bbx_xys, conf_thr=float(cfg.conf_thr)).item())
    B_out = _bspline_basis_matrix(t.cpu(), m=M2, degree=p, knots=U_new.cpu()).float().cpu()
    # For API compatibility: always report deltas in axis-angle space.
    delta_body_pose_aa = body - body0
    delta_global_orient_aa = glob - glob0
    out = {
        "body_pose_refined": body.detach().cpu(),
        "global_orient_refined": glob.detach().cpu(),
        "transl_refined": tr.detach().cpu(),
        "delta_body_pose": delta_body_pose_aa.detach().cpu(),
        "delta_global_orient": delta_global_orient_aa.detach().cpu(),
        "delta_transl": delta_tr.detach().cpu(),
        "u_body": (u_body.detach().cpu() if u_body is not None else None),
        "u_global_orient": (u_glob.detach().cpu() if u_glob is not None else None),
        "u_transl": (u_tr.detach().cpu() if u_tr is not None else None),
        "c_body": (_u_to_c(u_body, amp=float(amp_b), use_tanh=bool(cfg.use_tanh)).detach().cpu() if u_body is not None else None),
        "c_global_orient": (_u_to_c(u_glob, amp=float(amp_g), use_tanh=bool(cfg.use_tanh)).detach().cpu() if u_glob is not None else None),
        "c_transl": (_u_to_c(u_tr, amp=float(amp_t), use_tanh=bool(cfg.use_tanh)).detach().cpu() if u_tr is not None else None),
        "weights_B": B_out,
        "stats": {
            **stage1_out["stats"],
            "final_err": float(final_err),
            "M": int(M2),
            "stage2_selected_intervals": len(selected_interval_indices),
        },
    }
    print(
        f"[Spline-Opt] Stage2 optimize_time_sec={t_end - t_start:.3f} "
        f"final mean(||SMPL-GT||/s)={final_err:.6f}  (loss={last['loss']})"
    )
    return out
