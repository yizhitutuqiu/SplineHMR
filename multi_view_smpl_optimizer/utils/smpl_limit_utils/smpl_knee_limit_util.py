#!/usr/bin/env python3
"""
把“用 SKEL pose_limits 约束 SMPL/SMPL-X body_pose”的核心逻辑模块化。

设计目标（按你的要求）：
- 模块仅接受一个输入：SMPL body_pose (69D) 或 SMPL-X pose_body (63D)，自动适配；输出同维度。
- 模块内部仅从 npz 缓存加载 M 矩阵（SMPL-local -> SKEL-local 的 3x3），不在线计算。
- 对某个关节如果 M 未命中：直接跳过该关节的约束（保持原始旋转）。

注意：
- 这是“纯 pose 限制器”，不依赖 SMPL/SKEL forward（也不渲染/不读数据集）。
- elbow 的 2DOF（flexion + pro_sup）若想做到你之前的“cross-joint”版本，需要额外的 T-pose joints_ori；
  这里采用 1DOF delta（只 clamp elbow_flexion，保留其它分量），满足“只靠 M”即可运行的要求。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch


def _default_m_cache_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "smpl_skel_M_cache.npz")


def _m_key(*, smpl_joint_idx: int, skel_joint_idx: int) -> str:
    return f"smpl{int(smpl_joint_idx):02d}_skel{int(skel_joint_idx):02d}"


def _load_m_cache_npz(cache_path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(cache_path):
        return {}
    data: dict[str, np.ndarray] = {}
    with np.load(cache_path, allow_pickle=True) as z:
        for k in z.files:
            if k.startswith("meta_"):
                continue
            arr = z[k]
            if isinstance(arr, np.ndarray) and arr.shape == (3, 3):
                data[k] = arr.astype(np.float32)
    return data


def so3_log(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """rotation matrix -> axis-angle (so(3) log map). R: (...,3,3) -> (...,3)"""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((tr - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    w = torch.stack(
        [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    k = 0.5 * theta / (sin_theta + eps)
    return k.unsqueeze(-1) * w


def so3_exp(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """axis-angle -> rotation matrix. w: (...,3) -> (...,3,3)"""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)
    axis = w / (theta + eps)
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros],
        dim=-1,
    ).reshape(w.shape[:-1] + (3, 3))
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(w.shape[:-1] + (3, 3))
    theta_ = theta[..., 0].unsqueeze(-1).unsqueeze(-1)
    s = torch.sin(theta_)[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    c = torch.cos(theta_)[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    return I + s * K + (1.0 - c) * (K @ K)


def Rz(angle: torch.Tensor) -> torch.Tensor:
    """angle: (...,) -> (...,3,3)"""
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack([c, -s, z, s, c, z, z, z, o], dim=-1).reshape(angle.shape + (3, 3))


def _batch_rodrigues(aa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Minimal Rodrigues: aa (N,3) -> (N,3,3) ，避免依赖 smplx。
    """
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)  # (N,1)
    axis = aa / (theta + eps)
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    K = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1).reshape(aa.shape[:-1] + (3, 3))
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(aa.shape[:-1] + (3, 3))
    th = theta[..., 0].unsqueeze(-1).unsqueeze(-1)
    s = torch.sin(th)[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    c = torch.cos(th)[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    return I + s * K + (1.0 - c) * (K @ K)


@dataclass(frozen=True)
class _Limit:
    lo: float
    hi: float


@dataclass(frozen=True)
class _Mapping:
    name: str
    smpl_body_pose_idx: int  # 0..22
    smpl_joint_idx: int      # 1..23 (SMPL joint index)
    skel_joint_idx: int      # 0..23 (SKEL joint index)
    mode: str                # 'walker_knee_z' | 'axis_1dof_delta' | 'axis_2dof_delta' | 'axis_3dof_delta'
    limit1: _Limit
    limit2: _Limit | None = None
    limit3: _Limit | None = None
    axis1_skel_local: torch.Tensor | None = None  # (3,)
    axis2_skel_local: torch.Tensor | None = None  # (3,)
    axis3_skel_local: torch.Tensor | None = None  # (3,)
    flip1: float = 1.0
    flip2: float = 1.0
    flip3: float = 1.0


class SMPLPoseLimiterViaSKEL(torch.nn.Module):
    """
    训练/推理可用的 pose limiter：
    - 输入/输出：(...,63) 或 (...,69) torch.Tensor
    - 仅依赖：缓存 M + 内置 SKEL limits/axes（不在线算 M）
    """

    def __init__(self, *, m_cache_path: str | None = None, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.m_cache_path = m_cache_path or _default_m_cache_path()

        cache = _load_m_cache_npz(self.m_cache_path)
        # register as buffers (not params)
        mats = {}
        for k, v in cache.items():
            mats[k] = torch.from_numpy(v).to(self.device)
        self._mats = mats  # python dict of tensors

        # ===== 约束配置（来自 SKEL/kin_skel.py + SKEL/skel_model.py 中的 axis 定义）=====
        pi = float(np.pi)

        # knees (WalkerKnee): knee_angle_* in [0, 3/4*pi]
        knee_lim = _Limit(0.0, 0.75 * pi)

        # elbows (ulna): only clamp elbow_flexion_* in [0, 3/4*pi] using ulna axis
        elbow_flex_lim = _Limit(0.0, 0.75 * pi)
        # ulna axes from SKEL/skel_model.py:
        # ulna_r axis [[0.0494, 0.0366, 0.99810825]] flip=1
        # ulna_l axis [[-0.0494, -0.0366, 0.99810825]] flip=1
        ulna_r_axis = torch.tensor([0.0494, 0.0366, 0.99810825], dtype=torch.float32, device=self.device)
        ulna_l_axis = torch.tensor([-0.0494, -0.0366, 0.99810825], dtype=torch.float32, device=self.device)
        ulna_r_axis = ulna_r_axis / (torch.linalg.norm(ulna_r_axis) + 1e-8)
        ulna_l_axis = ulna_l_axis / (torch.linalg.norm(ulna_l_axis) + 1e-8)

        # wrists (hand): 2DOF clamp: flexion in [-pi/2,pi/2], deviation in [-pi/4,pi/4]
        wrist_flex_lim = _Limit(-0.5 * pi, 0.5 * pi)
        wrist_dev_lim = _Limit(-0.25 * pi, 0.25 * pi)
        # hand_r axis [[1,0,0],[0,0,-1]] flip=[1,1]
        # hand_l axis [[-1,0,0],[0,0,-1]] flip=[1,1]
        hand_r_a1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        hand_r_a2 = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        hand_l_a1 = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        hand_l_a2 = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)

        def _u(v: torch.Tensor) -> torch.Tensor:
            return v / (torch.linalg.norm(v) + 1e-8)

        # thorax/head (ConstantCurvatureJoint): 3DOF, each in [-pi/4, pi/4]
        cc_lim = _Limit(-0.25 * pi, 0.25 * pi)
        # axis definitions from SKEL/skel_model.py:
        # ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1,1,1])
        cc_a1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        cc_a2 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
        cc_a3 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)

        self.mappings: list[_Mapping] = [
            # knees: SMPL body_pose idx: L_Knee=3 (joint 4), R_Knee=4 (joint 5)
            _Mapping(
                name="R_Knee",
                smpl_body_pose_idx=4,
                smpl_joint_idx=5,
                skel_joint_idx=2,
                mode="walker_knee_z",
                limit1=knee_lim,
            ),
            _Mapping(
                name="L_Knee",
                smpl_body_pose_idx=3,
                smpl_joint_idx=4,
                skel_joint_idx=7,
                mode="walker_knee_z",
                limit1=knee_lim,
            ),
            # thorax/head: SMPL Neck=joint 12 (body_pose idx 11) -> SKEL thorax=12
            _Mapping(
                name="Neck_3dof",
                smpl_body_pose_idx=11,
                smpl_joint_idx=12,
                skel_joint_idx=12,
                mode="axis_3dof_delta",
                limit1=cc_lim,
                limit2=cc_lim,
                limit3=cc_lim,
                axis1_skel_local=_u(cc_a1),
                axis2_skel_local=_u(cc_a2),
                axis3_skel_local=_u(cc_a3),
                flip1=1.0,
                flip2=1.0,
                flip3=1.0,
            ),
            # SMPL Head=joint 15 (body_pose idx 14) -> SKEL head=13
            _Mapping(
                name="Head_3dof",
                smpl_body_pose_idx=14,
                smpl_joint_idx=15,
                skel_joint_idx=13,
                mode="axis_3dof_delta",
                limit1=cc_lim,
                limit2=cc_lim,
                limit3=cc_lim,
                axis1_skel_local=_u(cc_a1),
                axis2_skel_local=_u(cc_a2),
                axis3_skel_local=_u(cc_a3),
                flip1=1.0,
                flip2=1.0,
                flip3=1.0,
            ),
            # elbows: SMPL body_pose idx: L_Elbow=17 (joint 18), R_Elbow=18 (joint 19)
            _Mapping(
                name="R_Elbow_flex",
            smpl_body_pose_idx=18,
                smpl_joint_idx=19,
            skel_joint_idx=16,
                mode="axis_1dof_delta",
                limit1=elbow_flex_lim,
                axis1_skel_local=_u(ulna_r_axis),
                flip1=1.0,
            ),
            _Mapping(
                name="L_Elbow_flex",
            smpl_body_pose_idx=17,
                smpl_joint_idx=18,
            skel_joint_idx=21,
                mode="axis_1dof_delta",
                limit1=elbow_flex_lim,
                axis1_skel_local=_u(ulna_l_axis),
                flip1=1.0,
            ),
            # wrists: SMPL body_pose idx: L_Wrist=19 (joint 20), R_Wrist=20 (joint 21)
            _Mapping(
                name="R_Wrist_2dof",
            smpl_body_pose_idx=20,
                smpl_joint_idx=21,
            skel_joint_idx=18,
            mode="axis_2dof_delta",
                limit1=wrist_flex_lim,
                limit2=wrist_dev_lim,
                axis1_skel_local=_u(hand_r_a1),
                axis2_skel_local=_u(hand_r_a2),
                flip1=1.0,
                flip2=1.0,
            ),
            _Mapping(
                name="L_Wrist_2dof",
            smpl_body_pose_idx=19,
                smpl_joint_idx=20,
            skel_joint_idx=23,
            mode="axis_2dof_delta",
                limit1=wrist_flex_lim,
                limit2=wrist_dev_lim,
                axis1_skel_local=_u(hand_l_a1),
                axis2_skel_local=_u(hand_l_a2),
                flip1=1.0,
                flip2=1.0,
            ),
        ]

    def _get_M(self, smpl_joint_idx: int, skel_joint_idx: int) -> torch.Tensor | None:
        k = _m_key(smpl_joint_idx=smpl_joint_idx, skel_joint_idx=skel_joint_idx)
        return self._mats.get(k, None)

    def forward(self, body_pose: torch.Tensor) -> torch.Tensor:
        """
        body_pose: (...,63) or (...,69)
        return: same shape, clamped.
        """
        if not torch.is_tensor(body_pose):
            body_pose = torch.as_tensor(body_pose, dtype=torch.float32)
        x = body_pose.to(device=self.device, dtype=torch.float32)

        orig_last = int(x.shape[-1])
        if orig_last not in (63, 69):
            raise ValueError(f"body_pose last dim must be 63 or 69, got {tuple(x.shape)}")

        # to (...,69)
        if orig_last == 63:
            pad = torch.zeros(x.shape[:-1] + (6,), device=x.device, dtype=x.dtype)
            x69 = torch.cat([x, pad], dim=-1)
        else:
            x69 = x

        flat = x69.reshape(-1, 23, 3).clone()  # (N,23,3)
        N = flat.shape[0]

        for mp in self.mappings:
            M = self._get_M(mp.smpl_joint_idx, mp.skel_joint_idx)
            if M is None:
                # cache miss -> skip this joint
                continue
            M = M.to(device=flat.device, dtype=flat.dtype)  # (3,3)
            Mb = M.unsqueeze(0).expand(N, -1, -1)

            aa = flat[:, mp.smpl_body_pose_idx, :]  # (N,3)
            R_smpl = _batch_rodrigues(aa.reshape(-1, 3)).reshape(-1, 3, 3)  # (N,3,3)
            R_skel = Mb @ R_smpl @ Mb.transpose(1, 2)
            if mp.mode == "walker_knee_z":
                q_before = -torch.atan2(R_skel[:, 1, 0], R_skel[:, 0, 0])  # (N,)
                q_after = torch.clamp(q_before, mp.limit1.lo, mp.limit1.hi)
                R_skel_clamped = Rz(-q_after)
            elif mp.mode == "axis_1dof_delta":
                axis = mp.axis1_skel_local.to(device=flat.device, dtype=flat.dtype).view(1, 3)
                w = so3_log(R_skel)  # (N,3)
                q_before = (w * axis).sum(dim=-1) / float(mp.flip1)
                q_after = torch.clamp(q_before, mp.limit1.lo, mp.limit1.hi)
                dq = (q_after - q_before)
                w_delta = (dq * float(mp.flip1)).unsqueeze(-1) * axis  # (N,3)
                R_delta = so3_exp(w_delta)
                R_skel_clamped = R_delta @ R_skel
            elif mp.mode == "axis_2dof_delta":
                assert mp.limit2 is not None
                a1 = mp.axis1_skel_local.to(device=flat.device, dtype=flat.dtype)
                a2 = mp.axis2_skel_local.to(device=flat.device, dtype=flat.dtype)
                b1 = a1 * float(mp.flip1)
                b2 = a2 * float(mp.flip2)
                w = so3_log(R_skel)  # (N,3)
                A = torch.stack([b1, b2], dim=-1)  # (3,2)
                pinvA = torch.linalg.pinv(A)  # (2,3)
                q_before = (pinvA @ w.transpose(0, 1)).transpose(0, 1)  # (N,2)
                q1 = torch.clamp(q_before[:, 0], mp.limit1.lo, mp.limit1.hi)
                q2 = torch.clamp(q_before[:, 1], mp.limit2.lo, mp.limit2.hi)
                q_after = torch.stack([q1, q2], dim=-1)
                dq = q_after - q_before
                w_delta = dq[:, 0:1] * b1.view(1, 3) + dq[:, 1:2] * b2.view(1, 3)
                R_delta = so3_exp(w_delta)
                R_skel_clamped = R_delta @ R_skel
            elif mp.mode == "axis_3dof_delta":
                assert mp.limit2 is not None and mp.limit3 is not None
                assert mp.axis1_skel_local is not None and mp.axis2_skel_local is not None and mp.axis3_skel_local is not None
                a1 = mp.axis1_skel_local.to(device=flat.device, dtype=flat.dtype)
                a2 = mp.axis2_skel_local.to(device=flat.device, dtype=flat.dtype)
                a3 = mp.axis3_skel_local.to(device=flat.device, dtype=flat.dtype)
                b1 = a1 * float(mp.flip1)
                b2 = a2 * float(mp.flip2)
                b3 = a3 * float(mp.flip3)
                w = so3_log(R_skel)  # (N,3)
                A = torch.stack([b1, b2, b3], dim=-1)  # (3,3)
                pinvA = torch.linalg.pinv(A)  # (3,3)
                q_before = (pinvA @ w.transpose(0, 1)).transpose(0, 1)  # (N,3)
                q1 = torch.clamp(q_before[:, 0], mp.limit1.lo, mp.limit1.hi)
                q2 = torch.clamp(q_before[:, 1], mp.limit2.lo, mp.limit2.hi)
                q3 = torch.clamp(q_before[:, 2], mp.limit3.lo, mp.limit3.hi)
                q_after = torch.stack([q1, q2, q3], dim=-1)
                dq = q_after - q_before
                w_delta = (
                    dq[:, 0:1] * b1.view(1, 3)
                    + dq[:, 1:2] * b2.view(1, 3)
                    + dq[:, 2:3] * b3.view(1, 3)
                )
                R_delta = so3_exp(w_delta)
                R_skel_clamped = R_delta @ R_skel
            else:
                raise RuntimeError(f"Unknown mode: {mp.mode}")

            R_smpl_clamped = Mb.transpose(1, 2) @ R_skel_clamped @ Mb
            aa_new = so3_log(R_smpl_clamped)
            flat[:, mp.smpl_body_pose_idx, :] = aa_new

        out69 = flat.reshape(x69.shape)
        return out69[..., :orig_last]


def _make_test_pose(dim: int, device: torch.device) -> torch.Tensor:
    """
    生成一组“明显越界”的测试 pose：
    - knees: body_pose idx 3/4
    - elbows: idx 17/18
    - wrists: idx 19/20
    """
    if dim not in (63, 69):
        raise ValueError(dim)
    x = torch.zeros((1, dim), dtype=torch.float32, device=device)
    # knees (exceed)
    # 这里直接把 axis-angle 塞大一点，目的是触发 clamp（不追求生理正确）
    if dim == 63:
        # 63D 是 SMPL-X pose_body，不包含 wrists/elbows 的最后6维 SMPL joints 22/23；但 21 joints 仍在
        pass
    # Use 69D indexing semantics (23x3)
    x69 = x if dim == 69 else torch.cat([x, torch.zeros((1, 6), device=device)], dim=-1)
    aa = x69.view(1, 23, 3)
    aa[0, 3, :] = torch.tensor([-2.8, 2.0, 1.0], device=device)    # L_Knee (idx3)
    aa[0, 4, :] = torch.tensor([-2.8, 1.0, -2.0], device=device)   # R_Knee (idx4)
    aa[0, 17, :] = torch.tensor([0.0, 3.0, 0.0], device=device)   # L_Elbow (idx17)
    aa[0, 18, :] = torch.tensor([0.0, -3.0, 0.0], device=device)  # R_Elbow (idx18)
    aa[0, 19, :] = torch.tensor([2.0, 1.0, 2.0], device=device)   # L_Wrist (idx19)
    aa[0, 20, :] = torch.tensor([2.0, -3.0, -1.5], device=device) # R_Wrist (idx20)
    aa[0, 11, :] = torch.tensor([0.0, 0.0, 3.0], device=device)   # Neck (idx11)
    aa[0, 14, :] = torch.tensor([0.0, 3.0, 0.0], device=device)   # Head (idx14)
    out = aa.reshape(1, 69)[..., :dim]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--m_cache", type=str, default=None, help="M cache npz 路径（默认 smpl_experiment/smpl_skel_M_cache.npz）")
    parser.add_argument("--dim", type=int, default=69, choices=[63, 69], help="测试输入维度：63(SMPL-X pose_body) 或 69(SMPL body_pose)")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    limiter = SMPLPoseLimiterViaSKEL(m_cache_path=args.m_cache, device=dev)

    x = _make_test_pose(int(args.dim), dev)
    y = limiter(x)

    # 打印关键关节前后对比（用 69 的 joint 索引语义）
    x69 = x if x.shape[-1] == 69 else torch.cat([x, torch.zeros((1, 6), device=dev)], dim=-1)
    y69 = y if y.shape[-1] == 69 else torch.cat([y, torch.zeros((1, 6), device=dev)], dim=-1)
    xaa = x69.view(1, 23, 3)[0]
    yaa = y69.view(1, 23, 3)[0]

    print(f"[OK] loaded M cache from: {limiter.m_cache_path}  (keys={len(limiter._mats)})")
    for name, idx in [
        ("L_Knee", 3),
        ("R_Knee", 4),
        ("Neck", 11),
        ("Head", 14),
        ("L_Elbow", 17),
        ("R_Elbow", 18),
        ("L_Wrist", 19),
        ("R_Wrist", 20),
    ]:
        print(f"{name} body_pose_idx={idx}: {xaa[idx].detach().cpu().numpy()} -> {yaa[idx].detach().cpu().numpy()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


