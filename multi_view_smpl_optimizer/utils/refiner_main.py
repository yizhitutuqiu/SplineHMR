"""
Refiner 统一入口：从 bspline_body_pose_refiner 导入并封装。

- refine_body_pose_bspline_lbfgs: 与 bspline_body_pose_refiner 中完全相同的函数（直接导入）。
- BsplineRefineConfig: 与 bspline_body_pose_refiner 中完全相同的配置类（直接导入）。
- refine_body_pose_bspline_lbfgs_plus: 与 refine_body_pose_bspline_lbfgs 完全同签名；
  实现两阶段「时间解耦」优化：Stage 1 极稀疏节点全局吸收 → 按区间误差选高误差区间 →
  非均匀节点插入 → Stage 2 全局 LBFGS 再求解。
"""

from __future__ import annotations

from typing import Any

from .bspline_body_pose_refiner import (
    BsplineRefineConfig,
    refine_body_pose_bspline_lbfgs,
    _per_frame_normed_reproj_err,
    _refine_bspline_single_view_stage2,
    _make_coco17_model,
)
from .geo_transform import project_p2d

__all__ = [
    "refine_body_pose_bspline_lbfgs",
    "BsplineRefineConfig",
    "refine_body_pose_bspline_lbfgs_plus",
]


# Default threshold and top fraction for plus (temporal decoupling)
_PLUS_TAU = 0.1
_PLUS_TOP_FRAC = 0.5


def refine_body_pose_bspline_lbfgs_plus(
    *,
    body_pose: Any,
    betas: Any,
    global_orient: Any,
    transl: Any,
    K_fullimg: Any,
    bbx_xys: Any,
    coco17: Any,
    multi_view_joint: bool = False,
    body_pose_v1: Any = None,
    betas_v1: Any = None,
    global_orient_v1: Any = None,
    transl_v1: Any = None,
    K_fullimg_v1: Any = None,
    bbx_xys_v1: Any = None,
    coco17_v1: Any = None,
    cfg: Any = None,
    device: Any = "cuda",
    optimize_body_pose: bool = True,
    optimize_global_orient: bool = True,
    optimize_transl: bool = True,
    pose_limit_in_loss: bool = False,
    ankle_static_prob: Any = None,
    ground_normal_cam: Any = None,
    ankle_static_prob_v1: Any = None,
    ground_normal_cam_v1: Any = None,
    static_conf_logits: Any = None,
    static_conf_logits_v1: Any = None,
    ankle_ground_align_w: Any = None,
    multi_view_data: Any = None,
) -> dict:
    """
    与 refine_body_pose_bspline_lbfgs 完全同签名的两阶段（时间解耦）版本。

    Stage 1: 极稀疏节点，由 cfg.m_per_t 决定段长；典型用法为 m_per_t = fps（即 1 秒一段）由调用方传入。
    Stage 2: 按 m_per_t 帧为一段的区间平均重投影误差，选出 E > τ 的区间中 Top 50%，
             在这些区间内插入节点后再做一次全局 LBFGS。
    注意：3DPW 等 pipeline 在启用 bspline_plus 时会传 m_per_t=fps（1 秒）；非 plus 时多为 fps/2。
    """
    if cfg is None:
        cfg = BsplineRefineConfig()
    cfg = cfg.resolve()

    # Stage 1: 与原始接口一致的一遍优化
    out1 = refine_body_pose_bspline_lbfgs(
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        transl=transl,
        K_fullimg=K_fullimg,
        bbx_xys=bbx_xys,
        coco17=coco17,
        multi_view_joint=multi_view_joint,
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
        ankle_static_prob=ankle_static_prob,
        ground_normal_cam=ground_normal_cam,
        ankle_static_prob_v1=ankle_static_prob_v1,
        ground_normal_cam_v1=ground_normal_cam_v1,
        static_conf_logits=static_conf_logits,
        static_conf_logits_v1=static_conf_logits_v1,
        ankle_ground_align_w=ankle_ground_align_w,
        multi_view_data=multi_view_data,
    )

    # 仅单视角 2D 且未启用 k3d 时做 Stage 2
    if multi_view_joint or (hasattr(cfg, "k3d_gt") and getattr(cfg, "k3d_gt") is not None):
        return out1

    import torch
    T = int(body_pose.shape[0])
    m_per_t = int(getattr(cfg, "m_per_t", 10))
    if isinstance(m_per_t, float):
        m_per_t = max(1, int(round(m_per_t)))
    n_intervals = max(1, (T + m_per_t - 1) // m_per_t)

    # 用 Stage 1 的 body_pose_refined 算每帧重投影误差
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    smpl_coco17 = _make_coco17_model(cfg, dev)
    bp = out1["body_pose_refined"].to(dev)
    go = out1["global_orient_refined"].to(dev)
    tr = out1["transl_refined"].to(dev)
    K = (K_fullimg[:T] if (torch.is_tensor(K_fullimg) and K_fullimg.ndim == 3) else K_fullimg).to(dev)
    bbx = bbx_xys[:T].to(dev)
    coco = coco17[:T].to(dev)
    betas_d = (betas[:T] if (torch.is_tensor(betas) and betas.ndim == 2) else betas).to(dev)
    with torch.no_grad():
        j3d = smpl_coco17(body_pose=bp, betas=betas_d, global_orient=go, transl=tr)
        smpl2d = project_p2d(j3d, K=K)
    E_frame = _per_frame_normed_reproj_err(smpl2d, coco, bbx, conf_thr=float(cfg.conf_thr), use_max=False)

    # 每 1 秒（m_per_t 帧）区间内的平均误差
    E_interval = []
    for i in range(n_intervals):
        start = i * m_per_t
        end = min((i + 1) * m_per_t, T)
        if start >= end:
            continue
        E_interval.append((i, float(E_frame[start:end].mean().item())))
    if not E_interval:
        print("[refiner_plus] Stage2 skipped: no intervals (T or m_per_t). Result same as single-stage.")
        return out1

    # 筛选：E > τ 的区间，再取其中 Top 50%
    tau = float(getattr(cfg, "refiner_plus_tau", _PLUS_TAU))
    top_frac = float(getattr(cfg, "refiner_plus_top_frac", _PLUS_TOP_FRAC))
    above = [(i, e) for i, e in E_interval if e > tau]
    if not above:
        emax = max(e for _, e in E_interval) if E_interval else 0.0
        print(
            f"[refiner_plus] Stage2 skipped: no interval with E_global(t) > tau={tau} "
            f"(max_interval_err={emax:.4f}). Result same as single-stage."
        )
        return out1
    above.sort(key=lambda x: -x[1])
    n_take = max(1, int(len(above) * top_frac))
    selected = [x[0] for x in above[:n_take]]
    print(
        f"[refiner_plus] E_global(t)>tau={tau} intervals: {len(above)}, "
        f"selected top {int(top_frac*100)}%: {len(selected)} intervals (indices={selected[:10]}{'...' if len(selected)>10 else ''})"
    )

    insert_knots = max(0, int(getattr(cfg, "refiner_plus_insert_knots", 2)))
    out2 = _refine_bspline_single_view_stage2(
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        transl=transl,
        K_fullimg=K_fullimg,
        bbx_xys=bbx_xys,
        coco17=coco17,
        cfg=cfg,
        device=dev,
        stage1_out=out1,
        selected_interval_indices=selected,
        n_intervals=n_intervals,
        m_per_t=m_per_t,
        insert_knots_per_interval=insert_knots,
        optimize_body_pose=optimize_body_pose,
        optimize_global_orient=optimize_global_orient,
        optimize_transl=optimize_transl,
        pose_limit_in_loss=pose_limit_in_loss,
        ankle_static_prob=ankle_static_prob,
        ground_normal_cam=ground_normal_cam,
        static_conf_logits=static_conf_logits,
    )
    return out2
