from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


@dataclass
class BetaByLimbConfig:
    enable: bool = True
    betas_dim: int = 10
    betas_init_prior_w: float = 10.0
    max_iter: int = 60
    lr: float = 1.0
    line_search_fn: str | None = "strong_wolfe"
    limb_w: float = 1000.0
    betas_l2_w: float = 0.002
    betas_delta_amp: float = 3.0
    pelvis_idxs: tuple[int, int] = (11, 12)
    align_by_pelvis: bool = True
    max_frame_mpjpe_mm: float = 500.0
    mm_scale: float = 1000.0
    min_frames_per_limb: int = 20
    limb_inlier_min_ratio: float = 0.5
    limb_inlier_max_ratio: float = 1.5
    device: str | torch.device = "cuda"
    verbose: bool = True


def _kintree_for_kind(kind: str) -> list[tuple[int, int]]:
    k = str(kind).lower().strip()
    if k in {"wholebody23", "wb23", "openpose23"}:
        return [
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 6),
            (11, 12),
            (5, 11),
            (6, 12),
            (15, 17),
            (15, 18),
            (15, 19),
            (16, 20),
            (16, 21),
            (16, 22),
        ]
    return [
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (11, 12),
        (5, 11),
        (6, 12),
    ]


def _as_bool_mask(x: Any, *, shape: tuple[int, int]) -> torch.Tensor:
    if x is None:
        return torch.ones(shape, dtype=torch.bool)
    if torch.is_tensor(x):
        m = x.detach().cpu().bool()
    else:
        m = torch.as_tensor(np.asarray(x), dtype=torch.bool)
    if m.ndim != 2 or tuple(m.shape) != tuple(shape):
        return torch.ones(shape, dtype=torch.bool)
    return m


def _compute_limb_lengths(
    *,
    joints3d: torch.Tensor,
    valid_mask: torch.Tensor,
    kintree: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    j3d = joints3d.detach().cpu().float()
    T, J, _ = int(j3d.shape[0]), int(j3d.shape[1]), int(j3d.shape[2])
    vm = valid_mask.detach().cpu().bool()
    L = int(len(kintree))
    lengths = torch.full((T, L), float("nan"), dtype=torch.float32)
    mask = torch.zeros((T, L), dtype=torch.bool)
    for li, (a, b) in enumerate(kintree):
        if not (0 <= int(a) < J and 0 <= int(b) < J):
            continue
        va = vm[:, int(a)]
        vb = vm[:, int(b)]
        v = va & vb
        if not bool(v.any()):
            continue
        d = j3d[:, int(b), :] - j3d[:, int(a), :]
        l = torch.linalg.norm(d, dim=-1)
        ok = torch.isfinite(l) & v
        lengths[ok, li] = l[ok]
        mask[:, li] = ok
    return lengths, mask


def _robust_target_lengths(
    *,
    limb_lengths: torch.Tensor,
    limb_mask: torch.Tensor,
    min_ratio: float,
    max_ratio: float,
    min_frames_per_limb: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ll = limb_lengths.detach().cpu().float()
    m = limb_mask.detach().cpu().bool()
    T, L = int(ll.shape[0]), int(ll.shape[1])
    target = torch.full((L,), float("nan"), dtype=torch.float32)
    use = torch.zeros((L,), dtype=torch.bool)
    for li in range(L):
        x = ll[:, li]
        ok = m[:, li] & torch.isfinite(x)
        if int(ok.sum().item()) < int(min_frames_per_limb):
            continue
        med = torch.nanmedian(x[ok])
        if not torch.isfinite(med):
            continue
        inl = ok & (x >= med * float(min_ratio)) & (x <= med * float(max_ratio))
        if int(inl.sum().item()) < int(min_frames_per_limb):
            inl = ok
        targ = torch.nanmedian(x[inl])
        if not torch.isfinite(targ):
            continue
        target[li] = targ
        use[li] = True
    return target, use


def _mpjpe_mm(
    *,
    pred_j3d: torch.Tensor,
    gt_j3d: torch.Tensor,
    valid_mask: torch.Tensor,
    pelvis_idxs: tuple[int, int],
    align_by_pelvis: bool,
    max_frame_mm: float,
    mm_scale: float,
) -> tuple[float, dict[str, Any]]:
    p = pred_j3d.detach().cpu().float()
    g = gt_j3d.detach().cpu().float()
    T = int(min(p.shape[0], g.shape[0]))
    J = int(min(p.shape[1], g.shape[1]))
    p = p[:T, :J]
    g = g[:T, :J]
    vm = valid_mask.detach().cpu().bool()[:T, :J]
    finite = torch.isfinite(p).all(dim=-1) & torch.isfinite(g).all(dim=-1)
    m = vm & finite
    if not bool(m.any()):
        return float("nan"), {"frames_kept": 0, "frames_total": int(T)}

    if bool(align_by_pelvis):
        a, b = int(pelvis_idxs[0]), int(pelvis_idxs[1])
        if 0 <= a < J and 0 <= b < J:
            ok_p = m[:, a] & m[:, b]
            if bool(ok_p.any()):
                pelvis_p = (p[:, a, :] + p[:, b, :]) * 0.5
                pelvis_g = (g[:, a, :] + g[:, b, :]) * 0.5
                p = p - pelvis_p[:, None, :]
                g = g - pelvis_g[:, None, :]

    diff = (p - g) * float(mm_scale)
    dist = torch.linalg.norm(diff, dim=-1)
    dist = torch.where(m, dist, torch.tensor(float("nan")))
    frame = torch.nanmean(dist, dim=1)
    keep = torch.isfinite(frame) & (frame <= float(max_frame_mm))
    if not bool(keep.any()):
        return float("nan"), {"frames_kept": 0, "frames_total": int(T)}
    mpjpe = float(torch.nanmean(frame[keep]).item())
    return mpjpe, {"frames_kept": int(keep.sum().item()), "frames_total": int(T)}


def _limb_err_mm_stats(
    *,
    pred_len: torch.Tensor,
    target_len: torch.Tensor,
    limb_mask: torch.Tensor,
    use_limb: torch.Tensor,
    mm_scale: float,
) -> dict[str, float]:
    p = pred_len.detach().cpu().float()
    t = target_len.detach().cpu().float()
    m = limb_mask.detach().cpu().bool()
    u = use_limb.detach().cpu().bool()
    if p.ndim != 2 or m.ndim != 2 or t.ndim != 1 or u.ndim != 1:
        return {"mean_abs_mm": float("nan"), "median_abs_mm": float("nan"), "rmse_mm": float("nan")}
    T, L = int(p.shape[0]), int(p.shape[1])
    if int(t.numel()) != L or int(u.numel()) != L:
        return {"mean_abs_mm": float("nan"), "median_abs_mm": float("nan"), "rmse_mm": float("nan")}
    use2 = u.view(1, -1).expand(T, -1)
    ok = m & use2 & torch.isfinite(p) & torch.isfinite(t.view(1, -1))
    if not bool(ok.any()):
        return {"mean_abs_mm": float("nan"), "median_abs_mm": float("nan"), "rmse_mm": float("nan")}
    err = (p - t.view(1, -1)).abs() * float(mm_scale)
    v = err[ok]
    mean_abs = float(v.mean().item())
    median_abs = float(v.median().item())
    rmse = float(torch.sqrt((v * v).mean()).item())
    return {"mean_abs_mm": mean_abs, "median_abs_mm": median_abs, "rmse_mm": rmse}


def optimize_betas_by_limb_length(
    *,
    body_pose: torch.Tensor,
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    betas_init: torch.Tensor,
    k3d_gt: torch.Tensor,
    valid_mask: Any | None,
    k3d_kind: str,
    k3d_pred_fn: Any,
    cfg: BetaByLimbConfig | None = None,
) -> dict[str, Any]:
    from hmr4d.utils.pylogger import Log

    if cfg is None:
        cfg = BetaByLimbConfig()

    if not bool(cfg.enable):
        Log.info("[ShapeByLimb] skip: disabled")
        return {"betas_opt": betas_init.detach().cpu(), "stats": {"enabled": False, "reason": "disabled"}}

    if not torch.is_tensor(k3d_gt) or k3d_gt.ndim != 3 or int(k3d_gt.shape[-1]) != 3:
        Log.info("[ShapeByLimb] skip: invalid k3d_gt")
        return {"betas_opt": betas_init.detach().cpu(), "stats": {"enabled": False, "reason": "invalid_k3d_gt"}}

    dev = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device

    T = int(k3d_gt.shape[0])
    J = int(k3d_gt.shape[1])
    vm = _as_bool_mask(valid_mask, shape=(T, J))
    kintree = _kintree_for_kind(str(k3d_kind))
    if len(kintree) == 0:
        Log.info(f"[ShapeByLimb] skip: empty kintree for kind={str(k3d_kind)}")
        return {"betas_opt": betas_init.detach().cpu(), "stats": {"enabled": False, "reason": "empty_kintree"}}

    gt_len, gt_mask = _compute_limb_lengths(joints3d=k3d_gt, valid_mask=vm, kintree=kintree)
    target, use_limb = _robust_target_lengths(
        limb_lengths=gt_len,
        limb_mask=gt_mask,
        min_ratio=float(cfg.limb_inlier_min_ratio),
        max_ratio=float(cfg.limb_inlier_max_ratio),
        min_frames_per_limb=int(cfg.min_frames_per_limb),
    )
    if not bool(use_limb.any()):
        Log.info("[ShapeByLimb] skip: no valid limbs after filtering")
        return {"betas_opt": betas_init.detach().cpu(), "stats": {"enabled": False, "reason": "no_valid_limbs"}}

    bp = body_pose[:T].to(dev).float()
    go = global_orient[:T].to(dev).float()
    tr = transl[:T].to(dev).float()
    target_d = target.to(dev)
    use_limb_d = use_limb.to(dev)

    bet0 = betas_init.detach().cpu().float()
    if bet0.ndim == 2:
        bet0 = bet0[0]
    if int(bet0.numel()) < int(cfg.betas_dim):
        bet0 = torch.cat([bet0.view(-1), torch.zeros((int(cfg.betas_dim) - int(bet0.numel()),), dtype=torch.float32)], dim=0)
    bet0 = bet0[: int(cfg.betas_dim)]

    def _pred_lengths(betas_1d: torch.Tensor) -> torch.Tensor:
        b = betas_1d.view(1, -1).expand(T, -1)
        j3d = k3d_pred_fn(body_pose=bp, betas=b, global_orient=go, transl=tr, kind=str(k3d_kind))
        if not torch.is_tensor(j3d):
            raise TypeError("k3d_pred_fn must return torch.Tensor")
        if j3d.ndim != 3 or int(j3d.shape[-1]) != 3:
            raise ValueError(f"k3d_pred_fn returned invalid shape: {tuple(j3d.shape)}")
        j3d = j3d[:T, :J]
        L = int(len(kintree))
        out = torch.zeros((T, L), device=dev, dtype=torch.float32)
        for li, (a, b) in enumerate(kintree):
            if not (0 <= int(a) < int(j3d.shape[1]) and 0 <= int(b) < int(j3d.shape[1])):
                continue
            d = j3d[:, int(b), :] - j3d[:, int(a), :]
            out[:, li] = torch.linalg.norm(d, dim=-1)
        return out

    bet0_d = bet0.to(dev).clone().detach()
    amp = float(cfg.betas_delta_amp)
    u = torch.zeros_like(bet0_d, device=dev, dtype=torch.float32, requires_grad=True)

    def _bet_from_u(u_: torch.Tensor) -> torch.Tensor:
        if amp <= 0:
            return bet0_d
        return bet0_d + amp * torch.tanh(u_)

    with torch.no_grad():
        bet_base = bet0_d
        pred0 = _pred_lengths(bet_base)
        base_err = torch.nanmean(torch.abs(pred0[:, use_limb_d] - target_d[use_limb_d][None, :])).item()
        j3d0 = k3d_pred_fn(
            body_pose=bp,
            betas=bet_base.view(1, -1).expand(T, -1),
            global_orient=go,
            transl=tr,
            kind=str(k3d_kind),
        )
        base_mpjpe, base_mpjpe_stats = _mpjpe_mm(
            pred_j3d=j3d0,
            gt_j3d=k3d_gt,
            valid_mask=vm,
            pelvis_idxs=tuple(cfg.pelvis_idxs),
            align_by_pelvis=bool(cfg.align_by_pelvis),
            max_frame_mm=float(cfg.max_frame_mpjpe_mm),
            mm_scale=float(cfg.mm_scale),
        )
        base_limb_stats = _limb_err_mm_stats(
            pred_len=pred0,
            target_len=target,
            limb_mask=gt_mask,
            use_limb=use_limb,
            mm_scale=float(cfg.mm_scale),
        )
        if bool(cfg.verbose):
            Log.info(
                f"[ShapeByLimb] before mpjpe_mm={base_mpjpe:.3f} frames={base_mpjpe_stats.get('frames_kept', 0)}/{base_mpjpe_stats.get('frames_total', 0)} "
                f"limb_abs_mm(mean/med)={base_limb_stats['mean_abs_mm']:.2f}/{base_limb_stats['median_abs_mm']:.2f} rmse={base_limb_stats['rmse_mm']:.2f}"
            )

    opt = torch.optim.LBFGS([u], lr=float(cfg.lr), max_iter=int(cfg.max_iter), line_search_fn=cfg.line_search_fn)
    last: dict[str, float] = {}

    def _closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        bet = _bet_from_u(u)
        pred = _pred_lengths(bet)
        diff = pred - target_d.view(1, -1)
        diff = diff[:, use_limb_d]
        limb_loss = torch.mean(diff * diff)
        reg = torch.mean(bet * bet)
        init_prior = torch.mean((bet - bet0_d) * (bet - bet0_d))
        loss = (
            float(cfg.limb_w) * limb_loss
            + float(cfg.betas_init_prior_w) * init_prior
            + float(cfg.betas_l2_w) * reg
        )
        loss.backward()
        last["loss"] = float(loss.detach().cpu().item())
        last["limb"] = float(limb_loss.detach().cpu().item())
        last["reg"] = float(reg.detach().cpu().item())
        last["init_prior"] = float(init_prior.detach().cpu().item())
        return loss

    try:
        opt.step(_closure)
    except Exception as e:
        Log.info(f"[ShapeByLimb] failed: {type(e).__name__}: {e}")
        return {"betas_opt": betas_init.detach().cpu(), "stats": {"enabled": False, "reason": f"exception:{type(e).__name__}"}}

    bet_opt = _bet_from_u(u.detach()).detach().cpu().float()
    with torch.no_grad():
        bet_final = bet_opt.to(dev)
        pred1 = _pred_lengths(bet_final)
        final_err = torch.nanmean(torch.abs(pred1[:, use_limb_d] - target_d[use_limb_d][None, :])).item()
        j3d1 = k3d_pred_fn(
            body_pose=bp,
            betas=bet_final.view(1, -1).expand(T, -1),
            global_orient=go,
            transl=tr,
            kind=str(k3d_kind),
        )
        final_mpjpe, final_mpjpe_stats = _mpjpe_mm(
            pred_j3d=j3d1,
            gt_j3d=k3d_gt,
            valid_mask=vm,
            pelvis_idxs=tuple(cfg.pelvis_idxs),
            align_by_pelvis=bool(cfg.align_by_pelvis),
            max_frame_mm=float(cfg.max_frame_mpjpe_mm),
            mm_scale=float(cfg.mm_scale),
        )
        final_limb_stats = _limb_err_mm_stats(
            pred_len=pred1,
            target_len=target,
            limb_mask=gt_mask,
            use_limb=use_limb,
            mm_scale=float(cfg.mm_scale),
        )

    stats = {
        "enabled": True,
        "k3d_kind": str(k3d_kind),
        "num_joints": int(J),
        "num_limbs_all": int(len(kintree)),
        "num_limbs_used": int(use_limb.sum().item()),
        "betas_delta_amp": float(cfg.betas_delta_amp),
        "betas_init_prior_w": float(cfg.betas_init_prior_w),
        "baseline_abs_err_m": float(base_err),
        "final_abs_err_m": float(final_err),
        "baseline_mpjpe_mm": float(base_mpjpe),
        "final_mpjpe_mm": float(final_mpjpe),
        "baseline_mpjpe_frames_kept": int(base_mpjpe_stats.get("frames_kept", 0)),
        "final_mpjpe_frames_kept": int(final_mpjpe_stats.get("frames_kept", 0)),
        "baseline_limb_mean_abs_mm": float(base_limb_stats["mean_abs_mm"]),
        "final_limb_mean_abs_mm": float(final_limb_stats["mean_abs_mm"]),
        "baseline_limb_rmse_mm": float(base_limb_stats["rmse_mm"]),
        "final_limb_rmse_mm": float(final_limb_stats["rmse_mm"]),
        "loss": float(last.get("loss", float("nan"))),
        "limb_loss": float(last.get("limb", float("nan"))),
        "init_prior_loss": float(last.get("init_prior", float("nan"))),
        "betas_reg": float(last.get("reg", float("nan"))),
    }
    if bool(cfg.verbose):
        Log.info(
            f"[ShapeByLimb] done kind={stats['k3d_kind']} limbs_used={stats['num_limbs_used']}/{stats['num_limbs_all']} "
            f"abs_err_m: {stats['baseline_abs_err_m']:.6f}->{stats['final_abs_err_m']:.6f} "
            f"mpjpe_mm: {stats['baseline_mpjpe_mm']:.3f}->{stats['final_mpjpe_mm']:.3f} "
            f"limb_rmse_mm: {stats['baseline_limb_rmse_mm']:.2f}->{stats['final_limb_rmse_mm']:.2f} "
            f"loss={stats['loss']:.6f} init_prior={stats['init_prior_loss']:.6f} betas_maxabs={float(bet_opt.abs().max().item()):.3f}"
        )
    return {"betas_opt": bet_opt, "stats": stats, "target_limb_len": target.detach().cpu(), "use_limb": use_limb.detach().cpu()}
