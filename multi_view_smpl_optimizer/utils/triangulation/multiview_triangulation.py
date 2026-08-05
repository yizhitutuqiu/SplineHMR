from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from tqdm import tqdm

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    from scipy.optimize import least_squares

    _HAS_SCIPY = True
except Exception:
    least_squares = None
    _HAS_SCIPY = False


def _to_numpy(x, dtype=np.float64) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def _rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    """
    Rodrigues 向量转旋转矩阵（避免强依赖 cv2；使用稳定的近似/闭式）。
    rvec: (3,)
    """
    rvec = _to_numpy(rvec).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rvec / theta
    kx, ky, kz = k.tolist()
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R


def _invert_extrinsics(R_wc: np.ndarray, t_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    world-to-cam: Xc = R_wc Xw + t_wc
    cam-to-world: Xw = R_cw Xc + t_cw
    """
    R_wc = _to_numpy(R_wc).reshape(3, 3)
    t_wc = _to_numpy(t_wc).reshape(3)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw


def _make_P(K: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    K = _to_numpy(K).reshape(3, 3)
    R_wc = _to_numpy(R_wc).reshape(3, 3)
    t_wc = _to_numpy(t_wc).reshape(3)
    Rt = np.concatenate([R_wc, t_wc.reshape(3, 1)], axis=1)  # 3x4
    return K @ Rt


@dataclass(frozen=True)
class CameraModel:
    """
    相机模型（针孔）。

    约定默认使用 world-to-cam 外参：
      X_cam = R_wc @ X_world + t_wc
      x_img_h = K @ X_cam
      x_img = x_img_h[:2] / x_img_h[2]
    """

    K: np.ndarray  # 3x3
    D: np.ndarray | None  # (k,) 畸变系数，可为 None
    R_wc: np.ndarray  # 3x3
    t_wc: np.ndarray  # (3,)
    P: np.ndarray  # 3x4

    @staticmethod
    def from_matrices(
        K: np.ndarray,
        R: np.ndarray | None = None,
        t: np.ndarray | None = None,
        rvec: np.ndarray | None = None,
        tvec: np.ndarray | None = None,
        D: np.ndarray | None = None,
        extrinsics: Literal["world_to_cam", "cam_to_world"] = "world_to_cam",
    ) -> "CameraModel":
        K = _to_numpy(K)
        D_arr = None if D is None else _to_numpy(D).reshape(-1)
        if R is None:
            if rvec is None:
                raise ValueError("CameraModel.from_matrices: 需要提供 R 或 rvec")
            R = _rodrigues_to_R(_to_numpy(rvec).reshape(3))
        else:
            R = _to_numpy(R).reshape(3, 3)
        if t is None:
            if tvec is None:
                raise ValueError("CameraModel.from_matrices: 需要提供 t 或 tvec")
            t = _to_numpy(tvec).reshape(3)
        else:
            t = _to_numpy(t).reshape(3)

        if extrinsics == "cam_to_world":
            # 输入为 Xw = R_cw Xc + t_cw -> 转成 world-to-cam
            R_cw, t_cw = R, t
            R_wc = R_cw.T
            t_wc = -R_wc @ t_cw
        elif extrinsics == "world_to_cam":
            R_wc, t_wc = R, t
        else:
            raise ValueError(f"extrinsics must be 'world_to_cam' or 'cam_to_world', got {extrinsics!r}")

        P = _make_P(K, R_wc, t_wc)
        return CameraModel(
            K=K.astype(np.float64),
            D=None if D_arr is None else D_arr.astype(np.float64),
            R_wc=R_wc.astype(np.float64),
            t_wc=t_wc.astype(np.float64),
            P=P,
        )

    def project(self, X_w: np.ndarray) -> np.ndarray:
        """X_w: (...,3) -> (...,2) 像素坐标。"""
        X = _to_numpy(X_w).reshape(-1, 3)
        X_h = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)  # Nx4
        x_h = (self.P @ X_h.T).T  # Nx3
        z = x_h[:, 2:3]
        z = np.where(np.abs(z) < 1e-12, 1e-12, z)
        uv = x_h[:, :2] / z
        return uv.reshape((*X_w.shape[:-1], 2))

    def undistort_pixels(self, uv_px: np.ndarray) -> np.ndarray:
        """
        将像素点去畸变，输出等效针孔模型下的像素坐标（仍在像素空间）。
        若 D 为空或全 0，则原样返回。
        uv_px: (...,2)
        """
        if self.D is None:
            return _to_numpy(uv_px).astype(np.float64, copy=False)
        D = self.D
        if D.size == 0 or np.allclose(D, 0):
            return _to_numpy(uv_px).astype(np.float64, copy=False)
        if not _HAS_CV2:
            raise RuntimeError("检测到畸变系数 D，但当前环境无法导入 cv2 进行去畸变。")

        uv = _to_numpy(uv_px).reshape(-1, 1, 2).astype(np.float64)
        # 使用 P=K，让输出回到像素坐标系（而不是归一化坐标系）
        uv_ud = cv2.undistortPoints(uv, cameraMatrix=self.K, distCoeffs=D, P=self.K)  # (N,1,2)
        return uv_ud.reshape((*uv_px.shape[:-1], 2))

    def depth(self, X_w: np.ndarray) -> np.ndarray:
        """返回点在该相机坐标系下的深度 z。X_w: (...,3) -> (...,)"""
        X = _to_numpy(X_w).reshape(-1, 3)
        Xc = (self.R_wc @ X.T + self.t_wc.reshape(3, 1)).T
        return Xc[:, 2].reshape(X_w.shape[:-1])


@dataclass(frozen=True)
class TriangulationConfig:
    """多目三角化配置。"""

    # 2D 输入是否包含置信度 (u,v,conf)
    has_confidence: bool = True
    conf_threshold: float = 0.0
    conf_power: float = 1.0  # weight = conf ** conf_power

    # 非线性精修
    refine: bool = True
    loss: Literal["linear", "huber", "soft_l1", "cauchy", "arctan"] = "huber"
    f_scale_px: float = 3.0
    max_nfev: int = 50

    # 质量控制
    min_views: int = 2
    require_positive_depth: bool = True
    max_reproj_rmse_px: float | None = None


def _linear_dlt_triangulate(Ps: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """
    多目 DLT 三角化（线性），返回 X_world (3,)。
    Ps: (M,3,4), uvs: (M,2)
    """
    Ps = _to_numpy(Ps)
    uvs = _to_numpy(uvs)
    M = Ps.shape[0]
    A = np.zeros((2 * M, 4), dtype=np.float64)
    for i in range(M):
        P = Ps[i]
        u, v = float(uvs[i, 0]), float(uvs[i, 1])
        A[2 * i + 0] = u * P[2] - P[0]
        A[2 * i + 1] = v * P[2] - P[1]
    # solve A X = 0
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-12:
        return np.full((3,), np.nan, dtype=np.float64)
    X = X_h[:3] / X_h[3]
    return X.astype(np.float64)


def _refine_point_least_squares(
    X0: np.ndarray,
    cameras: list[CameraModel],
    view_ids: np.ndarray,
    uvs: np.ndarray,
    weights: np.ndarray,
    cfg: TriangulationConfig,
) -> tuple[np.ndarray, float]:
    """
    对单个点做重投影最小二乘精修。
    返回 (X, reproj_rmse_px)
    """
    if (not cfg.refine) or (not _HAS_SCIPY):
        # 直接返回线性解 + 线性重投影误差
        reproj = []
        for k, vid in enumerate(view_ids):
            uv_hat = cameras[int(vid)].project(X0[None])[0]
            err = (uv_hat - uvs[k]).reshape(2)
            reproj.append(err)
        reproj = np.stack(reproj, axis=0)  # (M,2)
        rmse = float(np.sqrt(np.mean((reproj**2).sum(axis=1))))
        return X0, rmse

    X0 = _to_numpy(X0).reshape(3)
    uvs = _to_numpy(uvs).reshape(-1, 2)
    weights = _to_numpy(weights).reshape(-1)
    w_sqrt = np.sqrt(np.clip(weights, 1e-12, None))

    def fun(x: np.ndarray) -> np.ndarray:
        X = x.reshape(3)
        res = []
        for k, vid in enumerate(view_ids):
            cam = cameras[int(vid)]
            uv_hat = cam.project(X[None])[0]
            r = (uv_hat - uvs[k]).reshape(2)
            res.append(w_sqrt[k] * r)
        return np.concatenate(res, axis=0)

    ls = least_squares(
        fun,
        X0,
        method="trf",
        loss=cfg.loss,
        f_scale=cfg.f_scale_px,
        max_nfev=int(cfg.max_nfev),
    )
    X = ls.x.astype(np.float64)

    # 计算未加权的 RMSE（每视角 2D 误差的 L2）
    reproj = []
    for k, vid in enumerate(view_ids):
        uv_hat = cameras[int(vid)].project(X[None])[0]
        err = (uv_hat - uvs[k]).reshape(2)
        reproj.append(err)
    reproj = np.stack(reproj, axis=0)
    rmse = float(np.sqrt(np.mean((reproj**2).sum(axis=1))))
    return X, rmse


def triangulate_multiview_keypoints(
    keypoints_2d: np.ndarray,
    cameras: Iterable[CameraModel],
    cfg: TriangulationConfig | None = None,
) -> dict:
    """
    多目三角化：输入 N(=V) 个视角同步的 2D 关键点序列，输出真实尺度、世界坐标系下的 3D 序列。

    Args:
        keypoints_2d: (V, T, J, 2) 或 (V, T, J, 3)；第 3 维可为 conf。
                     允许 NaN 表示缺失。
        cameras: 长度 V 的相机模型（针孔）。其外参应对应世界坐标系（默认 world-to-cam）。
        cfg: 三角化配置。

    Returns:
        {
          "keypoints_3d": (T,J,3) float64，世界坐标系；无效为 NaN
          "valid_mask": (T,J) bool
          "reproj_rmse_px": (T,J) float64（无效为 NaN）
          "num_views_used": (T,J) int32
        }
    """
    if cfg is None:
        cfg = TriangulationConfig()
    cams = list(cameras)
    V = len(cams)
    k2d = _to_numpy(keypoints_2d)
    if k2d.ndim != 4:
        raise ValueError(f"keypoints_2d 必须是 (V,T,J,C)，但得到 {k2d.shape}")
    if k2d.shape[0] != V:
        raise ValueError(f"keypoints_2d 的 V={k2d.shape[0]} 与 cameras 的 V={V} 不一致")
    if k2d.shape[-1] not in (2, 3):
        raise ValueError(f"keypoints_2d 最后一维必须是 2 或 3，但得到 {k2d.shape[-1]}")

    T, J = int(k2d.shape[1]), int(k2d.shape[2])
    out_3d = np.full((T, J, 3), np.nan, dtype=np.float64)
    out_valid = np.zeros((T, J), dtype=bool)
    out_rmse = np.full((T, J), np.nan, dtype=np.float64)
    out_views = np.zeros((T, J), dtype=np.int32)

    Ps = np.stack([c.P for c in cams], axis=0)  # (V,3,4)
    any_dist = any((c.D is not None) and (c.D.size > 0) and (not np.allclose(c.D, 0)) for c in cams)

    # 主循环：每个 (t,j) 独立三角化（方便处理缺失视角、任意关键点数）
    for t in tqdm(range(T), desc="Triangulation", unit="frame", leave=False):
        for j in range(J):
            obs = k2d[:, t, j]  # (V,2/3)
            uv = obs[:, :2]
            uv_ok = np.isfinite(uv).all(axis=1)

            if k2d.shape[-1] == 3 and cfg.has_confidence:
                conf = obs[:, 2]
                conf_ok = np.isfinite(conf) & (conf >= float(cfg.conf_threshold))
                ok = uv_ok & conf_ok
                w = np.clip(conf, 0.0, 1.0) ** float(cfg.conf_power)
            else:
                ok = uv_ok
                w = np.ones((V,), dtype=np.float64)

            view_ids = np.where(ok)[0]
            M = int(view_ids.shape[0])
            out_views[t, j] = M
            if M < int(cfg.min_views):
                continue

            uvs = uv[view_ids].astype(np.float64)
            if any_dist:
                # 对选中的视角逐个去畸变（返回像素坐标）
                uvs = np.stack([cams[int(vid)].undistort_pixels(uvs[k]) for k, vid in enumerate(view_ids)], axis=0)
            Ps_sel = Ps[view_ids].astype(np.float64)
            w_sel = w[view_ids].astype(np.float64)

            X0 = _linear_dlt_triangulate(Ps_sel, uvs)
            if not np.isfinite(X0).all():
                continue

            if bool(cfg.refine):
                X, rmse = _refine_point_least_squares(X0, cams, view_ids, uvs, w_sel, cfg)
                if not np.isfinite(X).all():
                    continue
            else:
                X = X0
                reproj = []
                for k, vid in enumerate(view_ids):
                    uv_hat = cams[int(vid)].project(X[None])[0]
                    err = (uv_hat - uvs[k]).reshape(2)
                    reproj.append(err)
                reproj = np.stack(reproj, axis=0)
                rmse = float(np.sqrt(np.mean((reproj**2).sum(axis=1))))

            if cfg.require_positive_depth:
                depths = np.array([cams[int(vid)].depth(X[None])[0] for vid in view_ids], dtype=np.float64)
                if not np.all(depths > 1e-6):
                    continue

            if cfg.max_reproj_rmse_px is not None and rmse > float(cfg.max_reproj_rmse_px):
                continue

            out_3d[t, j] = X
            out_valid[t, j] = True
            out_rmse[t, j] = rmse

    return {
        "keypoints_3d": out_3d,
        "valid_mask": out_valid,
        "reproj_rmse_px": out_rmse,
        "num_views_used": out_views,
    }
