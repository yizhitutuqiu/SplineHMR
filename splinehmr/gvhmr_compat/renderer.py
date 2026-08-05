from __future__ import annotations

"""Minimal mesh renderer adapted from GVHMR's hmr4d/utils/vis/renderer.py.

We vendor only the video-overlay mesh rendering path used by the SplineHMR demo,
so Spline-Opt inference does not require a full GVHMR checkout at runtime.
"""

import numpy as np
import torch
from pytorch3d.renderer import (
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes


def overlay_image_onto_background(image, mask, bbox, background, *, alpha: float = 1.0):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox_np = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox_np[1] : bbox_np[3], bbox_np[0] : bbox_np[2]]
    alpha = float(max(0.0, min(1.0, alpha)))
    if alpha >= 0.999:
        roi_image[mask] = image[mask]
    else:
        fg = image[mask].astype(np.float32)
        bg = roi_image[mask].astype(np.float32)
        roi_image[mask] = np.clip(alpha * fg + (1.0 - alpha) * bg, 0, 255).astype(roi_image.dtype)
    out_image[bbox_np[1] : bbox_np[3], bbox_np[0] : bbox_np[2]] = roi_image
    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype
    K = torch.zeros((K_org.shape[0], 4, 4), device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1

    image_sizes = []
    for idx, cur_bbox in enumerate(bbox):
        left, upper, right, lower = cur_bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]
        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        K[idx, 0, 2] = new_width - new_cx
        K[idx, 1, 2] = new_height - new_cy
        image_sizes.append((int(new_height), int(new_width)))
    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R is not None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T is not None:
        x3d = x3d + T.transpose(1, 2)
    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top
    new_left = torch.clamp(cx - width / 2 * scaleFactor, min=0, max=img_w - 1)
    new_right = torch.clamp(cx + width / 2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h - 1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)
    return torch.stack((new_left.detach(), new_top.detach(), new_right.detach(), new_bottom.detach())).int().float().T


class Renderer:
    def __init__(self, width, height, focal_length=None, device="cuda", faces=None, K=None, bin_size=None):
        self.width = int(width)
        self.height = int(height)
        self.bin_size = bin_size
        assert (focal_length is not None) ^ (K is not None), "focal_length and K are mutually exclusive"
        self.device = device
        if faces is not None:
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy(faces.astype("int64"))
            self.faces = faces.unsqueeze(0).to(self.device)
        self.initialize_camera_params(focal_length, K)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(image_size=self.image_sizes[0], blur_radius=1e-5, bin_size=self.bin_size),
            ),
            shader=SoftPhongShader(device=self.device, lights=self.lights),
        )

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)
        return PerspectiveCameras(device=self.device, R=self.R.mT, T=self.T, K=self.K_full, image_size=self.image_sizes, in_ndc=False)

    def initialize_camera_params(self, focal_length, K):
        self.R = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.T = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        if K is not None:
            self.K = K.float().reshape(1, 3, 3).to(self.device)
        else:
            self.K = torch.tensor(
                [[focal_length, 0, self.width / 2], [0, focal_length, self.height / 2], [0, 0, 1]],
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 3, 3)
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]], dtype=torch.float32, device=self.device)
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()

    def update_bbox(self, x3d, scale=2.0, mask=None):
        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))
        if mask is not None:
            x2d = x2d[:, ~mask]
        self.bboxes = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(self):
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]], dtype=torch.float32, device=self.device)
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, background=None, colors=(0.8, 0.8, 0.8), VI=50, alpha=1.0):
        self.update_bbox(vertices[::VI], scale=1.2)
        vertices = vertices.unsqueeze(0)
        if isinstance(colors, torch.Tensor):
            verts_features = colors.to(device=vertices.device, dtype=vertices.dtype)
            material_color = (0.8, 0.8, 0.8)
        else:
            color_vals = [float(c) / 255.0 if float(c) > 1 else float(c) for c in colors]
            material_color = tuple(color_vals)
            verts_features = torch.tensor(color_vals, device=vertices.device, dtype=vertices.dtype).reshape(1, 1, 3)
            verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)
        mesh = Meshes(verts=vertices, faces=self.faces, textures=textures)
        materials = Materials(device=self.device, specular_color=(material_color,), shininess=0)
        results = torch.flip(self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights), [1, 2])
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3
        if background is None:
            background = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy(), alpha=alpha)
        self.reset_bbox()
        return image
