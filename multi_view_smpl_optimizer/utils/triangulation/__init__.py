"""
多目三角化工具（真实尺度、世界坐标系输出）。

核心入口：
- `triangulate_multiview_keypoints`: 输入 (V,T,J,2/3) 的 2D 序列 + 相机参数，输出 (T,J,3) 的 3D 序列（world）。
"""

from .multiview_triangulation import (  # noqa: F401
    CameraModel,
    TriangulationConfig,
    triangulate_multiview_keypoints,
)

