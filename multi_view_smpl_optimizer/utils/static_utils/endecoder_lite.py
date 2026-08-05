import torch
import torch.nn as nn

from pytorch3d.transforms import axis_angle_to_matrix

from . import matrix
from ..body_model.smplx_lite import SmplxLite


class EnDecoderLite(nn.Module):
    """
    Minimal copy of GVHMR EnDecoder's FK interface:
      - fk_v2(body_pose, betas, global_orient=None, transl=None, get_intermediate=False)

    It uses SMPL-X skeleton (55 joints) but only the first 22 are used, identical to GVHMR.
    """

    def __init__(self, smplx_model_path: str):
        super().__init__()
        # Use the local-copied body_model implementation (user-provided), not GVHMR.
        self.smplx_model = SmplxLite(model_path=smplx_model_path, gender="neutral", num_betas=10)
        parents = self.smplx_model.parents[:22]
        self.register_buffer("parents_tensor", parents, False)
        self.parents = parents.tolist()

    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # B, L, 22, 3

        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat

