"""Differentiable camera pose representation for gradient-based optimization."""

import torch
import torch.nn as nn
from typing import Tuple


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to rotation matrix.

    Args:
        axis_angle: Axis-angle rotation vector [3] or [B, 3]

    Returns:
        Rotation matrix [3, 3] or [B, 3, 3]
    """
    # Handle batched vs single input
    squeeze_output = False
    if axis_angle.dim() == 1:
        axis_angle = axis_angle.unsqueeze(0)
        squeeze_output = True

    batch_size = axis_angle.shape[0]
    device = axis_angle.device
    dtype = axis_angle.dtype

    # Get angle (magnitude of axis-angle vector)
    theta = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]

    # Get normalized axis
    axis = axis_angle / theta  # [B, 3]

    # Rodrigues' rotation formula
    # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    # where K is the skew-symmetric matrix of the axis

    cos_theta = torch.cos(theta)  # [B, 1]
    sin_theta = torch.sin(theta)  # [B, 1]

    # Build skew-symmetric matrix K
    K = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    # Identity matrix
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    # Rodrigues formula
    R = I + sin_theta.unsqueeze(-1) * K + (1 - cos_theta.unsqueeze(-1)) * torch.bmm(K, K)

    if squeeze_output:
        R = R.squeeze(0)

    return R


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle representation.

    Args:
        R: Rotation matrix [3, 3] or [B, 3, 3]

    Returns:
        Axis-angle rotation vector [3] or [B, 3]
    """
    squeeze_output = False
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze_output = True

    batch_size = R.shape[0]
    device = R.device
    dtype = R.dtype

    # Compute angle from trace: trace(R) = 1 + 2*cos(theta)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # [B]

    # Compute axis from skew-symmetric part of R
    # axis = [R32-R23, R13-R31, R21-R12] / (2*sin(theta))
    axis = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=-1)  # [B, 3]

    # Handle small angles (near identity)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    sin_theta = sin_theta.clamp(min=1e-8)
    axis = axis / (2 * sin_theta)

    # Normalize and scale by angle
    axis_norm = torch.norm(axis, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis / axis_norm
    axis_angle = axis * theta.unsqueeze(-1)

    if squeeze_output:
        axis_angle = axis_angle.squeeze(0)

    return axis_angle


class DifferentiableCameraPose(nn.Module):
    """Differentiable camera pose for gradient-based optimization.

    Parameterizes camera pose as:
    - position: 3D position [3]
    - axis_angle: Rotation as axis-angle [3]

    This representation is differentiable and suitable for gradient descent.
    """

    def __init__(self, initial_c2w: torch.Tensor):
        """Initialize from a camera-to-world matrix.

        Args:
            initial_c2w: Initial camera-to-world matrix [3, 4] or [4, 4] or [1, 3, 4] or [1, 4, 4]
        """
        super().__init__()

        # Handle batched input
        if initial_c2w.dim() == 3:
            initial_c2w = initial_c2w.squeeze(0)

        # Extract position and rotation
        position = initial_c2w[:3, 3].clone()
        R = initial_c2w[:3, :3].clone()

        # Convert rotation matrix to axis-angle
        axis_angle = rotation_matrix_to_axis_angle(R)

        # Register as parameters for optimization
        self.position = nn.Parameter(position)
        self.axis_angle = nn.Parameter(axis_angle)

    def get_camera_to_world(self) -> torch.Tensor:
        """Get the camera-to-world transformation matrix.

        Returns:
            Camera-to-world matrix [1, 3, 4]
        """
        # Convert axis-angle to rotation matrix
        R = axis_angle_to_rotation_matrix(self.axis_angle)  # [3, 3]

        # Build camera-to-world matrix
        c2w = torch.zeros(3, 4, device=self.position.device, dtype=self.position.dtype)
        c2w[:3, :3] = R
        c2w[:3, 3] = self.position

        # Return with batch dimension
        return c2w.unsqueeze(0)  # [1, 3, 4]

    def get_position(self) -> torch.Tensor:
        """Get current camera position.

        Returns:
            Position [3]
        """
        return self.position

    def get_rotation_matrix(self) -> torch.Tensor:
        """Get current camera rotation matrix.

        Returns:
            Rotation matrix [3, 3]
        """
        return axis_angle_to_rotation_matrix(self.axis_angle)
