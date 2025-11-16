import torch
from torch import Tensor

def conics_to_semi_major_axis_points_analytic(
    means2d: Tensor,      # [N, 2]
    conics: Tensor,       # [N, 3] with [a, b, c] for [[a, b], [b, c]]
    scale_factor: float = 1.0,
    eps: float = 1e-12,
    num_interior: int = 0,  # number of uniform interior samples per ellipse
):
    """
    Returns:
      axis_points: [N, 5, 2] = [center, major_neg, major_pos, minor_neg, minor_pos]
      interior_points (optional): [N, num_interior, 2] (only if num_interior>0)
    """
    device = means2d.device
    dtype = means2d.dtype

    a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]  # [N]

    # Closed-form eigenvalues
    mu = 0.5 * (a + c)
    delta = 0.5 * (a - c)
    r = torch.sqrt(delta * delta + b * b + eps)
    lam_major = torch.clamp(mu + r, min=0.0)
    lam_minor = torch.clamp(mu - r, min=0.0)

    # Eigenvectors via rotation by theta
    theta = 0.5 * torch.atan2(2.0 * b, (a - c))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    v_major = torch.stack([cos_t, sin_t], dim=1)       # [N, 2]
    v_minor = torch.stack([-sin_t, cos_t], dim=1)      # [N, 2]

    # Axis (semi-axis) lengths
    major_len = torch.sqrt(lam_major) * scale_factor   # [N]
    minor_len = torch.sqrt(lam_minor) * scale_factor   # [N]

    # Axis endpoints
    major_pos = means2d + major_len.unsqueeze(1) * v_major
    major_neg = means2d - major_len.unsqueeze(1) * v_major
    minor_pos = means2d + minor_len.unsqueeze(1) * v_minor
    minor_neg = means2d - minor_len.unsqueeze(1) * v_minor

    axis_points = torch.stack(
        [means2d, major_neg, major_pos, minor_neg, minor_pos],
        dim=1
    )  # [N, 5, 2]

    if num_interior <= 0:
        return axis_points

    # ----- Uniform-in-area interior sampling -----
    # Sample in unit disk and map with principal-axis transform
    # r_hat = sqrt(U), phi ~ Uniform[0, 2pi]
    U = torch.rand(means2d.shape[0], num_interior, device=device, dtype=dtype)  # [N, S]
    r_hat = torch.sqrt(U)                                  # [N, S]
    phi = 2.0 * torch.pi * torch.rand_like(U)              # [N, S]
    cos_p, sin_p = torch.cos(phi), torch.sin(phi)          # [N, S]

    # Broadcast axis lengths to match [N, S, 1]
    major_len_exp = major_len[:, None, None]               # [N, 1, 1]
    minor_len_exp = minor_len[:, None, None]               # [N, 1, 1]

    # Scale cos/sin terms with r_hat and axis lengths
    w_major = major_len_exp * (r_hat[:, :, None] * cos_p[:, :, None])  # [N, S, 1]
    w_minor = minor_len_exp * (r_hat[:, :, None] * sin_p[:, :, None])  # [N, S, 1]

    vmaj = v_major[:, None, :]  # [N, 1, 2]
    vmin = v_minor[:, None, :]  # [N, 1, 2]
    m = means2d[:, None, :]     # [N, 1, 2]

    # Map unit-disk samples to ellipse
    interior_points = m + w_major * vmaj + w_minor * vmin  # [N, S, 2]

    return axis_points, interior_points