import torch
import numpy as np

# Ray helpers
def get_rays(H, W, K, c2w, device):
    """Takes camera calibration and pose to output ray origins and directions.
    Args:
        H: scalar. Number of pixels in height.
        W: scalar. Number of pixels in width.
        K: [3, 3]. Camera calibration matrix.
        c2w: [3, 4]. Camera frame to world frame pose matrix.
    Returns:
        rays_o: [H, W, 3]. Ray origins.
        rays_d: [H, W, 3]. Ray directions.
    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)

    return rays_o, rays_d

def get_rays_batch(H, W, K, c2w, device):
    """Just like get_rays, but for a batch of poses"""
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, None, :] * c2w[:, :3,:3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:, :3,-1].expand(rays_d.shape)

    rays_d = rays_d.permute(2, 0, 1, 3)
    rays_o = rays_o.permute(2, 0, 1, 3)
    return rays_o, rays_d

def look_at(location, target, up):
    z = (location - target)
    z /= torch.norm(z)
    x = torch.cross(up, z)
    x /= torch.norm(x)
    y = torch.cross(z, x)
    y /= torch.norm(y)

    R = torch.stack([x, y, z], dim=1)
    return R