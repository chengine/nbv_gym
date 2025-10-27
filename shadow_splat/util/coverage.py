import torch
from torch import Tensor
import math

from gsplat.cuda._wrapper import (
    fully_fused_projection,
)

@torch.no_grad()
def fibonacci_sphere(n_bins: int, device=None, dtype=None) -> Tensor:
    """Even-ish patch centers on S^2 via Fibonacci lattice. Shape [G,3], unit norm."""
    # Output dimension
    # [G, 3]
    G = n_bins
    i = torch.arange(G, device=device, dtype=torch.float32)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / G                   # in [-1,1]
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0)) # radius in xy
    theta = 2.0 * math.pi * (i / phi % 1.0)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    dirs = torch.stack([x, y, z], dim=-1)
    return dirs.to(device=device, dtype=dtype or torch.float32)

@torch.no_grad()
def dir_to_bin(bin_dirs: Tensor, view_dirs: Tensor) -> Tensor:
    """bin_dirs [G,3], view_dirs [N, 3] -> argmax."""
    dots = view_dirs @ bin_dirs.transpose(-2, -1) # [N, G]  
    return torch.argmax(dots, dim=-1) # [N]

@torch.no_grad()
### NOTE: THIS FUNCTION CAN BE INTEGRATED INTO THE RASTERIZATION CALL TO AVOID CALLING FULLY FUSED PROJECTION MULTIPLE TIMES
# HOWEVER, FOR READABILITY AND AVOIDING HAVING TO PASS IN THE TRAINING FLAG TO RASTERIZATION, WE KEEP IT SEPARATE
def update_view_coverage_for_frustum(
    means: Tensor, quats: Tensor, scales: Tensor,
    viewmats: Tensor, Ks: Tensor, width: int, height: int,
    coverage_counts: Tensor,       # [N, G]
    bin_dirs: Tensor,        # [G, 3]
    camera_model: str = "pinhole",
    eps2d: float = 0.3,
    near_plane: float = 1e-2,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
) -> None:
    """Find gaussians in current camera frustum and increment the sphere-bin for the
    camera optical axis. Everything is done in-place on coverage_counts.
    """
    N, G = coverage_counts.shape
    device = means.device
    # Project gaussians to figure out which are in the frustum.
    proj = fully_fused_projection(
        means, None, quats, scales,
        viewmats, Ks, width, height,
        eps2d=eps2d, packed=True,
        near_plane=near_plane, far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=False,
        calc_compensations=False,
        camera_model=camera_model,
    )
    # packed-mode tuple layout in gsplat>=1.0: (batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations)
    batch_ids, camera_ids, gaussian_ids, *_ = proj
    if gaussian_ids.numel() == 0:
        return False

    # One optical axis for this camera batch (we render one training cam at a time)
    camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
    # These two lines assume packed = True
    dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz, 3]
    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)

    # Now bin the directions into their respective patches on the unit sphere
    bin_dirs = bin_dirs / (torch.norm(bin_dirs, dim=-1, keepdim=True) + 1e-10)
    bin_idx = dir_to_bin(bin_dirs, dirs)

    # Increment that bin for all visible gaussians
    coverage_counts.index_put_((gaussian_ids, bin_idx), torch.ones_like(gaussian_ids, dtype=coverage_counts.dtype), accumulate=True)

    return True

@torch.no_grad()
# Call this function right after computing the spherical harmonics within rasterization!
def compute_coverage_per_gaussian(
    coverage_counts: Tensor,    # [N, G]    # Coverage counts for ALL gaussian in each bin
    bin_dirs: Tensor,           # [G, 3]    # Bin centers on the unit sphere
    masks: Tensor,       # [N]       # Which gaussians are visible to the current camera?
    inference_dirs: Tensor,     # [N, 3]    # Associated viewing directions of the camera
) -> Tensor:
    """For each gaussian: distance (angle, radians) from the current view direction to
    the closest bin whose count >= per-gaussian median count (and >0). If none, returns pi.
    Returns: [N] tensor of angles.
    """
    N, G = coverage_counts.shape
    device = coverage_counts.device
    dtype = coverage_counts.dtype

    # Normalize the dirs
    inference_dirs = inference_dirs / (torch.norm(inference_dirs, dim=-1, keepdim=True) + 1e-10)
    bin_dirs = bin_dirs / (torch.norm(bin_dirs, dim=-1, keepdim=True) + 1e-10)

    # Cosines with all bins (same for all gaussians)
    cos_all = (inference_dirs @ bin_dirs.transpose(-2, -1))         # [N, G]

    # Median threshold per gaussian
    med = coverage_counts.median(dim=1).values               # [N]

    # Mask of "well-covered" bins per gaussian: count >= median and >0
    invalid = (coverage_counts < med[:, None]) | (coverage_counts == 0)  # [N,G]

    # For each gaussian, want max cosine over allowed bins -> min angle
    # Fill disallowed with -inf
    # cos_masked = torch.where(good, cos_all.expand(N, -1), torch.full((N, G), -float('inf'), device=device, dtype=sphere_centers.dtype))
    # best_cos, _ = cos_masked.max(dim=1)                      # [N]
    # # If none allowed (all -inf), set angle = pi
    # no_hit = ~good.any(dim=1)
    # best_cos = torch.where(no_hit, torch.full_like(best_cos, -1.0), best_cos)
    # angles = torch.arccos(best_cos.clamp(-1.0, 1.0))         # [N], in [0, pi]
    # return angles

    # Make the cosine values -1 for invalid bins
    coverage_metric = ( (cos_all + 1.0) / 2.0 ) * (1. - invalid.to(torch.float32))

    # Find the max cosine per gaussian
    max_coverage, _ = coverage_metric.max(dim=-1)        # [N]

    return max_coverage
