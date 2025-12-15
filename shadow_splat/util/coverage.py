from typing import Optional

import torch
from torch import Tensor
import math
from typing import List

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
    # NOTE: DO WE HAVE TO WORRY ABOUT OVERFLOW HERE?
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
    # invalid = (coverage_counts < med[:, None]) | (coverage_counts == 0)  # [N,G]
    invalid = (coverage_counts == 0)

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

@torch.no_grad()
def spherical_gaussian_weights(
    input_view_dirs: Tensor, # [N, 3]
    bin_dirs: Tensor, # [G, 3]
    beta: Optional[float] = 1.0,
) -> Tensor:
    """
    """
    # Normalize the dirs
    input_view_dirs = input_view_dirs / (torch.norm(input_view_dirs, dim=-1, keepdim=True) + 1e-10)
    bin_dirs = bin_dirs / (torch.norm(bin_dirs, dim=-1, keepdim=True) + 1e-10)

    # Cosines with all bins (same for all gaussians)
    cos_all = (input_view_dirs @ bin_dirs.transpose(-2, -1))         # [N, G]

    # Compute the weights using a spherical gaussian kernel
    weights = torch.exp(beta * cos_all)     # [N, G]

    weights = weights / (torch.norm(weights, dim=-1, keepdim=True) + 1e-10)     # [N, G]

    return weights

@torch.no_grad()
def update_fig_for_frustum(
    means: Tensor, quats: Tensor, scales: Tensor,
    viewmats: Tensor, Ks: Tensor, width: int, height: int,
    depth_image: Tensor, variance_image: Tensor, bin_dirs: Tensor,
    fig: Tensor, view_fig: Tensor,
    camera_model: str = "pinhole",
    eps2d: float = 0.3,
    near_plane: float = 1e-2,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    concentration: Optional[float] = 1.0,
) -> None:
    """Find gaussians in current camera frustum and increment the sphere-bin for the
    camera optical axis. Everything is done in-place on coverage_counts.
    """
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
    batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations = proj
    if gaussian_ids.numel() == 0:
        return False

    eps = 1e-12
    denom = conics[:, 0] * conics[:, 2] - conics[:, 1]**2
    det_cov2d = 1.0 / torch.clamp(denom, min=eps)

    # One optical axis for this camera batch (we render one training cam at a time)
    camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
    # These two lines assume packed = True
    dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz, 3]
    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)

    # Now bin the directions into their respective patches on the unit sphere
    bin_dirs = bin_dirs / (torch.norm(bin_dirs, dim=-1, keepdim=True) + 1e-10)

    # Compute the weights using a spherical gaussian kernel
    sg_weights = spherical_gaussian_weights(dirs, bin_dirs, concentration)        # [N, G]

    # Compute the transmittance using the depth, variance, and Gaussian depths
    pixel_x = means2d[:, 0].long().clamp(0, width - 1)
    pixel_y = means2d[:, 1].long().clamp(0, height - 1)
    projected_pixel_ids = pixel_y * width + pixel_x  # shape [nnz]

    depth_image_flattened = depth_image.reshape(-1)
    variance_image_flattened = variance_image.reshape(-1)

    # Add minimum variance threshold to prevent division by very small numbers
    variance = torch.clamp(variance_image_flattened, min=1e-8)

    # Compute Gaussian distribution 
    normal_weights = torch.exp(-(1.0 / (2.0 * variance[projected_pixel_ids])) * (depths - depth_image_flattened[projected_pixel_ids]) ** 2)
    normal_weights = normal_weights / torch.sqrt(2.0 * math.pi * variance[projected_pixel_ids])
    normal_weights = (normal_weights **2) * torch.sqrt(det_cov2d)

    # Update the accumulated view transmittance
    combined_weight = (sg_weights**2) * normal_weights[:, None]  # [N, G]    

    # Update the accumulated transmittance
    fig.index_put_((gaussian_ids,), normal_weights.unsqueeze(1), accumulate=True)
    view_fig.index_put_((gaussian_ids,), combined_weight, accumulate=True)

    return None

@torch.no_grad()
def update_visibility_for_frustum(
    means: Tensor, quats: Tensor, scales: Tensor,
    viewmats: Tensor, Ks: Tensor, width: int, height: int, camera_id: int,
    depth_image: Tensor, variance_image: Tensor,
    visibility: Tensor, num_hits: Tensor,
    gaussian_ids: List[Tensor], camera_ids: List[Tensor],
    camera_model: str = "pinhole",
    eps2d: float = 0.3,
    near_plane: float = 1e-2,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
) -> None:
    """Find gaussians in current camera frustum and increment the sphere-bin for the
    camera optical axis. Everything is done in-place on coverage_counts.
    """
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
    batch_ids, _, gs_ids, radii, means2d, depths, conics, compensations = proj
    if gs_ids.numel() == 0:
        return False

    eps = 1e-12
    denom = conics[:, 0] * conics[:, 2] - conics[:, 1]**2
    det_cov2d = 1.0 / torch.clamp(denom, min=eps)

    # Compute the transmittance using the depth, variance, and Gaussian depths
    pixel_x = means2d[:, 0].long().clamp(0, width - 1)
    pixel_y = means2d[:, 1].long().clamp(0, height - 1)
    projected_pixel_ids = pixel_y * width + pixel_x  # shape [nnz]

    depth_image_flattened = depth_image.reshape(-1)
    variance_image_flattened = variance_image.reshape(-1)

    # Add minimum variance threshold to prevent division by very small numbers
    variance = torch.clamp(variance_image_flattened, min=1e-8)

    # Compute Gaussian distribution 
    normal_weights = torch.exp(-(1.0 / (2.0 * variance[projected_pixel_ids])) * (depths - depth_image_flattened[projected_pixel_ids]) ** 2)
    normal_weights = normal_weights / torch.sqrt(2.0 * math.pi * variance[projected_pixel_ids])
    normal_weights = (normal_weights **2) * torch.sqrt(det_cov2d)

    # Update the accumulated transmittance
    visibility.index_put_((camera_id.repeat(gs_ids.shape[0]), gs_ids), normal_weights, accumulate=False)    # [C, N]
    num_hits.index_put_((camera_id.repeat(gs_ids.shape[0]), gs_ids), torch.sqrt(det_cov2d), accumulate=False)    # [C, N]

    # Store camera ids and gaussian ids into list of tensors
    gaussian_ids.append(gs_ids)
    camera_ids.append(camera_id.repeat(gs_ids.shape[0]))

    return None

@torch.no_grad()
def compute_color_field_visibility_for_frustum(
    view_directions_train: Tensor, # [M, 3]
    view_directions_test: Tensor, # [N, 3]  # NOTE: The view directions of the candidate camera with respect to all Gaussians. Therefore, view_directions that are 0 are not in the frustum.
    gaussian_ids_train: Tensor, # [M]
    camera_ids_train: Tensor, # [M]
    visibility: Tensor, # [C, N]
    attribute: Tensor, # [N]        # This is W_tilde_beta_norm_sqr
    kappa: float = 1.0,
) -> None:
    """
    Compute the visibility of the color field for a given camera.
    """

    # TODO: Put in camera ids and gaussian ids into list of tensors

    dot_product, mask = dot_product_spherical_gaussians(
                        view_directions_train, # [M, 3]
                        view_directions_test, # [N, 3]  # NOTE: The view directions of the candidate camera with respect to all Gaussians. Therefore, view_directions that are 0 are not in the frustum.
                        gaussian_ids_train, # [M]
                        kappa = kappa)  # [M]

    gaussian_ids_valid = gaussian_ids_train[mask]
    camera_ids_valid = camera_ids_train[mask]

    visibility_expanded = visibility[camera_ids_valid, gaussian_ids_valid]      # M

    w_tilde_beta = (dot_product**2) * visibility_expanded

    # Use scatter_add to sum over all the cameras
    attribute.scatter_add_(0, gaussian_ids_valid, w_tilde_beta)     # N

    return None

@torch.no_grad()
def dot_product_spherical_gaussians(
    view_directions_train: Tensor, # [M, 3]
    view_directions_test: Tensor, # [N, 3]  # NOTE: The view directions of the candidate camera with respect to all Gaussians. Therefore, view_directions that are 0 are not in the frustum.
    gaussian_ids_train: Tensor, # [M]
    kappa: float = 1.0,
) -> Tensor:
    """
    Dot product between two spherical gaussians in continuous space. Note, we do NOT normalize the directions in this function. Make sure the directions
    are normalized before calling this function.
    """

    # Format the view_directions so that view_directions_test is the same size as view_directions_train
    view_directions_test_expanded = view_directions_test[gaussian_ids_train]

    # If the norm is 0, then remove the calculation
    view_directions_test_norm = torch.norm(view_directions_test_expanded, dim=-1, keepdim=True)

    mask = (view_directions_test_norm > 0)

    view_directions_train_valid = view_directions_train[mask]   
    view_directions_test_valid = view_directions_test_expanded[mask]

    # Compute the dot product
    dot_product = inner_ptilde(view_directions_train_valid, view_directions_test_valid, kappa)

    return dot_product, mask

@torch.no_grad()
def sinhc(z, eps=1e-4):
    """
    Safe sinh(z)/z with a Taylor expansion near z = 0.
    """
    abs_z = z.abs()
    out = torch.empty_like(z)

    small = abs_z < eps
    big = ~small

    # Taylor: sinh(z)/z ≈ 1 + z^2/6 + z^4/120
    z_small = z[small]
    out[small] = 1 + (z_small**2) / 6 + (z_small**4) / 120

    z_big = z[big]
    out[big] = torch.sinh(z_big) / z_big

    return out

@torch.no_grad()
def inner_ptilde(mu1, mu2, kappa, eps=1e-4):
    """
    <p_tilde(mu1), p_tilde(mu2)> for L2-normalized spherical Gaussians on S^2.
    mu1, mu2: (..., 3) unit vectors
    kappa: scalar tensor or broadcastable to mu1[...,0]
    """

    assert kappa > 0, "kappa must be positive"

    # s = ||mu1 + mu2||
    s = torch.linalg.norm(mu1 + mu2, dim=-1)

    # z = kappa * s
    z = kappa * s

    # sinhc(z) = sinh(z)/z, handled stably
    h = sinhc(z)

    # prefactor 2kappa / sinh(2kappa), also safe near kappa = 0
    two_kappa = 2 * kappa
    # small-kappa branch: sinh(2kappa) ≈ 2kappa + (2kappa)^3/6
    small_k = two_kappa.abs() < eps
    pref = torch.empty_like(two_kappa)

    if small_k.any():
        tk = two_kappa[small_k]
        sinh_2k_approx = tk + (tk**3) / 6
        pref[small_k] = two_kappa[small_k] / sinh_2k_approx

    if (~small_k).any():
        tk = two_kappa[~small_k]
        pref[~small_k] = tk / torch.sinh(tk)

    # final inner product
    # broadcasting: pref and h should be broadcastable to s's shape
    return pref * h
