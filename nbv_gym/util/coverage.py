from typing import Optional, Literal

import torch
from torch import Tensor
import math
from typing import List

from nerfstudio.cameras.cameras import Cameras, CameraType

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
def compute_coverage_metric(
    view_attributes: Tensor,
    bin_dirs: Tensor,
    masks: Tensor,
    inference_dirs: Tensor,
) -> Tensor:
    """Compute the coverage metric for a given set of gaussians."""
    coverage_metric = compute_coverage_per_gaussian(
        coverage_counts=view_attributes,
        bin_dirs=bin_dirs,
        masks=masks.squeeze(0),
        inference_dirs=inference_dirs.squeeze(0),
    )

    return coverage_metric.squeeze()

@torch.no_grad()
def compute_fig_metric(
    view_attributes: Tensor,
    bin_dirs: Tensor,
    masks: Tensor,
    inference_dirs: Tensor,
) -> Tensor:
    """Compute the fig metric for a given set of gaussians."""
    return torch.sqrt(view_attributes.squeeze())

@torch.no_grad()
def compute_fig_diag_metric(
    view_attributes: Tensor,
    bin_dirs: Tensor,
    masks: Tensor,
    inference_dirs: Tensor,
) -> Tensor:
    """Compute the fig metric for a given set of gaussians."""
    return - torch.sqrt(1./torch.sqrt(view_attributes.squeeze() + 1e-10))

@torch.no_grad()
def compute_view_fig_metric(
    view_attributes: Tensor,
    bin_dirs: Tensor,
    masks: Tensor,
    inference_dirs: Tensor,
    concentration: Optional[float] = 1.0,
) -> Tensor:
    """Compute the view fig metric for a given set of gaussians."""
    sg_weights = spherical_gaussian_weights(input_view_dirs=inference_dirs.squeeze(0), bin_dirs=bin_dirs, beta=concentration)       # [N, G]
    view_fig_metric = torch.sqrt(torch.sum(view_attributes * sg_weights, dim=-1, keepdim=True))       # [N, 1]

    return view_fig_metric.squeeze()

@torch.no_grad()
def compute_view_fig_diag_metric(
    view_attributes: Tensor,
    bin_dirs: Tensor,
    masks: Tensor,
    inference_dirs: Tensor,
    concentration: Optional[float] = 1.0,
) -> Tensor:
    """Compute the view fig diag metric for a given set of gaussians."""
    sg_weights = spherical_gaussian_weights(input_view_dirs=inference_dirs.squeeze(0), bin_dirs=bin_dirs, beta=concentration)       # [N, G]
    view_fig_diag_metric = -torch.sqrt(torch.sum(1./torch.sqrt(view_attributes + 1e-10) * sg_weights, dim=-1, keepdim=True))       # [N, 1]

    return view_fig_diag_metric.squeeze()

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
        return

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

def segment_softmax(logits: torch.Tensor, group_id: torch.LongTensor, num_groups: int, eps: float = 1e-12):
    # logits: [N]
    # group_id: [N] in [0, num_groups)
    device = logits.device
    neg_inf = torch.tensor(-float("inf"), device=device, dtype=logits.dtype)

    max_per = torch.full((num_groups,), neg_inf, device=device, dtype=logits.dtype)
    max_per.scatter_reduce_(0, group_id, logits, reduce="amax", include_self=True)

    exp_logits = torch.exp(logits - max_per[group_id])
    sumexp = torch.zeros((num_groups,), device=device, dtype=logits.dtype)
    sumexp.scatter_add_(0, group_id, exp_logits)

    return exp_logits / (sumexp[group_id] + eps)  # [N], sums to 1 per group


# @torch.no_grad()
# def update_fig_for_frustum(
#     means: Tensor, quats: Tensor, scales: Tensor,
#     viewmats: Tensor, Ks: Tensor, width: int, height: int,
#     depth_image: Tensor, variance_image: Tensor,
#     fig: Tensor,
#     camera_model: str = "pinhole",
#     eps2d: float = 0.3,
#     near_plane: float = 1e-2,
#     far_plane: float = 1e10,
#     radius_clip: float = 0.0,
# ) -> None:
#     """
#     """
#     device = means.device
#     # Project gaussians to figure out which are in the frustum.
#     proj = fully_fused_projection(
#         means, None, quats, scales,
#         viewmats, Ks, width, height,
#         eps2d=eps2d, packed=True,
#         near_plane=near_plane, far_plane=far_plane,
#         radius_clip=radius_clip,
#         sparse_grad=False,
#         calc_compensations=False,
#         camera_model=camera_model,
#     )
#     # packed-mode tuple layout in gsplat>=1.0: (batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations)
#     batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations = proj
#     if gaussian_ids.numel() == 0:
#         return False

#     eps = 1e-12
#     denom = conics[:, 0] * conics[:, 2] - conics[:, 1]**2
#     det_cov2d = 1.0 / torch.clamp(denom, min=eps)

#     # Compute the transmittance using the depth, variance, and Gaussian depths
#     pixel_x = means2d[:, 0].long().clamp(0, width - 1)
#     pixel_y = means2d[:, 1].long().clamp(0, height - 1)
#     projected_pixel_ids = pixel_y * width + pixel_x  # shape [nnz]

#     depth_image_flattened = depth_image.reshape(-1)
#     variance_image_flattened = variance_image.reshape(-1)

#     # Add minimum variance threshold to prevent division by very small numbers
#     variance = torch.clamp(variance_image_flattened, min=1e-8)

#     # Compute Gaussian distribution 
#     normal_weights = torch.exp(-(1.0 / (2.0 * variance[projected_pixel_ids])) * (depths - depth_image_flattened[projected_pixel_ids]) ** 2)
#     normal_weights = normal_weights / torch.sqrt(2.0 * math.pi * variance[projected_pixel_ids])
#     normal_weights = (normal_weights **2) * torch.sqrt(det_cov2d)

#     # Update the accumulated transmittance
#     fig.index_put_((gaussian_ids,), normal_weights.unsqueeze(1), accumulate=True)

# TODO: Still need to put back the sum over the 2D ellipse.
# @torch.no_grad()
# def update_fig_for_frustum(
#     means: Tensor, quats: Tensor, scales: Tensor,
#     viewmats: Tensor, Ks: Tensor, width: int, height: int,
#     depth_image: Tensor, variance_image: Tensor,
#     fig: Tensor,
#     camera_model: str = "pinhole",
#     eps2d: float = 0.3,
#     near_plane: float = 1e-2,
#     far_plane: float = 1e10,
#     radius_clip: float = 0.0,
#     # Optional: if you have per-pixel accumulated alpha from moment_rasterization / rasterization
#     alpha_image: Optional[Tensor] = None,
#     reduce: Literal["amax", "sum", "mean"] = "sum",
# ) -> None:
#     device = means.device

#     proj = fully_fused_projection(
#         means, None, quats, scales,
#         viewmats, Ks, width, height,
#         eps2d=eps2d, packed=True,
#         near_plane=near_plane, far_plane=far_plane,
#         radius_clip=radius_clip,
#         sparse_grad=False,
#         calc_compensations=False,
#         camera_model=camera_model,
#     )

#     batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations = proj
#     if gaussian_ids.numel() == 0:
#         return

#     eps = 1e-12
#     denom = conics[:, 0] * conics[:, 2] - conics[:, 1]**2
#     det_cov2d = 1.0 / torch.clamp(denom, min=eps)  # = 1/det(conic)

#     # Pixel ids
#     pixel_x = means2d[:, 0].long().clamp(0, width - 1)
#     pixel_y = means2d[:, 1].long().clamp(0, height - 1)
#     pix = pixel_y * width + pixel_x  # [nnz] in [0, H*W)

#     # If you have multiple cameras in this call, avoid mixing pixels across cameras:
#     HW = width * height
#     group_id = camera_ids.long() * HW + pix  # [nnz] unique per (camera, pixel)
#     num_groups = int(group_id.max().item()) + 1  # safe upper bound for scatter buffers

#     # Flatten depth/var (assume depth_image/variance_image are [C,H,W] or [H,W] compatible)
#     depth_flat = depth_image.reshape(-1)       # must align with per-camera layout if C>1
#     var_flat   = variance_image.reshape(-1)

#     # If depth_image includes cameras, you likely want per-camera indexing too:
#     # depth_flat should be shaped [C*H*W]. If depth_image is [C,H,W], this works.
#     depth_mu = depth_flat[camera_ids.long() * HW + pix]
#     var = torch.clamp(var_flat[camera_ids.long() * HW + pix], min=1e-8)

#     # ---- Softmax logits (log of your old "pdf^2 * sqrt(det_cov2d)" up to constants) ----
#     # old weight was: exp(-(d-mu)^2/var) / (2*pi*var) * sqrt(det_cov2d)
#     # logit = -(d-mu)^2/var - log(var) + 0.5*log(det_cov2d)  (dropping constant -log(2*pi))
#     diff = depths - depth_mu
#     logits = -(diff * diff) / var
#     logits = logits - torch.log(var) + 0.5 * torch.log(torch.clamp(det_cov2d, min=eps))

#     # ---- Segment softmax over group_id ----
#     neg_inf = torch.tensor(-float("inf"), device=device, dtype=logits.dtype)
#     max_per = torch.full((num_groups,), neg_inf, device=device, dtype=logits.dtype)
#     max_per.scatter_reduce_(0, group_id, logits, reduce="amax", include_self=True)

#     exp_logits = torch.exp(logits - max_per[group_id])
#     sumexp = torch.zeros((num_groups,), device=device, dtype=logits.dtype)
#     sumexp.scatter_add_(0, group_id, exp_logits)

#     probs = exp_logits / (sumexp[group_id] + 1e-12)  # in [0,1], sums to 1 per (camera,pixel)

#     # Optional: scale so sums match per-pixel opacity if you have it
#     if alpha_image is not None:
#         alpha_flat = alpha_image.reshape(-1)
#         alpha = alpha_flat[camera_ids.long() * HW + pix].clamp(0.0, 1.0)
#         weights = alpha * probs
#     else:
#         weights = probs

#     weights = weights**2 # * torch.sqrt(det_cov2d)

#     # Accumulate into fig per Gaussian
#     fig.scatter_reduce_(0, gaussian_ids, weights, reduce=reduce, include_self=True)

@torch.no_grad()
def update_fig_for_frustum(
    means: Tensor, quats: Tensor, scales: Tensor,
    viewmats: Tensor, Ks: Tensor, width: int, height: int,
    depth_image: Tensor, variance_image: Tensor,
    fig: Tensor,
    camera_model: str = "pinhole",
    eps2d: float = 0.3,
    near_plane: float = 1e-2,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    # new knobs
    reduce: Literal["amax", "sum", "mean"] = "sum",          # "sum" or "amax"
    alpha_image: Optional[Tensor] = None,  # if provided, primitive weights sum to alpha per pixel
) -> None:
    device = means.device

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
    batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations = proj
    if gaussian_ids.numel() == 0:
        return

    eps = 1e-12
    denom = conics[:, 0] * conics[:, 2] - conics[:, 1]**2
    det_cov2d = 1.0 / torch.clamp(denom, min=eps)                 # ~ det(Sigma_2D)
    footprint = torch.sqrt(torch.clamp(det_cov2d, min=eps))       # sqrt(det(Sigma_2D))

    # pixels
    pixel_x = means2d[:, 0].long().clamp(0, width - 1)
    pixel_y = means2d[:, 1].long().clamp(0, height - 1)
    pix = pixel_y * width + pixel_x
    HW = width * height

    # group by (camera, pixel)
    cam_ids = camera_ids.long()
    num_cams = int(cam_ids.max().item()) + 1
    group_id = cam_ids * HW + pix
    num_groups = num_cams * HW

    # index into per-camera images if available; otherwise treat as single camera [HW]
    depth_flat = depth_image.reshape(-1)
    var_flat   = variance_image.reshape(-1)

    if depth_flat.numel() == HW:
        idx_pix = pix
    else:
        idx_pix = cam_ids * HW + pix

    mu_depth = depth_flat[idx_pix]
    var = torch.clamp(var_flat[idx_pix], min=1e-8)

    # ---- primitive weights w in [0,1] via softmax (NO footprint here) ----
    diff = depths - mu_depth
    logits = -(diff * diff) / var
    # logits = logits - torch.log(var)  # optional, matches your old 1/var factor

    w_prim = segment_softmax(logits, group_id, num_groups)  # sums to 1 per (cam,pix)

    # optional: scale so per-pixel sum is alpha (still each w in [0,1])
    if alpha_image is not None:
        alpha_flat = alpha_image.reshape(-1)
        alpha = alpha_flat[idx_pix].clamp(0.0, 1.0)
        w_prim = w_prim * alpha

    # ---- what you actually want to accumulate/store ----
    contrib = (w_prim * w_prim) # * footprint  # may exceed 1 (OK per your description)

    fig.scatter_reduce_(0, gaussian_ids, contrib, reduce=reduce, include_self=True)

def update_view_fig_for_frustum(
    means: Tensor, quats: Tensor, scales: Tensor,
    viewmats: Tensor, Ks: Tensor, width: int, height: int,
    depth_image: Tensor, variance_image: Tensor, bin_dirs: Tensor,
    view_fig: Tensor,
    camera_model: str = "pinhole",
    eps2d: float = 0.3,
    near_plane: float = 1e-2,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    concentration: Optional[float] = 1.0,
    # new knobs
    reduce: Literal["amax", "sum", "mean"] = "sum",          # "sum" or "amax"
    alpha_image: Optional[Tensor] = None,  # if provided, primitive weights sum to alpha per pixel
) -> None:
    device = means.device

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
    batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations = proj
    if gaussian_ids.numel() == 0:
        return

    eps = 1e-12
    denom = conics[:, 0] * conics[:, 2] - conics[:, 1]**2
    det_cov2d = 1.0 / torch.clamp(denom, min=eps)
    footprint = torch.sqrt(torch.clamp(det_cov2d, min=eps))

    # directions for spherical gaussian weights
    camtoworlds = torch.inverse(viewmats)  # [C,4,4]
    dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [N,3]
    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)

    bin_dirs = bin_dirs / (torch.norm(bin_dirs, dim=-1, keepdim=True) + 1e-10)
    sg_weights = spherical_gaussian_weights(dirs, bin_dirs, concentration)  # [N,G]

    # pixels
    pixel_x = means2d[:, 0].long().clamp(0, width - 1)
    pixel_y = means2d[:, 1].long().clamp(0, height - 1)
    pix = pixel_y * width + pixel_x
    HW = width * height

    cam_ids = camera_ids.long()
    num_cams = int(cam_ids.max().item()) + 1
    group_id = cam_ids * HW + pix
    num_groups = num_cams * HW

    depth_flat = depth_image.reshape(-1)
    var_flat   = variance_image.reshape(-1)

    if depth_flat.numel() == HW:
        idx_pix = pix
    else:
        idx_pix = cam_ids * HW + pix

    mu_depth = depth_flat[idx_pix]
    var = torch.clamp(var_flat[idx_pix], min=1e-8)

    # primitive weights (NO footprint here)
    diff = depths - mu_depth
    logits = -(diff * diff) / var
    # logits = logits - torch.log(var)

    w_prim = segment_softmax(logits, group_id, num_groups)

    if alpha_image is not None:
        alpha_flat = alpha_image.reshape(-1)
        alpha = alpha_flat[idx_pix].clamp(0.0, 1.0)
        w_prim = w_prim * alpha

    contrib = (w_prim * w_prim) # * footprint              # [N]
    combined = (sg_weights * sg_weights) * contrib[:, None]  # [N,G]

    # scatter_reduce into view_fig by gaussian_id
    idx = gaussian_ids[:, None].expand(-1, combined.shape[1])
    view_fig.scatter_reduce_(0, idx, combined, reduce=reduce, include_self=True)

# @torch.no_grad()
# def update_view_fig_for_frustum(
#     means: Tensor, quats: Tensor, scales: Tensor,
#     viewmats: Tensor, Ks: Tensor, width: int, height: int,
#     depth_image: Tensor, variance_image: Tensor, bin_dirs: Tensor,
#     view_fig: Tensor,
#     camera_model: str = "pinhole",
#     eps2d: float = 0.3,
#     near_plane: float = 1e-2,
#     far_plane: float = 1e10,
#     radius_clip: float = 0.0,
#     concentration: Optional[float] = 1.0,
# ) -> None:
#     """
#     """
#     device = means.device
#     # Project gaussians to figure out which are in the frustum.
#     proj = fully_fused_projection(
#         means, None, quats, scales,
#         viewmats, Ks, width, height,
#         eps2d=eps2d, packed=True,
#         near_plane=near_plane, far_plane=far_plane,
#         radius_clip=radius_clip,
#         sparse_grad=False,
#         calc_compensations=False,
#         camera_model=camera_model,
#     )
#     # packed-mode tuple layout in gsplat>=1.0: (batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations)
#     batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations = proj
#     if gaussian_ids.numel() == 0:
#         return False

#     eps = 1e-12
#     denom = conics[:, 0] * conics[:, 2] - conics[:, 1]**2
#     det_cov2d = 1.0 / torch.clamp(denom, min=eps)

#     # One optical axis for this camera batch (we render one training cam at a time)
#     camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
#     # These two lines assume packed = True
#     dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz, 3]
#     dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)

#     # Now bin the directions into their respective patches on the unit sphere
#     bin_dirs = bin_dirs / (torch.norm(bin_dirs, dim=-1, keepdim=True) + 1e-10)

#     # Compute the weights using a spherical gaussian kernel
#     sg_weights = spherical_gaussian_weights(dirs, bin_dirs, concentration)        # [N, G]

#     # Compute the transmittance using the depth, variance, and Gaussian depths
#     pixel_x = means2d[:, 0].long().clamp(0, width - 1)
#     pixel_y = means2d[:, 1].long().clamp(0, height - 1)
#     projected_pixel_ids = pixel_y * width + pixel_x  # shape [nnz]

#     depth_image_flattened = depth_image.reshape(-1)
#     variance_image_flattened = variance_image.reshape(-1)

#     # Add minimum variance threshold to prevent division by very small numbers
#     variance = torch.clamp(variance_image_flattened, min=1e-8)

#     # Compute Gaussian distribution 
#     normal_weights = torch.exp(-(1.0 / (2.0 * variance[projected_pixel_ids])) * (depths - depth_image_flattened[projected_pixel_ids]) ** 2)
#     normal_weights = normal_weights / torch.sqrt(2.0 * math.pi * variance[projected_pixel_ids])
#     normal_weights = (normal_weights **2) * torch.sqrt(det_cov2d)

#     # Update the accumulated view transmittance
#     combined_weight = (sg_weights**2) * normal_weights[:, None]  # [N, G]    

#     # Update the accumulated transmittance
#     view_fig.index_put_((gaussian_ids,), combined_weight, accumulate=True)

# @torch.no_grad()
# def update_view_fig_for_frustum(
#     means: Tensor, quats: Tensor, scales: Tensor,
#     viewmats: Tensor, Ks: Tensor, width: int, height: int,
#     depth_image: Tensor, variance_image: Tensor, bin_dirs: Tensor,
#     view_fig: Tensor,
#     camera_model: str = "pinhole",
#     eps2d: float = 0.3,
#     near_plane: float = 1e-2,
#     far_plane: float = 1e10,
#     radius_clip: float = 0.0,
#     concentration: Optional[float] = 1.0,
#     # new: how to reduce when multiple updates hit same gaussian_id
#     reduce: Literal["amax", "sum", "mean"] = "sum",
#     alpha_image: Optional[Tensor] = None,  # optional per-pixel alpha if you want sum-to-alpha
# ) -> None:
#     device = means.device

#     proj = fully_fused_projection(
#         means, None, quats, scales,
#         viewmats, Ks, width, height,
#         eps2d=eps2d, packed=True,
#         near_plane=near_plane, far_plane=far_plane,
#         radius_clip=radius_clip,
#         sparse_grad=False,
#         calc_compensations=False,
#         camera_model=camera_model,
#     )

#     batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations = proj
#     if gaussian_ids.numel() == 0:
#         return

#     eps = 1e-12
#     denom = conics[:, 0] * conics[:, 2] - conics[:, 1] ** 2
#     det_cov2d = 1.0 / torch.clamp(denom, min=eps)  # scalar per nnz

#     # ---- directions + spherical gaussian kernel ----
#     camtoworlds = torch.inverse(viewmats)  # [C,4,4]
#     dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz,3]
#     dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)

#     bin_dirs = bin_dirs / (torch.norm(bin_dirs, dim=-1, keepdim=True) + 1e-10)

#     sg_weights = spherical_gaussian_weights(dirs, bin_dirs, concentration)  # [nnz, G]

#     # ---- pixel grouping ids ----
#     pixel_x = means2d[:, 0].long().clamp(0, width - 1)
#     pixel_y = means2d[:, 1].long().clamp(0, height - 1)
#     pix = pixel_y * width + pixel_x  # [nnz] in [0,HW)
#     HW = width * height

#     # group by (camera, pixel) so different cameras don't mix
#     group_id = camera_ids.long() * HW + pix  # [nnz]
#     num_groups = int(group_id.max().item()) + 1

#     # ---- gather per-(camera,pixel) depth mean/var (if provided) ----
#     depth_flat = depth_image.reshape(-1)
#     var_flat   = variance_image.reshape(-1)

#     # IMPORTANT: assumes depth/var are laid out as [C*H*W] if multiple cameras
#     idx_pix = camera_ids.long() * HW + pix
#     mu_depth = depth_flat[idx_pix]
#     var = torch.clamp(var_flat[idx_pix], min=1e-8)

#     # ---- softmax approximation for depth/footprint weights ----
#     # Use logits = log(old_weight) up to constants:
#     # old: exp(-(d-mu)^2/var) / (2*pi*var) * sqrt(det_cov2d)
#     # logit: -(d-mu)^2/var - log(var) + 0.5*log(det_cov2d) (+ const)
#     diff = depths - mu_depth
#     logits = -(diff * diff) / var
#     logits = logits - torch.log(var) + 0.5 * torch.log(torch.clamp(det_cov2d, min=eps))

#     # segment log-sum-exp for stability
#     neg_inf = torch.tensor(-float("inf"), device=device, dtype=logits.dtype)
#     max_per = torch.full((num_groups,), neg_inf, device=device, dtype=logits.dtype)
#     max_per.scatter_reduce_(0, group_id, logits, reduce="amax", include_self=True)

#     exp_logits = torch.exp(logits - max_per[group_id])
#     sumexp = torch.zeros((num_groups,), device=device, dtype=logits.dtype)
#     sumexp.scatter_add_(0, group_id, exp_logits)

#     depth_probs = exp_logits / (sumexp[group_id] + 1e-12)  # [nnz], sums to 1 per (cam,pix)

#     # optional: scale to per-pixel alpha if you have it
#     if alpha_image is not None:
#         alpha_flat = alpha_image.reshape(-1)
#         alpha = alpha_flat[idx_pix].clamp(0.0, 1.0)
#         depth_weights = alpha * depth_probs
#     else:
#         depth_weights = depth_probs

#     depth_weights = depth_weights**2 # * torch.sqrt(det_cov2d)

#     # ---- combine with spherical weights as before ----
#     combined_weight = (sg_weights ** 2) * depth_weights[:, None]  # [nnz, G]

#     # ---- scatter_reduce_ into view_fig by gaussian_id ----
#     # view_fig expected shape [num_gaussians, G]
#     # gaussian_ids shape [nnz]
#     # combined_weight shape [nnz, G]

#     # scatter_reduce along dim=0 using expanded indices
#     idx = gaussian_ids[:, None].expand(-1, combined_weight.shape[1])
#     view_fig.scatter_reduce_(0, idx, combined_weight, reduce=reduce, include_self=True)

# @torch.no_grad()
# def update_fig_color_field_for_frustum(
#     means: Tensor, quats: Tensor, scales: Tensor,
#     viewmats: Tensor, Ks: Tensor, width: int, height: int,
#     depth_image: Tensor, variance_image: Tensor,
#     visibility: List[Tensor],
#     gaussian_ids: List[Tensor], 
#     pointer_length: Tensor,
#     train_cam_pos_list: List[Tensor],
#     camera_model: str = "pinhole",
#     eps2d: float = 0.3,
#     near_plane: float = 1e-2,
#     far_plane: float = 1e10,
#     radius_clip: float = 0.0,
# ) -> None:
#     """
#     """
#     device = means.device
#     # Project gaussians to figure out which are in the frustum.
#     proj = fully_fused_projection(
#         means, None, quats, scales,
#         viewmats, Ks, width, height,
#         eps2d=eps2d, packed=True,
#         near_plane=near_plane, far_plane=far_plane,
#         radius_clip=radius_clip,
#         sparse_grad=False,
#         calc_compensations=False,
#         camera_model=camera_model,
#     )
#     # packed-mode tuple layout in gsplat>=1.0: (batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations)
#     batch_ids, _, gs_ids, radii, means2d, depths, conics, compensations = proj
#     if gs_ids.numel() == 0:
#         return False

#     eps = 1e-12
#     denom = conics[:, 0] * conics[:, 2] - conics[:, 1]**2
#     det_cov2d = 1.0 / torch.clamp(denom, min=eps)

#     # Compute the transmittance using the depth, variance, and Gaussian depths
#     pixel_x = means2d[:, 0].long().clamp(0, width - 1)
#     pixel_y = means2d[:, 1].long().clamp(0, height - 1)
#     projected_pixel_ids = pixel_y * width + pixel_x  # shape [nnz]

#     depth_image_flattened = depth_image.reshape(-1)
#     variance_image_flattened = variance_image.reshape(-1)

#     # Add minimum variance threshold to prevent division by very small numbers
#     variance = torch.clamp(variance_image_flattened, min=1e-8)

#     # Compute Gaussian distribution 
#     normal_weights = torch.exp(-(1.0 / (2.0 * variance[projected_pixel_ids])) * (depths - depth_image_flattened[projected_pixel_ids]) ** 2)
#     normal_weights = normal_weights / torch.sqrt(2.0 * math.pi * variance[projected_pixel_ids])
#     print(f"normal_weights: {normal_weights.max()}")
#     normal_weights = (normal_weights **2) * torch.sqrt(det_cov2d)

#     # Update the accumulated transmittance
#     visibility.append(normal_weights)    # [C, N]
#     # num_hits.index_put_((camera_id.repeat(gs_ids.shape[0]), gs_ids), torch.sqrt(det_cov2d), accumulate=False)    # [C, N]

#     # Store camera ids and gaussian ids into list of tensors
#     gaussian_ids.append(gs_ids)

#     pointer_length.scatter_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=pointer_length.dtype))

#     cam_pos = torch.inverse(viewmats)[..., :3, 3].squeeze()
#     train_cam_pos_list.append(cam_pos)

#     print(f"torch.sqrt(det_cov2d): {torch.sqrt(det_cov2d).max()}")

# @torch.no_grad()
# def update_fig_color_field_for_frustum(
#     means: Tensor, quats: Tensor, scales: Tensor,
#     viewmats: Tensor, Ks: Tensor, width: int, height: int,
#     depth_image: Tensor, variance_image: Tensor,
#     visibility_list: List[Tensor],
#     gaussian_ids_list: List[Tensor], 
#     pointer_length: Tensor,
#     train_cam_pos_list: List[Tensor],
#     camera_model: str = "pinhole",
#     eps2d: float = 0.3,
#     near_plane: float = 1e-2,
#     far_plane: float = 1e10,
#     radius_clip: float = 0.0,
#     alpha_image: Optional[Tensor] = None,
# ) -> None:
#     """
#     """
#     device = means.device

#     proj = fully_fused_projection(
#         means, None, quats, scales,
#         viewmats, Ks, width, height,
#         eps2d=eps2d, packed=True,
#         near_plane=near_plane, far_plane=far_plane,
#         radius_clip=radius_clip,
#         sparse_grad=False,
#         calc_compensations=False,
#         camera_model=camera_model,
#     )

#     batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations = proj
#     if gaussian_ids.numel() == 0:
#         return

#     eps = 1e-12
#     denom = conics[:, 0] * conics[:, 2] - conics[:, 1]**2
#     det_cov2d = 1.0 / torch.clamp(denom, min=eps)  # = 1/det(conic)

#     # Pixel ids
#     pixel_x = means2d[:, 0].long().clamp(0, width - 1)
#     pixel_y = means2d[:, 1].long().clamp(0, height - 1)
#     pix = pixel_y * width + pixel_x  # [nnz] in [0, H*W)

#     # If you have multiple cameras in this call, avoid mixing pixels across cameras:
#     HW = width * height
#     group_id = camera_ids.long() * HW + pix  # [nnz] unique per (camera, pixel)
#     num_groups = int(group_id.max().item()) + 1  # safe upper bound for scatter buffers

#     # Flatten depth/var (assume depth_image/variance_image are [C,H,W] or [H,W] compatible)
#     depth_flat = depth_image.reshape(-1)       # must align with per-camera layout if C>1
#     var_flat   = variance_image.reshape(-1)

#     # If depth_image includes cameras, you likely want per-camera indexing too:
#     # depth_flat should be shaped [C*H*W]. If depth_image is [C,H,W], this works.
#     depth_mu = depth_flat[camera_ids.long() * HW + pix]
#     var = torch.clamp(var_flat[camera_ids.long() * HW + pix], min=1e-8)

#     # ---- Softmax logits (log of your old "pdf^2 * sqrt(det_cov2d)" up to constants) ----
#     # old weight was: exp(-(d-mu)^2/var) / (2*pi*var) * sqrt(det_cov2d)
#     # logit = -(d-mu)^2/var - log(var) + 0.5*log(det_cov2d)  (dropping constant -log(2*pi))
#     diff = depths - depth_mu
#     logits = -(diff * diff) / var
#     logits = logits - torch.log(var) + 0.5 * torch.log(torch.clamp(det_cov2d, min=eps))

#     # ---- Segment softmax over group_id ----
#     neg_inf = torch.tensor(-float("inf"), device=device, dtype=logits.dtype)
#     max_per = torch.full((num_groups,), neg_inf, device=device, dtype=logits.dtype)
#     max_per.scatter_reduce_(0, group_id, logits, reduce="amax", include_self=True)

#     exp_logits = torch.exp(logits - max_per[group_id])
#     sumexp = torch.zeros((num_groups,), device=device, dtype=logits.dtype)
#     sumexp.scatter_add_(0, group_id, exp_logits)

#     probs = exp_logits / (sumexp[group_id] + 1e-12)  # in [0,1], sums to 1 per (camera,pixel)

#     # Optional: scale so sums match per-pixel opacity if you have it
#     if alpha_image is not None:
#         alpha_flat = alpha_image.reshape(-1)
#         alpha = alpha_flat[camera_ids.long() * HW + pix].clamp(0.0, 1.0)
#         weights = alpha * probs
#     else:
#         weights = probs

#     weights = weights**2 # * torch.sqrt(det_cov2d)

#     # Update the accumulated transmittance
#     visibility_list.append(weights)    # [C, N]
#     # num_hits.index_put_((camera_id.repeat(gs_ids.shape[0]), gs_ids), torch.sqrt(det_cov2d), accumulate=False)    # [C, N]

#     # Store camera ids and gaussian ids into list of tensors
#     gaussian_ids_list.append(gaussian_ids)

#     pointer_length.scatter_add_(0, gaussian_ids, torch.ones_like(gaussian_ids, dtype=pointer_length.dtype))

#     cam_pos = torch.inverse(viewmats)[..., :3, 3].squeeze()
#     train_cam_pos_list.append(cam_pos)

@torch.no_grad()
def update_fig_color_field_for_frustum(
    means: Tensor, quats: Tensor, scales: Tensor,
    viewmats: Tensor, Ks: Tensor, width: int, height: int,
    depth_image: Tensor, variance_image: Tensor,
    visibility: List[Tensor],
    gaussian_ids: List[Tensor],
    pointer_length: Tensor,
    train_cam_pos_list: List[Tensor],
    camera_model: str = "pinhole",
    eps2d: float = 0.3,
    near_plane: float = 1e-2,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    alpha_image: Optional[Tensor] = None,
) -> None:
    device = means.device

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
    # (batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations)
    batch_ids, camera_ids, gs_ids, radii, means2d, depths, conics, compensations = proj
    if gs_ids.numel() == 0:
        return False

    eps = 1e-12
    denom = conics[:, 0] * conics[:, 2] - conics[:, 1] ** 2
    det_cov2d = 1.0 / torch.clamp(denom, min=eps)          # ~ det(Sigma_2D)
    footprint = torch.sqrt(torch.clamp(det_cov2d, min=eps)) # sqrt(det(Sigma_2D))

    # Pixel ids (you said you render one training cam at a time)
    pixel_x = means2d[:, 0].long().clamp(0, width - 1)
    pixel_y = means2d[:, 1].long().clamp(0, height - 1)
    pix = pixel_y * width + pixel_x  # [nnz]
    HW = width * height

    depth_flat = depth_image.reshape(-1)       # [HW] (or [C*HW] if batched cameras)
    var_flat   = variance_image.reshape(-1)

    mu_depth = depth_flat[pix]
    var = torch.clamp(var_flat[pix], min=1e-8)

    # ---- primitive rendering weights via segment-softmax over each pixel ----
    # logit ~ log N(depths | mu_depth, var) up to constants (no footprint here!)
    diff = depths - mu_depth
    logits = -(diff * diff) / var
    # logits = logits - torch.log(var)  # optional but usually helps (matches your old 1/var factor)

    # segment softmax (stable)
    neg_inf = torch.tensor(-float("inf"), device=device, dtype=logits.dtype)
    max_per_pix = torch.full((HW,), neg_inf, device=device, dtype=logits.dtype)
    max_per_pix.scatter_reduce_(0, pix, logits, reduce="amax", include_self=True)

    exp_logits = torch.exp(logits - max_per_pix[pix])
    sumexp = torch.zeros((HW,), device=device, dtype=logits.dtype)
    sumexp.scatter_add_(0, pix, exp_logits)

    w_prim = exp_logits / (sumexp[pix] + 1e-12)  # in [0,1], sums to 1 per pixel

    if alpha_image is not None:
        alpha_flat = alpha_image.reshape(-1)
        alpha = alpha_flat[pix].clamp(0.0, 1.0)
        w_prim = w_prim * alpha

    # ---- your desired stored quantity: (primitive weight)^2 * footprint ----
    contrib = (w_prim * w_prim) # * footprint      # may exceed 1; that's OK by your definition

    visibility.append(contrib)   # [nnz]
    gaussian_ids.append(gs_ids)

    pointer_length.scatter_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=pointer_length.dtype))

    cam_pos = torch.inverse(viewmats)[..., :3, 3].squeeze()
    train_cam_pos_list.append(cam_pos)

@torch.no_grad()
def compute_fig_color_field_metric(
    training_cameras_positions: Tensor, # [C, 3]
    training_camera_ids: Tensor, # [M]
    training_visibilities: Tensor, # [M]
    view_attributes: Tensor, # [G, 2]
    means: Tensor, # [G, 3]
    inference_dirs: Tensor, # [G, 3]  # Only for Gaussians in the frustum (N <= G)
    masks: Tensor, # [G]
    kappa: float = 1.0,
) -> None:
    """
    """
    masks_squeezed = masks.squeeze()
    inference_dirs_squeezed = inference_dirs.squeeze()
    inference_dirs_squeezed = inference_dirs_squeezed / (torch.norm(inference_dirs_squeezed, dim=-1, keepdim=True) + 1e-10)

    camera_ids_pointer_start = view_attributes[:, 0].to(torch.int64)
    camera_ids_pointer_length = view_attributes[:, 1].to(torch.int64)

    gaussian_ids = gaussian_ids_from_csr(camera_ids_pointer_start, camera_ids_pointer_length) # [M]

    # Only keep the indices of Gaussians that are in the frustum
    mask_expanded = masks_squeezed[gaussian_ids] # [M]
    gaussian_ids_valid = gaussian_ids[mask_expanded] # [K]
    training_camera_ids_valid = training_camera_ids[mask_expanded] # [K]

    training_cameras_positions_expanded_valid = training_cameras_positions[training_camera_ids_valid]
    training_gaussian_positions_expanded_valid = means[gaussian_ids_valid] # [K, 3]

    training_view_directions = training_gaussian_positions_expanded_valid - training_cameras_positions_expanded_valid
    training_view_directions = training_view_directions / (torch.norm(training_view_directions, dim=-1, keepdim=True) + 1e-10) # [K, 3]

    test_view_directions = inference_dirs_squeezed[gaussian_ids_valid] # [K, 3]

    dot_product = dot_product_spherical_gaussians(
                        training_view_directions, # [K, 3]
                        test_view_directions, # [K, 3]  
                        kappa = kappa)  # -> [K]

    visibilities_valid = training_visibilities[mask_expanded]      # [K]

    w_tilde_beta = (dot_product**2) * visibilities_valid # [K]

    # Use scatter_add to sum over all the cameras
    outgoing_view_attributes = torch.zeros((means.shape[0]), device=means.device)
    outgoing_view_attributes.scatter_reduce_(0, gaussian_ids_valid, w_tilde_beta, "sum")     # [G, 1]

    # print(f"visibilities_valid: {visibilities_valid.max()}")
    # print(f"dot_product: {dot_product.max()}")
    # print(f"w_tilde_beta: {w_tilde_beta.max()}")
    # print(f"outgoing view attributes max: {outgoing_view_attributes}")

    return torch.sqrt(outgoing_view_attributes.squeeze())

def gaussian_ids_from_csr(pointer_start: torch.Tensor, pointer_length: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct gaussian_ids (length K) corresponding to entries in cam_pool,
    given CSR-style per-gaussian metadata.

    Args:
        obs_start: [G] int64 (not actually needed for this reconstruction)
        obs_len:   [G] int32/int64

    Returns:
        gaussian_ids: [K] int64, where K = obs_len.sum()
                     This aligns with cam_pool (and any parallel pools like w_pool).
    """
    G = pointer_length.numel()
    device = pointer_length.device
    lengths = pointer_length.to(torch.int64)

    gaussian_ids = torch.repeat_interleave(
        torch.arange(G, device=device, dtype=torch.int64),
        lengths
    )
    return gaussian_ids

# @torch.no_grad()
# def compute_color_field_visibility_for_frustum(
#     view_directions_train: Tensor, # [M, 3]
#     view_directions_test: Tensor, # [G, 3]  # NOTE: The view directions of the candidate camera with respect to all Gaussians. Therefore, view_directions that are 0 are not in the frustum.
#     gaussian_ids_train: Tensor, # [M]
#     camera_ids_train: Tensor, # [M]
#     visibility: Tensor, # [M]
#     attribute: Tensor, # [G]        # This is W_tilde_beta_norm_sqr
#     kappa: float = 1.0,
# ) -> None:
#     """
#     Compute the visibility of the color field for a given camera.
#     """

#     # TODO: Put in camera ids and gaussian ids into list of tensors

#     dot_product, mask = dot_product_spherical_gaussians(
#                         view_directions_train, # [M, 3]
#                         view_directions_test, # [N, 3]  # NOTE: The view directions of the candidate camera with respect to all Gaussians. Therefore, view_directions that are 0 are not in the frustum.
#                         gaussian_ids_train, # [M]
#                         kappa = kappa)  # [M]

#     gaussian_ids_valid = gaussian_ids_train[mask]
#     camera_ids_valid = camera_ids_train[mask]

#     visibility_expanded = visibility[camera_ids_valid, gaussian_ids_valid]      # M

#     w_tilde_beta = (dot_product**2) * visibility_expanded

#     # Use scatter_add to sum over all the cameras
#     attribute.scatter_add_(0, gaussian_ids_valid, w_tilde_beta)     # N

@torch.no_grad()
def dot_product_spherical_gaussians(
    view_directions_train: Tensor, # [M, 3]
    view_directions_test: Tensor, # [M, 3]  # NOTE: The view directions of the candidate camera with respect to all Gaussians. Therefore, view_directions that are 0 are not in the frustum.
    kappa: float = 1.0,
) -> Tensor:
    """
    Dot product between two spherical gaussians in continuous space. Note, we do NOT normalize the directions in this function. Make sure the directions
    are normalized before calling this function.
    """
    # Compute the dot product
    dot_product = inner_ptilde(view_directions_train, view_directions_test, kappa)

    return dot_product

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
    small_k = two_kappa < eps

    if small_k:
        tk = torch.tensor(two_kappa, device=mu1.device)
        sinh_2k_approx = tk + (tk**3) / 6
        pref = two_kappa / sinh_2k_approx
    else:
        tk = torch.tensor(two_kappa, device=mu1.device)
        pref = tk / torch.sinh(tk)

    # final inner product
    # broadcasting: pref and h should be broadcastable to s's shape
    return pref * h
