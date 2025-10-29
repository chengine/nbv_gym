import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import Literal

from gsplat.cuda._wrapper import (
    RollingShutterType,
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    fully_fused_projection,
    fully_fused_projection_2dgs,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    rasterize_to_pixels_2dgs,
    rasterize_to_pixels_eval3d,
    spherical_harmonics,
)
from gsplat.distributed import (
    all_gather_int32,
    all_gather_tensor_list,
    all_to_all_int32,
    all_to_all_tensor_list,
)
from gsplat.utils import depth_to_normal, get_projection_matrix

from gsplat.cuda._torch_impl import _fully_fused_projection

from nerfstudio.cameras.cameras import Cameras, CameraType
import matplotlib.pyplot as plt
from torchvision.transforms.functional import gaussian_blur
from shadow_splat.util.coverage import *
from shadow_splat.util.sampling import *


def generate_point_cloud_from_camera_depth(
    depth: torch.Tensor,
    K: Optional[torch.Tensor] = None,
    W: Optional[int] = None,
    H: Optional[int] = None,
    viewmat: Optional[torch.Tensor] = None,
    camera: Optional[Cameras] = None,
    camera_indices: Optional[torch.Tensor] = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    mask: Optional[torch.Tensor] = None,
    use_bounding_box: bool = False,
    bounding_box_min: Optional[torch.Tensor] = None,
    bounding_box_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if depth.ndim == 4:
        depth = depth.squeeze(0)
        depth = depth.squeeze(-1)
    elif depth.ndim == 3:
        depth = depth.squeeze(-1)

    if K is not None:
        if K.ndim == 3:
            K = K.squeeze(0)
        if viewmat.ndim == 3:
            viewmat = viewmat.squeeze(0)

    if camera is not None and camera_indices is not None:
        raybundle = camera.generate_rays(camera_indices)
        points = raybundle.origins + raybundle.directions * depth[..., None]
        view_direction = raybundle.directions

    # Project depth to 3D using K matrix
    # unnormalized pixel coordinates
    elif K is not None and viewmat is not None and W is not None and H is not None:
        u_coords = torch.arange(W, device=depth.device)
        v_coords = torch.arange(H, device=depth.device)

        # meshgrid
        U_grid, V_grid = torch.meshgrid(u_coords, v_coords, indexing="xy")

        # transformed points in camera frame
        # [u, v, 1] = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]] @ [x/z, y/z, 1]
        cam_pts_x = (U_grid - K[0, 2]) * depth / K[0, 0]
        cam_pts_y = (V_grid - K[1, 2]) * depth / K[1, 1]
        points = torch.stack((cam_pts_x, cam_pts_y, depth), axis=-1)

        c2w = torch.linalg.inv(viewmat)
        points = points @ c2w[:3, :3].T + c2w[:3, 3]
    else:
        raise ValueError("Either camera and camera_indices or K and viewmat must be provided")

    if mask is not None:
        points = points[mask]
        # view_direction = view_direction[mask]
    else:
        mask = torch.logical_and(depth > near_plane, depth < far_plane)
        points = points[mask]
        # view_direction = view_direction[mask]

    # if use_bounding_box:
    #     comp_l = torch.tensor(bounding_box_min, device=point.device)
    #     comp_m = torch.tensor(bounding_box_max, device=point.device)
    #     assert torch.all(
    #         comp_l < comp_m
    #     ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
    #     mask = torch.all(torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1)
    #     point = point[mask]
    #     view_direction = view_direction[mask]

    # The output will ostensibly be N x 3 while the depth and rgb is H x w x C, so we need to mask in order to know which values
    # in the rgb image to touch.
    return points, mask


def logistic_weighting(
    means2d: Tensor,  # [N, 2] where N is the number of gaussians in the frustum
    depths: Tensor,  # [N] where N is the number of gaussians in the frustum
    depth_image: Tensor,  # [H, W]
    variance_image: Tensor,  # [H, W]
    variance_factor: Optional[float] = 1.0,
    cutoff: Optional[float] = 0.3,
    hard_cutoff: Optional[bool] = False,
):
    # NOTE: Only evaluates the logistic distribution for the gaussians in the frustum!
    # Any logic that indexes into the total number of gaussians in the scene should be done outside of this function!

    H, W = depth_image.shape
    N, _ = means2d.shape

    assert N == depths.shape[0], "Number of means and depths must match"
    assert depth_image.shape == variance_image.shape, (
        "Depth and variance images must have the same shape"
    )

    # Check inputs for NaN values
    if torch.isnan(means2d).any():
        raise ValueError("NaN values detected in means2d")
    if torch.isnan(depths).any():
        raise ValueError("NaN values detected in depths")
    if torch.isnan(depth_image).any():
        raise ValueError("NaN values detected in depth_image")
    if torch.isnan(variance_image).any():
        raise ValueError("NaN values detected in variance_image")
    if variance_factor is not None and math.isnan(variance_factor):
        raise ValueError("NaN value detected in variance_factor")
    if cutoff is not None and math.isnan(cutoff):
        raise ValueError("NaN value detected in cutoff")

    # NOTE: These means correspond to Gaussians that are in the frustum!
    pixel_x = means2d[:, 0].long().clamp(0, W - 1)
    pixel_y = means2d[:, 1].long().clamp(0, H - 1)
    projected_pixel_ids = pixel_y * W + pixel_x  # shape [nnz]

    depth_image_flattened = depth_image.reshape(-1)
    variance_image_flattened = variance_image.reshape(-1)

    # Add minimum variance threshold to prevent division by very small numbers
    variance = torch.clamp(variance_image_flattened, min=1e-8)
    # s = torch.sqrt(variance_factor / (math.pi) ** 2 * variance)  # n_pixels
    s = variance_factor * torch.sqrt(3 * variance) / math.pi

    sigmoid_argument = (depths - depth_image_flattened[projected_pixel_ids]) / s[
        projected_pixel_ids
    ]

    sigmoid_weights = 1.0 - torch.sigmoid(sigmoid_argument)

    # Use learnable cutoff parameter for differentiable masking
    if hard_cutoff:
        # During evaluation, use the provided cutoff for consistency
        lit_mask = sigmoid_weights > cutoff
        sigmoid_weights = torch.clamp(sigmoid_weights + lit_mask, 0.0, 1.0)
    else:
        smooth_mask = torch.sigmoid((sigmoid_weights - cutoff) * 10.0)  # 10.0 controls sharpness
        # Apply the smooth mask to create differentiable lighting weights
        sigmoid_weights = smooth_mask + (1.0 - smooth_mask) * sigmoid_weights

    return sigmoid_weights


# one-tailed chebyshev weighting
def chebyshev_weighting(
    means2d: Tensor,  # [N, 2] where N is the number of gaussians in the frustum
    depths: Tensor,  # [N] where N is the number of gaussians in the frustum
    depth_image: Tensor,  # [H, W]
    variance_image: Tensor,  # [H, W]
    baseline: Optional[float] = None,
):
    H, W = depth_image.shape
    N, _ = means2d.shape

    assert N == depths.shape[0], "Number of means and depths must match"
    assert depth_image.shape == variance_image.shape, (
        "Depth and variance images must have the same shape"
    )

    # NOTE: These means correspond to Gaussians that are in the frustum!
    pixel_x = means2d[:, 0].long().clamp(0, W - 1)
    pixel_y = means2d[:, 1].long().clamp(0, H - 1)
    projected_pixel_ids = pixel_y * W + pixel_x  # shape [nnz]

    depth_image_flattened = depth_image.reshape(-1)
    variance_image_flattened = variance_image.reshape(-1)

    # Add minimum variance threshold to prevent division by very small numbers
    variance = torch.clamp(variance_image_flattened, min=1e-6)
    variance_per_gaussian = variance[projected_pixel_ids]

    depth_diff = depths - depth_image_flattened[projected_pixel_ids]
    weights = variance_per_gaussian / (variance_per_gaussian + torch.relu(depth_diff) ** 2)

    if baseline is not None:
        # Does softmax weighting
        weights_ambient = torch.stack(
            [weights.squeeze(), torch.ones_like(weights.squeeze()) * baseline], dim=-1
        )
        softmax_weights = torch.softmax(weights_ambient, dim=-1)
        weights = torch.sum(softmax_weights * weights_ambient, dim=-1)

        # Does linear mixing
        # weights = (1 - baseline) * weights + baseline

        # Does hard cutoff
        # weights = torch.max(weights, torch.ones_like(weights) * baseline)

    return weights


@torch.no_grad()
def project_points_packed(
    points_world: torch.Tensor,  # (N, 3)
    viewmats: torch.Tensor,  # (C, 4, 4), OpenCV-style (camera looks +Z)
    Ks: torch.Tensor,  # (C, 3, 3)
    width: int,
    height: int,
    near: float = 1e-6,
    far: float = float("inf"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized point projection akin to fully_fused_projection, but for points only.

    Returns:
        means2d:      (M, 2) pixel coords (u, v) of valid projections
        depths:       (M,)   camera-space z (> near)
        gaussian_ids: (M,)   indices into points_world for those valid projections

    Notes:
      • Assumes OpenCV convention for viewmats (world→camera, +Z forward).
      • If you have a single camera, pass viewmats[None, ...] and Ks[None, ...].
      • Multiple cameras: the same point can appear multiple times (once per camera).
    """
    assert points_world.ndim == 2 and points_world.shape[1] == 3, "points_world must be (N,3)"
    assert viewmats.ndim == 3 and viewmats.shape[-2:] == (4, 4), "viewmats must be (C,4,4)"
    assert Ks.ndim == 3 and Ks.shape[-2:] == (3, 3), "Ks must be (C,3,3)"
    C = viewmats.shape[0]
    N = points_world.shape[0]

    device = points_world.device
    dtype = points_world.dtype

    # Homogeneous transform world -> camera for all cameras
    ones = torch.ones(N, 1, device=device, dtype=dtype)  # (N,1)
    Xw_h = torch.cat([points_world, ones], dim=-1)  # (N,4)
    # cam_h: (C,N,4)  (broadcasted batch matmul)
    cam_h = torch.einsum("cab,nb->cna", viewmats.to(device, dtype), Xw_h)
    Xc = cam_h[..., :3]  # (C,N,3)
    z = Xc[..., 2]  # (C,N)

    # Depth validity
    z_ok = (z > near) & (z < far)

    # Project: p = K @ Xc  (batched per camera), then divide by z
    # P: (C,N,3)
    P = torch.einsum("cab,cnb->cna", Ks.to(device, dtype), Xc)
    z_safe = z.clamp_min(near)
    u = P[..., 0] / z_safe
    v = P[..., 1] / z_safe

    # Finite + image bounds
    finite = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(z)
    in_w = (u >= 0) & (u < (width - 1e-6))
    in_h = (v >= 0) & (v < (height - 1e-6))
    valid = z_ok & finite & in_w & in_h  # (C,N)

    if not valid.any():
        empty2 = torch.empty(0, 2, device=device, dtype=dtype)
        empty1 = torch.empty(0, device=device, dtype=dtype)
        return empty2, empty1, empty1  # means2d, depths, gaussian_ids

    # Gather packed outputs
    cam_ids, point_ids = valid.nonzero(as_tuple=True)  # (M,), (M,)
    means2d = torch.stack(
        [u[cam_ids, point_ids], v[cam_ids, point_ids]], dim=-1
    ).contiguous()  # (M,2)
    depths = z[cam_ids, point_ids].contiguous()  # (M,)
    gaussian_ids = point_ids.contiguous()  # (M,)

    return means2d, depths, gaussian_ids


def calculate_relighting_weights_from_point_cloud(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]       # NOTE: IMPORTANT! THESE SCALES MUST ALREADY BE POSITIVE
    opacities: Tensor,  # [N]       # NOTE: IMPORTANT! THESE OPACITIES MUST ALREADY BE [0, 1]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    point_cloud: Tensor,
    point_cloud_mask: Tensor,
    ambient: Optional[float] = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    tile_size: int = 16,
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    fix_variance: bool = False,
    use_2dgs: bool = False,
    distloss: bool = False,  # 2DGS only
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the relighting weights for a given light source for the scene."""

    N, _ = means.shape  # Number of gaussians in the scene
    assert N == opacities.shape[0], "Number of means and opacities must match"
    assert N == scales.shape[0], "Number of means and scales must match"
    assert N == quats.shape[0], "Number of means and quaternions must match"

    if not use_2dgs:
        moments, alphas, meta = moment_rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            viewmats,  # [C, 4, 4]
            Ks,  # [C, 3, 3]
            width,
            height,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            eps2d=eps2d,
            packed=True,
            tile_size=tile_size,
            sparse_grad=sparse_grad,
            absgrad=absgrad,
            rasterize_mode=rasterize_mode,
            channel_chunk=channel_chunk,
            distributed=distributed,
            camera_model=camera_model,
        )
    else:
        (moments, alphas, normals, normals_from_depth, distort, median, meta) = (
            moment_rasterization_2dgs(
                means,
                quats,
                scales,
                opacities,
                viewmats,
                Ks,
                width,
                height,
                near_plane=near_plane,
                far_plane=far_plane,
                radius_clip=radius_clip,
                eps2d=eps2d,
                packed=True,
                tile_size=tile_size,
                sparse_grad=sparse_grad,
                absgrad=absgrad,
                distloss=distloss,
            )
        )

    assert not torch.isnan(moments[..., 1]).any(), "Depth squared is nan"
    assert not torch.isnan(moments[..., 0]).any(), "Depth is nan"
    assert not torch.isinf(moments[..., 1]).any(), "Depth squared is inf"
    assert not torch.isinf(moments[..., 0]).any(), "Depth is inf"

    depth_image = moments[..., 0].squeeze()

    if fix_variance:
        # TODO: Implement fix_variance
        variance_image = torch.ones_like(depth_image)
    else:
        depth_sqr_image = moments[..., 1].squeeze()
        variance_image = depth_sqr_image - depth_image**2

    # Gaussian blur both the depth and variance images
    depth_image = gaussian_blur(depth_image[None], kernel_size=5, sigma=1.0)
    variance_image = gaussian_blur(variance_image[None], kernel_size=5, sigma=1.0)
    depth_image = depth_image.squeeze()
    variance_image = variance_image.squeeze().clamp(min=1e-3)

    assert torch.isnan(depth_image).any() == False, "Depth image is nan"
    assert torch.isnan(variance_image).any() == False, "Variance image is nan"

    # Project point cloud onto the light source camera
    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    means = point_cloud[..., :3]

    # Get rid of Nans
    means = means[~torch.isnan(means).any(dim=-1)]
    means = means[~torch.isinf(means).any(dim=-1)]

    covars = torch.randn(means.shape[0], 3, 3, device=means.device)
    covars = covars @ covars.transpose(-2, -1)

    ### NOTE: USING THE PROVIDED CUDA CODE RESULTS IN FLOATING POINT EXCEPTIONS (CORE DUMPED) ERRORS WHEN VIEWING SCENE FROM THE GROUND UP
    # Extract upper triangular part of covars, converting N x 3 x 3 to N x 6
    # i, j = torch.triu_indices(3, 3)  # order: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
    # covars = covars[:, i, j]

    # proj_results = fully_fused_projection(
    #     means,
    #     covars,
    #     None,
    #     None,
    #     viewmats,
    #     Ks,
    #     width,
    #     height,
    #     packed=True,
    #     near_plane=0.01,
    #     far_plane=1e10,
    #     camera_model=camera_model,
    # )

    # # The results are packed into shape [nnz, ...]. All elements are valid.
    # (   _,
    #     _,
    #     gaussian_ids,
    #     radii,
    #     means2d,
    #     depths,
    #     _,
    #     _,
    # ) = proj_results

    # assert (radii > 0).all(), "Radii are zero"

    radii, means2d, depths, conics, compensations = _fully_fused_projection(
        means,
        covars,
        viewmats,
        Ks,
        width,
        height,
        near_plane=0.01,
        far_plane=1e10,
        camera_model=camera_model,
    )
    radii = radii.squeeze(0)
    means2d = means2d.squeeze(0)
    depths = depths.squeeze(0)

    valid = (radii[:, 0] > 0) & (radii[:, 1] > 0)

    means2d = means2d[valid]
    depths = depths[valid]

    # This represents the fraction of light that is received by each gaussian in the frustum
    weights = chebyshev_weighting(
        means2d,  # [N, 2] where N is the number of gaussians in the frustum
        depths,  # [N] where N is the number of gaussians in the frustum
        depth_image,  # [H, W]
        variance_image,  # [H, W]
        baseline=ambient,
    )

    assert not torch.isnan(weights).any(), "Weights are nan"
    assert not torch.isinf(weights).any(), "Weights are inf"

    # We're going to make the assumption that if the pixel is not in the mask, we don't do anything to it.
    # TODO: Implement a flag that allows us to treat the pixels as dark if they're not in the mask.
    sel = point_cloud_mask.view(-1).nonzero(as_tuple=True)[0]  # [K]

    shadow_image = torch.zeros_like(point_cloud_mask).to(torch.float32)

    if len(sel) > 0:
        shadow_flat = shadow_image.reshape(-1)
        shadow_flat[sel[valid]] = 1.0 - weights
        shadow_image = shadow_flat.reshape(point_cloud_mask.shape)

    # shadow_image = torch.zeros_like(depth_image)

    return shadow_image, depth_image, variance_image


def calculate_relighting_weights(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]       # NOTE: IMPORTANT! THESE SCALES MUST ALREADY BE POSITIVE
    opacities: Tensor,  # [N]       # NOTE: IMPORTANT! THESE OPACITIES MUST ALREADY BE [0, 1]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    variance_factor: float,
    intensity: List[float],
    ambient: Optional[float] = None,
    background_ambient: Optional[float] = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    tile_size: int = 16,
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    fix_variance: bool = False,
    use_2dgs: bool = False,
    distloss: bool = False,  # 2DGS only
) -> Tuple[Tensor, Tensor, Dict]:
    """Compute the relighting weights for a given light source for the scene."""

    N, _ = means.shape  # Number of gaussians in the scene
    assert N == opacities.shape[0], "Number of means and opacities must match"
    assert N == scales.shape[0], "Number of means and scales must match"
    assert N == quats.shape[0], "Number of means and quaternions must match"

    if not use_2dgs:
        moments, alphas, meta = moment_rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            viewmats,  # [C, 4, 4]
            Ks,  # [C, 3, 3]
            width,
            height,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            eps2d=eps2d,
            packed=True,
            tile_size=tile_size,
            sparse_grad=sparse_grad,
            absgrad=absgrad,
            rasterize_mode=rasterize_mode,
            channel_chunk=channel_chunk,
            distributed=distributed,
            camera_model=camera_model,
        )
    else:
        (moments, alphas, normals, normals_from_depth, distort, median, meta) = (
            moment_rasterization_2dgs(
                means,
                quats,
                scales,
                opacities,
                viewmats,
                Ks,
                width,
                height,
                near_plane=near_plane,
                far_plane=far_plane,
                radius_clip=radius_clip,
                eps2d=eps2d,
                packed=True,
                tile_size=tile_size,
                sparse_grad=sparse_grad,
                absgrad=absgrad,
                distloss=distloss,
            )
        )

    assert not torch.isnan(moments[..., 1]).any(), "Depth squared is nan"
    assert not torch.isnan(moments[..., 0]).any(), "Depth is nan"
    assert not torch.isinf(moments[..., 1]).any(), "Depth squared is inf"
    assert not torch.isinf(moments[..., 0]).any(), "Depth is inf"

    depth_image = moments[..., 0].squeeze()

    if fix_variance:
        # TODO: Implement fix_variance
        variance_image = torch.ones_like(depth_image)
    else:
        depth_sqr_image = moments[..., 1].squeeze()
        variance_image = depth_sqr_image - depth_image**2

    # Gaussian blur both the depth and variance images
    depth_image = gaussian_blur(depth_image[None], kernel_size=49, sigma=3.0)
    variance_image = gaussian_blur(variance_image[None], kernel_size=49, sigma=3.0)
    depth_image = depth_image.squeeze()
    variance_image = variance_image.squeeze().clamp(min=5e-3)

    depths = meta["depths"]  # Depths of each gaussian in frustum
    means2d = meta["means2d"]  # 2D means of each gaussian in frustum
    gaussian_ids = meta["gaussian_ids"]  # Indices of the gaussians in the frustum
    conics = meta["conics"]

    assert (meta["radii"] >= 0.0).all(), "Radii must be non-negative"

    ### NOTE: UNCOMMENT THIS OUT IF YOU WANT TO ONLY COMPUTE WEIGHTS USING THE MEANS
    # This represents the fraction of light that is received by each gaussian in the frustum
    # weights = chebyshev_weighting(
    #     means2d,  # [N, 2] where N is the number of gaussians in the frustum
    #     depths,  # [N] where N is the number of gaussians in the frustum
    #     depth_image,  # [H, W]
    #     variance_image,  # [H, W]
    # )

    # Smoothing across whole Gaussian
    # Sample points in axis directions of the Gaussian
    assert torch.isnan(means2d).any() == False, "Means2d is nan"
    assert torch.isnan(depths).any() == False, "Depths is nan"
    assert torch.isnan(depth_image).any() == False, "Depth image is nan"
    assert torch.isnan(variance_image).any() == False, "Variance image is nan"

    extreme_points, interior_points = conics_to_semi_major_axis_points_analytic(
        means2d, conics, num_interior=50
    )
    sample_points = torch.cat([extreme_points, interior_points], dim=-2)

    num_points = sample_points.shape[-2]
    sample_points = sample_points.reshape(-1, 2)
    sample_depths = torch.repeat_interleave(depths, num_points)

    ### NOTE: THIS IS USING SAMPLES FROM THE FULL GAUSSIANS, NOT JUST THE MEANS, TO COMPUTE WEIGHTS
    weights = chebyshev_weighting(
        sample_points,  # [N, 2] where N is the number of gaussians in the frustum
        sample_depths,  # [N] where N is the number of gaussians in the frustum
        depth_image,  # [H, W]
        variance_image,  # [H, W]
        baseline=ambient,
    )

    weights = weights.reshape(means2d.shape[0], num_points)
    weights = weights.max(dim=1).values

    assert not torch.isnan(weights).any(), "Weights are nan"
    assert not torch.isinf(weights).any(), "Weights are inf"

    irradiance = torch.ones_like(means)
    irradiance_fraction = torch.ones_like(opacities)

    # The total intensity of the light that is received by each gaussian in the frustum is the product of the intensity of the light source and the fraction of light that is received by the gaussian
    # Additionally, if 2DGS, then a BRDF function is applied.
    if use_2dgs:
        cam2worlds = torch.inverse(viewmats)
        gaussian_normals = meta["normals"]
        gaussian_normals = gaussian_normals / torch.norm(gaussian_normals, dim=-1, keepdim=True)
        incident_dir = cam2worlds[meta["camera_ids"], :3, 3] - means[gaussian_ids, :]
        incident_dir = incident_dir / torch.norm(incident_dir, dim=-1, keepdim=True)
        cosine_angle = torch.sum(gaussian_normals * incident_dir, dim=-1)
        weights = weights * torch.relu(cosine_angle)

    # if ambient is not None:
    #     weights = weights + ambient

    irradiance[gaussian_ids] = torch.stack(
        [weights * intensity[0], weights * intensity[1], weights * intensity[2]], dim=-1
    )
    irradiance_fraction[gaussian_ids] = weights

    # if ambient is not None:
    #     irradiance += ambient  # ambient is in intensity space
    # if background_ambient is not None:
    #     irradiance[~gaussian_ids] += background_ambient
    # irradiance[~gaussian_ids] += intensity[0]

    return irradiance, irradiance_fraction, depth_image, variance_image


# Renders the accumulated or expected depth (first moment) and the accumulated
# or expected depth squared (second moment). Will add higher moments as necessary.
def moment_rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    covars: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    # TODO: Rewrite documentation to reflect the moment rendering

    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        **Multi-GPU Distributed Rasterization**: This function can be used in a multi-GPU
        distributed scenario by setting `distributed` to True. When `distributed` is True,
        a subset of total Gaussians could be passed into this function in each rank, and
        the function will collaboratively render a set of images using Gaussians from all ranks. Note
        to achieve balanced computation, it is recommended (not enforced) to have similar number of
        Gaussians in each rank. But we do enforce that the number of cameras to be rendered
        in each rank is the same. The function will return the rendered images
        corresponds to the input cameras in each rank, and allows for gradients to flow back to the
        Gaussians living in other ranks. For the details, please refer to the paper
        `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`_.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of images in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Support N-D Features**: If `sh_degree` is None,
        the `colors` is expected to be with shape [N, D] or [C, N, D], in which D is the channel of
        the features to be rendered. The computation is slow when D > 32 at the moment.
        If `sh_degree` is set, the `colors` is expected to be the SH coefficients with
        shape [N, K, 3] or [C, N, K, 3], where K is the number of SH bases. In this case, it is expected
        that :math:`(\\textit{sh_degree} + 1) ^ 2 \\leq K`, where `sh_degree` controls the
        activated bases in the SH coefficients.

    .. note::
        **Depth Rendering**: This function supports colors or/and depths via `render_mode`.
        The supported modes are "RGB", "D", "ED", "RGB+D", and "RGB+ED". "RGB" renders the
        colored image that respects the `colors` argument. "D" renders the accumulated z-depth
        :math:`\\sum_i w_i z_i`. "ED" renders the expected z-depth
        :math:`\\frac{\\sum_i w_i z_i}{\\sum_i w_i}`. "RGB+D" and "RGB+ED" render both
        the colored image and the depth, in which the depth is the last channel of the output.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [C, N, ...], which is faster but might consume more memory.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in pixel unit) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which is rarely happened in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered image more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        ortho: Whether to use orthographic projection. In such case fx and fy become the scaling
            factors to convert projected coordinates into pixel space and cx, cy become offsets.
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [N, 3, 3], Default is None.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, meta = rasterization(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])

    """
    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert quats.shape == (N, 4), quats.shape
        assert scales.shape == (N, 3), scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    # If in distributed mode, we distribute the projection computation over Gaussians
    # and the rasterize computation over cameras. So first we gather the cameras
    # from all ranks for projection.
    if distributed:
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Gather the number of Gaussians in each rank.
        N_world = all_gather_int32(world_size, N, device=device)

        # Enforce that the number of cameras is the same across all ranks.
        C_world = [C] * world_size
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])

        # Silently change C from local #Cameras to global #Cameras.
        C = len(viewmats)

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
    )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    # TODO: Get rid of colors variable
    if distributed:
        if packed:
            # count how many elements need to be sent to each rank
            cnts = torch.bincount(camera_ids, minlength=C)  # all cameras
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities],
                cnts,
                output_splits=collected_splits,
            )

            # before sending the data, we should turn the camera_ids from global to local.
            # i.e. the camera_ids produced by the projection stage are over all cameras world-wide,
            # so we need to turn them into camera_ids that are local to each rank.
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # and turn gaussian ids from local to global.
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # all to all communication across all ranks.
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

        else:
            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)

    # Rasterize to pixels
    moments = torch.stack((depths, depths**2), dim=-1)
    if backgrounds is not None:
        backgrounds = torch.zeros(C, 2, device=backgrounds.device)

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=C,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_images": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if moments.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (moments.shape[-1] + channel_chunk - 1) // channel_chunk
        render_moments, render_alphas = [], []
        for i in range(n_chunks):
            moments_chunk = moments[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_moments_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                moments_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_moments.append(render_moments_)
            render_alphas.append(render_alphas_)
        render_moments = torch.cat(render_moments, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_moments, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            moments,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )

    # We use expected depth
    render_moments = render_moments / render_alphas.clamp(min=1e-10)

    return render_moments, render_alphas, meta


# Regular rasterization, but allows to simultaneously render additional channels
# TODO: Do we want these additional channels to be view dependent?
def augmented_rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    additional_channels: Optional[Tensor] = None,  # [(C,) N, D2] or [(C,) N, K, D2]
    color_weights: Optional[Tensor] = None,  # [(C,) N, 3],
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    covars: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        **Multi-GPU Distributed Rasterization**: This function can be used in a multi-GPU
        distributed scenario by setting `distributed` to True. When `distributed` is True,
        a subset of total Gaussians could be passed into this function in each rank, and
        the function will collaboratively render a set of images using Gaussians from all ranks. Note
        to achieve balanced computation, it is recommended (not enforced) to have similar number of
        Gaussians in each rank. But we do enforce that the number of cameras to be rendered
        in each rank is the same. The function will return the rendered images
        corresponds to the input cameras in each rank, and allows for gradients to flow back to the
        Gaussians living in other ranks. For the details, please refer to the paper
        `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`_.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of images in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Support N-D Features**: If `sh_degree` is None,
        the `colors` is expected to be with shape [N, D] or [C, N, D], in which D is the channel of
        the features to be rendered. The computation is slow when D > 32 at the moment.
        If `sh_degree` is set, the `colors` is expected to be the SH coefficients with
        shape [N, K, 3] or [C, N, K, 3], where K is the number of SH bases. In this case, it is expected
        that :math:`(\\textit{sh_degree} + 1) ^ 2 \\leq K`, where `sh_degree` controls the
        activated bases in the SH coefficients.

    .. note::
        **Depth Rendering**: This function supports colors or/and depths via `render_mode`.
        The supported modes are "RGB", "D", "ED", "RGB+D", and "RGB+ED". "RGB" renders the
        colored image that respects the `colors` argument. "D" renders the accumulated z-depth
        :math:`\\sum_i w_i z_i`. "ED" renders the expected z-depth
        :math:`\\frac{\\sum_i w_i z_i}{\\sum_i w_i}`. "RGB+D" and "RGB+ED" render both
        the colored image and the depth, in which the depth is the last channel of the output.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [C, N, ...], which is faster but might consume more memory.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in pixel unit) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which is rarely happened in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered image more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        ortho: Whether to use orthographic projection. In such case fx and fy become the scaling
            factors to convert projected coordinates into pixel space and cx, cy become offsets.
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [N, 3, 3], Default is None.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, meta = rasterization(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])

    """
    # print('colors before', colors.isnan().any(), colors.isinf().any())
    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert quats.shape == (N, 4), quats.shape
        assert scales.shape == (N, 3), scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    if color_weights is not None:
        assert color_weights.shape == (N, 3), color_weights.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    # TODO: We may want to add more checks for the additional channels like with colors
    if additional_channels is not None:
        assert additional_channels.shape[0] == N, additional_channels.shape
        additional_channels = additional_channels[None].expand(C, -1, -1)

    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
        if distributed:
            assert colors.dim() == 2, "Distributed mode only supports per-Gaussian colors."
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or (
            colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
        if distributed:
            assert colors.dim() == 3, "Distributed mode only supports per-Gaussian colors."

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    # If in distributed mode, we distribute the projection computation over Gaussians
    # and the rasterize computation over cameras. So first we gather the cameras
    # from all ranks for projection.
    if distributed:
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Gather the number of Gaussians in each rank.
        N_world = all_gather_int32(world_size, N, device=device)

        # Enforce that the number of cameras is the same across all ranks.
        C_world = [C] * world_size
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])

        # Silently change C from local #Cameras to global #Cameras.
        C = len(viewmats)

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
    )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [N, D] or [C, N, D]
        if packed:
            if colors.dim() == 2:
                # Turn [N, D] into [nnz, D]
                colors = colors[gaussian_ids]
            else:
                # Turn [C, N, D] into [nnz, D]
                colors = colors[camera_ids, gaussian_ids]
        else:
            if colors.dim() == 2:
                # Turn [N, D] into [C, N, D]
                colors = colors.expand(C, -1, -1)
            else:
                # colors is already [C, N, D]
                pass

    else:
        # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
        camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
        if packed:
            dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz, 3]
            dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)
            masks = (radii > 0).any(-1)  # [nnz]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [nnz, 3]
                shs = colors[gaussian_ids, :, :]  # [nnz, K, 3]
            else:
                # Turn [C, N, K, 3] into [nnz, 3]
                shs = colors[camera_ids, gaussian_ids, :, :]  # [nnz, K, 3]
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [nnz, 3]
        else:
            dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
            dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)
            masks = (radii > 0).any(-1)  # [C, N]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [C, N, K, 3]
                shs = colors.expand(C, -1, -1, -1)  # [C, N, K, 3]
            else:
                # colors is already [C, N, K, 3]
                shs = colors
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    meta.update({"colors": colors})

    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    if distributed:
        if packed:
            # count how many elements need to be sent to each rank
            cnts = torch.bincount(camera_ids, minlength=C)  # all cameras
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities, colors],
                cnts,
                output_splits=collected_splits,
            )

            # before sending the data, we should turn the camera_ids from global to local.
            # i.e. the camera_ids produced by the projection stage are over all cameras world-wide,
            # so we need to turn them into camera_ids that are local to each rank.
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # and turn gaussian ids from local to global.
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # all to all communication across all ranks.
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

        else:
            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                    colors.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)
            colors = reshape_view(C, colors, N_world)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        if color_weights is not None:
            # The absolute color is the albedo (base color of Gaussian), the fraction of light reflected in each channel, times the intensity of the light incident on the Gaussian
            relit_colors = colors * color_weights
            colors = torch.cat((colors, relit_colors), dim=-1)

        if additional_channels is not None:
            colors = torch.cat((colors, additional_channels, depths[..., None]), dim=-1)
            background_channels_pad_size = additional_channels.shape[-1] + 1
        else:
            colors = torch.cat((colors, depths[..., None]), dim=-1)
            background_channels_pad_size = 1

        if backgrounds is not None and color_weights is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
        elif backgrounds is not None and color_weights is None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )

    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)

    else:  # RGB
        if color_weights is not None:
            # The absolute color is the albedo (base color of Gaussian), the fraction of light reflected in each channel, times the intensity of the light incident on the Gaussian
            relit_colors = colors * color_weights
            colors = torch.cat((colors, relit_colors), dim=-1)

        if additional_channels is not None:
            colors = torch.cat((colors, additional_channels), dim=-1)
            background_channels_pad_size = additional_channels.shape[-1]

        if (
            backgrounds is not None
            and color_weights is not None
            and additional_channels is not None
        ):
            backgrounds = torch.cat(
                [
                    backgrounds,
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
        elif backgrounds is not None and color_weights is not None and additional_channels is None:
            backgrounds = torch.cat([backgrounds, backgrounds], dim=-1)
        elif backgrounds is not None and color_weights is None and additional_channels is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=C,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    return render_colors, render_alphas, meta


# Renders the accumulated or expected depth (first moment) and the accumulated
# or expected variance (second moment) for 2DGS. Will add higher moments as necessary.
def moment_rasterization_2dgs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """Rasterize a set of 2D Gaussians (N) to a batch of image planes (C).

    This function supports a handful of features, similar to the :func:`rasterization` function.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad (Experimental): If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distloss: If true, use distortion regularization to get better geometry detail.
        depth_mode: render depth mode. Choose from expected depth and median depth.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **render_normals**: The rendered normals. [C, height, width, 3].

        **surf_normals**: surface normal from depth. [C, height, width, 3]

        **render_distort**: The rendered distortions. [C, height, width, 1].
        L1 version, different from L2 version in 2DGS paper.

        **render_median**: The rendered median depth. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (normals.shape, surf_normals.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 3])
        >>> print (distort.shape, median_depth.shape)
        torch.Size([1, 200, 300, 1]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'ray_transforms',
        'opacities', 'normals', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras', 'render_distort',
        'gradient_2dgs'])

    """

    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # Compute Ray-Splat intersection transformation.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        packed,
        sparse_grad,
    )

    if packed:
        (
            _,  # batch_ids
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = proj_results
        opacities = opacities[gaussian_ids]
    else:
        radii, means2d, depths, ray_transforms, normals = proj_results
        opacities = opacities.repeat(C, 1)
        camera_ids, gaussian_ids = None, None

    densify = torch.zeros_like(means2d, dtype=means.dtype, requires_grad=True, device="cuda")
    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=C,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # Rasterize to pixels
    colors = torch.stack((depths, depths**2), dim=-1)

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
        distloss=distloss,
    )
    render_normals_from_depth = None

    # normalize the accumulated depth to get the expected depth
    render_colors = render_colors / render_alphas.clamp(min=1e-10)

    if depth_mode == "expected":
        depth_for_normal = render_colors[..., 0:1]

    elif depth_mode == "median":
        depth_for_normal = render_median

    render_normals_from_depth = depth_to_normal(
        depth_for_normal, torch.linalg.inv(viewmats), Ks
    ).squeeze(0)

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "render_distort": render_distort,
        "gradient_2dgs": densify,  # This holds the gradient used for densification for 2dgs
    }

    render_normals = render_normals @ torch.linalg.inv(viewmats)[0, :3, :3].T

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
    )


# Regular rasterization, but allows to simultaneously render additional channels for 2DGS
# TODO: Do we want these additional channels to be view dependent?
def augmented_rasterization_2dgs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    additional_channels: Optional[Tensor] = None,  # [(C,) N, D2] or [(C,) N, K, D2]
    color_weights: Optional[Tensor] = None,  # [(C,) N, 3]
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """Rasterize a set of 2D Gaussians (N) to a batch of image planes (C).

    This function supports a handful of features, similar to the :func:`rasterization` function.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad (Experimental): If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distloss: If true, use distortion regularization to get better geometry detail.
        depth_mode: render depth mode. Choose from expected depth and median depth.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **render_normals**: The rendered normals. [C, height, width, 3].

        **surf_normals**: surface normal from depth. [C, height, width, 3]

        **render_distort**: The rendered distortions. [C, height, width, 1].
        L1 version, different from L2 version in 2DGS paper.

        **render_median**: The rendered median depth. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (normals.shape, surf_normals.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 3])
        >>> print (distort.shape, median_depth.shape)
        torch.Size([1, 200, 300, 1]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'ray_transforms',
        'opacities', 'normals', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras', 'render_distort',
        'gradient_2dgs'])

    """

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    channels = colors.shape[-1]

    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    if color_weights is not None:
        assert color_weights.shape == (N, 3), color_weights.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode
    if distloss:
        assert render_mode in [
            "D",
            "ED",
            "RGB+D",
            "RGB+ED",
        ], (
            f"distloss requires depth rendering, render_mode should be D, ED, RGB+D, RGB+ED, but got {render_mode}"
        )

    if additional_channels is not None:
        assert additional_channels.shape[0] == N, additional_channels.shape
        # additional_channels = additional_channels[None].expand(C, -1, -1)

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (colors.dim() == num_batch_dims + 2 and colors.shape[:-1] == batch_dims + (N,)) or (
            colors.dim() == num_batch_dims + 3 and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    # Compute Ray-Splat intersection transformation.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        packed,
        sparse_grad,
    )

    if packed:
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = proj_results
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]
        image_ids = batch_ids * C + camera_ids
    else:
        radii, means2d, depths, ray_transforms, normals = proj_results
        opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))  # [..., C, N]
        camera_ids, gaussian_ids = None, None
        image_ids = None

    densify = torch.zeros_like(
        means2d, dtype=means.dtype, requires_grad=True, device="cuda"
    )  # Identify intersecting tiles

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=I,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # TODO: SH also suport N-D.
    # Compute the per-view colors
    # if not (colors.dim() == 3 and sh_degree is None):  # silently support [C, N, D] color.
    #     colors = (
    #         colors[gaussian_ids] if packed else colors.expand(C, *([-1] * colors.dim()))
    #     )  # [nnz, D] or [C, N, 3]
    # else:
    #     if packed:
    #         colors = colors[camera_ids, gaussian_ids, :]

    if sh_degree is not None:  # SH coefficients
        camtoworlds = torch.inverse(viewmats)
        if packed:
            dirs = means[..., gaussian_ids, :] - camtoworlds[..., camera_ids, :3, 3]
        else:
            dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]

        if colors.dim() == num_batch_dims + 3:
            # Turn [..., N, K, 3] into [..., C, N, K, 3]
            shs = torch.broadcast_to(
                colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
            )  # [..., C, N, K, 3]
        else:
            # colors is already [..., C, N, K, 3]
            shs = colors
        colors = spherical_harmonics(
            sh_degree, dirs, shs, masks=(radii > 0).all(dim=-1)
        )  # [nnz, D] or [..., C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        if color_weights is not None:
            # The absolute color is the albedo (base color of Gaussian), the fraction of light reflected in each channel, times the intensity of the light incident on the Gaussian
            relit_colors = colors * color_weights
            colors = torch.cat((colors, relit_colors), dim=-1)

        if additional_channels is not None:
            colors = torch.cat((colors, additional_channels, depths[..., None]), dim=-1)
            background_channels_pad_size = additional_channels.shape[-1] + 1
        else:
            colors = torch.cat((colors, depths[..., None]), dim=-1)
            background_channels_pad_size = 1

        if backgrounds is not None and color_weights is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
        elif backgrounds is not None and color_weights is None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )

    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)

    else:  # RGB
        if color_weights is not None:
            # The absolute color is the albedo (base color of Gaussian), the fraction of light reflected in each channel, times the intensity of the light incident on the Gaussian
            relit_colors = colors * color_weights
            colors = torch.cat((colors, relit_colors), dim=-1)

        if additional_channels is not None:
            colors = torch.cat((colors, additional_channels), dim=-1)
            background_channels_pad_size = additional_channels.shape[-1]

        if (
            backgrounds is not None
            and color_weights is not None
            and additional_channels is not None
        ):
            backgrounds = torch.cat(
                [
                    backgrounds,
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
        elif backgrounds is not None and color_weights is not None and additional_channels is None:
            backgrounds = torch.cat([backgrounds, backgrounds], dim=-1)
        elif backgrounds is not None and color_weights is None and additional_channels is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
        distloss=distloss,
    )
    render_normals_from_depth = None
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode in ["RGB+ED", "RGB+D"]:
        # render_depths = render_colors[..., -1:]
        if depth_mode == "expected":
            depth_for_normal = render_colors[..., -1:]
        elif depth_mode == "median":
            depth_for_normal = render_median

        render_normals_from_depth = depth_to_normal(
            depth_for_normal, torch.linalg.inv(viewmats), Ks
        ).squeeze(0)

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "render_distort": render_distort,
        "gradient_2dgs": densify,  # This holds the gradient used for densification for 2dgs
    }

    render_normals = torch.einsum(
        "...ij,...hwj->...hwi", torch.linalg.inv(viewmats)[..., :3, :3], render_normals
    )

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
    )


def rasterization_with_coverage(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
    coverage_counts: Tensor,  # [..., N, G]
    bin_dirs: Tensor,  # [G, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
    segmented: bool = False,
    covars: Optional[Tensor] = None,
    with_ut: bool = False,
    with_eval3d: bool = False,
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        **Multi-GPU Distributed Rasterization**: This function can be used in a multi-GPU
        distributed scenario by setting `distributed` to True. When `distributed` is True,
        a subset of total Gaussians could be passed into this function in each rank, and
        the function will collaboratively render a set of images using Gaussians from all ranks. Note
        to achieve balanced computation, it is recommended (not enforced) to have similar number of
        Gaussians in each rank. But we do enforce that the number of cameras to be rendered
        in each rank is the same. The function will return the rendered images
        corresponds to the input cameras in each rank, and allows for gradients to flow back to the
        Gaussians living in other ranks. For the details, please refer to the paper
        `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`_.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of images in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Support N-D Features**: If `sh_degree` is None,
        the `colors` is expected to be with shape [..., N, D] or [..., C, N, D], in which D is the channel of
        the features to be rendered. The computation is slow when D > 32 at the moment.
        If `sh_degree` is set, the `colors` is expected to be the SH coefficients with
        shape [..., N, K, 3] or [..., C, N, K, 3], where K is the number of SH bases. In this case, it is expected
        that :math:`(\\textit{sh_degree} + 1) ^ 2 \\leq K`, where `sh_degree` controls the
        activated bases in the SH coefficients.

    .. note::
        **Depth Rendering**: This function supports colors or/and depths via `render_mode`.
        The supported modes are "RGB", "D", "ED", "RGB+D", and "RGB+ED". "RGB" renders the
        colored image that respects the `colors` argument. "D" renders the accumulated z-depth
        :math:`\\sum_i w_i z_i`. "ED" renders the expected z-depth
        :math:`\\frac{\\sum_i w_i z_i}{\\sum_i w_i}`. "RGB+D" and "RGB+ED" render both
        the colored image and the depth, in which the depth is the last channel of the output.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [..., C, N, ...], which is faster but might consume more memory.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in pixel unit) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which is rarely happened in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered image more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.

    .. note::
        **Camera Distortion and Rolling Shutter**: The function supports rendering with opencv
        distortion formula for pinhole and fisheye cameras (`radial_coeffs`, `tangential_coeffs`, `thin_prism_coeffs`).
        It also supports rolling shutter rendering with the `rolling_shutter` argument. We take
        reference from the paper `3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting
        <https://arxiv.org/abs/2412.12507>`_.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [..., N, 4]
        scales: The scales of the Gaussians. [..., N, 3]
        opacities: The opacities of the Gaussians. [..., N]
        colors: The colors of the Gaussians. [..., (C,) N, D] or [..., (C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [..., C, 4, 4]
        Ks: The camera intrinsics. [..., C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [..., (C,) N, K, 3] SH coefficients,
            else the `colors` should be [..., (C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [..., C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        camera_model: The camera model to use. Supported models are "pinhole", "ortho",
            "fisheye", and "ftheta". Default is "pinhole".
        segmented: Whether to use segmented radix sort. Default is False.
            Segmented radix sort performs sorting in segments, which is more efficient for the sorting operation itself.
            However, since it requires offset indices as input, additional global memory access is needed, which results
            in slower overall performance in most use cases.
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [..., N, 3, 3], Default is None.
        with_ut: Whether to use Unscented Transform (UT) for projection. Default is False.
        with_eval3d: Whether to calculate Gaussian response in 3D world space, instead
            of 2D image space. Default is False.
        radial_coeffs: Opencv pinhole/fisheye radial distortion coefficients. Default is None.
            For pinhole camera, the shape should be [..., C, 6]. For fisheye camera, the shape
            should be [..., C, 4].
        tangential_coeffs: Opencv pinhole tangential distortion coefficients. Default is None.
            The shape should be [..., C, 2] if provided.
        thin_prism_coeffs: Opencv pinhole thin prism distortion coefficients. Default is None.
            The shape should be [..., C, 4] if provided.
        ftheta_coeffs: F-Theta camera distortion coefficients shared for all cameras.
            Default is None. See `FThetaCameraDistortionParameters` for details.
        rolling_shutter: The rolling shutter type. Default `RollingShutterType.GLOBAL` means
            global shutter.
        viewmats_rs: The second viewmat when rolling shutter is used. Default is None.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [..., C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [..., C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, meta = rasterization(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])

    """
    meta = {}

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    assert means.shape == batch_dims + (N, 3), means.shape
    if covars is None:
        assert quats.shape == batch_dims + (N, 4), quats.shape
        assert scales.shape == batch_dims + (N, 3), scales.shape
    else:
        assert covars.shape == batch_dims + (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (colors.dim() == num_batch_dims + 2 and colors.shape[:-1] == batch_dims + (N,)) or (
            colors.dim() == num_batch_dims + 3 and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
        if distributed:
            assert colors.dim() == num_batch_dims + 2, (
                "Distributed mode only supports per-Gaussian colors."
            )
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
        if distributed:
            assert colors.dim() == num_batch_dims + 3, (
                "Distributed mode only supports per-Gaussian colors."
            )

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    if (
        radial_coeffs is not None
        or tangential_coeffs is not None
        or thin_prism_coeffs is not None
        or ftheta_coeffs is not None
        or rolling_shutter != RollingShutterType.GLOBAL
    ):
        assert with_ut, "Distortion and rolling shutter are only supported with `with_ut=True`."

    if rolling_shutter != RollingShutterType.GLOBAL:
        assert viewmats_rs is not None, "Rolling shutter requires to provide viewmats_rs."
    else:
        assert viewmats_rs is None, "viewmats_rs should be None for global rolling shutter."

    if with_ut or with_eval3d:
        assert (quats is not None) and (scales is not None), (
            "UT and eval3d requires to provide quats and scales."
        )
        assert packed is False, "Packed mode is not supported with UT."
        assert sparse_grad is False, "Sparse grad is not supported with UT."

    # Implement the multi-GPU strategy proposed in
    # `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`.
    #
    # If in distributed mode, we distribute the projection computation over Gaussians
    # and the rasterize computation over cameras. So first we gather the cameras
    # from all ranks for projection.
    if distributed:
        assert batch_dims == (), "Distributed mode does not support batch dimensions"
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Gather the number of Gaussians in each rank.
        N_world = all_gather_int32(world_size, N, device=device)

        # Enforce that the number of cameras is the same across all ranks.
        C_world = [C] * world_size
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])
        if viewmats_rs is not None:
            (viewmats_rs,) = all_gather_tensor_list(world_size, [viewmats_rs])

        # Silently change C from local #Cameras to global #Cameras.
        C = len(viewmats)

    if with_ut:
        proj_results = fully_fused_projection_with_ut(
            means,
            quats,
            scales,
            opacities,  # use opacities to compute a tigher bound for radii.
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            rolling_shutter=rolling_shutter,
            viewmats_rs=viewmats_rs,
        )

    else:
        # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
        proj_results = fully_fused_projection(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            packed=packed,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            sparse_grad=sparse_grad,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            opacities=opacities,  # use opacities to compute a tigher bound for radii.
        )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]  # [nnz]
        image_ids = batch_ids * C + camera_ids
    else:
        # The results are with shape [..., C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))  # [..., C, N]
        batch_ids, camera_ids, gaussian_ids = None, None, None
        image_ids = None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global batch and camera ids
            "batch_ids": batch_ids,
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    # Turn colors into [..., C, N, D] or [..., nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [..., N, D] or [..., C, N, D]
        if packed:
            if colors.dim() == num_batch_dims + 2:
                # Turn [..., N, D] into [nnz, D]
                colors = colors.view(B, N, -1)[batch_ids, gaussian_ids]
            else:
                # Turn [..., C, N, D] into [nnz, D]
                colors = colors.view(B, C, N, -1)[batch_ids, camera_ids, gaussian_ids]
        else:
            if colors.dim() == num_batch_dims + 2:
                # Turn [..., N, D] into [..., C, N, D]
                colors = torch.broadcast_to(colors[..., None, :, :], batch_dims + (C, N, -1))
            else:
                # colors is already [..., C, N, D]
                pass
    else:
        # Colors are SH coefficients, with shape [..., N, K, 3] or [..., C, N, K, 3]
        campos = torch.inverse(viewmats)[..., :3, 3]  # [..., C, 3]
        if viewmats_rs is not None:
            campos_rs = torch.inverse(viewmats_rs)[..., :3, 3]
            campos = 0.5 * (campos + campos_rs)  # [..., C, 3]
        if packed:
            dirs = (
                means.view(B, N, 3)[batch_ids, gaussian_ids]
                - campos.view(B, C, 3)[batch_ids, camera_ids]
            )  # [nnz, 3]
            masks = (radii > 0).all(dim=-1)  # [nnz]
            if colors.dim() == num_batch_dims + 3:
                # Turn [..., N, K, 3] into [nnz, 3]
                shs = colors.view(B, N, -1, 3)[batch_ids, gaussian_ids]  # [nnz, K, 3]
            else:
                # Turn [..., C, N, K, 3] into [nnz, 3]
                shs = colors.view(B, C, N, -1, 3)[
                    batch_ids, camera_ids, gaussian_ids
                ]  # [nnz, K, 3]
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [nnz, 3]
        else:
            dirs = means[..., None, :, :] - campos[..., None, :]  # [..., C, N, 3]
            masks = (radii > 0).all(dim=-1)  # [..., C, N]
            if colors.dim() == num_batch_dims + 3:
                # Turn [..., N, K, 3] into [..., C, N, K, 3]
                shs = torch.broadcast_to(colors[..., None, :, :, :], batch_dims + (C, N, -1, 3))
            else:
                # colors is already [..., C, N, K, 3]
                shs = colors
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [..., C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    coverage_metric = compute_coverage_per_gaussian(
        coverage_counts=coverage_counts,
        bin_dirs=bin_dirs,
        masks=masks.squeeze(0),
        inference_dirs=dirs.squeeze(0),
    )

    # Concatenate coverage_metric with colors
    colors = torch.cat((colors, coverage_metric[None, ..., None]), dim=-1)

    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    if distributed:
        if packed:
            # count how many elements need to be sent to each rank
            cnts = torch.bincount(camera_ids, minlength=C)  # all cameras
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities, colors],
                cnts,
                output_splits=collected_splits,
            )

            # before sending the data, we should turn the camera_ids from global to local.
            # i.e. the camera_ids produced by the projection stage are over all cameras world-wide,
            # so we need to turn them into camera_ids that are local to each rank.
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # and turn gaussian ids from local to global.
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # all to all communication across all ranks.
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

        else:
            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                    colors.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)
            colors = reshape_view(C, colors, N_world)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, 1), device=backgrounds.device),
                ],
                dim=-1,
            )
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(batch_dims + (C, 1), device=backgrounds.device)
    else:  # RGB
        pass

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=segmented,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_batches": B,
            "n_cameras": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            if with_eval3d:
                render_colors_, render_alphas_ = rasterize_to_pixels_eval3d(
                    means,
                    quats,
                    scales,
                    colors_chunk,
                    opacities,
                    viewmats,
                    Ks,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    camera_model=camera_model,
                    radial_coeffs=radial_coeffs,
                    tangential_coeffs=tangential_coeffs,
                    thin_prism_coeffs=thin_prism_coeffs,
                    ftheta_coeffs=ftheta_coeffs,
                    rolling_shutter=rolling_shutter,
                    viewmats_rs=viewmats_rs,
                )
            else:
                render_colors_, render_alphas_ = rasterize_to_pixels(
                    means2d,
                    conics,
                    colors_chunk,
                    opacities,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    packed=packed,
                    absgrad=absgrad,
                )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        if with_eval3d:
            render_colors, render_alphas = rasterize_to_pixels_eval3d(
                means,
                quats,
                scales,
                colors,
                opacities,
                viewmats,
                Ks,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds,
                camera_model=camera_model,
                radial_coeffs=radial_coeffs,
                tangential_coeffs=tangential_coeffs,
                thin_prism_coeffs=thin_prism_coeffs,
                ftheta_coeffs=ftheta_coeffs,
                rolling_shutter=rolling_shutter,
                viewmats_rs=viewmats_rs,
            )
        else:
            render_colors, render_alphas = rasterize_to_pixels(
                means2d,
                conics,
                colors,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds,
                packed=packed,
                absgrad=absgrad,
            )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    return render_colors, render_alphas, meta
