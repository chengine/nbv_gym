# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
import time
import torch
from pytorch_msssim import SSIM
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch.nn.functional as tf
import math
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB

from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.rendering import rasterization, rasterization_2dgs
from gsplat.cuda._wrapper import rasterize_to_indices_in_range_2dgs
from nerfacc import render_weight_from_alpha

from splatfacto_2dgs.model import Splatfacto2DGSModelConfig, Splatfacto2DGSModel
from shadow_splat.util.nerfstudio import resize_image, get_viewmat


# @torch.compile
def apply_weight_to_RGB(
    input: torch.Tensor,
    weights: torch.Tensor,
    intensity: torch.Tensor,
    tonemapping_type: Optional[Literal["linear", "reinhard", "luminance"]] = "linear",
    gamma_correction: Optional[bool] = True,
):
    ### TODO: How to handle spherical harmonics???
    if input.dim() == 3:  # higher order spherical harmonics
        new_color = input

    else:  # direct color
        new_color = weights * SH2RGB(input)  # NOTE: THIS IS THE ORIGINAL LINEAR WEIGHTING
        update_color_mask = weights.squeeze(-1) < 1.0

        updated_color = new_color[update_color_mask]

        # Apply tonemapping to the color
        updated_color[:, 0] *= intensity[0]
        updated_color[:, 1] *= intensity[1]
        updated_color[:, 2] *= intensity[2]

        # Does Reinhard tonemapping
        if tonemapping_type == "reinhard":
            updated_color /= updated_color + 1.0

        # Does luminance tonemapping
        elif tonemapping_type == "luminance":
            luminance = (
                0.2126 * updated_color[:, 0]
                + 0.7152 * updated_color[:, 1]
                + 0.0722 * updated_color[:, 2]
            )
            updated_color /= (luminance + 1.0)[:, None]

        # Does linear tonemapping
        elif tonemapping_type == "linear":
            updated_color = torch.clamp(updated_color, min=0.0, max=1.0)

        # Does gamma correction # NOTE: leads to nans during training
        # if gamma_correction:
        #     updated_color = updated_color ** (1.0 / 2.2)

        new_color[update_color_mask] = updated_color
        new_color = RGB2SH(new_color)

    return new_color


# @torch.compile
def prepare_weights(meta: Dict[str, torch.Tensor]):
    C, N = meta["means2d"].shape[:2]
    pixel_ids_x = meta["pixel_ids"] % meta["width"]
    pixel_ids_y = meta["pixel_ids"] // meta["width"]
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
    deltas = pixel_coords - meta["means2d"][meta["camera_ids"], meta["gs_ids"]]  # [M, 2]

    # 2DGS: ray_transforms are 2x2 matrices stored as [a, b, c, d]
    ray_transforms = meta["ray_transforms"][meta["camera_ids"], meta["gs_ids"]]  # [M, 4]

    # Reshape to 2x2 matrices
    transforms_2x2 = ray_transforms.view(-1, 2, 2)  # [M, 2, 2]

    # Apply transformation to deltas: transformed_deltas = T @ deltas
    transformed_deltas = torch.bmm(transforms_2x2, deltas.unsqueeze(-1)).squeeze(-1)  # [M, 2]

    # Compute squared distance in transformed space
    sigmas = 0.5 * torch.sum(transformed_deltas**2, dim=-1)  # [M]

    alphas = torch.clamp_max(
        meta["opacities"][meta["camera_ids"], meta["gs_ids"]] * torch.exp(-sigmas), 0.999
    )

    indices = meta["camera_ids"] * meta["height"] * meta["width"] + meta["pixel_ids"]
    total_pixels = C * meta["height"] * meta["width"]

    return alphas, indices, total_pixels


@dataclass
class ShadowSplat2DGSModelConfig(Splatfacto2DGSModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: ShadowSplat2DGSModel)
    # TODO: Additional params here


class ShadowSplat2DGSModel(Splatfacto2DGSModel):
    """Nerfstudio's implementation of Shadow Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: ShadowSplat2DGSModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        self.light_params = torch.nn.ParameterDict(
            {
                "intensity": torch.nn.Parameter(torch.log(torch.ones(3))),
                "cutoff": torch.nn.Parameter(torch.log(torch.tensor(0.5))),
                "variance_factor": torch.nn.Parameter(torch.logit(torch.tensor(0.01))),
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        self.shadow_fn = None
        self.lighting_weights = None

        ### TODO: Learn this lighting_fn!!!
        # self.reweighting_param = torch.nn.Parameter(torch.randn(1, device='cuda'))
        # self.reweighting_fn = lambda x: torch.pow(x, (torch.nn.functional.softplus(self.reweighting_param)))

    def update_light_source(
        self,
        light_source: Cameras,
        mask: Optional[torch.Tensor] = None,
        reduce: Optional[Literal["mean", "amax", "amin"]] = "amax",
        variance_factor: Optional[float] = None,
        intensity: Optional[List[float]] = None,
        cutoff: Optional[float] = None,
    ):
        """Update the light source, generating a new shadow function for the scene."""

        # Use learnable parameters if not provided
        if variance_factor is None:
            variance_factor = torch.exp(self.light_params["variance_factor"])
        if intensity is None:
            intensity = torch.exp(self.light_params["intensity"])
        if cutoff is None:
            cutoff = torch.sigmoid(self.light_params["cutoff"])

        ######
        # Investigate if this is optimizing the light source pose
        if self.training:
            assert light_source.shape[0] == 1, "Only one camera at a time"
            optimized_light_to_world = self.camera_optimizer.apply_to_camera(light_source)
        else:
            optimized_light_to_world = light_source.camera_to_worlds
        # optimized_light_to_world = light_source.camera_to_worlds
        #######

        # NOTE: ignore cropping for now
        opacities_crop = self.opacities
        means_crop = self.means
        features_dc_crop = self.features_dc
        features_rest_crop = self.features_rest
        scales_crop = self.scales
        quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop, features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        print("camera_scale_fac: ", camera_scale_fac)
        light_source.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_light_to_world)
        K = light_source.get_intrinsics_matrices().cuda()
        W, H = int(light_source.width.item()), int(light_source.height.item())
        self.last_size = (H, W)
        light_source.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if light_source.camera_type == CameraType.PERSPECTIVE.value:
            camera_model = "pinhole"
        elif light_source.camera_type == CameraType.ORTHOPHOTO.value:
            camera_model = "ortho"
        elif light_source.camera_type == CameraType.FISHEYE.value:
            camera_model = "fisheye"
        else:
            raise ValueError("Unknown camera type: %s", light_source.camera_type)

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        depth, alphas, normals, normals_from_depth, render_distort, render_median, meta = (
            rasterization_2dgs(
                means=means_crop,
                quats=quats_crop,  # rasterization does normalization internally
                scales=torch.exp(scales_crop),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                colors=colors_crop,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="ED",
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=self.strategy.absgrad
                if isinstance(self.strategy, DefaultStrategy)
                else False,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
                # camera_model=camera_model,
            )
        )

        # Compute 3D camera space coordinates once and reuse them
        w2c = torch.eye(4, device=light_source.camera_to_worlds[0].device)
        w2c[:3] = light_source.camera_to_worlds[0, :3]
        w2c = torch.linalg.inv(w2c)

        # Transform all means to camera space
        means_camera_space = (w2c[:3, :3] @ means_crop.T).T + w2c[:3, 3][None]

        meta["means2d"] = meta["means2d"].unsqueeze(0)
        meta["ray_transforms"] = meta["ray_transforms"].unsqueeze(0)
        meta["opacities"] = meta["opacities"].unsqueeze(0)

        # pixel_ids/gaussian ids are sorted in order of depth of gaussians

        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range_2dgs(
            0,
            1000,
            torch.ones(1, meta["height"], meta["width"], device=self.device),
            meta["means2d"],
            meta["ray_transforms"],
            meta["opacities"],
            meta["width"],
            meta["height"],
            meta["tile_size"],
            meta["isect_offsets"],
            meta["flatten_ids"],
        )

        meta["gs_ids"] = gs_ids
        meta["pixel_ids"] = pixel_ids
        meta["camera_ids"] = camera_ids

        alphas, indices, total_pixels = prepare_weights(meta)

        # Returns the weights and the transmittances
        weights, _ = render_weight_from_alpha(alphas, ray_indices=indices, n_rays=total_pixels)

        ### TODO: FIND ALL GAUSSIAN PIXEL INTERSECTIONS IN THE PROJECTION STEP (NOT THE RASTERIZATION STEP)
        ### TODO: REPLACE ADVANCED INDEXING WITH MULTIPLICATION AND ADDITION FOR FASTER COMPUTATION
        # Calculate the distance of rasterized Gaussians to the light source to get logistic parameters
        means_rasterized_camera = means_camera_space[gs_ids]
        distances = -means_rasterized_camera[:, 2]  # torch.norm(diff, dim=-1)

        # Use the weights to calculate the variance of the fitted logistic function for each ray
        depth_flattened = depth.reshape(-1)[pixel_ids]
        centered_distances_squared = (distances - depth_flattened) ** 2

        # Sum up the centered distance for each ray
        variance = torch.zeros(total_pixels, device=self.device)
        variance.scatter_add_(0, pixel_ids, weights * centered_distances_squared)

        # Add minimum variance threshold to prevent division by very small numbers
        variance = torch.clamp(variance, min=1e-8)

        s = torch.sqrt(variance_factor / (math.pi) ** 2 * variance)  # n_pixels

        # Apply the logistic function CDF weighting to all projected Gaussians
        xy = meta["means2d"].squeeze()  # shape [nnz, 2], in pixel coordinates
        H, W = meta["height"], meta["width"]
        pixel_x = xy[:, 0].long().clamp(0, W - 1)
        pixel_y = xy[:, 1].long().clamp(0, H - 1)
        projected_pixel_ids = pixel_y * W + pixel_x  # shape [nnz]
        projected_gs_ids = meta["gaussian_ids"].squeeze()

        means_projected_camera = means_camera_space[projected_gs_ids]
        projected_gs_distances = -means_projected_camera[:, 2]

        sigmoid_argument = (projected_gs_distances - depth.reshape(-1)[projected_pixel_ids]) / s[
            projected_pixel_ids
        ]

        sigmoid_weights = 1.0 - torch.sigmoid(sigmoid_argument)

        # Use learnable cutoff parameter for differentiable masking
        if self.training:
            smooth_mask = torch.sigmoid(
                (sigmoid_weights - cutoff) * 10.0
            )  # 10.0 controls sharpness
            # Apply the smooth mask to create differentiable lighting weights
            sigmoid_weights = smooth_mask + (1.0 - smooth_mask) * sigmoid_weights
        else:
            # During evaluation, use the provided cutoff for consistency
            lit_mask = sigmoid_weights > cutoff
            sigmoid_weights = torch.clamp(sigmoid_weights + lit_mask, 0.0, 1.0)

        # Way to reduce sigmoid weights into n_gaussians
        # NOTE: IF WE SET THE DEFAULT TO 1, THEN GAUSSIANS NOT IN THE FRUSTUM ARE WELL-LIT
        weights_ = torch.ones(self.means.shape[0], device=self.device)
        # TODO: Might need this if we find that only applying weighting to projected gaussians based only on their means is not sufficient
        # for accuracy
        # weights_.scatter_reduce_(0, projected_gs_ids, sigmoid_weights, reduce="amax", include_self=False)
        weights_[projected_gs_ids] = sigmoid_weights
        self.lighting_weights = weights_.unsqueeze(-1)

        # During training, detach the lighting weights to prevent gradient issues
        # if self.training:
        #     self.lighting_weights = self.lighting_weights.detach()

        self.shadow_fn = lambda x: apply_weight_to_RGB(x, self.lighting_weights, intensity)

        shadow_meta = {
            "distances": distances,
            "depth_flattened": depth_flattened,
            "centered_distances_squared": centered_distances_squared,
            "variance": variance,
            "s": s,
            "sigmoid_weights": sigmoid_weights,
            "sigmoid_argument": sigmoid_argument,
            "depth": depth,
        }
        return shadow_meta

    @property
    def features_dc(self):
        if self.shadow_fn is not None:
            return self.shadow_fn(self.gauss_params["features_dc"])
        else:
            return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        if self.shadow_fn is not None:
            return self.shadow_fn(self.gauss_params["features_rest"])
        else:
            return self.gauss_params["features_rest"]

    @property
    def albedo_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def albedo_rest(self):
        return self.gauss_params["features_rest"]

    def get_light_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            name: [self.light_params[name]] for name in ["intensity", "cutoff", "variance_factor"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        gps.update(self.get_light_param_groups())
        # gps['reweighting_param'] = [self.reweighting_param]
        return gps

    def forward(
        self, camera: Cameras, light: Optional[Cameras] = None
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Forward pass that takes a camera and optional light source.

        Args:
            camera: The camera(s) for which output images are rendered
            light: Optional light source camera for shadow computation

        Returns:
            Outputs of model (ie. rendered colors)
        """
        return self.get_outputs(camera, light)

    def get_outputs(
        self, camera: Cameras, light: Optional[Cameras] = None
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if light is not None:
            self.update_light_source(
                light,
                variance_factor=torch.exp(self.light_params["variance_factor"]),
                intensity=torch.exp(self.light_params["intensity"]),
                cutoff=torch.sigmoid(self.light_params["cutoff"]),
            )

        return super().get_outputs(camera)
