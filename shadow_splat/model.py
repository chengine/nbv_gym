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
import gc

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
import torch
from torch.nn import Parameter
from gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from shadow_splat.shadow_splat_rendering import (
    rasterization_with_coverage,
)


import math
from shadow_splat.lights import LightOptimizer, LightOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from nerfstudio.model_components.lib_bilagrid import (
    BilateralGrid,
    color_correct,
    slice,
    total_variation_loss,
)

import open3d as o3d

# import torch_geometric.nn.pool.knn as knn

from shadow_splat.util.nerfstudio import get_viewmat
from shadow_splat.shadow_splat_rendering import (
    calculate_relighting_weights_from_point_cloud,
    generate_point_cloud_from_camera_depth,
)
from shadow_splat.util.coverage import update_view_coverage_for_frustum, fibonacci_sphere

import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

to_homo = lambda x: torch.cat(
    [x, torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)], dim=-1
)


@dataclass
class ShadowSplatModelConfig(SplatfactoModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: ShadowSplatModel)
    light_optimizer: LightOptimizerConfig = field(
        default_factory=lambda: LightOptimizerConfig(mode="off")
    )
    # TODO: add shadow splat specific parameters here
    ambient: bool = (
        True  # Controls whether Gaussians outside the light frustum are set to ambient or to black
    )
    tone_mapping: Literal["linear", "luminance", "reinhard"] = "linear"
    gamma_correction: float = 1.0
    fix_variance: bool = False
    n_sphere_bins: int = 128


class ShadowSplatModel(SplatfactoModel):
    """Nerfstudio's implementation of Shadow Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: ShadowSplatModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        print("model init | seed_points", seed_points)
        super().__init__(*args, seed_points=seed_points, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        self.light_params = torch.nn.ParameterDict(
            {
                "intensity": torch.nn.Parameter(torch.log(torch.tensor(1.0))),
                "ambient": torch.nn.Parameter(torch.logit(torch.tensor(0.5))),  # in frustum
            }
        )

        self.light_optimizer: LightOptimizer = self.config.light_optimizer.setup(
            num_cameras=1, device="cpu"
        )

        self.last_training_light = None

        self.bin_dirs = fibonacci_sphere(n_bins=self.config.n_sphere_bins, device="cuda")
        self.coverage_counts = torch.nn.Parameter(
            torch.zeros((self.means.shape[0], self.config.n_sphere_bins), device="cuda")
        )

        self.gauss_params["coverage_counts"] = self.coverage_counts

        self.seen_cam_idx = []

    # TODO: What's the best way to return features_dc/rest to reflect the shadows conditioned on a light source?
    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def coverage_counts(self):
        return self.gauss_params["coverage_counts"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in [
                "means",
                "scales",
                "quats",
                "features_dc",
                "features_rest",
                "opacities",
                "coverage_counts",
            ]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        # TODO: add lighting params loading
        super().load_state_dict(dict, **kwargs)

    def step_post_backward(self, step):
        assert step == self.step
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                packed=False,
            )
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=self.info,
                lr=self.schedulers["means"].get_last_lr()[
                    0
                ],  # the learning rate for the "means" attribute of the GS
            )
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in [
                "means",
                "scales",
                "quats",
                "features_dc",
                "features_rest",
                "opacities",
                "coverage_counts",
            ]
        }

    def get_light_param_groups(self) -> Dict[str, List[Parameter]]:
        return {name: [self.light_params[name]] for name in ["intensity", "ambient"]}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        gps.update(self.get_light_param_groups())

        self.camera_optimizer.get_param_groups(param_groups=gps)
        self.light_optimizer.get_param_groups(param_groups=gps)
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
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # TODO: Implement multi-light support
        if light is not None:
            assert light.shape[0] == 1, "Only one light at a time"

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        features_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if camera.camera_type == CameraType.PERSPECTIVE.value:
            camera_model = "pinhole"
        elif camera.camera_type == CameraType.ORTHOPHOTO.value:
            camera_model = "ortho"
        elif camera.camera_type == CameraType.FISHEYE.value:
            camera_model = "fisheye"
        else:
            raise ValueError("Unknown camera type: %s", camera.camera_type)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            features_crop = torch.sigmoid(features_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization_with_coverage(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=features_crop,
            coverage_counts=self.coverage_counts,
            bin_dirs=self.bin_dirs,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
            camera_model=camera_model,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        if self.training:
            cam_idx = camera.metadata["cam_idx"]
            if cam_idx not in self.seen_cam_idx:
                is_updated = update_view_coverage_for_frustum(
                    means=means_crop,
                    quats=quats_crop,
                    scales=torch.exp(scales_crop),
                    viewmats=viewmat,
                    Ks=K,
                    width=W,
                    height=H,
                    coverage_counts=self.coverage_counts,
                    bin_dirs=self.bin_dirs,
                    camera_model=camera_model,
                    near_plane=0.01,
                    far_plane=1e10,
                )
                self.seen_cam_idx.append(cam_idx)

                # Put this in fancy text
                print(f"Updated coverage counts from camera {cam_idx}!")

        # If is_updated, then self.coverage_counts is updated in-place, otherwise self.coverage_counts is not updated

        if light is None:
            light = self.last_training_light

        if self.training and light is not None:
            self.last_training_light = light

        if light is not None:
            point_cloud, point_cloud_mask = generate_point_cloud_from_camera_depth(
                depth=render[:, ..., -1:],
                K=K,
                W=W,
                H=H,
                viewmat=viewmat,
                near_plane=0.1,
                far_plane=1e10,
                mask=None,
            )

            if self.training:
                optimized_light_to_world = self.light_optimizer.apply_to_camera(light)
            else:
                optimized_light_to_world = light.camera_to_worlds

            light_camera_to_world = optimized_light_to_world
            light.rescale_output_resolution(1 / camera_scale_fac)
            light_viewmat = get_viewmat(light_camera_to_world)
            light_K = light.get_intrinsics_matrices().cuda()
            light_W, light_H = int(light.width.item()), int(light.height.item())
            self.light_last_size = (light_H, light_W)
            light.rescale_output_resolution(camera_scale_fac)  # type: ignore

            if light.camera_type == CameraType.PERSPECTIVE.value:
                light_model = "pinhole"
            elif light.camera_type == CameraType.ORTHOPHOTO.value:
                light_model = "ortho"
            elif light.camera_type == CameraType.FISHEYE.value:
                light_model = "fisheye"
            else:
                raise ValueError("Unknown light type: %s", light.camera_type)

            shadow_img, light_depth_image, light_variance_image = (
                calculate_relighting_weights_from_point_cloud(
                    means=means_crop,  # [N, 3]
                    quats=quats_crop,  # [N, 4]
                    scales=torch.exp(
                        scales_crop
                    ),  # [N, 3]       # NOTE: IMPORTANT! THESE SCALES MUST ALREADY BE POSITIVE
                    opacities=torch.sigmoid(opacities_crop).squeeze(
                        -1
                    ),  # [N]       # NOTE: IMPORTANT! THESE OPACITIES MUST ALREADY BE [0, 1]
                    viewmats=light_viewmat,  # [C, 4, 4]
                    Ks=light_K,  # [C, 3, 3]
                    width=light_W,
                    height=light_H,
                    point_cloud=point_cloud,
                    point_cloud_mask=point_cloud_mask,
                    ambient=torch.sigmoid(self.light_params["ambient"]),
                    camera_model=light_model,
                )
            )
            shadow_img = shadow_img.unsqueeze(-1)
            light_depth_image = light_depth_image.unsqueeze(-1)
            light_variance_image = light_variance_image.unsqueeze(-1)
        else:
            shadow_img = torch.zeros((H, W, 1), device=self.device)
            light_depth_image = None
            light_variance_image = None

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        coverage = render[:, ..., 3:4].squeeze(0)
        lighted_dissimilarity = (1.0 - coverage) * (1.0 - shadow_img)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "coverage": coverage,  # type: ignore
            "shadow": shadow_img,  # type: ignore
            "light_depth": light_depth_image,  # type: ignore
            "light_variance": light_variance_image,  # type: ignore
            "lighted_dissimilarity": lighted_dissimilarity,  # type: ignore
        }  # type: ignore

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }

        # Losses for mcmc
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                mcmc_opacity_reg = (
                    self.config.mcmc_opacity_reg
                    * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
                loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg
            if self.config.mcmc_scale_reg > 0.0:
                mcmc_scale_reg = (
                    self.config.mcmc_scale_reg
                    * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                )
                loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            self.light_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        shadow_rgb = outputs["shadow"].repeat(1, 1, 3)
        combined_rgb = torch.cat([images_dict["img"], shadow_rgb], dim=1)
        images_dict["img"] = combined_rgb

        return metrics_dict, images_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = super().get_metrics_dict(outputs, batch)
        metrics_dict["intensity"] = torch.exp(self.light_params["intensity"])
        metrics_dict["ambient"] = torch.sigmoid(self.light_params["ambient"])
        self.light_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    @torch.no_grad()
    def coverage_score_for_camera(
        self, camera: Cameras, intrinsics_scale: float = 1.0
    ) -> torch.Tensor:
        """Compute coverage score for a candidate camera.

        Renders from the given camera and computes the sum of all pixel values in the
        coverage image. Higher scores indicate better coverage.

        Args:
            camera: Camera object to evaluate
            intrinsics_scale: Scale factor for camera intrinsics (for faster evaluation).
                Values < 1.0 downscale the resolution.

        Returns:
            Scalar tensor with the total coverage score (sum of all coverage pixels)
        """
        # Save current training state
        was_training = self.training

        # Set to eval mode for faster inference
        self.eval()

        # Optionally downscale camera intrinsics for faster evaluation
        if intrinsics_scale != 1.0:
            # Create a new camera with scaled intrinsics
            scaled_camera = Cameras(
                camera_to_worlds=camera.camera_to_worlds,
                fx=camera.fx * intrinsics_scale,
                fy=camera.fy * intrinsics_scale,
                cx=camera.cx * intrinsics_scale,
                cy=camera.cy * intrinsics_scale,
                width=(camera.width * intrinsics_scale).int(),
                height=(camera.height * intrinsics_scale).int(),
                camera_type=camera.camera_type,
                times=camera.times,
            ).to(camera.device)
            camera = scaled_camera

        # Render from camera - get_outputs will use render_mode="RGB+ED" which includes coverage
        outputs = self.get_outputs(camera, light=None)

        # Extract coverage from outputs
        if "coverage" in outputs:
            coverage = outputs["coverage"]  # [H, W] or [H, W, 1]
        else:
            # Fallback: coverage might be in render output
            # This should not happen if render_mode is set correctly, but handle gracefully
            coverage = torch.zeros((camera.height.item(), camera.width.item()), device=self.device)

        # Sum all pixel values to get total coverage score
        coverage_score = coverage.sum()

        # Restore training state
        if was_training:
            self.train()

        return coverage_score

    def uncertainty_score_for_camera(
        self,
        camera: Cameras,
        hessian: torch.Tensor,
        reduce_mode: str = "mean",
        lod: int = 8,
    ) -> torch.Tensor:
        """
        Compute BayesRays uncertainty score for a candidate camera.

        Renders from the given camera and computes per-pixel uncertainty using the Hessian,
        then aggregates to a single score. Higher scores indicate higher uncertainty.

        Args:
            camera: Camera object to evaluate
            hessian: Pre-computed Hessian tensor [((2^lod)+1)^3]
            reduce_mode: How to aggregate uncertainty - "mean" or "sum"
            lod: Level of detail (log2 of grid resolution) used to compute Hessian

        Returns:
            Scalar tensor with the aggregated uncertainty score
        """
        import types
        # from bayesrays.scripts.output_uncertainty import get_uncertainty, get_output_fn
        from shadow_splat.bayesrays_utils import get_uncertainty, get_output_fn

        # Save current training state
        was_training = self.training

        # Set to eval mode for inference
        self.eval()

        try:
            # Inject Hessian and uncertainty computation into model
            self.filter_out = False
            self.filter_thresh = 0.5
            self.hessian = hessian
            self.lod = lod
            self.get_uncertainty = types.MethodType(get_uncertainty, self)
            self.white_bg = False
            self.black_bg = False
            self.N = 4096 * 1000  # approx ray dataset size

            # Get the appropriate get_outputs function for uncertainty
            new_method = get_output_fn(self)
            self.get_outputs = types.MethodType(new_method, self)

            # Render from camera
            outputs = self.get_outputs(camera)

            # Extract uncertainty from outputs
            if "uncertainty" in outputs:
                uncertainty = outputs["uncertainty"]  # [H, W] or [H, W, 1]
            else:
                # Fallback: return low score if uncertainty not available
                uncertainty = torch.zeros((camera.height.item(), camera.width.item()), device=self.device)

            # Aggregate uncertainty to scalar score
            if reduce_mode.lower() == "mean":
                uncertainty_score = uncertainty.mean()
            elif reduce_mode.lower() == "sum":
                uncertainty_score = uncertainty.sum()
            else:
                raise ValueError(f"reduce_mode must be 'mean' or 'sum', got {reduce_mode}")

        finally:
            # Restore training state
            if was_training:
                self.train()

        return uncertainty_score

@dataclass
class FisherSplatModelConfig(SplatfactoModelConfig):
    """FisherRF Model Config, nerfstudio's implementation of FisherRFGaussian Splatting"""

    _target: Type = field(default_factory=lambda: FisherSplatModel)
    light_optimizer: LightOptimizerConfig = field(
        default_factory=lambda: LightOptimizerConfig(mode="off")
    )
    # TODO: add shadow splat specific parameters here
    ambient: bool = (
        True  # Controls whether Gaussians outside the light frustum are set to ambient or to black
    )

    render_uncertainty: bool = True
    """whether or not to render uncertainty during GS training. NOTE: This will slow down training significantly."""
    depth_uncertainty_weight: float = 1.0
    """weight of depth uncertainty with the Hessian"""
    rgb_uncertainty_weight: float = 1.0


@dataclass
class NerfactoModelWithUncertaintyConfig(NerfactoModelConfig):
    """Nerfacto model with BayesRays uncertainty scoring capability."""
    _target: Type = field(default_factory=lambda: NerfactoModelWithUncertainty)


class NerfactoModelWithUncertainty(NerfactoModel):
    """Nerfacto model extended with uncertainty scoring for view selection."""

    config: NerfactoModelWithUncertaintyConfig

    def uncertainty_score_for_camera(
        self,
        camera: "Cameras",
        hessian: torch.Tensor,
        reduce_mode: str = "mean",
        lod: int = 8,
        return_uncertainty_map: bool = False,
        downscale_factor: float = 4.0,
    ) -> torch.Tensor:
        """
        Score a candidate camera by rendering it and computing BayesRays uncertainty.

        Uses the pre-computed Hessian to generate an uncertainty map during
        rendering, then aggregates to a single scalar score. Higher scores indicate
        that this camera view would see high-uncertainty regions.

        Uses nerfstudio's eval-style downsampling for speed (4x default).

        Args:
            camera: Single camera to evaluate (Cameras object with size=1)
            hessian: Pre-computed Hessian uncertainty grid [((2^lod)+1)^3]
            reduce_mode: "mean" or "sum" for aggregating per-pixel uncertainty
            lod: Level of detail (log2 of grid resolution)
            return_uncertainty_map: If True, also return the full uncertainty map
            downscale_factor: Resolution downscaling factor for faster evaluation (default 4.0 = 1/4 resolution)

        Returns:
            If return_uncertainty_map=False: Scalar tensor with uncertainty score
            If return_uncertainty_map=True: Tuple of (score, uncertainty_map [H, W])
        """
        import types
        from copy import deepcopy
        from shadow_splat.bayesrays_utils import get_uncertainty, get_output_fn

        # Save original state
        was_training = self.training
        original_get_outputs = None
        if hasattr(self, "get_outputs"):
            original_get_outputs = self.get_outputs

        self.eval()

        try:
            # Inject Hessian and uncertainty computation into model
            self.filter_out = False
            self.filter_thresh = 0.5
            self.hessian = hessian
            self.lod = lod
            self.get_uncertainty = types.MethodType(get_uncertainty, self)
            self.white_bg = False
            self.black_bg = False
            self.N = 4096 * 1000  # approx ray dataset size

            # Compute inverse Hessian (uncertainty = 1/Hessian)
            reg_lambda = 1e-4 / ((2 ** lod) ** 3)
            H = hessian / self.N + reg_lambda
            self.un = 1 / H  # Uncertainty grid (inverse of regularized Hessian)

            with torch.no_grad():
                # Use eval-style downsampling like nerfstudio does
                # This makes uncertainty evaluation MUCH faster
                # print(f"[DEBUG uncertainty_score_for_camera] Input camera resolution: {camera.height.item() if hasattr(camera.height, 'item') else camera.height}x{camera.width.item() if hasattr(camera.width, 'item') else camera.width}")
                camera_eval = deepcopy(camera)
                if downscale_factor > 1.0:
                    # print(f"[DEBUG uncertainty_score_for_camera] Applying {downscale_factor}x downsampling")
                    camera_eval.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)
                # print(f"[DEBUG uncertainty_score_for_camera] Eval camera resolution: {camera_eval.height.item() if hasattr(camera_eval.height, 'item') else camera_eval.height}x{camera_eval.width.item() if hasattr(camera_eval.width, 'item') else camera_eval.width}")

                # Generate rays from downscaled camera
                ray_bundle = camera_eval.generate_rays(
                    camera_indices=torch.arange(camera_eval.size, device=self.device, dtype=torch.long)
                )

                # Use the same batch size as eval (4096 rays per chunk)
                eval_batch_size = getattr(self.config, "eval_num_rays_per_batch", 4096)

                # Flatten the rays if they're in image format [H, W, ...]
                if ray_bundle.origins.dim() == 3:
                    H_eval, W_eval = ray_bundle.origins.shape[0], ray_bundle.origins.shape[1]
                    num_rays_total = H_eval * W_eval

                    # Reshape ray_bundle by indexing all rays
                    h_indices, w_indices = torch.meshgrid(
                        torch.arange(H_eval, device=self.device),
                        torch.arange(W_eval, device=self.device),
                        indexing='ij'
                    )
                    flat_indices = (h_indices.flatten(), w_indices.flatten())
                    ray_bundle = ray_bundle[flat_indices]  # Use indexing to preserve all attributes
                else:
                    num_rays_total = ray_bundle.origins.shape[0]
                    H_eval, W_eval = None, None

                accumulated_uncertainty_per_ray = []
                min_uncertainty = -3.0
                max_uncertainty = 6.0

                # Process rays in batches (like pipeline's get_eval_image_metrics_and_images)
                for batch_start in range(0, num_rays_total, eval_batch_size):
                    batch_end = min(batch_start + eval_batch_size, num_rays_total)
                    batch_size = batch_end - batch_start

                    # Get batch of rays
                    batch_rays = ray_bundle[batch_start:batch_end]

                    # Ensure nears and fars are set (required by proposal sampler)
                    if batch_rays.nears is None:
                        if self.collider is not None:
                            batch_rays = self.collider(batch_rays)
                        else:
                            near_plane = getattr(self.config, 'near_plane', 0.05)
                            far_plane = getattr(self.config, 'far_plane', 1000.0)
                            batch_rays.nears = torch.full((batch_size, 1), near_plane, device=self.device)
                            batch_rays.fars = torch.full((batch_size, 1), far_plane, device=self.device)

                    # Get ray samples using the proposal sampler (standard forward pass)
                    ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
                        batch_rays, density_fns=self.density_fns
                    )

                    # Get field outputs
                    field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
                    points = ray_samples.frustums.get_positions()

                    # Look up uncertainty at each sample point using the Hessian
                    un_points = self.get_uncertainty(points)

                    # Get density and weights
                    density = field_outputs[FieldHeadNames.DENSITY]
                    weights = ray_samples.get_weights(density)

                    # Accumulate uncertainty along rays (volumetric rendering style)
                    batch_uncertainty_per_ray = torch.sum(weights * un_points, dim=-2)

                    # Alpha blending: add minimum uncertainty for unaccounted areas
                    batch_uncertainty_per_ray += (1 - torch.sum(weights, dim=-2)) * min_uncertainty

                    # Normalize to reasonable range
                    batch_uncertainty_per_ray = torch.clip(batch_uncertainty_per_ray, min_uncertainty, max_uncertainty)
                    batch_uncertainty_per_ray = (batch_uncertainty_per_ray - min_uncertainty) / (max_uncertainty - min_uncertainty)

                    accumulated_uncertainty_per_ray.append(batch_uncertainty_per_ray)

                # Concatenate all batches
                uncertainty_per_ray = torch.cat(accumulated_uncertainty_per_ray, dim=0)

                # Aggregate uncertainty to scalar score
                if reduce_mode.lower() == "mean":
                    uncertainty_score = uncertainty_per_ray.mean()
                elif reduce_mode.lower() == "sum":
                    uncertainty_score = uncertainty_per_ray.sum()
                else:
                    raise ValueError(f"reduce_mode must be 'mean' or 'sum', got {reduce_mode}")

                # Optionally reshape and return the full uncertainty map
                if return_uncertainty_map:
                    if H_eval is not None and W_eval is not None:
                        # Reshape from [num_rays] to [H, W]
                        uncertainty_map = uncertainty_per_ray.reshape(H_eval, W_eval)
                        # print(f"[DEBUG uncertainty_score_for_camera] Returning tuple: score={uncertainty_score.item()}, map shape={uncertainty_map.shape}")
                        return uncertainty_score, uncertainty_map
                    else:
                        # print(f"[DEBUG uncertainty_score_for_camera] Returning tuple (flat): score={uncertainty_score.item()}, per_ray shape={uncertainty_per_ray.shape}")
                        return uncertainty_score, uncertainty_per_ray
                else:
                    # print(f"[DEBUG uncertainty_score_for_camera] Returning scalar: {uncertainty_score.item()}")
                    return uncertainty_score

        except Exception as e:
            import traceback
            import sys
            print(f"\n[ERROR] uncertainty_score_for_camera failed with exception:")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            print(f"[ERROR] Exception message: {e}")
            print(f"[ERROR] Full traceback:")
            traceback.print_exc(file=sys.stdout)
            raise e

        finally:
            # Restore original state
            if was_training:
                self.train()

            # Restore original get_outputs method
            if original_get_outputs is not None:
                self.get_outputs = original_get_outputs

            # Clean up injected attributes
            for attr in [
                "filter_out",
                "filter_thresh",
                "hessian",
                "lod",
                "get_uncertainty",
                "white_bg",
                "black_bg",
                "N",
                "un",
            ]:
                if hasattr(self, attr):
                    delattr(self, attr)