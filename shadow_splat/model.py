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

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
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
from shadow_splat.util.coverage import (
    update_view_coverage_for_frustum,
    fibonacci_sphere,
    update_transmittance_metrics_for_frustum,
)

import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)


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
    n_sphere_bins: int = 128
    concentration: float = 5.0


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
        self.viewer_light = None

        ### THIS IS FOR COVERAGE ###
        self.bin_dirs = fibonacci_sphere(n_bins=self.config.n_sphere_bins, device="cuda")
        self.coverage_counts = torch.nn.Parameter(
            torch.zeros((self.means.shape[0], self.config.n_sphere_bins), device="cuda")
        )

        self.accumulated_transmittance = torch.nn.Parameter(
            torch.zeros((self.means.shape[0], 1), device="cuda")
        )

        self.accumulated_view_transmittance = torch.nn.Parameter(
            torch.zeros((self.means.shape[0], self.config.n_sphere_bins), device="cuda")
        )

        self.gauss_params["coverage_counts"] = self.coverage_counts
        self.gauss_params["accumulated_transmittance"] = self.accumulated_transmittance
        self.gauss_params["accumulated_view_transmittance"] = self.accumulated_view_transmittance

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

    @property
    def accumulated_transmittance(self):
        return self.gauss_params["accumulated_transmittance"]

    @property
    def accumulated_view_transmittance(self):
        return self.gauss_params["accumulated_view_transmittance"]

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
                "accumulated_transmittance",
                "accumulated_view_transmittance",
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
                "accumulated_transmittance",
                "accumulated_view_transmittance",
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

        # if self.config.output_depth_during_training or not self.training:
        #     render_mode = "RGB+ED"
        # else:
        #     render_mode = "RGB"

        # NOTE: We need to render depth (and variance) for coverage metrics
        render_mode = "RGB+ED"

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
            accumulated_transmittance=torch.sqrt(self.accumulated_transmittance.detach()),
            accumulated_view_transmittance=torch.sqrt(self.accumulated_view_transmittance.detach()),
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
            concentration=self.config.concentration,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        # If is_updated, then self.coverage_counts is updated in-place, otherwise self.coverage_counts is not updated

        # Check if the viewer has set a light
        if light is None and self.viewer_light is not None:
            light = self.viewer_light

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

        transmittance_img = render[:, ..., 4:5].squeeze(0)
        # lighted_transmittance_img = transmittance_img

        view_transmittance_img = render[:, ..., 5:6].squeeze(0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode in ["ED", "RGB+ED"]:
            depth_im = render[:, ..., -1:].squeeze(0)
            depth_sqr_im = render[:, ..., -2:-1].squeeze(0)
            variance_img = depth_sqr_im - depth_im**2

            depth_im = torch.where(alpha.squeeze(0) > 0, depth_im, depth_im.detach().max())
            variance_img = torch.where(alpha.squeeze(0) > 0, variance_img, 0.0)

        else:
            depth_im = None
            variance_img = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        if self.training:
            cam_idx = camera.metadata["cam_idx"]
            if cam_idx not in self.seen_cam_idx:
                ### UPDATE COVERAGE METRICS ###
                is_updated_coverage = update_view_coverage_for_frustum(
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

                is_updated_transmittance = update_transmittance_metrics_for_frustum(
                    means=means_crop,
                    quats=quats_crop,
                    scales=torch.exp(scales_crop),
                    viewmats=viewmat,
                    Ks=K,
                    width=W,
                    height=H,
                    depth_image=depth_im,
                    variance_image=variance_img,
                    bin_dirs=self.bin_dirs,
                    accumulated_transmittance=self.accumulated_transmittance,
                    accumulated_view_transmittance=self.accumulated_view_transmittance,
                    camera_model=camera_model,
                    near_plane=0.01,
                    far_plane=1e10,
                    concentration=self.config.concentration,
                )

                ### END ###
                self.seen_cam_idx.append(cam_idx)

                # Put this in fancy text
                print(f"Updated coverage counts from camera {cam_idx}!")

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "variance": variance_img,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "coverage": coverage,  # type: ignore
            "transmittance": transmittance_img,  # type: ignore
            "view_transmittance": view_transmittance_img,  # type: ignore
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
        self, camera: Cameras, intrinsics_scale: float = 1.0, metric: str = "coverage"
    ) -> torch.Tensor:
        """Compute coverage score for a candidate camera.

        Renders from the given camera and computes the sum of all pixel values in the
        coverage image. Higher scores indicate better coverage (more Gaussians seen from
        more directions). Lower scores indicate novel/uncovered areas.

        Args:
            camera: Camera object to evaluate
            intrinsics_scale: Scale factor for camera intrinsics (for faster evaluation).
                Values < 1.0 downscale the resolution.

        Returns:
            Scalar tensor with the total coverage score (sum of all coverage pixels).
            Lower values indicate views that see poorly-covered areas.
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
        if metric in outputs:
            coverage = outputs[metric]  # [H, W] or [H, W, 1]
        else:
            # Fallback: coverage might be in render output
            # This should not happen if render_mode is set correctly, but handle gracefully
            coverage = torch.zeros((camera.height.item(), camera.width.item()), device=self.device)
            print(f"{metric} not found in outputs")

        # Sum all pixel values to get total coverage score
        coverage_score = coverage.sum()

        # Restore training state
        if was_training:
            self.train()

        return coverage_score
