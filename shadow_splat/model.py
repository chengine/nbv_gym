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
    moment_rasterization,
)

try:
    from modified_diff_gaussian_rasterization_depth import (
        GaussianRasterizer as ModifiedGaussianRasterizer,
        GaussianRasterizationSettings,
    )
    from einops import repeat, reduce, rearrange
except ImportError:
    print("Please install GaussianRasterizationDepth or einops")

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

from shadow_splat.util.nerfstudio import get_viewmat
from shadow_splat.shadow_splat_rendering import (
    calculate_relighting_weights_from_point_cloud,
    generate_point_cloud_from_camera_depth,
)
from shadow_splat.util.coverage import (
    update_view_coverage_for_frustum,
    fibonacci_sphere,
    update_fig_for_frustum,
)

import matplotlib.pyplot as plt


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
    n_sphere_bins: int = 128
    """Number of bins on the unit sphere for coverage computation."""
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
        self.fig = torch.nn.Parameter(torch.zeros((self.means.shape[0], 1), device="cuda"))
        self.view_fig = torch.nn.Parameter(
            torch.zeros((self.means.shape[0], self.config.n_sphere_bins), device="cuda")
        )

        self.gauss_params["coverage_counts"] = self.coverage_counts
        self.gauss_params["fig"] = self.fig
        self.gauss_params["view_fig"] = self.view_fig

        self.seen_cam_idx = []

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
    def fig(self):
        return self.gauss_params["fig"]

    @property
    def view_fig(self):
        return self.gauss_params["view_fig"]

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
                "fig",
                "view_fig",
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
                "fig",
                "view_fig",
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

        # # TODO: Implement multi-light support
        # print(light.shape)
        # if light is not None:
        #     assert light.shape[0] == 1, f"Only one light at a time, light shape: {light.shape}"

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

        # TODO: HARDCODED. MAKE THIS MORE ELEGANT.
        camera_model = "pinhole"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            features_crop = torch.sigmoid(features_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        if self.training:
            coverage_counts = None
            fig = None
            view_fig = None
        else:
            coverage_counts = self.coverage_counts.detach()
            fig = self.fig.detach()
            view_fig = self.view_fig.detach()

        render, alpha, self.info = rasterization_with_coverage(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=features_crop,
            coverage_counts=coverage_counts,
            fig=fig,
            view_fig=view_fig,
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

        # Check if the viewer has set a light
        if light is None and self.viewer_light is not None and not self.training:
            light = self.viewer_light

        if light is None:
            light = self.last_training_light

        if self.training and light is not None:
            self.last_training_light = light

        if light is not None and not self.training:
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

            # if self.training:
            #     optimized_light_to_world = self.light_optimizer.apply_to_camera(light)
            # else:
            optimized_light_to_world = light.camera_to_worlds
            light_camera_to_world = optimized_light_to_world.cuda()
            light.rescale_output_resolution(1 / camera_scale_fac)
            light_K = light.get_intrinsics_matrices().cuda()
            light_W, light_H = int(light.width.item()), int(light.height.item())
            self.light_last_size = (light_H, light_W)
            light.rescale_output_resolution(camera_scale_fac)  # type: ignore

            if light_camera_to_world.dim() < 3:
                light_camera_to_world = light_camera_to_world.unsqueeze(0)
                light_K = light_K.unsqueeze(0)

            light_viewmat = get_viewmat(light_camera_to_world)

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
            shadow_img = None
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

        if not self.training:
            coverage = render[:, ..., 3:4].squeeze(0)
            if shadow_img is not None:
                coverage_lit = (1.0 - coverage) * (1.0 - shadow_img)
            else:
                coverage_lit = None
            fig_img = render[:, ..., 4:5].squeeze(0)
            view_fig_img = render[:, ..., 5:6].squeeze(0)
        else:
            coverage = None
            coverage_lit = None
            fig_img = None
            view_fig_img = None

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode in ["ED", "RGB+ED"]:
            depth_im = render[:, ..., -1:].squeeze(0)
            depth_im = torch.where(alpha.squeeze(0) > 0, depth_im, depth_im.detach().max())

            if not self.training:
                depth_sqr_im = render[:, ..., -2:-1].squeeze(0)
                variance_img = depth_sqr_im - depth_im**2
                variance_img = torch.where(alpha.squeeze(0) > 0, variance_img, 0.0)
            else:
                variance_img = None

        else:
            depth_im = None
            variance_img = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        # if self.training:
        #     cam_idx = camera.metadata["cam_idx"]
        #     if cam_idx not in self.seen_cam_idx:
        #         ### UPDATE COVERAGE METRICS ###
        #         is_updated_coverage = update_view_coverage_for_frustum(
        #             means=means_crop,
        #             quats=quats_crop,
        #             scales=torch.exp(scales_crop),
        #             viewmats=viewmat,
        #             Ks=K,
        #             width=W,
        #             height=H,
        #             coverage_counts=self.coverage_counts,
        #             bin_dirs=self.bin_dirs,
        #             camera_model=camera_model,
        #             near_plane=0.01,
        #             far_plane=1e10,
        #         )

        #         # is_updated_fig = update_fig_for_frustum(
        #         #     means=means_crop,
        #         #     quats=quats_crop,
        #         #     scales=torch.exp(scales_crop),
        #         #     viewmats=viewmat,
        #         #     Ks=K,
        #         #     width=W,
        #         #     height=H,
        #         #     depth_image=depth_im,
        #         #     variance_image=variance_img,
        #         #     bin_dirs=self.bin_dirs,
        #         #     fig=self.fig,
        #         #     view_fig=self.view_fig,
        #         #     camera_model=camera_model,
        #         #     near_plane=0.01,
        #         #     far_plane=1e10,
        #         #     concentration=self.config.concentration,
        #         # )

        #         ### END ###
        #         self.seen_cam_idx.append(cam_idx)

        #         # Put this in fancy text
        #         print(f"Updated coverage counts from camera {cam_idx}!")

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "variance": variance_img,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "coverage": coverage,  # type: ignore
            "fig": fig_img,  # type: ignore
            "view_fig": view_fig_img,  # type: ignore
            "shadow": shadow_img,  # type: ignore
            "light_depth": light_depth_image,  # type: ignore
            "light_variance": light_variance_image,  # type: ignore
            "coverage_lit": coverage_lit,  # type: ignore
        }  # type: ignore

    @torch.no_grad()
    def update_coverage(self, cameras: List[Cameras]):
        # Update coverage counts based on all cameras in the camera batch, conditioned on the current state of the scene

        # TODO: Might be able to optimize this by batching the update_view_coverage_for_frustum calls.
        for camera in cameras:
            camera = camera.to(self.device)

            camera_scale_fac = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / camera_scale_fac)
            viewmat = get_viewmat(camera.camera_to_worlds)
            K = camera.get_intrinsics_matrices().cuda()
            W, H = int(camera.width.item()), int(camera.height.item())
            camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

            # NOTE: HARDCODED. MAKE THIS MORE ELEGANT.
            camera_model = "pinhole"

            is_updated_coverage = update_view_coverage_for_frustum(
                means=self.means,
                quats=self.quats,
                scales=torch.exp(self.scales),
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

    @torch.no_grad()
    def update_fig(self, cameras: List[Cameras]):
        # Update fig based on all cameras in the camera batch, conditioned on the current state of the scene
        for camera in cameras:
            camera = camera.to(self.device)

            camera_scale_fac = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / camera_scale_fac)
            viewmat = get_viewmat(camera.camera_to_worlds)
            K = camera.get_intrinsics_matrices().cuda()
            W, H = int(camera.width.item()), int(camera.height.item())
            camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

            # NOTE: HARDCODED. MAKE THIS MORE ELEGANT.
            camera_model = "pinhole"

            moments, alphas, meta = moment_rasterization(
                self.means,  # [N, 3]
                self.quats,  # [N, 4]
                torch.exp(self.scales),  # [N, 3]
                torch.sigmoid(self.opacities).squeeze(-1),  # [N]
                viewmat,  # [C, 4, 4]
                K,  # [C, 3, 3]
                W,
                H,
                near_plane=0.01,
                far_plane=1e10,
                radius_clip=3.0,
                eps2d=0.3,
                packed=True,
                tile_size=16,
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.config.rasterize_mode,
                channel_chunk=32,
                distributed=False,
                camera_model=camera_model,
            )

            depth_image = moments[..., 0].squeeze(0)
            depth_image = torch.where(
                alphas.squeeze(0).squeeze(-1) > 0, depth_image, depth_image.detach().max()
            )

            depth_sqr_image = moments[..., 1].squeeze(0)
            variance_image = depth_sqr_image - depth_image**2
            variance_image = torch.where(alphas.squeeze(0).squeeze(-1) > 0, variance_image, 0.0)

            is_updated_fig = update_fig_for_frustum(
                means=self.means,
                quats=self.quats,
                scales=torch.exp(self.scales),
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                depth_image=depth_image,
                variance_image=variance_image,
                bin_dirs=self.bin_dirs,
                fig=self.fig,
                view_fig=self.view_fig,
                camera_model=camera_model,
                near_plane=0.01,
                far_plane=1e10,
                concentration=self.config.concentration,
            )

    @torch.no_grad()
    def reset_coverage(self):
        self.gauss_params["coverage_counts"].zero_()

    @torch.no_grad()
    def reset_fig(self):
        self.gauss_params["fig"].zero_()
        self.gauss_params["view_fig"].zero_()

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
        # shadow_rgb = outputs["shadow"].repeat(1, 1, 3)
        # combined_rgb = torch.cat([images_dict["img"], shadow_rgb], dim=1)
        # images_dict["img"] = combined_rgb

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
        self, camera: Cameras, light=None, intrinsics_scale: float = 1.0, metric: str = "coverage"
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

        # Clear self.info before rendering to free memory from previous renders
        # This is critical for preventing memory leaks during view selection
        old_info = self.info
        self.info = {}

        # Optionally downscale camera intrinsics for faster evaluation
        # scaled_camera = None
        # if intrinsics_scale != 1.0:
        #     # Create a new camera with scaled intrinsics
        #     scaled_camera = Cameras(
        #         camera_to_worlds=camera.camera_to_worlds,
        #         fx=camera.fx * intrinsics_scale,
        #         fy=camera.fy * intrinsics_scale,
        #         cx=camera.cx * intrinsics_scale,
        #         cy=camera.cy * intrinsics_scale,
        #         width=(camera.width * intrinsics_scale).int(),
        #         height=(camera.height * intrinsics_scale).int(),
        #         camera_type=camera.camera_type,
        #         times=camera.times,
        #     ).to(camera.device)
        #     camera = scaled_camera

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)

        outputs = None
        coverage_score = None
        try:
            # Render from camera - get_outputs will use render_mode="RGB+ED" which includes coverage
            outputs = self.get_outputs(camera, light=light)

            # Extract coverage from outputs
            if metric in outputs:
                coverage = outputs[metric]  # [H, W] or [H, W, 1]
            else:
                # Fallback: coverage might be in render output
                # This should not happen if render_mode is set correctly, but handle gracefully
                coverage = torch.zeros(
                    (camera.height.item(), camera.width.item()), device=self.device
                )
                print(f"{metric} not found in outputs")

            valid_mask = outputs["accumulation"] > 0

            # Sum all pixel values where alphas is > 0 to get total coverage score
            # Detach to avoid keeping references to the computation graph
            coverage_score = ((coverage * valid_mask).sum() / valid_mask.sum()).detach()
            # Delete coverage tensor after extracting score
            del coverage

        finally:
            # Cleanup: explicitly delete intermediate outputs and clear self.info
            # This is critical for preventing memory leaks during view selection
            if outputs is not None:
                # Delete all tensors in outputs dict
                for key, value in list(outputs.items()):
                    if isinstance(value, torch.Tensor):
                        del value
                outputs.clear()
                del outputs

            # Clear self.info to free all intermediate tensors
            if isinstance(self.info, dict):
                for key, value in list(self.info.items()):
                    if isinstance(value, torch.Tensor):
                        del value
                self.info.clear()

            # Clear old_info references
            if isinstance(old_info, dict):
                for key, value in list(old_info.items()):
                    if isinstance(value, torch.Tensor):
                        del value
                old_info.clear()

            # Restore training state
            if was_training:
                self.train()

            # Force garbage collection and CUDA cache clearing
            gc.collect()
            torch.cuda.empty_cache()

        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
        return coverage_score


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


class FisherSplatModel(SplatfactoModel):
    """Nerfstudio's implementation of FisherRF Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: FisherSplatModelConfig

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

    # TODO: What's the best way to return features_dc/rest to reflect the shadows conditioned on a light source?
    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

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

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=features_crop,
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
        )

        if light is None:
            light = self.last_training_light

        if self.training and light is not None:
            self.last_training_light = light

        if light is not None and not self.training:
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
            shadow_img = None
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

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode in ["ED", "RGB+ED"]:
            depth_im = render[:, ..., -1:].squeeze(0)
            depth_im = torch.where(alpha.squeeze(0) > 0, depth_im, depth_im.detach().max())

            if not self.training:
                depth_sqr_im = render[:, ..., -2:-1].squeeze(0)
                variance_img = depth_sqr_im - depth_im**2
                variance_img = torch.where(alpha.squeeze(0) > 0, variance_img, 0.0)
            else:
                variance_img = None

        else:
            depth_im = None
            variance_img = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        if self.config.render_uncertainty and not self.training:
            # render uncertainty as in FisherRF (ECCV 2024, Jiang et al.)
            rgb_weight = self.config.rgb_uncertainty_weight
            depth_weight = self.config.depth_uncertainty_weight

            uncertainties = self.render_uncertainty_rgb_depth(
                [camera], [camera], rgb_weight=rgb_weight, depth_weight=depth_weight
            )
            uncertainty = uncertainties[0].unsqueeze(2)
        else:
            uncertainty = None

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "fisher_info": uncertainty,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "shadow": shadow_img,  # type: ignore
            "light_depth": light_depth_image,  # type: ignore
            "light_variance": light_variance_image,  # type: ignore
        }  # type: ignore

    @torch.no_grad()
    def prepare_rasterizer(
        self, camera: Cameras
    ) -> Tuple[ModifiedGaussianRasterizer, List[torch.Tensor]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}  # type: ignore
        # print(camera.shape)
        # assert camera.shape[0] == 1, "Only one camera at a time"

        optimized_camera_to_world = camera.camera_to_worlds

        # move to the GPU
        camera = camera.to(self.device)
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        optimized_camera_to_world = optimized_camera_to_world.squeeze()
        R = optimized_camera_to_world[:3, :3]  # 3 x 3
        T = optimized_camera_to_world[:3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        opacities_crop = self.opacities
        means_crop = self.means
        features_dc_crop = self.features_dc
        features_rest_crop = self.features_rest
        scales_crop = self.scales
        quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        opacities = torch.sigmoid(opacities_crop)

        fovx = 2 * torch.atan(camera.width / (2 * camera.fx))
        fovy = 2 * torch.atan(camera.height / (2 * camera.fy))
        tanfovx = math.tan(fovx * 0.5)
        tanfovy = math.tan(fovy * 0.5)
        bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        scaling_modifier = 1.0
        projmat = projection_matrix(0.01, 100.0, fovx, fovy, device="cuda").cuda()

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            sh_degree_to_use = None

        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmat.t(),
            projmatrix=viewmat.t() @ projmat.t(),
            sh_degree=sh_degree_to_use,
            campos=viewmat.inverse()[:3, 3],
            prefiltered=False,
            debug=False,
        )
        rasterizer = ModifiedGaussianRasterizer(raster_settings=raster_settings)

        # Create temporary varaibles to avoid side effects of the backward engine
        # this also addresses the issues of normalization for quaterions
        means3D = means_crop.clone().requires_grad_(True)
        shs = colors_crop.clone().requires_grad_(True)
        opacities = opacities.clone().requires_grad_(True)
        scales = torch.exp(scales_crop.clone()).requires_grad_(True)
        rotations = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
        rotations.requires_grad_(True)

        params = [means3D, shs, opacities, scales, rotations]

        return rasterizer, params

    @torch.no_grad()
    def compute_diag_H_rgb_depth(self, camera: Cameras, compute_rgb_H=False):
        """
        Compute diagonal hessian, on rgb or depth.

        return dict:
        rgb: rendering from 3D-GS renderer (H, W, C)
        depth: depth map from 3D-GS renderer (H, W)
        H: list of diag hessian on gaussians in the order of:
            means3D, shs, opacities, scales, rotations
        """
        # pdb.set_trace()
        # print(f"camera: {camera}")
        # print(f"camera.shape: {camera.shape}")
        rasterizer, params = self.prepare_rasterizer(camera)
        means3D, shs, opacities, scales, rotations = params

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        with torch.enable_grad():
            rendered_image, rendered_depth, radii = rasterizer(
                means3D=means3D,
                means2D=screenspace_points,
                shs=shs,
                colors_precomp=None,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )
            if compute_rgb_H:
                rendered_image.backward(gradient=torch.ones_like(rendered_image))
            else:
                rendered_depth.backward(gradient=torch.ones_like(rendered_depth))

        cur_H = [p.grad.detach().clone() for p in params]  # type: ignore

        rgb = rearrange(rendered_image, "c h w -> h w c")

        return {"rgb": rgb, "H": cur_H, "depth": rendered_depth}  # type: ignore

    @torch.no_grad()
    def render_uncertainty_rgb_depth(
        self,
        train_cameras: Iterable[Cameras],
        test_cameras: Iterable[Cameras],
        rgb_weight=1.0,
        depth_weight=1.0,
    ):
        H_per_gaussian = torch.zeros(
            self.opacities.shape[0], device=self.opacities.device, dtype=self.opacities.dtype
        )

        # Optionally downscale camera intrinsics for faster evaluation
        camera_scale_fac = self._get_downscale_factor()

        # go through provided training cameras
        for train_cam in train_cameras:
            train_cam = train_cam.to(self.device)
            train_cam.rescale_output_resolution(1 / camera_scale_fac)

            # get rgb uncertainty
            H_info_rgb = self.compute_diag_H_rgb_depth(train_cam, compute_rgb_H=True)
            H_info_rgb["H"] = [p * rgb_weight for p in H_info_rgb["H"]]
            H_per_gaussian += sum([reduce(p, "n ... -> n", "sum") for p in H_info_rgb["H"]])

            # get depth uncertainty
            H_info_depth = self.compute_diag_H_rgb_depth(train_cam, compute_rgb_H=False)
            H_info_depth["H"] = [p * depth_weight for p in H_info_depth["H"]]
            H_per_gaussian += sum([reduce(p, "n ... -> n", "sum") for p in H_info_depth["H"]])

            train_cam.rescale_output_resolution(camera_scale_fac)

        hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3)
        uncern_maps = []
        for test_cam in test_cameras:
            test_cam = test_cam.to(self.device)
            test_cam.rescale_output_resolution(1 / camera_scale_fac)

            rasterizer, params = self.prepare_rasterizer(test_cam)
            means3D, shs, opacities, scales, rotations = params

            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
            screenspace_points = (
                torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
                + 0
            )
            try:
                screenspace_points.retain_grad()
            except:
                pass

            # cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params]) # type: ignore
            pts3d_homo = to_homo(means3D)
            pts3d_cam = pts3d_homo @ rasterizer.raster_settings.viewmatrix
            gaussian_depths = pts3d_cam[:, 2, None]

            cur_hessian_color = hessian_color * gaussian_depths.clamp(min=0)
            rendered_image, rendered_depth, radii = rasterizer(
                means3D=means3D,
                means2D=screenspace_points,
                shs=None,
                colors_precomp=cur_hessian_color,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )
            rendered_image[0] = rendered_image[0]
            uncern_maps.append(rendered_image[0])

            test_cam.rescale_output_resolution(camera_scale_fac)

        return uncern_maps

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
        # shadow_rgb = outputs["shadow"].repeat(1, 1, 3)
        # combined_rgb = torch.cat([images_dict["img"], shadow_rgb], dim=1)
        # images_dict["img"] = combined_rgb

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
        self,
        training_cameras: List[Cameras],
        test_camera: List[Cameras],
        intrinsics_scale: float = 1.0,
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

        # Clear self.info before rendering to free memory from previous renders
        # This is critical for preventing memory leaks during view selection
        old_info = self.info
        self.info = {}

        # outputs = None
        coverage_score = None
        try:
            # Render from camera - get_outputs will use render_mode="RGB+ED" which includes coverage
            uncertainty = self.render_uncertainty_rgb_depth(
                training_cameras,
                test_camera,
                rgb_weight=self.config.rgb_uncertainty_weight,
                depth_weight=self.config.depth_uncertainty_weight,
            )

            # Sum all pixel values to get total coverage score
            # Detach to avoid keeping references to the computation graph
            coverage_score = []
            for unc_map in uncertainty:
                valid_mask = unc_map > 0
                coverage_score.append(-(unc_map * valid_mask).sum() / valid_mask.sum())
            # Delete uncertainty tensor after extracting score
            del uncertainty

        finally:
            # Cleanup: explicitly delete intermediate outputs and clear self.info
            # This is critical for preventing memory leaks during view selection
            # if outputs is not None:
            #     # Delete all tensors in outputs dict
            #     for key, value in list(outputs.items()):
            #         if isinstance(value, torch.Tensor):
            #             del value
            #     outputs.clear()
            #     del outputs

            # Clear self.info to free all intermediate tensors
            if isinstance(self.info, dict):
                for key, value in list(self.info.items()):
                    if isinstance(value, torch.Tensor):
                        del value
                self.info.clear()

            # Clear old_info references
            if isinstance(old_info, dict):
                for key, value in list(old_info.items()):
                    if isinstance(value, torch.Tensor):
                        del value
                old_info.clear()

            # Restore training state
            if was_training:
                self.train()

            # Force garbage collection and CUDA cache clearing
            gc.collect()
            torch.cuda.empty_cache()

        return coverage_score
