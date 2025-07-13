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
    moment_rasterization,
    augmented_rasterization,
    moment_rasterization_2dgs,
    augmented_rasterization_2dgs,
    calculate_relighting_weights,
)

# torch.autograd.set_detect_anomaly(True)
import math
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB
from nerfstudio.model_components.lib_bilagrid import (
    BilateralGrid,
    color_correct,
    slice,
    total_variation_loss,
)
from shadow_splat.util.nerfstudio import get_viewmat
import matplotlib.pyplot as plt


@dataclass
class ShadowSplatModelConfig(SplatfactoModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: ShadowSplatModel)
    # TODO: add shadow splat specific parameters here
    ambient: bool = (
        True  # Controls whether Gaussians outside the light frustum are set to ambient or to black
    )
    tone_mapping: Literal["linear", "luminance", "reinhard"] = "linear"
    gamma_correction: float = 1.0
    fix_variance: bool = False


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
                "intensity": torch.nn.Parameter(torch.log(torch.ones(3))),
                "cutoff": torch.nn.Parameter(torch.log(torch.tensor(0.5))),
                "variance_factor": torch.nn.Parameter(torch.logit(torch.tensor(0.01))),
            }
        )

        self.irradiance = None
        self.irradiance_fraction = None

    # TODO: What's the best way to return features_dc/rest to reflect the shadows conditioned on a light source?
    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def albedo_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def albedo_rest(self):
        return self.gauss_params["features_rest"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
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

    def compute_irradiance(
        self,
        light: Cameras,
        variance_factor: Optional[float] = None,
        intensity: Optional[float] = None,
        cutoff: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the irradiance for a given camera and light source.

        Args:
            camera: The camera(s) for which output images are rendered
            light: Optional light source camera for shadow computation
        """
        camera_scale_fac = self._get_downscale_factor()
        opacities_crop = self.opacities
        means_crop = self.means
        scales_crop = self.scales
        quats_crop = self.quats

        # TODO: Implement light intrinsic optimization
        light_camera_to_world = light.camera_to_worlds
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

        if self.training:
            hard_cutoff = False
        else:
            hard_cutoff = True

        # print lighting params values
        if variance_factor is None:
            variance_factor = torch.exp(self.light_params["variance_factor"])
        if intensity is None:
            intensity = torch.exp(self.light_params["intensity"])
        if cutoff is None:
            cutoff = torch.sigmoid(self.light_params["cutoff"])
        # print(f"variance_factor: {variance_factor:.4f}")
        # print(f"intensity: {intensity:.4f}")
        # print(f"cutoff: {cutoff:.4f}")

        irradiance, irradiance_fraction = calculate_relighting_weights(
            means=means_crop,  # [N, 3]
            quats=quats_crop,  # [N, 4]
            scales=torch.exp(scales_crop),  # [N, 3]
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),  # [N]
            viewmats=light_viewmat,  # [C, 4, 4]
            Ks=light_K,  # [C, 3, 3]
            width=light_W,
            height=light_H,
            light=light,
            variance_factor=variance_factor,
            intensity=intensity,
            cutoff=cutoff,
            hard_cutoff=hard_cutoff,
            ambient=self.config.ambient,
            near_plane=0.01,
            far_plane=1e10,
            depth_mode="absolute",
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
            camera_model=light_model,
            distloss=False,  # 2DGS only
            fix_variance=self.config.fix_variance,
        )
        self.irradiance = irradiance
        self.irradiance_fraction = irradiance_fraction
        return irradiance, irradiance_fraction

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
            albedo_dc_crop = self.albedo_dc[crop_ids]
            albedo_rest_crop = self.albedo_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            albedo_dc_crop = self.albedo_dc
            albedo_rest_crop = self.albedo_rest
            scales_crop = self.scales
            quats_crop = self.quats

        albedo_crop = torch.cat((albedo_dc_crop[:, None, :], albedo_rest_crop), dim=1)

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

        if light is not None:
            # TODO: Implement light intrinsic optimization
            irradiance, irradiance_fraction = self.compute_irradiance(light)
            # print(f"max irradiance: {irradiance.max()}, min irradiance: {irradiance.min()}")
        elif self.irradiance is not None and self.irradiance.shape[0] == self.means.shape[0]:
            # NOTE: during training, the sizes occasionally mismatch right after training
            # For now we just display albedo in the viewer for this one frame as a workaround
            irradiance = self.irradiance
            irradiance_fraction = self.irradiance_fraction
        else:
            irradiance, irradiance_fraction = None, None

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            albedo_crop = torch.sigmoid(albedo_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        # First 3 channels of render are albedo, next 3 are relit color (intensity), then shadow, then depth
        render, alpha, self.info = augmented_rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=albedo_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            additional_channels=(
                irradiance_fraction.reshape(-1, 1) if irradiance_fraction is not None else None
            ),  # [(C,) N, D2] or [(C,) N, K, D2]
            color_weights=irradiance,  # [(C,) N, 3],
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
            camera_model=camera_model,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        albedo_rgb = render[:, ..., :3] + (1 - alpha) * background
        albedo_rgb = torch.clamp(albedo_rgb, 0.0, 1.0)

        if irradiance is not None:
            # Apply tone mapping and gamma correction
            relit_intensity = (
                render[:, ..., 3:6] + (1 - alpha) * background
            )  # NOTE: Should we be mixing with the background?

            if self.config.tone_mapping == "reinhard":
                relit_rgb = relit_intensity / (relit_intensity + 1.0)
            # Does luminance tonemapping
            elif self.config.tone_mapping == "luminance":
                luminance = (
                    0.2126 * relit_intensity[..., 0]
                    + 0.7152 * relit_intensity[..., 1]
                    + 0.0722 * relit_intensity[..., 2]
                )
                relit_rgb = relit_intensity / (luminance + 1.0)[..., None]
            # Does linear tonemapping
            elif self.config.tone_mapping == "linear":
                relit_rgb = torch.clamp(relit_intensity, min=0.0, max=1.0)

            # Does gamma correction # NOTE: leads to nans during training
            relit_rgb = relit_rgb ** (1.0 / self.config.gamma_correction)
            if torch.isnan(relit_rgb).any():
                raise ValueError("Relit rgb is nan")
        else:
            relit_rgb = albedo_rgb

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                albedo_rgb = self._apply_bilateral_grid(
                    albedo_rgb, camera.metadata["cam_idx"], H, W
                )

                if light is not None:
                    relit_rgb = self._apply_bilateral_grid(
                        relit_rgb, camera.metadata["cam_idx"], H, W
                    )

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)

            if light is not None:
                # NOTE: currently broken
                shadow_im = render[:, ..., -2:-1]
            else:
                shadow_im = None
        else:
            depth_im = None
            if light is not None:
                shadow_im = render[:, ..., -1:]
            else:
                shadow_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": relit_rgb.squeeze(0),  # type: ignore
            "albedo": albedo_rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "shadow": shadow_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

    ### NOTE: CHANGED ALL REFERENCES TO RGB TO RGB_RELIGHT!!!
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # print(self.get_gt_img(batch["image"]).shape)
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]
        # lit_mask = outputs["shadow_weights"] > self.light_params["cutoff"]
        # lit_mask = torch.sigmoid(10 * (outputs["shadow_weights"] - self.light_params["cutoff"]))
        # gt_img = gt_img * lit_mask
        # pred_img = pred_img * lit_mask

        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # ax[0].imshow(gt_img.cpu().numpy())
        # ax[1].imshow(pred_img.detach().cpu().numpy())
        # ax[2].imshow(outputs["rgb"].detach().cpu().numpy())
        # # ax[3].imshow(outputs["shadow"])
        # plt.show()

        # Check if the gt img has nans
        # print(gt_img.shape)

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            print("Using mask")
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
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        predicted_rgb = outputs["rgb"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
