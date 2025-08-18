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
    augmented_rasterization,
    calculate_relighting_weights,
)

# torch.autograd.set_detect_anomaly(True)
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

import torch_geometric.nn.pool.knn as knn

from shadow_splat.util.nerfstudio import get_viewmat
from shadow_splat.shadow_splat_rendering import lumen_rasterization, calculate_relighting_weights_from_point_cloud, generate_point_cloud_from_camera_depth, chebyshev_weighting, render_equirect_from_sh, render_equirect_from_sg, focal_bce
import matplotlib.pyplot as plt

@dataclass
class LumenModelConfig(SplatfactoModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Lumen"""

    _target: Type = field(default_factory=lambda: LumenModel)
    # # TODO: add shadow splat specific parameters here
    # ambient: bool = (
    #     True  # Controls whether Gaussians outside the light frustum are set to ambient or to black
    # )
    tone_mapping: Literal["linear", "luminance", "reinhard"] = "linear"
    # gamma_correction: float = 1.0
    # fix_variance: bool = False
    # light_transport_sh_degree: int = 3
    light_transport_sg_degree: int = 50
    env_map_sg_degree: int = 3

class LumenModel(SplatfactoModel):
    """Nerfstudio's implementation of Lumen

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: LumenModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        print("model init | seed_points", seed_points)
        # raise
        super().__init__(*args, seed_points=seed_points, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        num_points = self.means.shape[0]

        # TODO: FOR SPHERICAL HARMONICS
        # sh_degree_lumen = num_sh_bases(self.config.light_transport_sh_degree)
        # visibility = torch.nn.Parameter(0.5*torch.ones((num_points, sh_degree_lumen, 1)))        # Initialize to no visibility
        # env_map = torch.nn.Parameter(torch.randn((1, sh_degree_lumen, 3)))
        # self.gauss_params["visibility"] = visibility

        # self.light_params = torch.nn.ParameterDict(
        #     {
        #         "env_map": env_map,
        #     }
        # )

        # TODO: FOR SPHERICAL GAUSSIANS
        visibility_mu = torch.nn.Parameter(torch.randn((num_points, self.config.light_transport_sg_degree, 3)))     # Need to be normalized
        visibility_kappa = torch.nn.Parameter(2*torch.ones((num_points, self.config.light_transport_sg_degree)))  # Need to be non-negative (softplus)
        visibility_logits = torch.nn.Parameter(torch.zeros((num_points, self.config.light_transport_sg_degree))) # Need to be non-negative and sum to 1 (softmax)
        visibility_scale = torch.nn.Parameter(-5*torch.ones((num_points)))  # Need to be non-negative and less than 1 (sigmoid)
        
        self.gauss_params["visibility_mu"] = visibility_mu
        self.gauss_params["visibility_kappa"] = visibility_kappa
        self.gauss_params["visibility_logits"] = visibility_logits
        self.gauss_params["visibility_scale"] = visibility_scale

        env_map_mu = torch.nn.Parameter(torch.randn((self.config.env_map_sg_degree, 1, 3)))     # Need to be normalized
        env_map_kappa = torch.nn.Parameter(100*torch.ones((self.config.env_map_sg_degree, 1)))     # Need to be non-negative (softplus)
        env_map_logits = torch.nn.Parameter(torch.zeros((self.config.env_map_sg_degree, 1)))     # Need to be non-negative and sum to 1 (softmax)
        env_map_scale = torch.nn.Parameter(torch.ones(1))     # Need to be non-negative and less than 1 (sigmoid)
        
        self.light_params = torch.nn.ParameterDict(
            {
                "env_map_mu": env_map_mu,
                "env_map_kappa": env_map_kappa,
                "env_map_logits": env_map_logits,
                "env_map_scale": env_map_scale,
            }
        )

        self.env_map_transform = torch.eye(4)

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

    @property
    def visibility_mu(self):
        return self.gauss_params["visibility_mu"]

    @property
    def visibility_kappa(self):
        return self.gauss_params["visibility_kappa"]

    @property
    def visibility_logits(self):
        return self.gauss_params["visibility_logits"]

    @property
    def visibility_scale(self):
        return self.gauss_params["visibility_scale"]

    @property
    def env_map_mu(self):
        return self.light_params["env_map_mu"]

    @property
    def env_map_kappa(self):
        return self.light_params["env_map_kappa"]

    @property
    def env_map_logits(self):
        return self.light_params["env_map_logits"]

    @property
    def env_map_scale(self):
        return self.light_params["env_map_scale"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities", 
                      "visibility_mu", "visibility_kappa", "visibility_logits", "visibility_scale"]:
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
                packed=True,
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
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities", 
                         "visibility_mu", "visibility_kappa", "visibility_logits", "visibility_scale"]
        }

    def get_light_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            name: [self.light_params[name]]
            for name in ["env_map_mu", "env_map_kappa", "env_map_logits", "env_map_scale"]
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
        return gps

    def forward(
        self, camera: Cameras
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Forward pass that takes a camera.

        Args:
            camera: The camera(s) for which output images are rendered
            light: Optional light source camera for shadow computation

        Returns:
            Outputs of model (ie. rendered colors)
        """
        return self.get_outputs(camera)

    def get_outputs(
        self, camera: Cameras
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

            visibility_mu_crop = self.visibility_mu[crop_ids]
            visibility_kappa_crop = self.visibility_kappa[crop_ids]
            visibility_logits_crop = self.visibility_logits[crop_ids]
            visibility_scale_crop = self.visibility_scale[crop_ids]

        else:
            opacities_crop = self.opacities
            means_crop = self.means
            albedo_dc_crop = self.albedo_dc
            albedo_rest_crop = self.albedo_rest
            scales_crop = self.scales
            quats_crop = self.quats

            visibility_mu_crop = self.visibility_mu
            visibility_kappa_crop = self.visibility_kappa
            visibility_logits_crop = self.visibility_logits
            visibility_scale_crop = self.visibility_scale

        albedo_crop = torch.cat((albedo_dc_crop[:, None, :], albedo_rest_crop), dim=1)

        if self.training:
            env_map_mu = self.env_map_mu
        else:
            env_map_mu = self.env_map_mu @ self.env_map_transform[:3, :3].to(self.env_map_mu.device)

        env_map_kappa = self.env_map_kappa
        env_map_logits = self.env_map_logits
        env_map_scale = self.env_map_scale

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
            albedo_crop = torch.sigmoid(albedo_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        # First 3 channels of render are albedo, next 3 are lumen color (intensity), then depth, then depth^2
        render, alpha, self.info = lumen_rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=albedo_crop,
            visibility=[visibility_mu_crop / torch.linalg.norm(visibility_mu_crop, dim=-1, keepdim=True), 
                        torch.nn.functional.softplus(visibility_kappa_crop), 
                        torch.softmax(visibility_logits_crop, dim=-1), 
                        torch.sigmoid(visibility_scale_crop)],
            env_map=[ (env_map_mu / torch.linalg.norm(env_map_mu, dim=-1, keepdim=True)).repeat(1, 3, 1), 
                     torch.nn.functional.softplus(env_map_kappa).repeat(1, 3), 
                     torch.softmax(env_map_logits, dim=0).repeat(1, 3), 
                     torch.nn.functional.softplus(env_map_scale).repeat(3)],
            viewmats=viewmat,  # [C, 4, 4]
            Ks=K,  # [C, 3, 3]
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            eps2d=0.3,
            sh_degree=sh_degree_to_use,
            # sh_degree_lumen=self.config.light_transport_sh_degree,
            tile_size=16,
            backgrounds=None,
            render_mode=render_mode,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
            distributed=False,
            camera_model=camera_model,
        )

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        albedo_rgb = render[:, ..., :3] + (1 - alpha) * background
        albedo_rgb = torch.clamp(albedo_rgb, 0.0, 1.0)

        # NOTE: IN ORDER FOR THESE TO BE VALID, ED (EXPECTED DEPTH) MUST BE ON!!!
        depth_image = render[..., -2].squeeze()
        depth_sqr_image = render[..., -1].squeeze()
        variance_image = depth_sqr_image - depth_image**2

        if self.training:
            vis_target = chebyshev_weighting(
                self.info["means2d"],
                self.info["depths"],
                depth_image,
                variance_image,
            ).squeeze()

            self.vis = self.info["visibilities"].squeeze()
            self.vis_target = vis_target.detach()

            # assert self.vis is not None, "self.vis is None"
            # print(self.vis)
            # print("Visibility", self.vis.min(), self.vis.max())

        ### TODO!!!
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
        # relit_rgb = relit_rgb ** (1.0 / self.config.gamma_correction)
        if torch.isnan(relit_rgb).any():
            raise ValueError("Relit rgb is nan")
    
        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                albedo_rgb = self._apply_bilateral_grid(
                    albedo_rgb, camera.metadata["cam_idx"], H, W
                )

                relit_rgb = self._apply_bilateral_grid(
                    relit_rgb, camera.metadata["cam_idx"], H, W
                )

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)

        else:
            depth_im = None
 
        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        ### TODO!!!
        # Visualize environment map
        # env_map_image = render_equirect_from_sh(
        #     self.env_map,                 # (N, N_coeffs, 3) or (1, N_coeffs, 3)
        #     self.config.light_transport_sh_degree,
        #     H = relit_rgb.shape[1],
        #     W = relit_rgb.shape[2],
        #     phi_offset = 0.0,  # yaw (radians), positive rotates to the left
        #     v_flip = False,     # flip vertically
        #     convention = "y-up", # "y-up" (x=sinθcosφ, y=cosθ, z=sinθsinφ) or "z-up"
        #     chunk = 131072,      # to limit memory
        # )

        env_map_image = render_equirect_from_sg(
                (env_map_mu / torch.linalg.norm(env_map_mu, dim=-1, keepdim=True)).repeat(1, 3, 1), 
                torch.nn.functional.softplus(env_map_kappa).repeat(1, 3), 
                torch.softmax(env_map_logits, dim=0).repeat(1, 3), 
                torch.nn.functional.softplus(env_map_scale).repeat(3),
                H = relit_rgb.shape[1],
                W = relit_rgb.shape[2],
                phi_offset = 0.0,  # yaw (radians), positive rotates to the left
                v_flip = False,     # flip vertically
                convention = "y-up", # "y-up" (x=sinθcosφ, y=cosθ, z=sinθsinφ) or "z-up"
                chunk = 1000000,      # to limit memory
        )

        return {
            "rgb": relit_rgb.squeeze(0),  # type: ignore
            "albedo": albedo_rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "variance": variance_image.reshape(*depth_im.shape) if depth_im is not None else None,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "env_map": env_map_image.squeeze(0),  # type: ignore
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

        # ====== ALBEDO TV LOSS ======
        albedo_img = outputs["albedo"]
        albedo_tv_loss = 0.001 * total_variation_loss(albedo_img)
        # ====== ALBEDO TV LOSS ======

        # ====== VISIBILITY MATCHING LOSS ======
        # Up weight high visibility regions
        # vis_loss = torch.mean( (self.vis - self.vis_target)**2) # + 0.1 * torch.mean(torch.abs(self.vis))
        # vis_loss = torch.mean( (self.vis - self.vis_target)**2) # + 0.1 * torch.mean(torch.abs(self.vis))
        # print("Vis loss", torch.mean(torch.abs(self.vis - self.vis_target)).item(), self.vis.max(), self.vis.min(), self.vis.mean())#, "Vis regularizer", torch.mean(torch.abs(self.vis)).item())
        
        # Fraction of visibility target above 0.9 or below 0.1
        vis_target_high = torch.where(self.vis_target > 0.9, torch.ones_like(self.vis_target), torch.zeros_like(self.vis_target))
        vis_target_low = torch.where(self.vis_target < 0.1, torch.ones_like(self.vis_target), torch.zeros_like(self.vis_target))
        vis_target_high = vis_target_high.sum() / vis_target_high.numel()
        vis_target_low = vis_target_low.sum() / vis_target_low.numel()
        print("Visibility target", vis_target_high, vis_target_low)

        high = self.vis_target > 0.7
        low = self.vis_target < 0.3
        print("Reconstruction loss on positive labels", torch.mean( (self.vis[high] - self.vis_target[high])**2))
        print("Reconstruction loss on negative labels", torch.mean( (self.vis[low] - self.vis_target[low])**2))

        smallest_mask = torch.min(high.sum(), low.sum())
        high_indices = torch.arange(self.vis.shape[0], device=high.device)[high]
        low_indices = torch.arange(self.vis.shape[0], device=low.device)[low]

        # Of the indices, randomly sample smallest_mask
        high_indices = high_indices[torch.randperm(high_indices.shape[0])[:smallest_mask]]
        low_indices = low_indices[torch.randperm(low_indices.shape[0])[:smallest_mask]]

        #print("High indices", high_indices.shape, "Low indices", low_indices.shape)
        print("Reconstruction loss on positive labels", torch.mean( (self.vis[high_indices] - self.vis_target[high_indices])**2))
        print("Reconstruction loss on negative labels", torch.mean( (self.vis[low_indices] - self.vis_target[low_indices])**2))

        vis_loss = torch.mean( (self.vis[high_indices] - self.vis_target[high_indices])**2) + torch.mean( (self.vis[low_indices] - self.vis_target[low_indices])**2)

        # ====== VISIBILITY MATCHING LOSS ======

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
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss + albedo_tv_loss + vis_loss,
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
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        combined_rgb = torch.cat([images_dict["img"], outputs["albedo"], outputs["depth"], outputs["variance"]], dim=1)
        images_dict["img"] = combined_rgb

        return metrics_dict, images_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = super().get_metrics_dict(outputs, batch)
        return metrics_dict

@dataclass
class ShadowSplatModelConfig(SplatfactoModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: ShadowSplatModel)
    light_optimizer: LightOptimizerConfig = field(default_factory=lambda: LightOptimizerConfig(mode="off"))
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
        print("model init | seed_points", seed_points)
        # raise
        super().__init__(*args, seed_points=seed_points, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        self.light_params = torch.nn.ParameterDict(
            {
                "intensity": torch.nn.Parameter(torch.log(torch.tensor(1.0))),
                "ambient": torch.nn.Parameter(torch.logit(torch.tensor(0.5))),  # in frustum
                # "background_ambient": torch.nn.Parameter(torch.log(torch.tensor(1.0))),  # outside
                # "variance_factor": torch.nn.Parameter(torch.log(torch.tensor(0.01))),
            }
        )

        self.light_optimizer: LightOptimizer = self.config.light_optimizer.setup(
            num_cameras=1, device="cpu"
        )

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
            name: [self.light_params[name]]
            for name in ["intensity", "ambient"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        gps.update(self.get_light_param_groups())
        # gps['reweighting_param'] = [self.reweighting_param]

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

    # def compute_irradiance(
    #     self,
    #     light: Cameras,
    #     variance_factor: Optional[float] = None,
    #     intensity: Optional[float] = None,
    #     ambient: Optional[float] = None,
    #     background_ambient: Optional[float] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Get the irradiance for a given camera and light source.

    #     Args:
    #         camera: The camera(s) for which output images are rendered
    #         light: Optional light source camera for shadow computation
    #     """
    #     camera_scale_fac = self._get_downscale_factor()
    #     opacities_crop = self.opacities
    #     means_crop = self.means
    #     scales_crop = self.scales
    #     quats_crop = self.quats

    #     if self.training:
    #         assert light.shape[0] == 1, "Only one light at a time"
    #         optimized_light_to_world = self.light_optimizer.apply_to_camera(light)
    #     else:
    #         optimized_light_to_world = light.camera_to_worlds

    #     # print("Optimized light to world", optimized_light_to_world)

    #     # TODO: Implement light intrinsic optimization
    #     light_camera_to_world = optimized_light_to_world
    #     light.rescale_output_resolution(1 / camera_scale_fac)
    #     light_viewmat = get_viewmat(light_camera_to_world)
    #     light_K = light.get_intrinsics_matrices().cuda()
    #     light_W, light_H = int(light.width.item()), int(light.height.item())
    #     self.light_last_size = (light_H, light_W)
    #     light.rescale_output_resolution(camera_scale_fac)  # type: ignore

    #     if light.camera_type == CameraType.PERSPECTIVE.value:
    #         light_model = "pinhole"
    #     elif light.camera_type == CameraType.ORTHOPHOTO.value:
    #         light_model = "ortho"
    #     elif light.camera_type == CameraType.FISHEYE.value:
    #         light_model = "fisheye"
    #     else:
    #         raise ValueError("Unknown light type: %s", light.camera_type)

    #     # if self.training:
    #     #     hard_cutoff = False
    #     # else:
    #     #     hard_cutoff = True

    #     # print lighting params values
    #     if variance_factor is None:
    #         variance_factor = torch.exp(self.light_params["variance_factor"])
    #     if intensity is None:
    #         intensity = torch.exp(self.light_params["intensity"]) * torch.ones(3).to(self.device)
    #     if ambient is None:
    #         ambient = torch.nn.functional.sigmoid(self.light_params["ambient"])
    #     # if background_ambient is None:
    #     #     background_ambient = torch.exp(self.light_params["background_ambient"])

    #     # Check for NaN values in the input tensors
    #     if torch.isnan(means_crop).any():
    #         raise ValueError("NaN values detected in means_crop")
    #     if torch.isnan(quats_crop).any():
    #         raise ValueError("NaN values detected in quats_crop")
    #     if torch.isnan(scales_crop).any():
    #         raise ValueError("NaN values detected in scales_crop")

    #     irradiance, irradiance_fraction, light_depth_image, light_variance_image = calculate_relighting_weights(
    #         means=means_crop,  # [N, 3]
    #         quats=quats_crop,  # [N, 4]
    #         scales=torch.exp(scales_crop),  # [N, 3]
    #         opacities=torch.sigmoid(opacities_crop).squeeze(-1),  # [N]
    #         viewmats=light_viewmat,  # [C, 4, 4]
    #         Ks=light_K,  # [C, 3, 3]
    #         width=light_W,
    #         height=light_H,
    #         variance_factor=variance_factor,
    #         intensity=intensity,
    #         ambient=ambient,
    #         background_ambient=None,
    #         near_plane=0.01,
    #         far_plane=1e10,
    #         sparse_grad=False,
    #         absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
    #         rasterize_mode=self.config.rasterize_mode,
    #         camera_model=light_model,
    #         distloss=False,  # 2DGS only
    #         fix_variance=self.config.fix_variance,
    #     )
    #     self.irradiance = irradiance
    #     self.irradiance_fraction = irradiance_fraction
    #     self.light_depth_image = light_depth_image
    #     self.light_variance_image = light_variance_image
    #     return irradiance, irradiance_fraction, light_depth_image, light_variance_image

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
        # render, alpha, self.info = augmented_rasterization(
        #     means=means_crop,
        #     quats=quats_crop,  # rasterization does normalization internally
        #     scales=torch.exp(scales_crop),
        #     opacities=torch.sigmoid(opacities_crop).squeeze(-1),
        #     colors=albedo_crop,
        #     viewmats=viewmat,  # [1, 4, 4]
        #     Ks=K,  # [1, 3, 3]
        #     width=W,
        #     height=H,
        #     packed=False,
        #     near_plane=0.01,
        #     far_plane=1e10,
        #     render_mode=render_mode,
        #     sh_degree=sh_degree_to_use,
        #     additional_channels=None,  # [(C,) N, D2] or [(C,) N, K, D2]
        #     color_weights=None,  # [(C,) N, 3],
        #     sparse_grad=False,
        #     absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
        #     rasterize_mode=self.config.rasterize_mode,
        #     camera_model=camera_model,
        #     # set some threshold to disregrad small gaussians for faster rendering.
        #     # radius_clip=3.0,
        # )

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=albedo_crop,
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
                assert light.shape[0] == 1, "Only one light at a time"
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

            # Check for NaN values in the input tensors
            if torch.isnan(means_crop).any():
                raise ValueError("NaN values detected in means_crop")
            if torch.isnan(quats_crop).any():
                raise ValueError("NaN values detected in quats_crop")
            if torch.isnan(scales_crop).any():
                raise ValueError("NaN values detected in scales_crop")

            relit_rgb, shadow_img, light_depth_image, light_variance_image = calculate_relighting_weights_from_point_cloud(
                means=means_crop,  # [N, 3]
                quats=quats_crop,  # [N, 4]
                scales=torch.exp(scales_crop),  # [N, 3]       # NOTE: IMPORTANT! THESE SCALES MUST ALREADY BE POSITIVE
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),  # [N]       # NOTE: IMPORTANT! THESE OPACITIES MUST ALREADY BE [0, 1]
                viewmats=light_viewmat,  # [C, 4, 4]
                Ks=light_K,  # [C, 3, 3]
                width=light_W,
                height=light_H,
                point_cloud=point_cloud,
                point_cloud_mask=point_cloud_mask,
                rgb_image=render[:, ..., :3].squeeze(),
                intensity=torch.exp(self.light_params["intensity"]),
                ambient=torch.sigmoid(self.light_params["ambient"]),
                camera_model=light_model,
            )

        else:
            relit_rgb = render[:, ..., :3]
            shadow_img = None
            light_depth_image = None
            light_variance_image = None

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        albedo_rgb = render[:, ..., :3] + (1 - alpha) * background
        albedo_rgb = torch.clamp(albedo_rgb, 0.0, 1.0)

        # Apply tone mapping and gamma correction
        relit_rgb = (
            relit_rgb + (1 - alpha) * background
        )  # NOTE: Should we be mixing with the background?

        if self.config.tone_mapping == "reinhard":
            relit_rgb = relit_rgb / (relit_rgb + 1.0)
        # Does luminance tonemapping
        elif self.config.tone_mapping == "luminance":
            luminance = (
                0.2126 * relit_rgb[..., 0]
                + 0.7152 * relit_rgb[..., 1]
                + 0.0722 * relit_rgb[..., 2]
            )
            relit_rgb = relit_rgb / (luminance + 1.0)[..., None]
        # Does linear tonemapping
        elif self.config.tone_mapping == "linear":
            relit_rgb = torch.clamp(relit_rgb, min=0.0, max=1.0)

        # Does gamma correction # NOTE: leads to nans during training
        # relit_rgb = relit_rgb ** (1.0 / self.config.gamma_correction)
        if torch.isnan(relit_rgb).any():
            raise ValueError("Relit rgb is nan")
  
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
        else:
            depth_im = None
           
        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        # if self.training:
        #     low_visibility_mask = (irradiance_fraction < 0.3).reshape(-1)
        #     low_visibility_points = means_crop[low_visibility_mask]
        #     ind = knn(low_visibility_points.detach(), low_visibility_points.detach(), 4)
        #     neighbor_ind = ind[1].reshape(-1, 4)[:, 1:].reshape(-1)
            
        #     # ind is 2d tensor of shape [N, 3]. Select the colors of the 3 neighbors
        #     neighbor_colors = self.info["colors"].squeeze()[low_visibility_mask][neighbor_ind]
        #     neighbor_colors = neighbor_colors.reshape(-1, 3, 3)

        #     # Compute loss that penalizes them to be close to each other
        #     mean_neighbor_colors = torch.mean(neighbor_colors, dim=1)
   
        #     self.neighbor_color_loss = torch.mean((1 - irradiance_fraction)[low_visibility_mask][:, None, None] *(neighbor_colors - mean_neighbor_colors[:, None, :])**2)
        return {
            "rgb": relit_rgb.squeeze(0),  # type: ignore
            "albedo": albedo_rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "shadow": shadow_img.unsqueeze(-1) if shadow_img is not None else torch.zeros_like(depth_im),  # type: ignore
            "light_depth": light_depth_image.unsqueeze(-1) if light_depth_image is not None else torch.zeros_like(depth_im),  # type: ignore
            "light_variance": light_variance_image.unsqueeze(-1) if light_variance_image is not None else torch.zeros_like(depth_im),  # type: ignore
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

        # ====== ALBEDO TV LOSS ======
        albedo_img = outputs["albedo"]
        albedo_tv_loss = 0.0 #0.1 * total_variation_loss(albedo_img)
        # ====== ALBEDO TV LOSS ======

        # Variance loss #
        light_variance_img = outputs["light_variance"]
        light_depth_img = outputs["light_depth"]
        light_tv_loss = 0.0 # torch.mean(light_variance_img) #+ total_variation_loss(light_depth_img)
        # ====== Variance loss ======

        # Neighbor color loss #
        # neighbor_color_loss = 0.1*self.neighbor_color_loss
        # ====== Neighbor color loss ======

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
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss + light_tv_loss, #+ neighbor_color_loss,
            "scale_reg": scale_reg,
            "albedo_tv_loss": albedo_tv_loss,
            "light_tv_loss": light_tv_loss,
            # "neighbor_color_loss": neighbor_color_loss,
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
        combined_rgb = torch.cat([images_dict["img"], outputs["albedo"], shadow_rgb], dim=1)
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
        metrics_dict["ambient"] = torch.exp(self.light_params["ambient"])
        # metrics_dict["background_ambient"] = torch.exp(self.light_params["background_ambient"])
        self.light_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict
