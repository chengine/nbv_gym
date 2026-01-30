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
Contains model classes for Coverage Splatting and Fisher-RF Splatting.
"""

from __future__ import annotations
import math

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Type, Union, Any
from functools import partial
import torch
from torch.nn import Parameter
from gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from nbv_gym.util.more_rendering import (
    rasterization_with_view_attributes,
    moment_rasterization,
)

### For Fisher-RF Splatting
try:
    from modified_diff_gaussian_rasterization_depth import (
        GaussianRasterizer as ModifiedGaussianRasterizer,
        GaussianRasterizationSettings,
    )
    from einops import repeat, reduce, rearrange
except ImportError:
    print("Please install GaussianRasterizationDepth and/or einops")

###

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from nerfstudio.model_components.lib_bilagrid import (
    BilateralGrid,
    color_correct,
    slice,
    total_variation_loss,
)

from nbv_gym.util.nerfstudio import get_viewmat

# For fisher-splat
from nbv_gym.util.camera_utils import projection_matrix, to_homo

# For computing view metrics
from nbv_gym.util.coverage import (
    fibonacci_sphere,
    compute_coverage_metric,
    compute_fig_metric,
    compute_fig_diag_metric,
    compute_view_fig_metric,
    compute_view_fig_diag_metric,
    compute_fig_color_field_metric,
    update_view_coverage_for_frustum,
    update_fig_for_frustum,
    update_view_fig_for_frustum,
    update_fig_color_field_for_frustum,
)

@dataclass
class NBVSplatModelConfig(SplatfactoModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: NBVSplatModel)

    n_sphere_bins: int = 128
    """Number of bins on the unit sphere for coverage computation."""
    concentration: float = 5.0
    """Concentration parameter for the spherical gaussian kernel."""

    # Fisher-RF uncertainty settings
    fisher_rf_depth_weight: float = 1.0
    """Weight of depth uncertainty in Fisher-RF Hessian computation."""
    fisher_rf_rgb_weight: float = 1.0
    """Weight of RGB uncertainty in Fisher-RF Hessian computation."""

class NBVSplatModel(SplatfactoModel):
    """Nerfstudio's implementation of Shadow Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: NBVSplatModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        print('kwargs', kwargs, 'args', args)
        super().__init__(*args, seed_points=seed_points, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        ### THIS IS FOR PARAMETRIZING THE UNIT SPHERE ###
        self.bin_dirs = fibonacci_sphere(n_bins=self.config.n_sphere_bins, device="cuda")

    def setup_view_metric(
        self,
        view_metric: Literal["coverage", "fig", "view_fig", "fig_diag", "view_fig_diag", "fig_color_field", "fisher_rf", None]):

        self.view_metric = view_metric

        # Initialize the view_attributes tensor to be correct shapes
        # NOTE: We update gauss_params directly because view_attributes is a read-only property

        if view_metric in [None, "fig", "fig_diag"]:
            # None: Dummy variables (not used)
            # fig: The running sum of rendering weights over camera views per-Gaussian
            # fig_diag: Is the same object as fig. Only difference is when we render the fig_diag, we take the reciprocal.
            new_view_attributes = torch.nn.Parameter(torch.zeros((self.means.shape[0]), device="cuda"))

        elif view_metric in ["coverage", "view_fig", "view_fig_diag"]:
            # coverage: The running counts of the hits on a Gaussian per patch of the unit viewing direction sphere
            # view_fig: The running sum of rendering weights of the color field per patch of the unit viewing direction sphere
            # view_fig_diag: Is the same object as view_fig. Only difference is when we render the view_fig_diag, we take the reciprocal.
            new_view_attributes = torch.nn.Parameter(torch.zeros((self.means.shape[0], self.config.n_sphere_bins), device="cuda"))

        elif view_metric in ["fig_color_field"]:
            # fig_color_field: Column 0 is the start index of the camera_ids tensor. Column 1 is the length after the start index.
            # This information allows us to implicitly compute visible gaussian_ids for all training cameras, while the camera_ids is already stored.
            new_view_attributes = torch.nn.Parameter(torch.zeros((self.means.shape[0], 2), device="cuda"))

        elif view_metric in ["fisher_rf"]:
            # fisher_rf: Per-Gaussian accumulated Hessian from training cameras (Fisher information)
            new_view_attributes = torch.nn.Parameter(torch.zeros((self.means.shape[0]), device="cuda"))

        else:
            raise ValueError(f"Unknown view metric: {view_metric}")

        self.gauss_params["view_attributes"] = new_view_attributes

        # Initialize the per-Gaussian view attribute function

        if view_metric == "fig":
            self.view_attributes_fn = partial(compute_fig_metric, bin_dirs=self.bin_dirs)
        
        elif view_metric == "fig_diag":
            self.view_attributes_fn = partial(compute_fig_diag_metric, bin_dirs=self.bin_dirs)

        elif view_metric == "coverage":
            self.view_attributes_fn = partial(compute_coverage_metric, bin_dirs=self.bin_dirs)

        elif view_metric == "view_fig":
            self.view_attributes_fn = partial(compute_view_fig_metric, bin_dirs=self.bin_dirs, concentration=self.config.concentration)

        elif view_metric == "view_fig_diag":
            self.view_attributes_fn = partial(compute_view_fig_diag_metric, bin_dirs=self.bin_dirs, concentration=self.config.concentration)

        elif view_metric == "fig_color_field":
            self.view_attributes_fn = None

        elif view_metric == "fisher_rf":
            # Fisher-RF uses per-Gaussian Hessian; rendering uses reciprocal as uncertainty
            self.view_attributes_fn = None  # Handled specially in get_outputs

    @torch.no_grad()
    # Initialize the update_view_attributes function
    def update_view_attributes(self, cameras: List[Cameras], reset_view_attributes: bool = True):
        # Reset the view attributes to 0 (to get accurate accumulation. If you want accumulation, turn this off)
        if reset_view_attributes:
            self.reset_view_attributes()

        if self.view_metric == "fig_color_field":
            visibility_list = []
            gaussian_ids_list = []
            train_cam_pos_list = []

        # Update view attributes based on all cameras in the camera batch, conditioned on the current state of the scene
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

            if self.view_metric == "coverage":
                # Updates the coverage view_attributes in place
                update_view_coverage_for_frustum(
                means=self.means,
                quats=self.quats,
                scales=torch.exp(self.scales),
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                coverage_counts=self.view_attributes,
                bin_dirs=self.bin_dirs,
                camera_model=camera_model,
                near_plane=0.01,
                far_plane=1e10,
                radius_clip=3.0,
                )

            elif self.view_metric in ["fig", "fig_diag", "view_fig", "view_fig_diag", "fig_color_field"]:

                # View metrics requiring rendering weights require the rasterization of the rendering weights.
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

                if self.view_metric in ["fig", "fig_diag"]:
                    update_fig_for_frustum(
                        means=self.means,
                        quats=self.quats,
                        scales=torch.exp(self.scales),
                        viewmats=viewmat,
                        Ks=K,
                        width=W,
                        height=H,
                        depth_image=depth_image,
                        variance_image=variance_image,
                        fig=self.view_attributes,
                        camera_model=camera_model,
                        near_plane=0.01,
                        far_plane=1e10,
                        radius_clip=3.0,
                        alpha_image=alphas.squeeze(0).squeeze(-1),
                    )
                elif self.view_metric in ["view_fig", "view_fig_diag"]:
                    update_view_fig_for_frustum(
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
                        view_fig=self.view_attributes,
                        camera_model=camera_model,
                        eps2d=0.3,
                        near_plane=0.01,
                        far_plane=1e10,
                        radius_clip=3.0,
                        concentration=self.config.concentration,
                        alpha_image=alphas.squeeze(0).squeeze(-1),
                    )
                elif self.view_metric == "fig_color_field":
                    update_fig_color_field_for_frustum(
                        means=self.means,
                        quats=self.quats,
                        scales=torch.exp(self.scales),
                        viewmats=viewmat,
                        Ks=K,
                        width=W,
                        height=H,
                        depth_image=depth_image,
                        variance_image=variance_image,
                        visibility_list=visibility_list,
                        gaussian_ids_list=gaussian_ids_list, 
                        pointer_length=self.view_attributes[:, 1],
                        train_cam_pos_list=train_cam_pos_list,
                        camera_model=camera_model,
                        eps2d=0.3,
                        near_plane=0.01,
                        far_plane=1e10,
                        radius_clip=3.0,
                        alpha_image=alphas.squeeze(0).squeeze(-1),
                    )

        # Only do this for fig_color_field
        if self.view_metric == "fig_color_field":
            self.view_attributes[1:, 0] = self.view_attributes[:, 1].cumsum(0)[:-1]

            train_cam_pos = torch.stack(train_cam_pos_list, dim=0)

            # Allocate pooled storage (K = total #pairs)
            M = int(self.view_attributes[:, 1].sum().item())
            cam_pool = torch.empty(M, dtype=torch.int64, device=self.device)   # C<500 fits
            w_pool   = torch.empty(M, dtype=self.view_attributes.dtype, device=self.device)  # optional

            cursor = self.view_attributes[:, 0].to(torch.int64).clone()

            for camera_id, gs_ids in enumerate(gaussian_ids_list):
                pos = cursor[gs_ids]            # where to write this camera's entries
                cam_pool[pos] = camera_id
                w_pool[pos]   = visibility_list[camera_id]   # optional

                # advance cursor for these gaussians
                cursor.index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.int64))

            self.cam_pool = cam_pool
            self.w_pool = w_pool
            self.train_cam_pos = train_cam_pos

            self.view_attributes_fn = partial(compute_fig_color_field_metric, 
            training_cameras_positions=train_cam_pos,
            training_camera_ids=cam_pool,
            training_visibilities=w_pool,
            kappa=self.config.concentration)

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def view_attributes(self):
        return self.gauss_params["view_attributes"]

    @torch.no_grad()
    def reset_view_attributes(self):
        self.view_attributes.zero_()
        self.gauss_params["view_attributes"].zero_()

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
            # Don't remap view_attributes - they may have different shapes

        # Remove view_attributes from checkpoint if shape doesn't match
        # (view_attributes are not needed for RGB rendering)
        if "gauss_params.view_attributes" in dict:
            checkpoint_shape = dict["gauss_params.view_attributes"].shape
            model_shape = self.gauss_params["view_attributes"].shape
            if len(checkpoint_shape) != len(model_shape) or checkpoint_shape[1:] != model_shape[1:]:
                del dict["gauss_params.view_attributes"]
        if "view_attributes" in dict:
            del dict["view_attributes"]

        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            if name == "view_attributes":
                continue  # Skip view_attributes - keep model's initialized shape
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))

        # Resize view_attributes to match the new number of Gaussians
        # but keep the model's expected shape for the remaining dimensions
        view_attr_shape = self.gauss_params["view_attributes"].shape
        new_view_attr_shape = (newp,) + view_attr_shape[1:]
        self.gauss_params["view_attributes"] = torch.nn.Parameter(
            torch.zeros(new_view_attr_shape, device=self.device)
        )

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
                "view_attributes",
            ]
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
        return gps

    def forward(
        self, camera: Cameras
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Forward pass that takes a camera.

        Args:
            camera: The camera(s) for which output images are rendered

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

        # TODO: HARDCODED. MAKE THIS MORE ELEGANT. This is supposedly supposed to fix how nerfstudio handles fisheye cameras.
        camera_model = "pinhole"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            features_crop = torch.sigmoid(features_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        if self.view_metric is not None and self.view_attributes_fn is not None:
            render_view_attributes_fn = partial(self.view_attributes_fn, view_attributes=self.view_attributes.detach())
        else:
            # For cases where view_attributes_fn is None (e.g., fisher_rf which uses FisherSplatModel instead)
            render_view_attributes_fn = None

        if self.view_metric == "fig_color_field" and render_view_attributes_fn is not None:
            render_view_attributes_fn = partial(render_view_attributes_fn, means=means_crop)

        render, alpha, self.info = rasterization_with_view_attributes(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=features_crop,
            render_view_attributes_fn=render_view_attributes_fn if not self.training else None,  # We do not want to compute view attributes during training, slowing down training
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
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )

        alpha = alpha[:, ...]
        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if not self.training and self.view_metric is not None:
            view_metric = render[:, ..., 3:4]
        else:
            view_metric = torch.zeros((H, W, 1), device=self.device)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode in ["ED", "RGB+ED"]:
            depth_im = render[:, ..., -1:].squeeze(0)
            depth_im = torch.where(alpha.squeeze(0) > 0, depth_im, depth_im.detach().max())
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "view_metric": view_metric.squeeze(0),  # type: ignore
        }  # type: ignore

    @torch.no_grad()
    def view_metric_score_for_camera(
        self,
        camera: Cameras,
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

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)

        # Render from camera - get_outputs will use render_mode="RGB+ED" which includes coverage
        outputs = self.get_outputs(camera)
        view_metric = outputs["view_metric"]
        valid_mask = outputs["accumulation"] > 0

        # Sum all pixel values where alphas is > 0 to get total view metric score
        # Detach to avoid keeping references to the computation graph
        view_metric_score = ((view_metric * valid_mask).sum() / valid_mask.sum()).detach()

        # Restore training state
        if was_training:
            self.train()

        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
        return view_metric_score

    def view_metric_score_for_camera_differentiable(
        self,
        camera: Cameras,
        intrinsics_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute coverage score for a candidate camera (differentiable version).

        Same as view_metric_score_for_camera but allows gradients to flow through.
        Required for gradient descent optimization of camera pose.

        Args:
            camera: Camera object to evaluate (with differentiable camera_to_worlds)
            intrinsics_scale: Scale factor for camera intrinsics (for faster evaluation).
                Values < 1.0 downscale the resolution.

        Returns:
            Scalar tensor with the total coverage score (sum of all coverage pixels).
            Lower values indicate views that see poorly-covered areas.
        """
        # Save current training state
        was_training = self.training

        # Set to eval mode for inference
        self.eval()

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)

        # Render from camera - get_outputs will use render_mode="RGB+ED" which includes coverage
        outputs = self.get_outputs(camera)
        view_metric = outputs["view_metric"]
        valid_mask = outputs["accumulation"] > 0

        # Sum all pixel values where alphas is > 0 to get total view metric score
        # Do NOT detach - allow gradients to flow
        view_metric_score = (view_metric * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        # Restore training state
        if was_training:
            self.train()

        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
        return view_metric_score

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
        return metrics_dict, images_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = super().get_metrics_dict(outputs, batch)
        return metrics_dict

    # @torch.no_grad()
    # def coverage_score_for_camera_rollouts(
    #     self,
    #     training_cameras: List[Cameras],
    #     test_cameras: List[Cameras],
    #     light=None,
    #     metric: Literal["coverage", "fig", "view_fig", "coverage_lit"] = "coverage",
    #     num_rollouts: int = 10,
    # ) -> torch.Tensor:
    #     """Compute coverage score for set of candidate cameras.

    #     Renders from the given camera and computes the sum of all pixel values in the
    #     coverage image. Higher scores indicate better coverage (more Gaussians seen from
    #     more directions). Lower scores indicate novel/uncovered areas.

    #     Args:
    #         camera: Camera object to evaluate
    #         intrinsics_scale: Scale factor for camera intrinsics (for faster evaluation).
    #             Values < 1.0 downscale the resolution.

    #     Returns:
    #         Scalar tensor with the total coverage score (sum of all coverage pixels).
    #         Lower values indicate views that see poorly-covered areas.
    #     """
    #     # Save current training state
    #     was_training = self.training

    #     # Set to eval mode for faster inference
    #     self.eval()

    #     if self.coverage_metric in ["coverage", "coverage_lit"]:
    #         update_fn = self.update_coverage
    #         reset_fn = self.reset_coverage

    #     elif self.coverage_metric in ["fig", "view_fig"]:
    #         update_fn = self.update_fig
    #         reset_fn = self.reset_fig

    #     else:
    #         raise ValueError(f"Invalid coverage metric: {self.coverage_metric}")

    #     raise NotImplementedError("Not implemented")

    #     # Update coverage metric for training cameras

    #     picks = min(num_rollouts, len(test_cameras))
    #     for _ in range(picks):
    #         if len(candidate_indices) == 0:
    #             break

    #         # Score the current pool
    #         scores: List[tuple[int, float]] = []

    #         for test_cam in test_cameras:
    #             s = self.coverage_score_for_camera(
    #                 test_cam, intrinsics_scale=self.intrinsics_scale, metric=self.coverage_metric
    #             )  # lower is better
    #             scores.append((cam_idx, float(s.item())))

    #         # Pick the lowest score
    #         scores.sort(key=lambda x: x[1])
    #         best_idx = scores[0][0]
    #         selected_indices.append(best_idx)
    #         candidate_indices.remove(best_idx)

    #         # Update internal state incrementally for ShadowSplatModel
    #         best_cam = all_cameras[best_idx:best_idx+1]
    #         if self.coverage_metric == "coverage":
    #             model.update_coverage([best_cam])
    #         else:  # "fig" or "view_fig"
    #             model.update_fig([best_cam])

    #         print("Updating hypothetical coverage/fig using camera", best_idx)

    #         # Memory hygiene between iterations
    #         if hasattr(model, "info"):
    #             if isinstance(model.info, dict):
    #                 for _, v in list(model.info.items()):
    #                     if isinstance(v, torch.Tensor):
    #                         del v
    #                 model.info.clear()
    #             model.info = {}
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    #     camera_scale_fac = self._get_downscale_factor()
    #     camera.rescale_output_resolution(1 / camera_scale_fac)

    #     # Render from camera - get_outputs will use render_mode="RGB+ED" which includes coverage
    #     outputs = self.get_outputs(camera, light=light)

    #     # Extract coverage from outputs
    #     if metric in outputs:
    #         coverage = outputs[metric]  # [H, W] or [H, W, 1]
    #     else:
    #         # Fallback: coverage might be in render output
    #         # This should not happen if render_mode is set correctly, but handle gracefully
    #         coverage = torch.zeros((camera.height.item(), camera.width.item()), device=self.device)
    #         print(f"{metric} not found in outputs")

    #     valid_mask = outputs["accumulation"] > 0

    #     # Sum all pixel values where alphas is > 0 to get total coverage score
    #     # Detach to avoid keeping references to the computation graph
    #     coverage_score = ((coverage * valid_mask).sum() / valid_mask.sum()).detach()

    #     # Restore training state
    #     if was_training:
    #         self.train()

    #     camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
    #     return coverage_score

@dataclass
class FisherSplatModelConfig(SplatfactoModelConfig):
    """Minimal FisherRF Model Config for Fisher-RF view selection.

    This model uses the modified Gaussian rasterizer from the FisherRF paper
    to compute per-Gaussian Hessians for uncertainty-based view selection.
    """

    _target: Type = field(default_factory=lambda: FisherSplatModel)

    depth_uncertainty_weight: float = 1.0
    """Weight of depth uncertainty in the Hessian computation."""
    rgb_uncertainty_weight: float = 1.0
    """Weight of RGB uncertainty in the Hessian computation."""


class FisherSplatModel(SplatfactoModel):
    """Minimal FisherRF Gaussian Splatting model for uncertainty-based view selection.

    This model extends SplatfactoModel with methods for computing Fisher information
    (diagonal Hessian) for uncertainty-based next-best-view selection.

    The core methods are:
    - prepare_rasterizer(): Prepares the modified Gaussian rasterizer
    - compute_diag_H_rgb_depth(): Computes diagonal Hessian on RGB/depth
    - render_uncertainty_rgb_depth(): Renders uncertainty maps
    - coverage_score_for_camera(): Scores cameras by uncertainty
    """

    config: FisherSplatModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(*args, seed_points=seed_points, **kwargs)

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # Resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, remap parameter names
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    @torch.no_grad()
    def prepare_rasterizer(
        self, camera: Cameras
    ) -> Tuple["ModifiedGaussianRasterizer", List[torch.Tensor]]:
        """Prepare the modified Gaussian rasterizer for Fisher-RF uncertainty computation.

        Args:
            camera: Camera to render from

        Returns:
            Tuple of (rasterizer, params) where params is [means3D, shs, opacities, scales, rotations]
        """
        if not isinstance(camera, Cameras):
            raise ValueError("Called prepare_rasterizer with not a camera")

        optimized_camera_to_world = camera.camera_to_worlds

        # Move to the GPU
        camera = camera.to(self.device)
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)

        # Compute view matrix
        optimized_camera_to_world = optimized_camera_to_world.squeeze()
        R = optimized_camera_to_world[:3, :3]
        T = optimized_camera_to_world[:3, 3:4]

        # Flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit

        # Analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv

        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        # Get Gaussian parameters
        opacities_crop = self.opacities
        means_crop = self.means
        features_dc_crop = self.features_dc
        features_rest_crop = self.features_rest
        scales_crop = self.scales
        quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        # Rescale camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)

        opacities = torch.sigmoid(opacities_crop)

        # Compute projection matrix
        fovx = 2 * torch.atan(camera.width / (2 * camera.fx))
        fovy = 2 * torch.atan(camera.height / (2 * camera.fy))
        tanfovx = math.tan(fovx * 0.5)
        tanfovy = math.tan(fovy * 0.5)
        bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
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
            scale_modifier=1.0,
            viewmatrix=viewmat.t(),
            projmatrix=viewmat.t() @ projmat.t(),
            sh_degree=sh_degree_to_use,
            campos=viewmat.inverse()[:3, 3],
            prefiltered=False,
            debug=False,
        )
        rasterizer = ModifiedGaussianRasterizer(raster_settings=raster_settings)

        # Create temporary variables to avoid side effects of the backward engine
        means3D = means_crop.clone().requires_grad_(True)
        shs = colors_crop.clone().requires_grad_(True)
        opacities = opacities.clone().requires_grad_(True)
        scales = torch.exp(scales_crop.clone()).requires_grad_(True)
        rotations = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
        rotations.requires_grad_(True)

        params = [means3D, shs, opacities, scales, rotations]

        return rasterizer, params

    @torch.no_grad()
    def compute_diag_H_rgb_depth(self, camera: Cameras, compute_rgb_H: bool = False) -> Dict[str, Any]:
        """Compute diagonal Hessian on RGB or depth.

        Args:
            camera: Camera to render from
            compute_rgb_H: If True, compute Hessian w.r.t. RGB. If False, compute w.r.t. depth.

        Returns:
            Dict with keys:
                - 'rgb': Rendered RGB image (H, W, C)
                - 'depth': Depth map (H, W)
                - 'H': List of diagonal Hessians for [means3D, shs, opacities, scales, rotations]
        """
        rasterizer, params = self.prepare_rasterizer(camera)
        means3D, shs, opacities, scales, rotations = params

        # Create zero tensor for screen-space points gradients
        screenspace_points = (
            torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        )
        try:
            screenspace_points.retain_grad()
        except Exception:
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

        cur_H = [p.grad.detach().clone() for p in params]
        rgb = rearrange(rendered_image, "c h w -> h w c")

        return {"rgb": rgb, "H": cur_H, "depth": rendered_depth}

    @torch.no_grad()
    def render_uncertainty_rgb_depth(
        self,
        train_cameras: Iterable[Cameras],
        test_cameras: Iterable[Cameras],
        rgb_weight: float = 1.0,
        depth_weight: float = 1.0,
    ) -> List[torch.Tensor]:
        """Render uncertainty maps using Fisher-RF method.

        Args:
            train_cameras: Training cameras to compute Hessian from
            test_cameras: Test cameras to render uncertainty for
            rgb_weight: Weight for RGB Hessian
            depth_weight: Weight for depth Hessian

        Returns:
            List of uncertainty maps, one per test camera
        """
        H_per_gaussian = torch.zeros(
            self.opacities.shape[0], device=self.opacities.device, dtype=self.opacities.dtype
        )

        # Downscale camera intrinsics for faster evaluation
        camera_scale_fac = self._get_downscale_factor()

        # Go through provided training cameras
        for i, train_cam in enumerate(train_cameras):
            print(f"Processing training camera {i} of {len(train_cameras)}")
            train_cam = train_cam.to(self.device)
            train_cam.rescale_output_resolution(1 / camera_scale_fac)

            # Get RGB uncertainty
            H_info_rgb = self.compute_diag_H_rgb_depth(train_cam, compute_rgb_H=True)
            H_info_rgb["H"] = [p * rgb_weight for p in H_info_rgb["H"]]
            H_per_gaussian += sum([reduce(p, "n ... -> n", "sum") for p in H_info_rgb["H"]])

            # Get depth uncertainty
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

            # Create zero tensor for screen-space points
            screenspace_points = (
                torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
            )
            try:
                screenspace_points.retain_grad()
            except Exception:
                pass

            # Compute depth-weighted Hessian color
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
            uncern_maps.append(rendered_image[0])

            test_cam.rescale_output_resolution(camera_scale_fac)

        return uncern_maps

    @torch.no_grad()
    def coverage_score_for_camera(
        self,
        training_cameras: List[Cameras],
        test_cameras: List[Cameras],
        intrinsics_scale: float = 1.0,
    ) -> List[torch.Tensor]:
        """Compute uncertainty scores for candidate cameras.

        Computes Fisher information-based uncertainty for each test camera
        given the training cameras. Lower scores indicate views with higher
        uncertainty (more novel/uncovered areas).

        Args:
            training_cameras: List of training cameras to compute Hessian from
            test_cameras: List of test cameras to score
            intrinsics_scale: Scale factor for camera intrinsics (for efficiency)

        Returns:
            List of scalar tensors with uncertainty scores (lower = more uncertain = better to select)
        """
        was_training = self.training
        self.eval()

        uncertainty = self.render_uncertainty_rgb_depth(
            training_cameras,
            test_cameras,
            rgb_weight=self.config.rgb_uncertainty_weight,
            depth_weight=self.config.depth_uncertainty_weight,
        )

        # Score each test camera: negate so that higher uncertainty = lower score
        coverage_scores = []
        for unc_map in uncertainty:
            valid_mask = unc_map > 0
            score = -((unc_map * valid_mask).sum() / valid_mask.sum().clamp(min=1))
            coverage_scores.append(score)

        if was_training:
            self.train()

        return coverage_scores
