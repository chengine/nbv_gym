"""Custom pipeline for Shadow Splat"""

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler

from shadow_splat.datamanager import ShadowSplatDataManagerConfig
from shadow_splat.datamanager import ViewSelectionDataManagerConfig
from shadow_splat.model import ShadowSplatModelConfig
from shadow_splat.view_selector import create_view_selector


@dataclass
class ShadowSplatPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ShadowSplatPipeline)
    """target class to instantiate"""
    datamanager: ShadowSplatDataManagerConfig = ShadowSplatDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ShadowSplatModelConfig()
    """specifies the model config"""
    disable_light: bool = False
    """specifies whether to train without light"""


class ShadowSplatPipeline(VanillaPipeline):
    def __init__(
        self,
        config: ShadowSplatPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        cameras, batch, light = self.datamanager.next_train(step)

        if self.config.disable_light:
            model_outputs = self._model(cameras)
        else:
            model_outputs = self._model(cameras, light)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch, light = self.datamanager.next_eval(step)
        if self.config.disable_light:
            model_outputs = self.model(ray_bundle)
        else:
            model_outputs = self.model(ray_bundle, light)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch, light = self.datamanager.next_eval_image(step)
        if self.config.disable_light:
            outputs = self.model(camera)
        else:
            outputs = self.model(camera, light)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict


@dataclass
class ViewSelectionPipelineConfig(VanillaPipelineConfig):
    """Configuration for progressive view selection pipeline"""

    _target: Type = field(default_factory=lambda: ViewSelectionPipeline)
    datamanager: ViewSelectionDataManagerConfig = ViewSelectionDataManagerConfig()
    model: ModelConfig = ShadowSplatModelConfig()
    disable_light: bool = False

    # Progressive view selection controls
    add_every_n_steps: int = 1000
    add_num_views: int = 1
    view_selector: Optional[str] = None
    """View selection mode: 'random', 'all', 'optics', 'bayes', or dotted path to custom ViewSelector class.
    If None, defaults to random selection."""

    # Optics view selector configuration
    optics_intrinsics_scale: float = 1.0
    """Scale factor for camera intrinsics during coverage scoring (for efficiency). Values < 1.0 downscale."""
    optics_use_kdtree_filter: bool = False
    """Whether to use KD-tree filtering to reduce candidate pool for optics selection."""
    optics_num_nearest_neighbors: int = 5
    """Number of nearest neighbors to consider when using KD-tree filtering for optics selection."""

    # BayesRays view selector configuration
    bayes_reduce_mode: str = "mean"
    """How to aggregate per-pixel uncertainty for BayesRays: 'mean' or 'sum'."""
    bayes_lod: int = 8
    """Level of detail (log2 of grid resolution) for Hessian computation in BayesRays."""
    bayes_max_hessian_batches: Optional[int] = None
    """Maximum number of training batches to use for Hessian computation. None = use all."""


class ViewSelectionPipeline(VanillaPipeline):
    def __init__(
        self,
        config: ViewSelectionPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )

        # Create and set view selector if specified
        if config.view_selector is not None and hasattr(self.datamanager, "view_selector"):
            # Pass optics-specific config if using optics selector
            if config.view_selector == "optics":
                view_selector = create_view_selector(
                    mode=config.view_selector,
                    num_nearest_neighbors=config.optics_num_nearest_neighbors,
                    intrinsics_scale=config.optics_intrinsics_scale,
                    use_kdtree_filter=config.optics_use_kdtree_filter,
                )
            # Pass BayesRays-specific config if using BayesRays selector
            elif config.view_selector == "bayes" or config.view_selector == "bayesrays":
                view_selector = create_view_selector(
                    mode=config.view_selector,
                    bayes_reduce_mode=config.bayes_reduce_mode,
                    bayes_lod=config.bayes_lod,
                )
            else:
                view_selector = create_view_selector(config.view_selector)
            self.datamanager.view_selector = view_selector
            self._view_selector = view_selector
        else:
            self._view_selector = None

        # Initialize Hessian computer if using BayesRays selector
        self._hessian_computer = None
        if config.view_selector == "bayes" or config.view_selector == "bayesrays":
            try:
                from shadow_splat.bayesrays_utils import HessianComputer

                self._hessian_computer = HessianComputer(lod=config.bayes_lod, device=self.device)
            except ImportError:
                raise ImportError(
                    "BayesRays selector requires bayesrays_utils module. "
                    "Ensure bayesrays and related dependencies are installed."
                )

        self._cached_hessian = None
        self._hessian_computed_at_step = -1

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """Get training loss dict and conditionally expand the active view set."""
        # Expand active set on schedule
        if (
            getattr(self.config, "add_every_n_steps", 0) > 0
            and step > 0
            and step % self.config.add_every_n_steps == 0
            and hasattr(self.datamanager, "expand_active_set")
        ):
            # Compute Hessian if using BayesRays selector
            hessian = None
            if self._hessian_computer is not None:
                try:
                    # Only recompute Hessian if we haven't already at this step
                    if self._hessian_computed_at_step != step:
                        hessian = self._hessian_computer.compute_hessian_from_datamanager(
                            model=self._model,
                            datamanager=self.datamanager,
                            max_batches=self.config.bayes_max_hessian_batches,
                        )
                        self._cached_hessian = hessian
                        self._hessian_computed_at_step = step
                    else:
                        hessian = self._cached_hessian
                except Exception as e:
                    print(f"Warning: Failed to compute Hessian for view selection: {e}")
                    hessian = None

            try:
                self.datamanager.expand_active_set(
                    k=self.config.add_num_views, step=step, model=self._model, pipeline=self, hessian=hessian
                )
            except Exception as e:
                print(f"Warning: Failed to expand active set: {e}")

        # Handle both 3-tuple (cameras, batch, light) and 2-tuple (ray_bundle, batch)
        result = self.datamanager.next_train(step)
        if len(result) == 3:
            cameras, batch, light = result
            if self.config.disable_light:
                model_outputs = self._model(cameras)
            else:
                model_outputs = self._model(cameras, light)
        else:
            ray_bundle, batch = result
            # Ray-batched datamanager returns RayBundle directly
            model_outputs = self._model(ray_bundle)

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        self.eval()
        result = self.datamanager.next_eval(step)
        if len(result) == 3:
            ray_bundle, batch, light = result
        else:
            ray_bundle, batch = result
            light = None

        if self.config.disable_light:
            model_outputs = self.model(ray_bundle)
        else:
            model_outputs = self.model(ray_bundle, light)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        result = self.datamanager.next_eval_image(step)
        if len(result) == 3:
            camera, batch, light = result
            # Convert Cameras to RayBundle for the model
            ray_bundle = camera.generate_rays(
                camera_indices=torch.arange(
                    camera.size, device=self.device, dtype=torch.long
                )
            )
            if self.config.disable_light:
                outputs = self.model(ray_bundle)
            else:
                outputs = self.model(ray_bundle, light)
        else:
            ray_bundle, batch = result
            # Ray-batched datamanager case (should rarely reach here for eval_image)
            outputs = self.model(ray_bundle)

        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        # Use original camera for num_rays calculation
        if len(result) == 3:
            camera = result[0]
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict
