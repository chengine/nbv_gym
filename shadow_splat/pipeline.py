"""Custom pipeline for Shadow Splat"""

import torch
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler

from shadow_splat.datamanager import ShadowSplatDataManagerConfig
from shadow_splat.datamanager import ViewSelectionDataManagerConfig
from shadow_splat.model import ShadowSplatModelConfig, FisherSplatModelConfig
from shadow_splat.view_selector import create_view_selector
import time


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
class FisherSplatPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: FisherSplatPipeline)
    """target class to instantiate"""
    datamanager: ShadowSplatDataManagerConfig = ShadowSplatDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = FisherSplatModelConfig()
    """specifies the model config"""
    disable_light: bool = False
    """specifies whether to train without light"""


class FisherSplatPipeline(ShadowSplatPipeline):
    def __init__(
        self,
        config: FisherSplatPipelineConfig,
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
    """View selection mode: 'random', 'all', 'optics', or dotted path to custom ViewSelector class.
    If None, defaults to random selection."""

    # Initial view selection
    start_num_views: int = 10
    """Number of initial views to randomly select at the start of training."""
    initial_view_seed: Optional[int] = None
    """Random seed for initial view selection. If None, uses a random seed. Set to a value to get reproducible initial views per scene."""

    # Optics view selector configuration
    optics_coverage_metric: str = "coverage"
    """Coverage metric to use for optics selection."""
    optics_intrinsics_scale: float = 1.0
    """Scale factor for camera intrinsics during coverage scoring (for efficiency). Values < 1.0 downscale."""
    optics_use_kdtree_filter: bool = False
    """Whether to use KD-tree filtering to reduce candidate pool for optics selection."""
    optics_num_nearest_neighbors: int = 5
    """Number of nearest neighbors to consider when using KD-tree filtering for optics selection."""


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
        # Copy initial view selection parameters from pipeline config to datamanager config
        config.datamanager.start_num_views = config.start_num_views
        config.datamanager.initial_view_seed = config.initial_view_seed

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
                    coverage_metric=config.optics_coverage_metric,
                    num_nearest_neighbors=config.optics_num_nearest_neighbors,
                    intrinsics_scale=config.optics_intrinsics_scale,
                    use_kdtree_filter=config.optics_use_kdtree_filter,
                )
            else:
                view_selector = create_view_selector(config.view_selector)
            self.datamanager.view_selector = view_selector
            self._view_selector = view_selector
            if config.view_selector == "all":
                self.datamanager.active_train_indices = self.datamanager.all_train_indices.copy()
                self.datamanager.active_unseen_cameras = self.datamanager.all_train_indices.copy()
        else:
            self._view_selector = None

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
            # # Clear model.info before view selection to free memory from previous renders
            # # This is critical for preventing memory leaks during view selection
            # if hasattr(self._model, "info"):
            #     if isinstance(self._model.info, dict):
            #         for key, value in list(self._model.info.items()):
            #             if isinstance(value, torch.Tensor):
            #                 del value
            #         self._model.info.clear()
            #     self._model.info = {}

            self._model.training = False

            self.datamanager.expand_active_set(
                k=self.config.add_num_views, step=step, model=self._model, pipeline=self
            )

            # finally:
            #     # Aggressive cleanup after view expansion
            #     # Clear model.info again to ensure all tensors from view selection are freed
            #     if hasattr(self._model, "info"):
            #         if isinstance(self._model.info, dict):
            #             for key, value in list(self._model.info.items()):
            #                 if isinstance(value, torch.Tensor):
            #                     del value
            #             self._model.info.clear()
            #         self._model.info = {}
            # Restore training state
            self._model.training = True

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
class FisherViewSelectionPipelineConfig(VanillaPipelineConfig):
    """Configuration for progressive view selection pipeline"""

    _target: Type = field(default_factory=lambda: FisherViewSelectionPipeline)
    datamanager: ViewSelectionDataManagerConfig = ViewSelectionDataManagerConfig()
    model: ModelConfig = FisherSplatModelConfig()
    disable_light: bool = False

    # Progressive view selection controls
    add_every_n_steps: int = 1000
    add_num_views: int = 1
    view_selector: Optional[str] = None
    """View selection mode: 'random', 'all', 'optics', or dotted path to custom ViewSelector class.
    If None, defaults to random selection."""

    # Initial view selection
    start_num_views: int = 10
    """Number of initial views to randomly select at the start of training."""
    initial_view_seed: Optional[int] = None
    """Random seed for initial view selection. If None, uses a random seed. Set to a value to get reproducible initial views per scene."""

    # Optics view selector configuration
    optics_intrinsics_scale: float = 1.0
    """Scale factor for camera intrinsics during coverage scoring (for efficiency). Values < 1.0 downscale."""
    optics_use_kdtree_filter: bool = False
    """Whether to use KD-tree filtering to reduce candidate pool for optics selection."""
    optics_num_nearest_neighbors: int = 5
    """Number of nearest neighbors to consider when using KD-tree filtering for optics selection."""


class FisherViewSelectionPipeline(VanillaPipelineConfig):
    def __init__(
        self,
        config: FisherViewSelectionPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        # Copy initial view selection parameters from pipeline config to datamanager config
        config.datamanager.start_num_views = config.start_num_views
        config.datamanager.initial_view_seed = config.initial_view_seed

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
            else:
                view_selector = create_view_selector(config.view_selector)
            self.datamanager.view_selector = view_selector
            self._view_selector = view_selector
        else:
            self._view_selector = None

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
            self.datamanager.expand_active_set(
                k=self.config.add_num_views, step=step, model=self._model, pipeline=self
            )

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
