"""Custom pipeline for Coverage Splatting"""

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

from nbv_gym.datamanager import ViewSelectionDataManagerConfig
from nbv_gym.model import NBVSplatModelConfig
from nbv_gym.view_selector import create_view_selector

@dataclass
class ViewSelectionPipelineConfig(VanillaPipelineConfig):
    """Configuration for progressive view selection pipeline"""

    _target: Type = field(default_factory=lambda: ViewSelectionPipeline)
    datamanager: ViewSelectionDataManagerConfig = ViewSelectionDataManagerConfig()
    model: ModelConfig = NBVSplatModelConfig()

    # Progressive view selection controls
    add_every_n_steps: int = 1000
    add_num_views: int = 1
    view_selector: Literal["random", "all", "basic"] = "all"
    """View selection mode: 'random', 'all', 'basic'."""

    # Initial view selection
    start_num_views: int = 10
    """Number of initial views to randomly select at the start of training."""
    initial_view_seed: Optional[int] = None
    """Random seed for initial view selection. If None, uses a random seed. Set to a value to get reproducible initial views per scene."""

    # View selector configuration
    view_metric: Literal["coverage", "fig", "view_fig", "fig_diag", "view_fig_diag", "fig_color_field", "fisher_rf"] = "coverage"
    """view selection metric to use."""
    view_selection_intrinsics_scale: float = 1.0
    """Scale factor for camera intrinsics during view metric scoring (for efficiency). Values < 1.0 downscale."""
    view_selection_use_kdtree_filter: bool = False
    """Whether to use KD-tree filtering to reduce candidate pool for view selection."""
    view_selection_num_nearest_neighbors: int = 5
    """Number of nearest neighbors to consider when using KD-tree filtering for view selection."""

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
        if hasattr(self.datamanager, "view_selector"):
            # Pass KD-tree and view metric config based on selector mode
            if config.view_selector == "basic":
                view_selector = create_view_selector(
                    mode=config.view_selector,
                    view_metric=config.view_metric,
                    num_nearest_neighbors=config.view_selection_num_nearest_neighbors,
                    intrinsics_scale=config.view_selection_intrinsics_scale,
                    use_kdtree_filter=config.view_selection_use_kdtree_filter,
                )
                self._model.setup_view_metric(config.view_metric)
                self.model.setup_view_metric(config.view_metric)
            else:
                # Pass KD-tree parameters for other selectors (random, all, etc.)
                view_selector = create_view_selector(
                    mode=config.view_selector,
                    use_kdtree_filter=config.view_selection_use_kdtree_filter,
                    num_nearest_neighbors=config.view_selection_num_nearest_neighbors,
                )
                self._model.setup_view_metric(None)
                self.model.setup_view_metric(None)

            self.datamanager.view_selector = view_selector
            self._view_selector = view_selector
            if config.view_selector == "all":
                self.datamanager.active_train_indices = self.datamanager.all_train_indices.copy()
                self.datamanager.active_unseen_cameras = self.datamanager.all_train_indices.copy()
        else:
            self._view_selector = None
            raise ValueError(f"View selector not found! Please check the view_selector config.")

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

            self._model.training = False

            self.datamanager.expand_active_set(
                k=self.config.add_num_views, step=step, model=self._model, pipeline=self
            )

            self._model.training = True

        cameras, batch = self.datamanager.next_train(step)
        model_outputs = self._model(cameras)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict
