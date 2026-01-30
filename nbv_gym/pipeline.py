"""Custom pipeline for Coverage Splatting"""

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type, Optional

from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler

from nbv_gym.datamanager import ViewSelectionDataManagerConfig
from nbv_gym.model import NBVSplatModelConfig, FisherSplatModelConfig
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
    view_selector: Literal["random", "all", "basic", "gradient_descent"] = "all"
    """View selection mode: 'random', 'all', 'basic', 'gradient_descent'."""

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

    # Gradient descent view selector configuration
    external_3dgs_config_path: Optional[Path] = None
    """Path to external 'all views' 3DGS config.yml for RGB rendering (required for gradient_descent selector)."""
    gd_num_gradient_steps: int = 50
    """Number of gradient descent steps for pose optimization."""
    gd_learning_rate_position: float = 0.001
    """Learning rate for camera position optimization."""
    gd_learning_rate_rotation: float = 0.001
    """Learning rate for camera rotation optimization."""
    gd_include_neighbor_frame: bool = True
    """Whether to include the original dataset neighbor frame in addition to the optimized frame."""

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

        # Conditionally select model config based on view_metric
        # For fisher_rf, use FisherSplatModel which has working Fisher-RF uncertainty computation
        if config.view_metric == "fisher_rf" and config.view_selector == "basic":
            config.model = FisherSplatModelConfig()

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
                # Only call setup_view_metric for NBVSplatModel (not FisherSplatModel)
                # FisherSplatModel handles fisher_rf internally and doesn't need view_metric setup
                if hasattr(self._model, "setup_view_metric"):
                    self._model.setup_view_metric(config.view_metric)
                if hasattr(self.model, "setup_view_metric"):
                    self.model.setup_view_metric(config.view_metric)
            elif config.view_selector == "gradient_descent":
                # Gradient descent view selector with external 3DGS rendering
                view_selector = create_view_selector(
                    mode=config.view_selector,
                    view_metric=config.view_metric,
                    num_nearest_neighbors=config.view_selection_num_nearest_neighbors,
                    intrinsics_scale=config.view_selection_intrinsics_scale,
                    use_kdtree_filter=config.view_selection_use_kdtree_filter,
                    external_3dgs_config_path=config.external_3dgs_config_path,
                    gd_num_gradient_steps=config.gd_num_gradient_steps,
                    gd_learning_rate_position=config.gd_learning_rate_position,
                    gd_learning_rate_rotation=config.gd_learning_rate_rotation,
                    gd_include_neighbor_frame=config.gd_include_neighbor_frame,
                )
                # Setup view metric for the model
                if hasattr(self._model, "setup_view_metric"):
                    self._model.setup_view_metric(config.view_metric)
                if hasattr(self.model, "setup_view_metric"):
                    self.model.setup_view_metric(config.view_metric)
            else:
                # Pass KD-tree parameters for other selectors (random, all, etc.)
                view_selector = create_view_selector(
                    mode=config.view_selector,
                    use_kdtree_filter=config.view_selection_use_kdtree_filter,
                    num_nearest_neighbors=config.view_selection_num_nearest_neighbors,
                )
                # Setup view metric for the model (use configured view_metric, not None)
                if hasattr(self._model, "setup_view_metric"):
                    self._model.setup_view_metric(config.view_metric)
                if hasattr(self.model, "setup_view_metric"):
                    self.model.setup_view_metric(config.view_metric)

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
