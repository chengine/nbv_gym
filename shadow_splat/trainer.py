"""
Custom trainer for Shadow Splat which uses the custom viewer.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Type

from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
from nerfstudio.utils import writer, profiler
from nerfstudio.utils.rich_utils import CONSOLE

from shadow_splat.viewer import ShadowSplatViewer


class ShadowSplatTrainer(Trainer):
    """Custom trainer that uses the ShadowSplat custom viewer."""

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions."""
        # Call the parent setup but skip the viewer creation part
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled - use our custom viewer
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            # Use our custom viewer instead of the standard one
            self.viewer_state = ShadowSplatViewer(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
                trainer=self,
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging,
            max_iter=self.config.max_num_iterations,
            banner_messages=banner_messages,
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

    def _after_train(self) -> None:
        super()._after_train()
        print(self.pipeline.datamanager.log_added_views)
        # Save view selection log to outputs folder
        if (
            hasattr(self.pipeline.datamanager, "log_added_views")
            and self.pipeline.datamanager.log_added_views
        ):
            import json

            view_log_path = self.config.get_base_dir() / "view_selection_log.json"
            with open(view_log_path, "w") as f:
                json.dump(self.pipeline.datamanager.log_added_views, f, indent=2)

            CONSOLE.log(f"View selection log saved to: {view_log_path}")


@dataclass
class ShadowSplatTrainerConfig(TrainerConfig):
    """Configuration for ShadowSplat training regimen"""

    _target: Type = field(default_factory=lambda: ShadowSplatTrainer)
    """target class to instantiate"""
