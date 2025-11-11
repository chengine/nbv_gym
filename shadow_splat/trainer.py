"""
Custom trainer for Shadow Splat which uses the custom viewer.
"""

import dataclasses
import functools
import torch
from dataclasses import dataclass, field
from typing import Literal, Type

from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
from nerfstudio.utils import writer, profiler
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.utils.decorators import check_eval_enabled

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

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(
                    step=step
                )
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images, run_at_zero=True):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(
                name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step
            )


@dataclass
class ShadowSplatTrainerConfig(TrainerConfig):
    """Configuration for ShadowSplat training regimen"""

    _target: Type = field(default_factory=lambda: ShadowSplatTrainer)
    """target class to instantiate"""
