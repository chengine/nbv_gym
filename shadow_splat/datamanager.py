# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Datamanager.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Optional
import torch
from copy import deepcopy

from nerfstudio.cameras.cameras import Cameras
from rich.console import Console
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils import writer

from shadow_splat.view_selector import AllViewSelector

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

CONSOLE = Console(width=120)


@dataclass
class ShadowSplatDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: ShadowSplatDataManager)


class ShadowSplatDataManager(FullImageDatamanager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ShadowSplatDataManagerConfig

    def __init__(
        self,
        config: ShadowSplatDataManagerConfig,
        device: Union[torch.device, str] = "cuda:0",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )
        self.current_light = None

    def next_train(self, step: int) -> Tuple[Cameras, Dict, Cameras]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(
            random.randint(0, len(self.train_unseen_cameras) - 1)
        )
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        # NOTE: changed for RGBA images
        data = deepcopy(self.cached_train[image_idx])
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        cameras = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if cameras.metadata is None:
            cameras.metadata = {}
        cameras.metadata["cam_idx"] = image_idx

        # NOTE: Added
        # if self.train_dataparser_outputs.lights is not None:
        if (
            hasattr(self.train_dataparser_outputs, "lights")
            and self.train_dataparser_outputs.lights is not None
        ):
            light = self.train_dataparser_outputs.lights[image_idx : image_idx + 1].to(self.device)
            self.current_light = light
        else:
            light = None

        return cameras, data, light

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch
        Returns a Camera instead of raybundle
        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        if self.config.cache_images == "disk":
            camera, data = next(self.iter_eval_image_dataloader)[0]
            return camera, data
        image_idx = self.eval_unseen_cameras.pop(
            random.randint(0, len(self.eval_unseen_cameras) - 1)
        )
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = self.cached_eval[image_idx]
        data = data.copy()
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)

        if (
            hasattr(self.train_dataparser_outputs, "lights")
            and self.train_dataparser_outputs.lights is not None
        ):
            light = self.train_dataparser_outputs.lights[image_idx : image_idx + 1].to(self.device)
            self.current_light = light
        else:
            light = None

        return camera, data, light


@dataclass
class ViewSelectionDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: ViewSelectionDataManager)
    start_num_views: int = 10
    """Number of initial views to randomly select at the start of training."""
    initial_view_seed: Optional[int] = None
    """Random seed for initial view selection. If None, uses a random seed. Set to a value to get reproducible initial views per scene."""


class ViewSelectionDataManager(FullImageDatamanager):  # pylint: disable=abstract-method
    """DataManager that progressively grows an active subset of training views.

    Only samples training images from `active_train_indices`, which starts with a
    small subset and can be expanded over time via `expand_active_set`.
    """

    config: ViewSelectionDataManagerConfig

    def __init__(
        self,
        config: ViewSelectionDataManagerConfig,
        device: Union[torch.device, str] = "cuda:0",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        view_selector=None,  # Optional ViewSelector instance
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

        self.current_light = None
        self.view_selector = view_selector

        # Initialize active set
        num_train_images = len(self.train_dataset)
        self.all_train_indices = list(range(num_train_images))
        start_k = max(1, min(self.config.start_num_views, num_train_images))

        # Use seed for initial view selection if provided (for reproducibility)
        # Always use random selection for initial views, regardless of view_selector
        if self.config.initial_view_seed is not None:
            rng = random.Random(self.config.initial_view_seed)
            self.active_train_indices = (
                rng.sample(self.all_train_indices, k=start_k) if num_train_images > 0 else []
            )
        else:
            self.active_train_indices = (
                random.sample(self.all_train_indices, k=start_k) if num_train_images > 0 else []
            )
        # If the view selector is all, set the active train indices to all the train indices
        if self.view_selector is not None and isinstance(self.view_selector, AllViewSelector):
            self.active_train_indices = self.all_train_indices.copy()
            self.active_unseen_cameras = self.all_train_indices.copy()
        else:
            self.active_unseen_cameras = list(self.active_train_indices)

        # Log initial indices to wandb table
        # self._log_active_indices_to_wandb(
        #     iteration=0,
        #     newly_added_indices=self.active_train_indices.copy(),
        # )

    def _log_active_indices_to_wandb(self, iteration: int, newly_added_indices: list[int]) -> None:
        """Log active indices to wandb as a table row.

        This logs each index update as a new row in a wandb table. All rows are tracked
        and can be downloaded via the wandb API or UI for visualization (e.g., with plotly).

        Args:
            iteration: Current iteration number (0 for initialization)
            newly_added_indices: List of newly added indices (or all indices for initialization)
        """
        if not WANDB_AVAILABLE:
            return

        try:
            # Check if wandb is initialized
            if wandb.run is None:
                return

            # Create table row data
            # Sort indices for consistency and easier visualization
            sorted_newly_added = sorted(newly_added_indices)
            sorted_total_active = sorted(self.active_train_indices)

            table_data = {
                "iteration": iteration,
                "newly_added_indices": sorted_newly_added,  # Keep as list for better serialization
                "newly_added_indices_str": str(
                    sorted_newly_added
                ),  # String version for table display
                "num_newly_added": len(newly_added_indices),
                "total_active_indices": sorted_total_active,  # Keep as list for better serialization
                "total_active_indices_str": str(
                    sorted_total_active
                ),  # String version for table display
                "num_total_active": len(self.active_train_indices),
            }

            # Log as a table using wandb.Table for visualization in wandb UI
            table = wandb.Table(
                columns=[
                    "iteration",
                    "newly_added_indices",
                    "num_newly_added",
                    "total_active_indices",
                    "num_total_active",
                ],
                data=[
                    [
                        table_data["iteration"],
                        table_data["newly_added_indices_str"],
                        table_data["num_newly_added"],
                        table_data["total_active_indices_str"],
                        table_data["num_total_active"],
                    ]
                ],
            )

            # Log the table at the current step
            # Each log creates a new table version, which wandb tracks
            # You can download all table versions via the wandb API or UI
            wandb.log({"View Selection/Active Indices Table": table}, step=iteration)

            # Also log structured data as JSON strings for easier programmatic access
            # This makes it easier to download and process the data programmatically
            wandb.log(
                {
                    "View Selection/Active Indices/iteration": iteration,
                    "View Selection/Active Indices/newly_added_json": json.dumps(
                        sorted_newly_added
                    ),
                    "View Selection/Active Indices/num_newly_added": table_data["num_newly_added"],
                    "View Selection/Active Indices/total_active_json": json.dumps(
                        sorted_total_active
                    ),
                    "View Selection/Active Indices/num_total_active": table_data[
                        "num_total_active"
                    ],
                },
                step=iteration,
            )

        except Exception:
            # Silently fail if wandb logging fails
            pass

    def expand_active_set(self, k: int = 1, step: Optional[int] = None, **kwargs) -> None:
        """Expand the active set by adding up to k remaining indices using the view selector."""
        if not self.all_train_indices:
            return
        remaining = list(set(self.all_train_indices) - set(self.active_train_indices))
        if not remaining:
            return

        # Use view selector if available, otherwise fall back to random
        if self.view_selector is not None:
            add = self.view_selector.select_views(
                active_indices=self.active_train_indices,
                remaining_indices=remaining,
                num_to_select=k,
                step=step,
                datamanager=self,
                **kwargs,
            )
        else:
            # Fallback to random selection
            add = random.sample(remaining, k=min(max(1, k), len(remaining)))

        self.active_train_indices.extend(add)
        # Make newly added indices available for immediate sampling
        self.active_unseen_cameras.extend(add)

        # Log to wandb/tensorboard if step is provided
        if step is not None:
            try:
                writer.put_scalar(name="View Selection/Views Added", scalar=len(add), step=step)
                writer.put_scalar(
                    name="View Selection/Total Active Views",
                    scalar=len(self.active_train_indices),
                    step=step,
                )
            except Exception:
                # Silently fail if writer is not initialized (e.g., wandb not enabled)
                pass

            # Log to wandb table
            # self._log_active_indices_to_wandb(
            #     iteration=step,
            #     newly_added_indices=add,
            # )

        print(
            f"Expanded active set to {len(self.active_train_indices)} views (added {len(add)} views)"
        )

    def next_train(self, step: int) -> Tuple[Cameras, Dict, Cameras]:
        """Return the next training batch restricted to the active view subset.

        Returns a `Cameras` object instead of a ray bundle to match the existing
        ShadowSplat pipeline/model interface.
        """
        if not self.active_unseen_cameras:
            # Refill from the active set when we have seen all active cameras
            self.active_unseen_cameras = list(self.active_train_indices)

        image_idx = self.active_unseen_cameras.pop(
            random.randint(0, len(self.active_unseen_cameras) - 1)
        )

        data = deepcopy(self.cached_train[image_idx])
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        cameras = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if cameras.metadata is None:
            cameras.metadata = {}
        cameras.metadata["cam_idx"] = image_idx

        if (
            hasattr(self.train_dataparser_outputs, "lights")
            and self.train_dataparser_outputs.lights is not None
        ):
            light = self.train_dataparser_outputs.lights[image_idx : image_idx + 1].to(self.device)
            self.current_light = light
        else:
            light = None

        return cameras, data, light

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch.
        Mirrors the ShadowSplatDataManager behavior to return a Camera.
        """
        if self.config.cache_images == "disk":
            camera, data = next(self.iter_eval_image_dataloader)[0]
            return camera, data
        image_idx = self.eval_unseen_cameras.pop(
            random.randint(0, len(self.eval_unseen_cameras) - 1)
        )
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = self.cached_eval[image_idx]
        data = data.copy()
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)

        if (
            hasattr(self.train_dataparser_outputs, "lights")
            and self.train_dataparser_outputs.lights is not None
        ):
            light = self.train_dataparser_outputs.lights[image_idx : image_idx + 1].to(self.device)
            self.current_light = light
        else:
            light = None

        return camera, data, light
