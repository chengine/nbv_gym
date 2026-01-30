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
from typing import Dict, List, Literal, Tuple, Type, Union, Optional
import torch
from copy import deepcopy

from nerfstudio.cameras.cameras import Cameras
from rich.console import Console
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils import writer
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

CONSOLE = Console(width=120)

@dataclass
class ViewSelectionDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: ViewSelectionDataManager)
    # NOTE: changed for saving memory
    cache_images: Literal["cpu", "gpu", "disk"] = "cpu"
    cache_images_type: Literal["uint8", "float32"] = "uint8"
    start_num_views: int = 10
    """Number of initial views to randomly select at the start of training."""
    initial_view_seed: Optional[int] = 0
    """Random seed for initial view selection. If None, uses a random seed. Set to a value to get reproducible initial views per scene."""
    bias_views: bool = False
    """Whether to restrict available views to subset of indices."""

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

        self.view_selector = view_selector

        # Initialize synthetic view storage for gradient descent view selector
        self.synthetic_cameras: List[Cameras] = []
        self.synthetic_images: List[torch.Tensor] = []

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
        self.active_unseen_cameras = list(self.active_train_indices)

        self.log_added_views = []

        # NOTE: Used to bias the training viewset. Typically not used.
        if self.config.bias_views:
            self.available_indices = self.all_train_indices[
                : int(0.2 * len(self.all_train_indices))
            ]
            for i, idx in enumerate(
                self.all_train_indices[int(0.2 * len(self.all_train_indices)) :]
            ):
                if i % 10 == 0:
                    self.available_indices.append(idx)
        else:
            self.available_indices = self.all_train_indices

    def add_synthetic_view(self, camera: Cameras, image: torch.Tensor) -> int:
        """Add an optimized camera + rendered image to the training set.

        Args:
            camera: Optimized camera pose
            image: Rendered RGB image from external model [H, W, 3]

        Returns:
            Negative index representing this synthetic view
        """
        self.synthetic_cameras.append(camera)
        self.synthetic_images.append(image)
        # Use negative indices to distinguish from dataset indices
        # -1 = first synthetic, -2 = second synthetic, etc.
        synthetic_idx = -len(self.synthetic_cameras)
        return synthetic_idx

    def expand_active_set(self, k: int = 1, step: Optional[int] = None, **kwargs) -> None:
        """Expand the active set by adding up to k remaining indices using the view selector."""
        if not self.all_train_indices:
            return
        # Filter out negative indices (synthetic views) when computing remaining
        real_active_indices = [idx for idx in self.active_train_indices if idx >= 0]
        remaining = list(set(self.available_indices) - set(real_active_indices))
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

        # Handle sentinel value [-1] from gradient descent selector
        # This indicates synthetic view was already added via add_synthetic_view
        if add == [-1]:
            # Synthetic view already added, just log it
            add = []  # Don't add anything else to active_train_indices

        self.active_train_indices.extend(add)
        # Make newly added indices available for immediate sampling
        self.active_unseen_cameras.extend(add)

        self.log_added_views.append(add)

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

        num_synthetic = sum(1 for idx in self.active_train_indices if idx < 0)
        num_real = len(self.active_train_indices) - num_synthetic

        print(
            f"Expanded active set to {len(self.active_train_indices)} views "
            f"({num_real} real + {num_synthetic} synthetic) (added {len(add)} dataset views)"
        )

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Return the next training batch restricted to the active view subset.

        Returns a `Cameras` object instead of a ray bundle to match the existing
        Coverage Splatting pipeline/model interface.

        Handles both positive indices (dataset views) and negative indices (synthetic views).
        """
        # print(f"Active train cameras: {len(self.active_train_indices)}")
        if not self.active_unseen_cameras:
            # Refill from the active set when we have seen all active cameras
            self.active_unseen_cameras = list(self.active_train_indices)

        image_idx = self.active_unseen_cameras.pop(
            random.randint(0, len(self.active_unseen_cameras) - 1)
        )

        # Handle synthetic views (negative indices)
        if image_idx < 0:
            # Convert negative index: -1 -> 0, -2 -> 1, etc.
            synthetic_idx = -image_idx - 1
            if synthetic_idx >= len(self.synthetic_cameras):
                raise IndexError(f"Synthetic view index {synthetic_idx} out of range. "
                               f"Only {len(self.synthetic_cameras)} synthetic views available.")

            cameras = self.synthetic_cameras[synthetic_idx].to(self.device)
            if cameras.metadata is None:
                cameras.metadata = {}
            cameras.metadata["cam_idx"] = image_idx  # Keep negative index for tracking

            data = {"image": self.synthetic_images[synthetic_idx].to(self.device)}
            return cameras, data

        # Handle regular dataset views (positive indices)
        # data = deepcopy(self.cached_train[image_idx])
        # Avoid deepcopy - just reference and move to device
        data = self.cached_train[image_idx]
        # Create a shallow copy of the dict, but don't copy the image tensor yet
        data = {k: v for k, v in data.items()}
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        cameras = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if cameras.metadata is None:
            cameras.metadata = {}
        cameras.metadata["cam_idx"] = image_idx

        return cameras, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch.
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
        # data = data.copy()
        data = {k: v for k, v in data.items()}
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)

        return camera, data
