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

import random
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union
import torch
from copy import deepcopy

from nerfstudio.cameras.cameras import Cameras
from rich.progress import Console

CONSOLE = Console(width=120)
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from typing import (
    Dict,
    Literal,
    Tuple,
    Type,
    Union,
)


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
        # print(self.train_dataparser_outputs.lights)
        print("datamanager init | num cameras: ", len(self.train_dataset.cameras))

    def next_train(self, step: int) -> Tuple[Cameras, Dict, Cameras]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(
            random.randint(0, len(self.train_unseen_cameras) - 1)
        )
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        # TODO: Add functionality for RGBA images
        data = deepcopy(self.cached_train[image_idx])
        data["image"] = data["image"].to(self.device)[..., :3]

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        cameras = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if cameras.metadata is None:
            cameras.metadata = {}

        cameras.metadata["cam_idx"] = image_idx

        # NOTE: Added
        light = self.train_dataparser_outputs.lights[image_idx : image_idx + 1].to(self.device)

        return cameras, data, light

    # def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
    #     """Returns the next evaluation batch
    #     Returns a Camera instead of raybundle"""
    #     self.eval_count += 1
    #     if self.config.cache_images == "disk":
    #         camera, data = next(self.iter_eval_image_dataloader)[0]
    #         camera = camera.to(self.device)
    #         data = get_dict_to_torch(data, self.device)
    #         return camera, data

    #     return self.next_eval_image(step=step)

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

        light = (
            self.dataparser.get_dataparser_outputs(split=self.test_split)
            .lights[image_idx : image_idx + 1]
            .to(self.device)
        )

        return camera, data, light
