"""Custom dataparser for Shadow Splat"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig

from nerfstudio.utils.io import load_from_json


@dataclass
class ShadowSplatDataParserConfig(NerfstudioDataParserConfig):
    """Dataset config"""

    _target: Type = field(default_factory=lambda: ShadowSplatDataParser)
    # TODO: add more fields here
    pass


@dataclass
class ShadowSplatDataparserOutputs(DataparserOutputs):
    # TODO: add lighting info fields
    lights: Cameras = None
    """Camera object storing light source information for each frame"""

    @classmethod
    def from_parent(cls, parent: DataparserOutputs):
        # Extract field names from the parent class
        parent_field_names = {f.name for f in fields(DataparserOutputs)}

        # Filter the parent instance's __dict__ to only include parent fields
        parent_data = {name: getattr(parent, name) for name in parent_field_names}

        # Initialize the child with parent fields; new fields will remain uninitialized
        return cls(**parent_data)


class ShadowSplatDataParser(Nerfstudio):
    """Shadow Splat DatasetParser"""

    config: ShadowSplatDataParserConfig

    def _generate_dataparser_outputs(self, split="train") -> ShadowSplatDataparserOutputs:
        # Call parent method
        dataparser_outputs = ShadowSplatDataparserOutputs.from_parent(
            super()._generate_dataparser_outputs(split)
        )

        meta = load_from_json(self.config.data / "transforms.json")
        data_dir = self.config.data

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        print("dataparser | num frames", len(frames))

        # Load 3D points
        # print("dataparser | config.load_3D_points", self.config.load_3D_points)
        # if self.config.load_3D_points:  # NOTE: for some reason this is False even though we set it to True in the method config
        print("dataparser | loading 3D points")
        if "ply_file_path" in meta:
            ply_file_path = data_dir / meta["ply_file_path"]
            if ply_file_path:
                sparse_points = self._load_3D_points(
                    ply_file_path,
                    dataparser_outputs.dataparser_transform,
                    dataparser_outputs.dataparser_scale,
                )
                if sparse_points is not None:
                    dataparser_outputs.metadata.update(sparse_points)

        light_poses = []
        for frame in frames:
            light_poses.append(np.array(frame["light_pose"]))

        light_poses = torch.from_numpy(np.array(light_poses).astype(np.float32))

        lights = Cameras(
            fx=meta["light_intrinsics"]["fl_x"],
            fy=meta["light_intrinsics"]["fl_y"],
            cx=meta["light_intrinsics"]["cx"],
            cy=meta["light_intrinsics"]["cy"],
            height=meta["light_intrinsics"]["h"],
            width=meta["light_intrinsics"]["w"],
            camera_to_worlds=light_poses[:, :3, :4],
            camera_type=CameraType.ORTHOPHOTO,
        )
        dataparser_outputs.lights = lights

        return dataparser_outputs
