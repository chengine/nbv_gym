"""Custom dataparser for Shadow Splat"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.io import load_from_json


@dataclass
class ShadowSplatDataParserConfig(NerfstudioDataParserConfig):
    """Dataset config"""

    _target: Type = field(default_factory=lambda: ShadowSplatDataParser)
    load_3D_points: bool = True
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
        # Call parent method first - this already filters frames by split
        dataparser_outputs = ShadowSplatDataparserOutputs.from_parent(
            super()._generate_dataparser_outputs(split)
        )

        # Get the already-filtered camera filenames for this split
        camera_filenames = dataparser_outputs.image_filenames
        print("dataparser | num cameras", len(camera_filenames))

        # Load metadata only if we need light info
        # Check if any frame has light_pose by sampling first frame
        meta = load_from_json(self.config.data / "transforms.json")
        data_dir = self.config.data

        # Build a mapping from filename to frame data ONLY for frames in this split
        # This avoids processing all frames in the JSON
        filename_to_frame = {}
        camera_filenames_set = set(camera_filenames)

        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            if fname in camera_filenames_set:
                filename_to_frame[fname] = frame

        # Check if we need to process light info
        has_light_info = False
        if filename_to_frame:
            # Check first frame that's actually in our split
            first_frame = next(iter(filename_to_frame.values()))
            has_light_info = "light_pose" in first_frame

        # Process light info only if needed and only for frames in this split
        if has_light_info:
            CONSOLE.log("LOADING LIGHTS")
            light_poses = []
            for camera_filename in camera_filenames:
                if camera_filename in filename_to_frame:
                    frame = filename_to_frame[camera_filename]
                    light_poses.append(np.array(frame["light_pose"]))
                else:
                    # Fallback: create zero pose if frame not found
                    CONSOLE.print(
                        f"[yellow]Warning: Frame {camera_filename} not found in metadata, using zero pose"
                    )
                    light_poses.append(np.zeros((4, 4), dtype=np.float32))

            light_poses = torch.from_numpy(np.array(light_poses).astype(np.float32))

            # Transform light poses with dataparser transform and scale
            colmap_path = self.config.data / "colmap/sparse/0"
            if not colmap_path.exists():
                transform_4x4 = torch.eye(4)
                transform_4x4[:3, :4] = dataparser_outputs.dataparser_transform
                light_poses = light_poses @ transform_4x4
                light_poses[:, :3, 3] *= dataparser_outputs.dataparser_scale

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
