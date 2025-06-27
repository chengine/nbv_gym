from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from typing_extensions import Annotated
import pickle
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib as mpl
from tqdm import tqdm
import open3d as o3d

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import SplatfactoModel
from shadow_splat.model import ShadowSplatModel

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
from nerfstudio.data.datasets.base_dataset import InputDataset

class GaussianSplat():
    def __init__(self, config_path: Path, res_factor=None,
        test_mode: Literal["test", "val", "inference"] = "inference",
        dataset_mode: Literal["train", "val", "test"] = 'test',
        device: Union[torch.device, str] = "cpu"
    ) -> None:
        # config path
        self.config_path = config_path

        # camera rescale resolution factor
        self.res_factor = res_factor

        # device
        self.device = device

        # initialize pipeline
        self.init_pipeline(test_mode)

        # load dataset
        self.load_dataset(dataset_mode)

        # load cameras
        self.get_cameras()

    def init_pipeline(self,
        test_mode: Literal["test", "val", "inference"]
    ):
        # Get config and pipeline
        self.config, self.pipeline, _, _ = eval_setup(
            self.config_path, 
            test_mode=test_mode,
        )

    def load_dataset(self,
        dataset_mode: Literal["train", "val", "test"]
    ):
        # return dataset
        if dataset_mode == "train":
            self.dataset = self.pipeline.datamanager.train_dataset
        elif dataset_mode in ["val", "test"]:
            self.dataset = self.pipeline.datamanager.eval_dataset
        else:
            ValueError('Incorrect value for datset_mode. Accepted values include: dataset_mode: Literal["train", "val", "test"].')

    def get_cameras(self):
        # Camera object contains camera intrinsics and extrinsics
        self.cameras = self.dataset.cameras.to(self.device)

        if self.res_factor is not None:
            self.cameras.rescale_output_resolution(self.res_factor)

        return self.cameras

    def get_light_source(self):
        
        if isinstance(self.pipeline.model, ShadowSplatModel):
            self.light_source = self.pipeline.datamanager.train_dataparser_outputs.lights.to(self.device)
        else:
            self.light_source = None

        return self.light_source

    def get_poses(self):
        return self.cameras.camera_to_worlds
    
    def get_light_source_poses(self):
        if isinstance(self.pipeline.model, ShadowSplatModel):
            return self.light_source.camera_to_worlds
        else:
            return None
    
    def get_images(self):
        # images
        images = [self.dataset.get_image_float32(image_idx)
                  for image_idx 
                  in range(len(self.dataset._dataparser_outputs.image_filenames))]
        
        return images
    
    def get_camera_intrinsics(self):
        K = self.cameras[0].get_intrinsics_matrices().squeeze()
        # width and height
        W = int(self.cameras[0].width.item())
        H = int(self.cameras[0].height.item())
        return H, W, K

    def render(self, camera, 
               light_source: Optional[torch.Tensor] = None,
               ):
        
        # render outputs
        if isinstance(self.pipeline.model, SplatfactoModel):
            with torch.no_grad():
                outputs = self.pipeline.model(camera)
        elif isinstance(self.pipeline.model, ShadowSplatModel):
            with torch.no_grad():
                outputs = self.pipeline.model(camera, light_source)

        return outputs
    
# def load_dataset(data_path: Path,
#     dataset_mode: Literal["train", "val", "test", "all"] # 'all' uses the entire dataset.
# ):
#     # init Nerfstudio dataset config
#     nerfstudio_data_parser_config = NerfstudioDataParserConfig(data=data_path,
#                                                                eval_mode="all" if dataset_mode == "all" else "fraction")

#     # init data parser
#     nerfstudio_data_parser = Nerfstudio(nerfstudio_data_parser_config)

#     # data parser outputs
#     data_parser_ouputs = nerfstudio_data_parser._generate_dataparser_outputs(split=dataset_mode
#                                                                              if dataset_mode != "all" else "val")

#     # load dataset
#     dataset = InputDataset(data_parser_ouputs)
    
#     return dataset

# def load_model(config_path: Path):
#     # rescale factor
#     res_factor = None

#     # device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # initialize NeRF
#     gsplat = GaussianSplat(config_path=config_path,
#                 res_factor=res_factor,
#                 test_mode="test", # [options: "test", "inference", "val"]
#                 dataset_mode="val",
#                 device=device)
    
#     return gsplat