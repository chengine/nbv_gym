import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig

from shadow_splat.model import ShadowSplatModelConfig


@dataclass
class ShadowSplatPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ShadowSplatPipeline)
    """target class to instantiate"""
    datamanager: FullImageDatamanagerConfig = FullImageDatamanagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ShadowSplatModelConfig()
    """specifies the model config"""


class ShadowSplatPipeline(VanillaPipeline):
    def __init__(
        self,
        config: ShadowSplatPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config=config, 
                         device=device, 
                         test_mode=test_mode, 
                         world_size=world_size, 
                         local_rank=local_rank, 
                         grad_scaler=grad_scaler)