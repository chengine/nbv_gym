"""Custom pipeline for Shadow Splat"""

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler

from shadow_splat.datamanager import ShadowSplatDataManagerConfig
from shadow_splat.model import ShadowSplatModelConfig


@dataclass
class ShadowSplatPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ShadowSplatPipeline)
    """target class to instantiate"""
    datamanager: ShadowSplatDataManagerConfig = ShadowSplatDataManagerConfig()
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
        
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        camera, data, light = self.datamanager.next_train(step)
        
        self._model.update_light_source(light)

        model_outputs = self._model(camera)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, data)
        loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

        return model_outputs, loss_dict, metrics_dict