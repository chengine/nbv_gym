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
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        cameras, batch, light = self.datamanager.next_train(step)
        # import torch

        # with torch.no_grad():
        #     if step >= 500:
        #         self._model.update_light_source(light)  # added
        # if step >= 500:
        #     model_outputs = self._model(cameras, light)
        # else:
        #     model_outputs = self._model(cameras)
        model_outputs = self._model(cameras, light)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(batch["image"].detach().cpu().numpy())
        # ax[1].imshow(model_outputs["rgb"].detach().cpu().numpy())
        # plt.show()

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict
