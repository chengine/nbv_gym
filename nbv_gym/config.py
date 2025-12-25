"""
NBV-Gym configuration file.
"""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from nbv_gym.model import NBVSplatModelConfig
from nbv_gym.datamanager import ViewSelectionDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nbv_gym.pipeline import (
    ViewSelectionPipelineConfig,
)

# NOTE: The NBVTrainer import is needed for some reason to make the viewer work
from nbv_gym.trainer import NBVTrainerConfig, NBVTrainer

nbv_splat = MethodSpecification(
    NBVTrainerConfig(
        method_name="nbv-splat",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=1000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30001,
        mixed_precision=False,
        pipeline=ViewSelectionPipelineConfig(
            add_every_n_steps=200,
            add_num_views=1,
            view_selector="all",
            datamanager=ViewSelectionDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=NBVSplatModelConfig(
                sh_degree=3,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "view_attributes": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Config for NBV Gaussian Splatting",
)

# fisher_splat = MethodSpecification(
#     NBVTrainerConfig(
#         method_name="fisher-splat",
#         steps_per_eval_image=500,
#         steps_per_eval_batch=500,
#         steps_per_save=1000,
#         steps_per_eval_all_images=1000,
#         max_num_iterations=30001,
#         mixed_precision=False,
#         pipeline=ViewSelectionPipelineConfig(
#             add_every_n_steps=200,
#             add_num_views=1,
#             view_selector="all",
#             datamanager=ViewSelectionDataManagerConfig(
#                 dataparser=NerfstudioDataParserConfig(load_3D_points=True),
#                 cache_images_type="uint8",
#             ),
#             model=FisherSplatModelConfig(
#                 sh_degree=3,
#             ),
#         ),
#         optimizers={
#             "means": {
#                 "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(
#                     lr_final=1.6e-6,
#                     max_steps=30000,
#                 ),
#             },
#             "features_dc": {
#                 "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
#                 "scheduler": None,
#             },
#             "features_rest": {
#                 "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
#                 "scheduler": None,
#             },
#             "opacities": {
#                 "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
#                 "scheduler": None,
#             },
#             "scales": {
#                 "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
#                 "scheduler": None,
#             },
#             "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
#             "camera_opt": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(
#                     lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
#                 ),
#             },
#             "bilateral_grid": {
#                 "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(
#                     lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
#                 ),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Config for Fisher-RF Splatting",
# )
