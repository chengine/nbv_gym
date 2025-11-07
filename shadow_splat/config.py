"""
Shadow Splat configuration file.
"""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from shadow_splat.model import ShadowSplatModelConfig, FisherSplatModelConfig
from shadow_splat.dataparser import ShadowSplatDataParserConfig
from shadow_splat.datamanager import ShadowSplatDataManagerConfig, ViewSelectionDataManagerConfig
from shadow_splat.pipeline import ShadowSplatPipelineConfig, ViewSelectionPipelineConfig, FisherSplatPipelineConfig

# NOTE: The ShadowSplatTrainer import is needed for some reason to make the viewer work
from shadow_splat.trainer import ShadowSplatTrainerConfig, ShadowSplatTrainer

shadow_splat = MethodSpecification(
    ShadowSplatTrainerConfig(
        method_name="shadow-splat",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=700,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=ViewSelectionPipelineConfig(
            # pipeline=ShadowSplatPipelineConfig(
            add_every_n_steps=1000,
            add_num_views=5,
            view_selector="random",
            datamanager=ViewSelectionDataManagerConfig(
                # datamanager=ShadowSplatDataManagerConfig(
                dataparser=ShadowSplatDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=ShadowSplatModelConfig(
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
            "coverage_counts": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "fig": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "view_fig": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "light_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "intensity": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "ambient": {"optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Config for ShadowSplat",
)

fisher_splat = MethodSpecification(
    ShadowSplatTrainerConfig(
        method_name="fisher-splat",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=700,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=FisherSplatPipelineConfig(
            # # pipeline=ShadowSplatPipelineConfig(
            # add_every_n_steps=1000,
            # add_num_views=5,
            # view_selector="random",
            # datamanager=ViewSelectionDataManagerConfig(
            #     # datamanager=ShadowSplatDataManagerConfig(
            #     dataparser=ShadowSplatDataParserConfig(load_3D_points=True),
            #     cache_images_type="uint8",
            # ),
            # model=ShadowSplatModelConfig(
            #     sh_degree=3,
            # ),
                    # pipeline=ShadowSplatPipelineConfig(
            datamanager=ShadowSplatDataManagerConfig(
                dataparser=ShadowSplatDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=FisherSplatModelConfig(
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
            "light_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "intensity": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "ambient": {"optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Config for FisherSplat",
)