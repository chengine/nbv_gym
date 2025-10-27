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

from shadow_splat.model import ShadowSplatModelConfig
from shadow_splat.dataparser import ShadowSplatDataParserConfig
from shadow_splat.datamanager import ShadowSplatDataManagerConfig
from shadow_splat.pipeline import ShadowSplatPipelineConfig
from shadow_splat.model_2dgs import ShadowSplat2DGSModelConfig

# NOTE: The ShadowSplatTrainer import is needed for some reason to make the viewer work
from shadow_splat.trainer import ShadowSplatTrainerConfig, ShadowSplatTrainer

shadow_splat = MethodSpecification(
    ShadowSplatTrainerConfig(
        method_name="shadow-splat",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=700,
        steps_per_eval_all_images=1000,
        max_num_iterations=10000,
        mixed_precision=False,
        pipeline=ShadowSplatPipelineConfig(
            datamanager=ShadowSplatDataManagerConfig(
                dataparser=ShadowSplatDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=ShadowSplatModelConfig(
                sh_degree=3,
                # random_scale=0.5,
                # reset_alpha_every=1000000,
                # cull_alpha_thresh=0.001,
                # warmup_length=100000,
                # refine_every=100000,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                # "optimizer": AdamOptimizerConfig(lr=1e-5),
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
                # "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                # "optimizer": AdamOptimizerConfig(lr=1e-8),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                # "optimizer": AdamOptimizerConfig(lr=1e-5),
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
            # "background_ambient": {
            #     "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            #     "scheduler": None,
            # },
            # "variance_factor": {
            #     "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            #     "scheduler": None,
            # },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Config for ShadowSplat",
)

shadow_splat_2dgs = MethodSpecification(
    ShadowSplatTrainerConfig(
        method_name="shadow-splat-2dgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=700,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=ShadowSplatPipelineConfig(
            datamanager=ShadowSplatDataManagerConfig(
                dataparser=ShadowSplatDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=ShadowSplat2DGSModelConfig(
                # sh_degree=0,
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
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "intensity": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "cutoff": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "variance_factor": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Config for ShadowSplat2DGS",
)
