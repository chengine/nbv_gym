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
from nerfstudio.models.nerfacto import NerfactoModelConfig

from shadow_splat.model import ShadowSplatModelConfig,  NerfactoModelWithUncertaintyConfig
from shadow_splat.dataparser import ShadowSplatDataParserConfig
from shadow_splat.datamanager import ShadowSplatDataManagerConfig, ViewSelectionDataManagerConfig
from shadow_splat.bayes_rays_datamanager import BayesRaysParallelDataManagerConfig
from shadow_splat.pipeline import ShadowSplatPipelineConfig, ViewSelectionPipelineConfig
# from shadow_splat.model_2dgs import ShadowSplat2DGSModelConfig

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
            add_every_n_steps=50,
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

# shadow_splat_2dgs = MethodSpecification(
#     ShadowSplatTrainerConfig(
#         method_name="shadow-splat-2dgs",
#         steps_per_eval_image=100,
#         steps_per_eval_batch=0,
#         steps_per_save=700,
#         steps_per_eval_all_images=1000,
#         max_num_iterations=30000,
#         mixed_precision=False,
#         pipeline=ShadowSplatPipelineConfig(
#             datamanager=ShadowSplatDataManagerConfig(
#                 dataparser=ShadowSplatDataParserConfig(load_3D_points=True),
#                 cache_images_type="uint8",
#             ),
#             model=ShadowSplat2DGSModelConfig(
#                 # sh_degree=0,
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
#             "intensity": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
#             "cutoff": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
#             "variance_factor": {
#                 "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
#                 "scheduler": None,
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Config for ShadowSplat2DGS",
# )

bayes_rays = MethodSpecification(
    TrainerConfig(
        method_name="bayes-rays",
        steps_per_eval_image=0,  # Disabled to avoid OOM on full image evals with ray-batched training
        steps_per_eval_batch=500,  # Use batch eval instead (faster and lower memory)
        steps_per_save=1000,
        steps_per_eval_all_images=1000,  # Disabled for ray-batched training
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=ViewSelectionPipelineConfig(
            # Use ray-batched datamanager with view selection (nerfacto)
            datamanager=BayesRaysParallelDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,  # Standard nerfacto setting
                eval_num_rays_per_batch=32768,
                start_num_views=10,  # Start with 10 views (warm start), expand via BayesRays
            ),
            model=NerfactoModelWithUncertaintyConfig(
                eval_num_rays_per_chunk=32768,
                average_init_density=0.01,
                ),  # Use standard nerfacto (no wrapper needed)
            add_every_n_steps=200,
            add_num_views=1,
            view_selector="random",
            bayes_reduce_mode="mean",
            bayes_lod=8,
            bayes_max_hessian_batches=None,
            disable_light=True,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=0
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=0
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfacto with BayesRays progressive view selection (ray-batched)",
)