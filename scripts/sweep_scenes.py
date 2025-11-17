import subprocess
import pathlib
import shlex
import os
from datetime import datetime
import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)

# Optional WandB support - set to True to enable logging
USE_WANDB = True

# WandB configuration
WANDB_ENTITY = "chengine-stanford-university"  # Set to your WandB username if needed, otherwise uses default
WANDB_PROJECT_PREFIX = "next-best-view"

# If WandB is enabled, try to initialize it
if USE_WANDB:
    try:
        import wandb
        # Try to login - will use existing credentials or prompt
        wandb.login(anonymous="allow")
    except ImportError:
        print("Warning: WandB not installed. Install with: pip install wandb")
        USE_WANDB = False
    except Exception as e:
        print(f"Warning: Could not initialize WandB: {e}")
        USE_WANDB = False

# -----------------------------
# Batch experiment configuration
# -----------------------------
PROJECT_NAME = "next-best-view"

# Datasets to sweep
BASE_DATA_DIR = pathlib.Path("/home/admin/StanfordMSL/shadow_splat/data")
SCENES = [
"caterpillar",
"train",
"ignatius",
"shiny_statue_6pm",
"space_laces_4pm",
"chair_3pm",
# "master_chief_cycles"
"bicycle",
"counter",
"flowers",
"garden",
"stump",
"treehill",
# "kitchen",
# "bonsai",
# "room",
]
DATASETS = [str(BASE_DATA_DIR / s) for s in SCENES]

# Methods / information gain metrics to compare.
# Valid entries: "coverage", "fig", "view_fig", "fisher_info", "random", "bayes", "bayes-rays"
METHODS = [
    # "coverage",
    # "fig",
    # "view_fig",
    # "fisher_info",
    # "random",
    # "nerf-random",  # NeRF with random view selection
    # "nerfacto",  # Nerfacto
    "bayes-rays"  # BayesRays with Nerfacto ray-batched training
]

# Visualization backends. Example: "viewer+wandb" or just "wandb"
VIS = "wandb"

# Quit the viewer on train completion to avoid hangs in sweeps
QUIT_VIEWER_ON_DONE = True

# Use filename-based eval mode for reproducibility across datasets
EVAL_MODE = "filename"

# -----------------------------
# Internal helpers
# -----------------------------

def build_cmd(dataset: str, method: str) -> list[str]:
    """Construct an ns-train command for a (dataset, method) pair.

    Rules:
      - coverage / fig / view_fig: model=shadow-splat, view-selector=optics,
        and set --pipeline.optics-coverage-metric accordingly.
      - fisher_info: model=fisher-splat, view-selector=optics,
        and set --pipeline.optics-coverage-metric=fisher_info.
      - random: model=shadow-splat, view-selector=random, no metric flag.
      - bayes-rays: method=bayes-rays, uses Nerfacto with ray-batched BayesRays
        view selection, no optics args needed.
    """
    dataset_name = pathlib.Path(dataset.rstrip("/\\")).name
    ts = datetime.now().strftime("%Y%m%d-%H%M")

    if method == "nerf-random":
        model = "bayes-rays"
        view_selector = "random"
        extra_metric_args = []
        exp_suffix = f"{dataset_name}__{method}"
    elif method == "nerfacto":
        model = "nerfacto"
        view_selector = "all"
        extra_metric_args = []
        exp_suffix = f"{dataset_name}__{method}"
    elif method in {"coverage", "fig", "view_fig"}:
        model = "shadow-splat"
        view_selector = "optics"
        extra_metric_args = ["--pipeline.optics-coverage-metric", method]
        exp_suffix = f"{dataset_name}__{method}__optics"
    elif method == "fisher_info":
        model = "fisher-splat"
        view_selector = "optics"
        extra_metric_args = ["--pipeline.optics-coverage-metric", method]
        exp_suffix = f"{dataset_name}__{method}__optics"
    elif method == "bayes-rays":
        model = "bayes-rays"
        view_selector = "bayes"  # Uses Hessian-based uncertainty
        extra_metric_args = []
        exp_suffix = f"{dataset_name}__{method}"
    else:
        raise ValueError(f"Unknown method: {method}")

    project_for_dataset = f"{PROJECT_NAME}__{dataset_name}"
    exp_name = f"{exp_suffix}__{ts}"
    run_name = f"{dataset_name}__{method}___{ts}"

    # Set WandB environment variables for metadata
    if USE_WANDB:
        os.environ["WANDB_PROJECT"] = project_for_dataset
        if WANDB_ENTITY:
            os.environ["WANDB_ENTITY"] = WANDB_ENTITY
        os.environ["WANDB_TAGS"] = f"method:{method},dataset:{dataset_name},model:{model}"
        os.environ["WANDB_NOTES"] = f"View selection: {view_selector}, Dataset: {dataset_name}"

    try:
        cmd = [
            "ns-train",
            model,
            f"--vis={VIS}",
            "--project-name", project_for_dataset,
            "--experiment-name", exp_name,
        ]


        cmd.extend(["--viewer.quit-on-train-completion", str(QUIT_VIEWER_ON_DONE)])

        # BayesRays uses standard nerfstudio data format, others use shadow-splat
        if method == "bayes-rays":
            cmd.extend(["--data", dataset,
                        "--machine.seed" , str(seed)])
        elif method == "nerf-random":
            cmd.extend(["--data", dataset])
        elif method == "nerfacto":
            cmd.extend([
                "--data", dataset,
                "--steps_per_eval_all_images", "1000",
            ])
        else:
            # Gaussian splat methods use shadow-splat-data parser
            cmd.extend([
                "--pipeline.view-selector", view_selector,
                "--pipeline.optics-coverage-metric", method,
                "shadow-splat-data",
                "--data", dataset,
            ])

        # Add eval mode if specified
        # cmd.extend(["--eval-mode", EVAL_MODE])
    except Exception as e:
        print(f"Error building command for {dataset} with {method}: {e}")

    cmd += extra_metric_args
    return cmd

def main() -> None:
    # Set WandB run group for this sweep (groups all runs together)
    if USE_WANDB:
        sweep_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.environ["WANDB_RUN_GROUP"] = sweep_id
        print(f"🎯 Starting sweep with group ID: {sweep_id}")

    total_runs = len(DATASETS) * len(METHODS)
    current_run = 0

    for ds in DATASETS:
        for method in METHODS:
            current_run += 1
            print(f"\n📊 Run {current_run}/{total_runs}: {pathlib.Path(ds).name} with {method}")

            cmd = build_cmd(ds, method)
            print("Command:", " ".join(shlex.quote(c) for c in cmd))

            try:
                subprocess.run(cmd, check=True)
                print(f"✓ Completed: {pathlib.Path(ds).name} with {method}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {pathlib.Path(ds).name} with {method}")
                print(f"  Error: {e}")
                # Continue with next run instead of stopping
                continue

    print(f"\n✅ Sweep complete! {total_runs} runs attempted.")


if __name__ == "__main__":
    main()