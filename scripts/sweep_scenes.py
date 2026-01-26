import subprocess
import pathlib
import shlex
from datetime import datetime

# -----------------------------
# Batch experiment configuration
# -----------------------------
PROJECT_NAME = "next-best-view-updated"

# Datasets to sweep
BASE_DATA_DIR = pathlib.Path("/home/chengine/Research/data")
SCENES = [
"caterpillar",
"train",
"ignatius",
"shiny_statue_6pm",
"space_laces_4pm",
"chair_3pm",
"mipnerf360/bicycle",
"mipnerf360/bonsai",
"mipnerf360/counter",
"mipnerf360/flowers",
"mipnerf360/garden",
"mipnerf360/kitchen",
"mipnerf360/room",
"mipnerf360/stump",
"mipnerf360/treehill",
]
DATASETS = [str(BASE_DATA_DIR / s) for s in SCENES]

# Methods / information gain metrics to compare.
# Valid entries: "coverage", "fig", "view_fig", "fisher_info", "random"
METHODS = [
    "coverage",
    # "fig_color_field",
    # "fisher_info",
    "random",
    "all",
    # "fig",
    # "view_fig",
    # "fig_diag",
    # "view_fig_diag",
]

# Visualization backends. Example: "viewer+wandb" or just "wandb"
VIS = "wandb"

# Quit the viewer on train completion to avoid hangs in sweeps
QUIT_VIEWER_ON_DONE = True

# Biased dataset
BIASED = False

MAX_ITERATIONS = 30000

SEED = 0

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
    """
    dataset_name = pathlib.Path(dataset.rstrip("/\\")).name
    ts = datetime.now().strftime("%Y%m%d-%H%M")

    if method == "random":
        model = "nbv-splat"
        view_selector = "random"
        extra_metric_args = []
        exp_suffix = f"{dataset_name}__{method}"
    elif method in {"coverage", "fig", "view_fig", "fig_diag", "view_fig_diag", "fig_color_field"}:
        model = "nbv-splat"
        view_selector = "basic"
        extra_metric_args = ["--pipeline.view-metric", method]
        exp_suffix = f"{dataset_name}__{method}"
    elif method == "all":
        model = "nbv-splat"
        view_selector = "all"
        extra_metric_args = []
        exp_suffix = f"{dataset_name}__{method}"
    else:
        raise ValueError(f"Unknown method: {method}")

    project_for_dataset = f"{PROJECT_NAME}__{dataset_name}"
    exp_name = f"{exp_suffix}__{ts}"

    try:
        cmd = [
            "ns-train",
            model,
            f"--vis={VIS}",
            "--project-name",
            project_for_dataset,
            "--experiment-name",
            exp_name,
            "--machine.seed",
            str(SEED),
            "--max-num-iterations",
            str(MAX_ITERATIONS),
            "--pipeline.view-selector",
            view_selector,
            "--pipeline.datamanager.bias-views",
            str(BIASED),
            "--pipeline.initial-view-seed",
            str(0),
            "--viewer.quit-on-train-completion",
            str(QUIT_VIEWER_ON_DONE),
            # "shadow-splat-data",
            "--data",
            dataset,
        ]
        cmd += extra_metric_args
    except Exception as e:
        print(f"Error building command for {dataset} with {method}: {e}")

    return cmd


def main() -> None:
    for ds in DATASETS:
        for method in METHODS:
            cmd = build_cmd(ds, method)
            print("Running:", " ".join(shlex.quote(c) for c in cmd))
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
