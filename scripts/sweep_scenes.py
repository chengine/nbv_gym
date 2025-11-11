import subprocess
import pathlib
import shlex
from datetime import datetime

# -----------------------------
# Batch experiment configuration
# -----------------------------
PROJECT_NAME = "next-best-view"

# Datasets to sweep
# BASE_DATA_DIR = pathlib.Path("/home/chengine/Research/data")
BASE_DATA_DIR = pathlib.Path("data")
SCENES = [
    "tandt/caterpillar",
    "tandt/train",
    "tandt/ignatius",
    "captures/shiny_statue_6pm",
    "captures/space_laces_4pm",
    "captures/chair_3pm",
]
DATASETS = [str(BASE_DATA_DIR / s) for s in SCENES]

# Methods / information gain metrics to compare.
# Valid entries: "coverage", "fig", "view_fig", "fisher_info", "random"
METHODS = [
    "coverage",
    "fig",
    "view_fig",
    "fisher_info",
    "random",
]

# Visualization backends. Example: "viewer+wandb" or just "wandb"
VIS = "viewer+wandb"

# Quit the viewer on train completion to avoid hangs in sweeps
QUIT_VIEWER_ON_DONE = True

# Biased dataset
BIASED = True

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
        model = "shadow-splat"
        view_selector = "random"
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
            "--pipeline.view-selector",
            view_selector,
            "--pipeline.optics-coverage-metric",
            method,
            "--pipeline.datamanager.bias-views",
            str(BIASED),
            "--viewer.quit-on-train-completion",
            str(QUIT_VIEWER_ON_DONE),
            # Data spec (keep consistent with existing scripts)
            "shadow-splat-data",
            "--data",
            dataset,
        ]
    except Exception as e:
        print(f"Error building command for {dataset} with {method}: {e}")

    # cmd += extra_metric_args
    return cmd


def main() -> None:
    for ds in DATASETS:
        for method in METHODS:
            cmd = build_cmd(ds, method)
            print("Running:", " ".join(shlex.quote(c) for c in cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
