import os, subprocess, wandb, datetime, pathlib, shlex


PROJECT_NAME = "shadow-splat"
DATASETS = [
    "data/blender/master_chief",
    "data/blender/perseverance",
    "data/captures/chair_3pm/",
    "data/captures/space_laces_4pm/",
    "data/captures/shiny_statue_6pm/",
]
VIEW_SELECTORS = [
    "random",
    "optics",
]


def main():
    for ds in DATASETS:
        name = pathlib.Path(ds).name
        method = "shadow-splat"

        for vs in VIEW_SELECTORS:
            run_name = f"{name}__{vs}"
            cmd = [
                "ns-train",
                method,
                "--data",
                ds,
                "--vis=wandb",
                "--project-name",
                PROJECT_NAME,
                "--experiment-name",
                run_name,
                "--pipeline.view-selector",
                vs,
                "--viewer.quit-on-train-completion",
                "True",
            ]
            print("Running:", " ".join(shlex.quote(c) for c in cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
