import os, subprocess, wandb, datetime, pathlib, shlex


def main():
    wandb.init(project="shadow-splat")  # project default if env not set

    ds = wandb.config["data"]
    method = wandb.config["method"]
    seed = int(wandb.config.get("seed", 0))

    # Group by timestamp when sweep starts (write once via env or make here)
    if not os.getenv("WANDB_RUN_GROUP"):
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.environ["WANDB_RUN_GROUP"] = ts

    dataset_name = pathlib.Path(ds).name
    run_name = f"{dataset_name}__{method}__s{seed}"

    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_TAGS"] = f"dataset:{dataset_name},method:{method},seed:{seed},nerfstudio"

    cmd = [
        "ns-train",
        method,
        "--data",
        ds,
        "--seed",
        str(seed),
        "--project-name",
        "shadow-splat",
        "--experiment-name",
        os.environ["WANDB_RUN_GROUP"],
        "--run-name",
        run_name,
        "--max-num-iterations",
        "30000",
        "--mixed-precision",
        "True",
    ]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
