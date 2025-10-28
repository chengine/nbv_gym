# %%
import os
import torch
import numpy as np
from scipy.spatial import KDTree
import json
import time
import datetime
import argparse
import yaml
from pathlib import Path

from nerf_utils import *
from planner_utils import *

import shutil
from PIL import Image

from tabulate import tabulate

script_dir = os.path.dirname(os.path.abspath(__file__))

# ==============================
# Config helpers
# ==============================
def _exp(p): return os.path.expanduser(p) if isinstance(p, str) else p

def load_config(cfg_path: str | None):
    """Load YAML config; return {} if not found."""
    if cfg_path is None:
        cfg_path = str(Path(script_dir) / "config.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def cfg_get(cfg, path, default):
    """Nested get with default: path like 'data.train_fraction'."""
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

#########################################################################################
# -----------------------------------  Data Loading  ---------------------------------- #
#########################################################################################

def run_eval(
    data_dir,
    method_name,
    scene_name,
    num_data_aug_steps = 50,
    num_train_iterations: int = 1000,
    view_cached_results: bool = False,
    cfg: dict | None = None,
):
    """
    NOTE: All parameters are still honored from call-site, but `cfg` (if provided)
    can override them. Absent keys fall back to the original defaults.
    """
    cfg = cfg or {}

    # Top-level toggles/knobs (YAML can override)
    method_name = cfg_get(cfg, "method.name", method_name)
    view_cached_results = cfg_get(cfg, "runtime.view_cached_results", view_cached_results)

    if not view_cached_results:
        # data directory
        data_dir = _exp(cfg_get(cfg, "data.data_dir", data_dir))

        # Load the poses
        if os.path.exists(
            os.path.join(_exp(data_dir), "transforms_original.json")
        ):
            with open(
                os.path.join(_exp(data_dir), "transforms_original.json")
            ) as f:
                meta = json.load(f)
        else:
            with open(os.path.join(_exp(data_dir), "transforms.json")) as f:
                meta = json.load(f)

            shutil.copy(
                os.path.join(_exp(data_dir), "transforms.json"),
                os.path.join(_exp(data_dir), "transforms_original.json"),
            )

        # fraction for training (YAML)
        train_fraction: float = float(cfg_get(cfg, "data.train_fraction", 0.8))

        # total number of frames
        num_frames = len(meta["frames"])

        # create a training dataset and a test dataset
        train_ind = np.linspace(
            0, num_frames - 1, np.ceil(train_fraction * num_frames).astype(int)
        ).astype(int)
        test_ind = np.setdiff1d(np.arange(num_frames), train_ind)

        # training  and test frames
        frames_data = np.array(meta["frames"])
        train_frames_data = frames_data[train_ind]
        test_frames_data = frames_data[test_ind]

        try:
            # camera intrinsics (prefer scene-level)
            cam_intrinsics = {}
            cam_intrinsics["fx"] = torch.tensor([meta["fl_x"]])
            cam_intrinsics["fy"] = torch.tensor([meta["fl_y"]])
            cam_intrinsics["cx"] = torch.tensor([meta["cx"]])
            cam_intrinsics["cy"] = torch.tensor([meta["cy"]])
            cam_intrinsics["h"] = torch.tensor([meta["h"]])
            cam_intrinsics["w"] = torch.tensor([meta["w"]])
            cam_intrinsics["cam_camera_type"] = torch.tensor([[1]])
        except KeyError:
            # camera intrinsics from first frame
            cam_intrinsics = {}
            cam_intrinsics["fx"] = torch.tensor([meta["frames"][0]["fl_x"]])
            cam_intrinsics["fy"] = torch.tensor([meta["frames"][0]["fl_y"]])
            cam_intrinsics["cx"] = torch.tensor([meta["frames"][0]["cx"]])
            cam_intrinsics["cy"] = torch.tensor([meta["frames"][0]["cy"]])
            cam_intrinsics["h"] = torch.tensor([meta["frames"][0]["h"]])
            cam_intrinsics["w"] = torch.tensor([meta["frames"][0]["w"]])
            cam_intrinsics["cam_camera_type"] = torch.tensor([[1]])

        # Optional intrinsics downscale for candidate eval (YAML)
        intr_scale = float(cfg_get(cfg, "planner.intrinsics_scale", 0.25))

        # training and test poses
        train_poses = [torch.tensor(frame["transform_matrix"]) for frame in train_frames_data]
        train_poses = torch.stack(train_poses, dim=0)
        test_poses = [torch.tensor(frame["transform_matrix"]) for frame in test_frames_data]
        test_poses = torch.stack(test_poses, dim=0)

        # Camera origins
        train_camera_origins = train_poses[..., :3, -1]
        total_train_kdtree = KDTree(train_camera_origins.numpy().reshape(-1, 3))

        # Choose a fixed frame to be the root
        root_idx = int(cfg_get(cfg, "planner.root_index", 0))
        root_pose = train_poses[root_idx]

        # Find the nearest neighbors to the root (base set size k)
        k_init = int(cfg_get(cfg, "data.k_init_neighbors", 10))
        _, nearest_indices = total_train_kdtree.query(
            train_camera_origins[root_idx].numpy().reshape(1, 3), k=k_init
        )

        # Get the nearest neighbors' poses. This includes the root itself.
        training_poses = train_poses[
            nearest_indices
        ]  # This forms the base set. This is invariant between all methods.

        # Update meta['frames'] to solely include the frames corresponding to training_poses
        entire_train_frames = train_frames_data.copy()
        training_indices = nearest_indices.flatten().tolist()
        curr_train_frames_data = [train_frames_data[i] for i in training_indices]
        meta["frames"] = curr_train_frames_data

        # Save the updated transforms.json
        with open(os.path.join(_exp(data_dir), "transforms.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # Form the queryable set without the nearest neighbors
        queryable_poses = np.delete(train_poses.numpy(), nearest_indices, axis=0)
        queryable_poses = torch.tensor(queryable_poses)
        queryable_camera_origins = queryable_poses[..., :3, -1]
        queryable_kdtree = KDTree(queryable_camera_origins.reshape(-1, 3))

        ### Run the baselines and method
        metric_list = []
        times_list = []
        best_poses_list = []
        best_poses_list.append(root_pose)
        training_sets = [training_poses]

        # method stats
        perf_stats = {}

        # number of dataset augmentation steps (YAML)
        num_data_aug_steps = int(cfg_get(cfg, "incremental.num_data_aug_steps", num_data_aug_steps))
        # number of training iterations per (re)train (YAML)
        num_train_iterations = int(cfg_get(cfg, "training.num_train_iterations", num_train_iterations))
        # option to save the rendered images (YAML)
        save_rendered_images: bool = bool(cfg_get(cfg, "rendering.save_rendered_images", False))

        ### Train the initial dataset for the method
        c_time = datetime.datetime.now()
        # output directory root (YAML override)
        out_root = cfg_get(
            cfg, "output.root",
            str(Path(script_dir) / f"nerf_data/outputs/{method_name}/scenes/{scene_name}")
        )
        output_dir: Path = Path(out_root, c_time.strftime("%Y_%m_%d_%H_%M_%S")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir)

        # Train initial GSplat
        train_gsplat(
            data_dir=data_dir,
            output_dir=output_dir,
            num_train_iterations=num_train_iterations,
            load_from_checkpoint=bool(cfg_get(cfg, "training.load_from_checkpoint_initial", False)),
            checkpoint_file=cfg_get(cfg, "training.initial_checkpoint_file", None),
        )

        # photometric stats
        psnr, ssim, lpips = compute_photometric_scores(
            output_dir=output_dir,
            data_dir=data_dir,
            test_frames_data=test_frames_data,
            cam_intrinsics=cam_intrinsics,
            method_name=method_name,
            run_name=scene_name,
            save_rendered_images=save_rendered_images,
        )

        # cache the stats
        if not method_name in perf_stats:
            perf_stats[method_name] = [
                {
                    "augmentation_step": "n/a",
                    "directory": Path(output_dir).stem,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                    "best_poses": best_poses_list[-1].tolist(),
                }
            ]
        else:
            perf_stats[method_name].append(
                {
                    "augmentation_step": "n/a",
                    "directory": Path(output_dir).stem,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                    "best_poses": best_poses_list[-1].tolist(),
                }
            )

        # Save the performance stats to the output directory
        perf_stats_output_path: Path = Path(
            f"{script_dir}/nerf_data/results/{method_name}/scenes/{scene_name}/perf_stats.json"
        ).resolve()
        perf_stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(perf_stats_output_path, "w") as f:
            json.dump(perf_stats, f, indent=4)

        # Load cached performance stats if available
        if perf_stats_output_path.exists():
            with open(perf_stats_output_path, "r") as f:
                perf_stats = json.load(f)

        # Tabulate the data
        print(tabulate(perf_stats, headers="keys", tablefmt="grid"))

        # option to save the rendered image at each selected pose (YAML)
        save_render_selected_best_pose: bool = bool(
            cfg_get(cfg, "rendering.save_render_selected_best_pose", False)
        )

        # ----------------------------------   Dataset Augmentation   ------------------------- #
        root_pose = training_poses[-1]

        # neighbors to query per step (YAML)
        num_nearest_neighbors_for_querying_uncertainty = int(
            cfg_get(cfg, "planner.num_nearest_neighbors_for_querying_uncertainty", 5)
        )
        # number of poses added per augmentation step (YAML)
        num_poses_to_add_per_augmentation_step: int = int(
            cfg_get(cfg, "planner.num_poses_to_add_per_augmentation_step", 5)
        )

        # load from checkpoint when training the Splat (YAML)
        load_from_checkpoint: bool = bool(cfg_get(cfg, "training.load_from_checkpoint", True))

        # path to the checkpoint of the field
        checkpoint_path: Path = get_path_to_checkpoint(output_dir)

        try:
            for i in tqdm(range(num_data_aug_steps), desc="Data Augmentation"):
                # Find latest config.yml inside output_dir tree
                config_path = None
                for root, dirs, files in os.walk(output_dir):
                    if "config.yml" in files:
                        config_path = Path(os.path.join(root, "config.yml"))
                        break
                if config_path is None:
                    raise FileNotFoundError("No config.yml file found in the output directory tree.")
                print(f"config_path: {config_path}")

                nerf = NeRF(config_path, cam_intrinsics=cam_intrinsics, dataset_mode="train")

                tnow = time.time()
                torch.cuda.synchronize()

                # Dead-reckon steps
                for j in range(num_poses_to_add_per_augmentation_step):
                    # Candidate set: nearest neighbors to root_pose among queryable
                    _, nearest = queryable_kdtree.query(
                        root_pose[:3, -1].numpy().reshape(1, 3),
                        k=num_nearest_neighbors_for_querying_uncertainty
                    )
                    candidate_poses = queryable_poses[nearest]

                    # Downscaled intrinsics for candidate scoring
                    cam_intrinsics_copy = cam_intrinsics.copy()
                    for k in cam_intrinsics_copy:
                        if k != "cam_camera_type":
                            if k in ["h", "w"]:
                                cam_intrinsics_copy[k] = (cam_intrinsics_copy[k].float() * intr_scale).int()
                            else:
                                cam_intrinsics_copy[k] = cam_intrinsics_copy[k].float() * intr_scale

                    # Coverage-based selection (kept identical call)
                    best_pose, best_idx, acq_score = evaluate_next_view_3dgs_coverage(
                        nerf, candidate_poses, camera_info=cam_intrinsics_copy,
                    )

                    if save_render_selected_best_pose:
                        # directory for renders of selected pose
                        output_dir_render_best_pose: Path = Path(
                            f"{output_dir}/images_render_selected_pose_by_method/aug_idx_{i}"
                        )
                        output_dir_render_best_pose.mkdir(parents=True, exist_ok=True)

                        rgb_img = method.render(pose=best_pose)["rgb"]
                        rgb_img = (rgb_img * 255).detach().cpu().numpy().astype(np.uint8)
                        Image.fromarray(rgb_img).save(f"{output_dir_render_best_pose}/{j:05}.png")

                    # Remove the pose corresponding to best_idx from the candidates
                    queryable_poses = torch.cat(
                        [
                            queryable_poses[: nearest[0][best_idx]],
                            queryable_poses[nearest[0][best_idx] + 1 :],
                        ],
                        dim=0,
                    )
                    queryable_camera_origins = queryable_poses[..., :3, -1]
                    queryable_kdtree = KDTree(queryable_camera_origins.reshape(-1, 3))

                    # Add best pose to the training set
                    training_poses = torch.cat([training_poses, best_pose.unsqueeze(0)], dim=0)

                    # Save the best pose
                    best_poses_list.append(best_pose)

                    # Advance root
                    root_pose = best_pose

                torch.cuda.synchronize()
                times_list.append(time.time() - tnow)

                # Create new training set in transforms.json
                new_training_indices = []
                for pose in training_poses:
                    for ii, frame in enumerate(entire_train_frames):
                        if (
                            torch.equal(torch.tensor(frame["transform_matrix"]), pose)
                            and ii not in training_indices
                        ):
                            new_training_indices.append(ii)
                            break

                # update the training indices and frames
                training_indices.extend(new_training_indices)
                meta["frames"].extend([entire_train_frames[jj] for jj in new_training_indices])

                # snapshot previous transforms
                shutil.copy(
                    os.path.join(_exp(data_dir), "transforms.json"),
                    os.path.join(_exp(data_dir), f"transforms_{str(i).zfill(3)}.json"),
                )
                # Save the updated transforms.json
                with open(os.path.join(_exp(data_dir), "transforms.json"), "w") as f:
                    json.dump(meta, f, indent=4)

                # New round output dir
                c_time = datetime.datetime.now()
                out_root = cfg_get(
                    cfg, "output.root",
                    str(Path(script_dir) / f"nerf_data/outputs/{method_name}/scenes/{scene_name}")
                )
                output_dir: Path = Path(out_root, c_time.strftime("%Y_%m_%d_%H_%M_%S")).resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dir = str(output_dir)

                # Train GSplat for this round
                train_gsplat(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    num_train_iterations=num_train_iterations,
                    load_from_checkpoint=load_from_checkpoint,
                    checkpoint_file=checkpoint_path,
                )

                # update checkpoint path
                checkpoint_path: Path = get_path_to_checkpoint(output_dir)

                # photometric stats
                psnr, ssim, lpips = compute_photometric_scores(
                    output_dir=output_dir,
                    data_dir=data_dir,
                    test_frames_data=test_frames_data,
                    cam_intrinsics=cam_intrinsics,
                    save_rendered_images=save_rendered_images,
                )
                # cache stats
                if not method_name in perf_stats:
                    perf_stats[method_name] = [
                        {
                            "augmentation_step": i,
                            "directory": Path(output_dir).stem,
                            "psnr": psnr,
                            "ssim": ssim,
                            "lpips": lpips,
                            "runtime": times_list[-1],
                            "best_poses": [pose.tolist() for pose in best_poses_list[-5:]]
                        }
                    ]
                else:
                    perf_stats[method_name].append(
                        {
                            "augmentation_step": i,
                            "directory": Path(output_dir).stem,
                            "psnr": psnr,
                            "ssim": ssim,
                            "lpips": lpips,
                            "runtime": times_list[-1],
                            "best_poses": [pose.tolist() for pose in best_poses_list[-5:]]
                        }
                    )

                with open(perf_stats_output_path, "w") as f:
                    json.dump(perf_stats, f, indent=4)

        except IndexError as excp:
            # handle the error gracefully
            print(excp)
        finally:
            # Load cached performance stats if available
            if perf_stats_output_path.exists():
                with open(perf_stats_output_path, "r") as f:
                    perf_stats = json.load(f)

            # Tabulate the filtered data
            headers_to_include = ["augmentation_step", "directory", "psnr", "ssim", "lpips", "runtime"]
            filtered_perf_stats = [
                {key: stat[key] for key in headers_to_include if key in stat}
                for stat_list in perf_stats.values()
                for stat in stat_list
            ]
            print(tabulate(filtered_perf_stats, headers="keys", tablefmt="grid"))

    else:
        # view cached results
        perf_stats_output_path: Path = Path(
            f"{script_dir}/nerf_data/results/{method_name}/scenes/{scene_name}/perf_stats.json"
        ).resolve()

        # Load cached performance stats if available
        if perf_stats_output_path.exists():
            with open(perf_stats_output_path, "r") as f:
                perf_stats = json.load(f)
        else:
            raise FileNotFoundError("The file containing the performance stats was not found!")

        # Tabulate the filtered data
        headers_to_include = ["augmentation_step", "directory", "psnr", "ssim", "lpips", "runtime"]
        filtered_perf_stats = [
            {key: stat[key] for key in headers_to_include if key in stat}
            for stat_list in perf_stats.values()
            for stat in stat_list
        ]
        print(tabulate(filtered_perf_stats, headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(script_dir) / "config.yaml"),
                        help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # method name
    method_name = cfg_get(cfg, "method.name", "3dgs_coverage")

    # data dirs: list of dicts (each with 'data_dir'), preserve your default
    all_data_dirs = cfg_get(cfg, "data.dirs", None)

    if all_data_dirs is None:
        raise ValueError("No data directories provided in the config file!")

    # option to view pre-cached results
    view_cached_results: bool = bool(cfg_get(cfg, "runtime.view_cached_results", False))

    # top-level training knobs (still per-run defaults, can be overridden in cfg)
    num_data_aug_steps = int(cfg_get(cfg, "incremental.num_data_aug_steps", 50))
    num_train_iterations = int(cfg_get(cfg, "training.num_train_iterations", 1000))

    # run the evaluation
    for config in all_data_dirs:
        run_eval(
            data_dir=config["data_dir"],
            method_name=method_name,
            scene_name=Path(_exp(config["data_dir"])).stem,
            view_cached_results=view_cached_results,
            num_data_aug_steps=num_data_aug_steps,
            num_train_iterations=num_train_iterations,
            cfg=cfg,
        )
