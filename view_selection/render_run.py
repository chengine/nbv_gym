# %%
import os
import torch
import numpy as np
from scipy.spatial import KDTree
import json
import time
import datetime

from nerf_utils import *
from planner_utils import *

import shutil
from PIL import Image

import pdb
from tabulate import tabulate
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))



#%% Script to render from a previously completed baseline eval ###

def run_render(
    data_dir,
    method_name,
    run_name,
    scene_name,
    num_data_aug_steps = 50, # number of dataset augmentation steps
    num_train_iterations: int = 1000, # number of training iterations (which could vary per augmentation step)
    view_cached_results: bool = False
):
    
    # if not view_cached_results:
    # data directory
    data_dir = os.path.expanduser(data_dir)

    # Load the poses
    if os.path.exists(
        os.path.join(os.path.expanduser(data_dir), "transforms_original.json")
    ):
        with open(
            os.path.join(os.path.expanduser(data_dir), "transforms_original.json")
        ) as f:
            meta = json.load(f)
    else:
        with open(os.path.join(os.path.expanduser(data_dir), "transforms.json")) as f:
            meta = json.load(f)

        shutil.copy(
            os.path.join(os.path.expanduser(data_dir), "transforms.json"),
            os.path.join(os.path.expanduser(data_dir), "transforms_original.json"),
        )

    # fraction for training
    train_fraction: float = 0.8

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
        # camera intrinsics
        cam_intrinsics = {}
        cam_intrinsics["fx"] = torch.tensor([meta["fl_x"]])
        cam_intrinsics["fy"] = torch.tensor([meta["fl_y"]])
        cam_intrinsics["cx"] = torch.tensor([meta["cx"]])
        cam_intrinsics["cy"] = torch.tensor([meta["cy"]])
        cam_intrinsics["h"] = torch.tensor([meta["h"]])
        cam_intrinsics["w"] = torch.tensor([meta["w"]])
        cam_intrinsics["cam_camera_type"] = torch.tensor([[1]])
    except KeyError:
        # camera intrinsics
        cam_intrinsics = {}
        cam_intrinsics["fx"] = torch.tensor([meta["frames"][0]["fl_x"]])
        cam_intrinsics["fy"] = torch.tensor([meta["frames"][0]["fl_y"]])
        cam_intrinsics["cx"] = torch.tensor([meta["frames"][0]["cx"]])
        cam_intrinsics["cy"] = torch.tensor([meta["frames"][0]["cy"]])
        cam_intrinsics["h"] = torch.tensor([meta["frames"][0]["h"]])
        cam_intrinsics["w"] = torch.tensor([meta["frames"][0]["w"]])
        cam_intrinsics["cam_camera_type"] = torch.tensor([[1]])
        

    # training and test poses
    train_poses = [torch.tensor(frame["transform_matrix"]) for frame in train_frames_data]
    train_poses = torch.stack(train_poses, dim=0)
    test_poses = [torch.tensor(frame["transform_matrix"]) for frame in test_frames_data]
    test_poses = torch.stack(test_poses, dim=0)

    # Camera origins
    train_camera_origins = train_poses[..., :3, -1]

    total_train_kdtree = KDTree(train_camera_origins.numpy().reshape(-1, 3))

    if not view_cached_results:
        # Choose a fixed frame to be the root
        root_idx = 0
        # root_idx = np.random.randint(0, len(train_poses))
        root_pose = train_poses[root_idx]

        # Find the nearest neighbors to the root
        k = 10
        _, nearest_indices = total_train_kdtree.query(
            train_camera_origins[root_idx].numpy().reshape(1, 3), k=k
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
        with open(os.path.join(os.path.expanduser(data_dir), "transforms.json"), "w") as f:
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

        #%%######################################################################################
        # ----------------------------------   Evaluations   ---------------------------------- #
        #########################################################################################

        # # method name
        method_name = "fisher_rf"

        # method stats
        perf_stats = {}

        # # number of dataset augmentation steps
        # num_data_aug_steps = 50
        # # number of training iterations (which could vary per augmentation step)
        # num_train_iterations: int = 1000

        # option to save the rendered images
        save_rendered_images: bool = True

        ### Train the initial dataset for the method
        # current time
        c_time = datetime.datetime.now()

        # output directory
        # output_dir: Path = Path(script_dir,
        #                         f"nerf_data/outputs/{method_name}"
        #                         ).resolve()
        output_dir: Path = Path(
            script_dir,
            f"nerf_data/outputs/{method_name}/scenes/{scene_name}/{c_time.strftime('%Y_%m_%d_%H_%M_%S')}"
        ).resolve()

        # create directory, if necessary
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir)

        #%%######################################################################################
        # ----------------------------------   Dataset Augmentation   ------------------------- #
        #########################################################################################

        # Find the pose furthest from the original root pose
        root_pose = training_poses[-1]

        # number of neighbors to query the uncertainty when selecting poses during a data augmentation iteration
        num_nearest_neighbors_for_querying_uncertainty = 5

        # number of poses to add per augmentaton step
        num_poses_to_add_per_augmentation_step: int = 5

        # load from checkpoint when training the Splat
        load_from_checkpoint: bool = True

        # path to the checkpoint of the field
        checkpoint_path: Path = get_path_to_checkpoint(output_dir)

        try:
            for i in tqdm(range(num_data_aug_steps), desc="Data Augmentation"):
                # num_train_iterations = 1000 * (i + 1)
                config_path = None
                for root, dirs, files in os.walk(output_dir):
                    if "config.yml" in files:
                        config_path = Path(os.path.join(root, "config.yml"))
                        break

                if config_path is None:
                    raise FileNotFoundError("No config.yml file found in the output directory tree.")
                print(f"config_path: {config_path}")

                method = NeRF(config_path, cam_intrinsics=cam_intrinsics, dataset_mode="train")

                tnow = time.time()
                torch.cuda.synchronize()
                
                # Number of steps to dead-reckon the algorithm. Note that the methods don't update their underlying representation during this time, except for VISTA (the view diversity; not the point cloud)!
                for j in range(num_poses_to_add_per_augmentation_step):
                    # Find the k nearest neighbors to the root in the queryable set

                    _, nearest = queryable_kdtree.query(
                        root_pose[:3, -1].numpy().reshape(1, 3), k=num_nearest_neighbors_for_querying_uncertainty
                    )
                    candidate_poses = queryable_poses[nearest]

                    # Evaluate the methods on these candidate poses.
                    cam_intrinsics_copy = cam_intrinsics.copy()
                    for k in cam_intrinsics_copy:
                        if k != "cam_camera_type":
                            if k in ["h", "w"]:
                                cam_intrinsics_copy[k] = (cam_intrinsics_copy[k].float() * 0.25).int()
                            else:
                                cam_intrinsics_copy[k] = cam_intrinsics_copy[k].float() * 0.25
                    best_pose, best_idx, acq_score = evaluate_next_view_fisher(
                        method, root_pose, candidate_poses, rgb_weight=1.0,
                        camera_info=cam_intrinsics_copy,
                    )
                    
                    if save_render_selected_best_pose:
                        # output directory for renders associated with the best pose
                        output_dir_render_best_pose: Path = Path(f"{output_dir}/images_render_selected_pose_by_method/aug_idx_{i}")

                        # create directory, if necessary
                        output_dir_render_best_pose.mkdir(parents=True, exist_ok=True)
                        
                        rgb_img = method.render(pose=best_pose)["rgb"]
                        rgb_img = (rgb_img * 255).detach().cpu().numpy().astype(np.uint8)
                        
                        # save the image
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

                    # # Save the rendered image
                    # rendered_rgb = method.render(pose=best_pose)["rgb"]
                    # rendered_rgb = rendered_rgb.cpu()
                    # # rendered RGB
                    # Image.fromarray((rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)).save(
                    #     Path(f"{img_output_filename}/next_best_view_{i}_{j}.png")
                    #     )

                    root_pose = best_pose

                torch.cuda.synchronize()
                times_list.append(time.time() - tnow)

                # Create new training set
                # Update meta['frames'] to include the new training poses
                # new_training_indices = [i for i, frame in enumerate(original_frames) if torch.equal(torch.tensor(frame['transform_matrix']), training_poses[-1])]
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
                
                shutil.copy(
                    os.path.join(os.path.expanduser(data_dir), "transforms.json"),
                    os.path.join(
                        os.path.expanduser(data_dir),
                        f"transforms_{str(i).zfill(3)}.json",
                    ),
                )
                # Save the updated transforms.json
                with open(
                    os.path.join(os.path.expanduser(data_dir), "transforms.json"),
                    "w",
                ) as f:
                    json.dump(meta, f, indent=4)

                # current time
                c_time = datetime.datetime.now()

                # output directory
                output_dir: Path = Path(
                    script_dir,
                    f"nerf_data/outputs/{method_name}/scenes/{scene_name}/{c_time.strftime('%Y_%m_%d_%H_%M_%S')}"
                ).resolve()

                # create directory, if necessary
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dir = str(output_dir)
                
                # train the GSplat
                train_gsplat(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    num_train_iterations=num_train_iterations,
                    load_from_checkpoint=load_from_checkpoint,
                    checkpoint_file=checkpoint_path,
                )
                
                # time.sleep(1)
                    
                # path to the checkpoint of the field
                checkpoint_path: Path = get_path_to_checkpoint(output_dir)

                # compute the photometric stats
                psnr, ssim, lpips = compute_photometric_scores(
                    output_dir=output_dir,
                    data_dir=data_dir,
                    test_frames_data=test_frames_data,
                    cam_intrinsics=cam_intrinsics,
                    save_rendered_images=save_rendered_images,
                )
                # cache the stats
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
                    
                    
                # method.train(training_poses)        # TODO: We might want to scale the number of training iterations based on the size of the training set.
                # metric = method.calculate_metric(training_poses)        # This is the metric that we want to maximize like PSNR on the test set.
                # metric_list.append(metric)
                # training_sets.append(training_poses)

                #########################################################################################
                # -------------------------------   Visualize Results   ------------------------------- #
                #########################################################################################

                # # Save the performance stats to the output directory
                # perf_stats_output_path: Path = Path(script_dir,
                #                                     f"nerf_data/outputs/tests/perf_stats.json").resolve()

                with open(perf_stats_output_path, "w") as f:
                    json.dump(perf_stats, f, indent=4)

            # Tabulate the data
            # print(tabulate(perf_stats, headers="keys", tablefmt="grid"))
            # %%
        except IndexError as excp:
            # handle the error gracefully
            print(excp)
        finally:
            # Load the stats from the wandb-summary.json file
            # Loop through the output directories to retrieve wandb stats
            # method_name = "fisher_rf"
            # output_base_dir = Path(script_dir,
            #                        f"nerf_data/outputs/fisher_rf").resolve()
            # perf_stats_output_path: Path = Path(script_dir,
            #                                     f"nerf_data/outputs/tests/perf_stats.json").resolve()

            # # performance stats output directory
            # perf_stats_output_path: Path = Path(f"{script_dir}/nerf_data/outputs/{method_name}/perf_stats.json").resolve()

            # Load cached performance stats if available
            if perf_stats_output_path.exists():
                with open(perf_stats_output_path, "r") as f:
                    perf_stats = json.load(f)

            # Tabulate the data
            # Specify the headers to include in the table
            headers_to_include = ["augmentation_step", "directory", "psnr", "ssim", "lpips", "runtime"]

            # Filter the performance stats to include only the specified headers
            filtered_perf_stats = [
                {key: stat[key] for key in headers_to_include if key in stat}
                for stat_list in perf_stats.values()
                for stat in stat_list
            ]

            # Tabulate the filtered data
            print(tabulate(filtered_perf_stats, headers="keys", tablefmt="grid"))
            # %%
    else:
        # view cached results
        # path to the file
        perf_stats_output_path: Path = Path(f"{script_dir}/nerf_data/results/{method_name}/scenes/{scene_name}/perf_stats.json").resolve()
        
        # Load cached performance stats if available
        if perf_stats_output_path.exists():
            with open(perf_stats_output_path, "r") as f:
                perf_stats = json.load(f)
        else:
            raise FileNotFoundError("The file containing the performance stats was not found!")

        # Tabulate the data
        # Specify the headers to include in the table
        headers_to_include = ["augmentation_step", "directory", "psnr", "ssim", "lpips", "runtime"]

        # Filter the performance stats to include only the specified headers
        filtered_perf_stats = [
            {key: stat[key] for key in headers_to_include if key in stat}
            for stat_list in perf_stats.values()
            for stat in stat_list
        ]

        # Tabulate the filtered data
        print(tabulate(filtered_perf_stats, headers="keys", tablefmt="grid"))

        # Render the results (RGB and Uncertainty)
        for stat in filtered_perf_stats:
            directory = stat["directory"]
            # output directory
            output_dir: Path = Path(
            script_dir,
            f"nerf_data/outputs/{method_name}/scenes/{scene_name}/{directory}"
            ).resolve()

            # render from prior run
            render_from_checkpoint(
            output_dir=output_dir,
            data_dir=data_dir,
            test_frames_data=test_frames_data,
            cam_intrinsics=cam_intrinsics,
            method_name=method_name,
            run_name=run_name,
            )
        # for directory in filtered_perf_stats["directory"]:
        #     # output directory
        #     output_dir: Path = Path(
        #         script_dir,
        #         f"nerf_data/outputs/{method_name}/scenes/{scene_name}/{directory}"
        #     ).resolve()

        #     # render from prior run
        #     render_from_checkpoint(
        #         output_dir=output_dir,
        #         data_dir=data_dir,
        #         test_frames_data=test_frames_data,
        #         cam_intrinsics=cam_intrinsics,
        #         method_name=method_name,
        #         run_name=run_name,
        #     )
        
if __name__ == "__main__":
    # method name
    method_name = "fisher_rf"
    # run name
    run_name = "march_twentysixth"
    
    # data directories
    all_data_dirs = [
        # {
        #     "data_dir": "~/StanfordMSL/LL-Baselines/vista/nerf_data/kitchen",
        # },
        {
            "data_dir": "~/StanfordMSL/LL-Baselines/vista/nerf_data/plane",
        },
        # {
        #     "data_dir": "~/StanfordMSL/LL-Baselines/vista/nerf_data/poster",
        # },
        # {
        #     "data_dir": "~/StanfordMSL/LL-Baselines/vista/nerf_data/old_union",
        # },
        # {
        #     "data_dir": "~/StanfordMSL/LL-Baselines/vista/nerf_data/registered",
        # },
        # {
        #     "data_dir": "~/StanfordMSL/LL-Baselines/vista/nerf_data/flight",
        # },
    ]
    
    # option to view pre-cached results
    view_cached_results: bool = True
    
    # run the evaluation
    for config in all_data_dirs:
        run_render(
            data_dir=config["data_dir"],
            method_name=method_name,
            run_name=run_name,
            scene_name=Path(config["data_dir"]).stem,
            view_cached_results=view_cached_results,
        )