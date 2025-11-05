# %%
import os
import torch
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))

#%% Main script to run VISTA and the baselines in a simulated scenario ###

#########################################################################################
# -----------------------------------  Data Loading  ---------------------------------- #
#########################################################################################

# Directory where the data is stored
# workspace_dir = "~/StanfordMSL/FisherRF/LL-FisherRF/"
# data_dir = "nerf_data/statues"
# ndata_dir = "/home/admin/StanfordMSL/FisherRF/LL-FisherRF/nerf_data/statues"

# data_dir = "~/StanfordMSL/LL-Baselines/vista/nerf_data/statues"
# data_dir = "~/Research/NeRF_Data/statues"

def run_eval(
    data_dir,
    method_name,
    scene_name,
    num_data_aug_steps = 50, # number of dataset augmentation steps
    num_train_iterations: int = 1000, # number of training iterations (which could vary per augmentation step)
    view_cached_results: bool = False
):
    # config_path = Path(
    #     f"{os.path.expanduser('~/StanfordMSL/FisherRF/LL-FisherRF/nerf_data/outputs/statues/splatfacto/2025-01-27_153802/config.yml')}"
    #     if "MSL" in data_dir
    #     else f"{os.path.expanduser('~/Research/Gaussian_Splatting/outputs/nerfstudio/statues/splatfacto/2025-01-28_092010/config.yml')}"
    # )
    
    if not view_cached_results:
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

        # method_name = "fisher_rf"
        # # current time
        # c_time = datetime.datetime.now()
        # # output directory
        # output_dir: Path = Path(
        #     f"./outputs/{method_name}/{c_time.strftime('%Y_%m_%d_%H_%M_%S')}"
        # ).resolve()

        # # create directory, if necessary
        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_dir = str(output_dir)

        # train_all(method_name,data_dir=data_dir, output_dir=output_dir)

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
        train_poses = torch.stack(train_poses, dim=0).to(device)
        test_poses = [torch.tensor(frame["transform_matrix"]) for frame in test_frames_data]
        test_poses = torch.stack(test_poses, dim=0).to(device)

        # Camera origins
        train_camera_origins = train_poses[..., :3, -1]

        # TODO: Might be prudent to store a subset of the data as a test set
        # camera_poses = camera_origins.numpy().reshape(-1, 3)
        # train_indices, test_indices = cluster_and_exclude(camera_poses,
        #                                                    n_clusters=3,
        #                                                    exclude_ratio=0.4,
        #                                                    random_state=42)

        # train_origins, holdout_origins = camera_poses[train_indices], camera_poses[test_indices]

        total_train_kdtree = KDTree(train_camera_origins.cpu().numpy().reshape(-1, 3))

        # Choose a fixed frame to be the root
        root_idx = 0
        # root_idx = np.random.randint(0, len(train_poses))
        root_pose = train_poses[root_idx]

        # Find the nearest neighbors to the root
        k = 10
        _, nearest_indices = total_train_kdtree.query(
            train_camera_origins[root_idx].cpu().numpy().reshape(1, 3), k=k
        )

        # Get the nearest neighbors' poses. This includes the root itself.
        training_poses = train_poses[
            nearest_indices
        ]  # This forms the base set. This is invariant between all methods.
        training_poses = training_poses.squeeze()

        # Update meta['frames'] to solely include the frames corresponding to training_poses
        entire_train_frames = train_frames_data.copy()
        training_indices = nearest_indices.flatten().tolist()
        curr_train_frames_data = [train_frames_data[i] for i in training_indices]
        meta["frames"] = curr_train_frames_data

        # Save the updated transforms.json
        with open(os.path.join(os.path.expanduser(data_dir), "transforms.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # Form the queryable set without the nearest neighbors
        queryable_poses = np.delete(train_poses.cpu().numpy(), nearest_indices, axis=0)
        queryable_poses = torch.tensor(queryable_poses)
        queryable_camera_origins = queryable_poses[..., :3, -1]
        queryable_kdtree = KDTree(queryable_camera_origins.cpu().numpy().reshape(-1, 3))

        ### Run the baselines and method
        metric_list = []
        times_list = []
        best_poses_list = []
        best_poses_list.append(root_pose)
        training_sets = [training_poses]

        # # output directory for rendered images
        # img_output_dir = Path(f"/home/admin/StanfordMSL/FisherRF/LL-FisherRF/render_results/rgb")
        # # filename
        # img_output_filename = Path(f"{img_output_dir}/rendered_rgb")
        # # create directory, if necessary
        # img_output_filename.mkdir(parents=True, exist_ok=True)

        # # Render all the views in training_poses
        # method = NeRF(config_path, dataset_mode="train")
        # for idx, pose in enumerate(training_poses):
        #     rendered_rgb = method.render(pose=pose)["rgb"]
        #     rendered_rgb = rendered_rgb.cpu()
        #     # Save the rendered RGB image
        #     Image.fromarray((rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)).save(
        #         Path(f"{img_output_filename}/training_view_{idx}.png")
        #     )

        #%%######################################################################################
        # ----------------------------------   Evaluations   ---------------------------------- #
        #########################################################################################

        # # method name
        # method_name = "bayes_rays"

        # method stats
        perf_stats = {}

        # # number of dataset augmentation steps
        # num_data_aug_steps = 50

        # # number of training iterations (which could vary per augmentation step)
        # num_train_iterations: int = 1000

        # option to save the rendered images
        save_rendered_images: bool = False

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
        # ----------------------------------   Train the Gsplat   ------------------------------#
        #########################################################################################
        train_gsplat(
            data_dir=data_dir,
            output_dir=output_dir,
            num_train_iterations=num_train_iterations,
        )

        #%%# compute the photometric stats
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
        #%%
        # Save the performance stats to the output directory
        perf_stats_output_path: Path = Path(f"{script_dir}/nerf_data/results/{method_name}/scenes/{scene_name}/perf_stats.json").resolve()

        # create directory, if necessary
        perf_stats_output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"script_dir: {script_dir}")
        print(f"perf_stats_output_path: {perf_stats_output_path}")

        with open(perf_stats_output_path, "w") as f:
            json.dump(perf_stats, f, indent=4)

        #%%
        # perf_stats_output_path: Path = Path(f"{script_dir}/nerf_data/outputs/tests/perf_stats.json").resolve()

        # Load cached performance stats if available
        if perf_stats_output_path.exists():
            with open(perf_stats_output_path, "r") as f:
                perf_stats = json.load(f)

        # Tabulate the data
        print(tabulate(perf_stats, headers="keys", tablefmt="grid"))

        #%%######################################################################################
        # ----------------------------------   VISTA Definition   ------------------------- #
        #########################################################################################
        # Take training camera origins and make a point cloud using Open3d
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(train_camera_origins.cpu().numpy())

        # Create a bounding box
        bounding_box = pointcloud.get_axis_aligned_bounding_box()
        far_clip = ( torch.tensor(bounding_box.get_max_bound()) - torch.tensor(bounding_box.get_min_bound()) ).norm().item()

        param_dict = {
            'discretizations': torch.tensor([100, 100, 100], device=device),
            # 'lower_bound': torch.tensor(bounding_box.get_min_bound(), device=device),
            # 'upper_bound': torch.tensor(bounding_box.get_max_bound(), device=device),
            'lower_bound': torch.tensor([-4., -4., -4.], device=device),
            'upper_bound': torch.tensor([4., 4., 4.], device=device),
            'voxel_grid_values': None,
            'voxel_grid_binary': None,
            'far_clip': far_clip,
            'val_ndim': 3,
        }
        cam_intrinsics["far_clip"] = far_clip
        cam_intrinsics["near_clip"] = 1.

        K = torch.tensor([
            [cam_intrinsics["fx"].item(), 0., cam_intrinsics["cx"].item()],
            [0., cam_intrinsics["fy"].item(), cam_intrinsics["cy"].item()],
            [0., 0., 1.]
        ], device=device)

        cam_intrinsics["K"] = K
        
        # option to save the rendered image at each selected pose
        save_render_selected_best_pose: bool = False

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
                
                cameras = {
                    "c2w": training_poses,
                    "K": K,
                    "far_clip": far_clip,
                    "near_clip": 0.,
                }

                pointcloud = method.generate_point_cloud()
                pointcloud["cameras"] = cameras
                # pointcloud = method.generate_RGBD_point_cloud(far_clip = far_clip, samples_per_view = 10000)
                vista = VistaCoverage(param_dict, pointcloud, 3, device)
                
                # pcd = o3d.geometry.PointCloud()
                # # pcd.points = o3d.utility.Vector3dVector(pointcloud['points'].cpu().numpy())
                # # pcd.colors = o3d.utility.Vector3dVector(pointcloud['values'][:, :3].cpu().numpy())
                # pcd.points = o3d.utility.Vector3dVector(method.pipeline.model.means.detach().cpu().numpy())
                # pcd.colors = o3d.utility.Vector3dVector(method.pipeline.model.features_dc.detach().cpu().numpy())

                # pcd_traj = o3d.geometry.PointCloud()
                # pcd_traj.points = o3d.utility.Vector3dVector(train_poses.cpu().numpy()[:, :3, -1])
                # pcd_traj.paint_uniform_color([1.0, 0.1, 0.1])

                # mesh = vista.create_mesh()
                # o3d.visualization.draw_geometries([mesh, pcd_traj])

                
                # Number of steps to dead-reckon the algorithm. Note that the methods don't update their underlying representation during this time, except for VISTA (the view diversity; not the point cloud)!
                for j in range(num_poses_to_add_per_augmentation_step):
                    # Find the k nearest neighbors to the root in the queryable set

                    _, nearest = queryable_kdtree.query(
                        root_pose[:3, -1].cpu().numpy().reshape(1, 3), k=num_nearest_neighbors_for_querying_uncertainty
                    )
                    candidate_poses = queryable_poses[nearest].squeeze().to(device)

                    # Evaluate the methods on these candidate poses.
                    best_pose, best_idx = evaluate_next_view_vista(
                        vista, candidate_poses, camera_info=cam_intrinsics, device=device
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
                            torch.equal(torch.tensor(frame["transform_matrix"]).to(device), pose)
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
                            "best_poses": torch.stack(best_poses_list[-5:], dim=0).tolist(),
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
                            "best_poses": torch.stack(best_poses_list[-5:], dim=0).tolist(),
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
        
if __name__ == "__main__":
    # method name
    method_name = "vista"
    
    # data directories
    all_data_dirs = [
        {
            "data_dir": "~/Research/NeRF_Data/lincoln_labs/kitchen",
        },
        {
            "data_dir": "~/Research/NeRF_Data/lincoln_labs/plane",
        },
        {
            "data_dir": "~/Research/NeRF_Data/lincoln_labs/poster",
        },
        {
            "data_dir": "~/Research/NeRF_Data/lincoln_labs/old_union",
        },
        {
            "data_dir": "~/Research/NeRF_Data/lincoln_labs/registered",
        },
        {
            "data_dir": "~/Research/NeRF_Data/lincoln_labs/flight",
        },
    ]
    
    # option to view pre-cached results
    view_cached_results: bool = False
    
    # run the evaluation
    for config in all_data_dirs:
        run_eval(
            data_dir=config["data_dir"],
            method_name=method_name,
            scene_name=Path(config["data_dir"]).stem,
            view_cached_results=view_cached_results,
        )
    
    