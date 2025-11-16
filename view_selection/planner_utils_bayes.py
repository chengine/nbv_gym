import numpy as np
from scipy.spatial import KDTree
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

import torch

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
# from nerfstudio.configs.base_config import 
# from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerf_utils import *

import types

from bayesrays.scripts.uncertainty import ComputeUncertainty
from bayesrays.scripts.render_uncertainty import _render_trajectory_video

from bayesrays.scripts.output_uncertainty import get_output_fn, get_uncertainty

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn

import pdb


from sklearn.cluster import KMeans
import os
import cv2

def cluster_and_exclude(poses, n_clusters=3, exclude_ratio=0.4, random_state=42):
    """
    Splits poses so that approximately `exclude_ratio` of all poses
    in random clusters are excluded entirely.
    
    :param poses: np.ndarray of shape (N, D), representing camera poses in some D-dimensional space.
    :param n_clusters: How many clusters to create.
    :param exclude_ratio: Fraction of total samples to exclude in "missing" clusters.
    :param random_state: For reproducibility.
    :return: (train_indices, test_indices)
    """
    # Fit k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(poses)

    # Count how many poses in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Shuffle cluster labels to randomize which clusters get removed
    rng = np.random.default_rng(random_state)
    shuffled_clusters = rng.permutation(unique_labels)

    # Keep picking clusters to exclude until we hit the exclude_ratio
    total_samples = len(poses)
    excluded_set = set()
    cum_excluded = 0
    for c in shuffled_clusters:
        if cum_excluded / total_samples >= exclude_ratio:
            break
        # Mark all poses from this cluster as excluded
        cluster_indices = np.where(labels == c)[0]
        for idx in cluster_indices:
            excluded_set.add(idx)
        cum_excluded += len(cluster_indices)

    all_indices = set(range(total_samples))
    train_indices = np.array(list(all_indices - excluded_set))
    test_indices = np.array(list(excluded_set))

    return train_indices, test_indices

def evaluate_next_view(
    self,                             # Requires the VanillaPipeline instance here
    current_pose: np.ndarray,
    training_poses: List[np.ndarray], 
    rgb_weight: float,
    camera_info: Optional[dict] = None
):
    """
    Filter candidates based on k nearest neighbors
    from the current view, then call fisher_single_view on feasible views.
    """

    # current_cam = training_cams[current_view_idx]
    current_cam,_,_ = self.generate_output_cameras([current_pose], width = 640, height = 360)
    training_cams,_,_ = self.generate_output_cameras(training_poses, width = 640, height = 360)
    # pdb.set_trace()

    current_pose = current_cam[0].camera_to_worlds.cpu().numpy()
    print(f"shape of, type of, and current pose: {current_pose.shape} {type(current_pose)} {current_pose}")

    training_poses = [pose.numpy() for pose in training_poses]
    selected_view, acq_scores = self.pipeline.fisher_calc_from_views(
        training_cams,
        training_poses,
        rgb_weight,
        camera_info=camera_info
    )

    # selected_pose = feasible_candidates[selected_view]
    selected_pose = torch.tensor(training_poses[selected_view])
    print(f"Selected view: {selected_view}, Pose: {selected_pose}")
    return  selected_pose, selected_view, acq_scores

def get_feasible_candidates(
    current_pose: np.ndarray,
    candidate_views: List[np.ndarray],
    max_distance: float
) -> List[np.ndarray]:
    """
    Filters candidate views so that they are within a certain distance of the current view.
    """
    feasible = []
    for cand_pose in candidate_views:
        dist = pose_distance(current_pose, cand_pose)
        if dist <= max_distance:
            feasible.append(cand_pose)
    return feasible


def plan_next_view(
    self,                             # Requires the VanillaPipeline instance here
    current_view_idx: int,
    training_views: List[int], training_cams,
    candidate_views: List[np.ndarray], candidate_cams,
    rgb_weight: float,
    k_nearest_neighbors: int,
    # max_distance: float,
    camera_info: Optional[dict] = None
):
    """
    Filter candidates based on k nearest neighbors
    from the current view, then call fisher_single_view on feasible views.
    """

    # pdb.set_trace()
    current_cam = training_cams[current_view_idx]
    current_pose = current_cam.camera_to_worlds.cpu().numpy()
    print(f"shape of, type of, and current pose: {current_pose.shape} {type(current_pose)} {current_pose}")

    # Extract a list of poses from training_cams
    training_poses = [cam.camera_to_worlds.cpu().numpy() for cam in training_cams]
    # print(f"Size of training_poses: {len(training_poses)}")
    # print(f"Size of an individual element in training_poses: {training_poses[0].shape}")

    # camera_tree = build_kd_tree(training_poses)
    translations = np.array([pose[:, 3] for pose in training_poses])  # shape (N, 3)

    # 2. Create a KD-tree using the flattened poses.
    camera_tree = KDTree(translations)

    # Get the k-nearest neighbors to the current view
    distances, indices = camera_tree.query(current_pose[:,3], k=k_nearest_neighbors)
    nearest_poses = [training_poses[i] for i in indices]

    # print(f"current cam: {current_cam}")
    # print(f"training cams shape: {training_cams.shape}")
    # print(f"type training cams: {type(training_cams)}")

    selected_idx, acq_scores = self.pipeline.fisher_calc_from_views(
        training_cams,
        nearest_poses,
        rgb_weight,
        camera_info=camera_info
    )

    # selected_pose = feasible_candidates[selected_view]
    selected_pose = nearest_poses[selected_idx]
    print(f"Selected view: {selected_idx}, Pose: {selected_pose}")
    return selected_pose, selected_idx, acq_scores    

def evaluate_bayes_uncertainty_from_camera_path(    
    self,                             # Requires the VanillaPipeline instance here
    config_path,
    data_dir,
    unc_path,
    output_path,                
    training_poses: List[np.ndarray],
    camera_info: Optional[dict] = None
):
    """
    Filter candidates based on k nearest neighbors
    from the current view, then evaluate the uncertainty of the model using Bayes Rays.
    """

    width = camera_info["w"].item()
    height = camera_info["h"].item()
    fx = camera_info["fx"].item()
    fy = camera_info["fy"].item()
    cx = camera_info["cx"].item()
    cy = camera_info["cy"].item()

    training_cams,_,_ = self.generate_output_cameras(training_poses, w = width, h = height, fx = fx, fy = fy, cx = cx, cy = cy)

    # current_cam,_,_ = self.generate_output_cameras([current_pose], width = 640, height = 360)
    # training_cams,_,_ = self.generate_output_cameras(training_poses, width = 640, height = 360)
    # camera_names = [f"camera_{i}" for i in range(len(training_cams))]
    # pdb.set_trace()

    # current_pose = current_cam[0].camera_to_worlds.cpu().numpy()
    # print(f"shape of, type of, and current pose: {current_pose.shape} {type(current_pose)} {current_pose}")


    training_poses = [pose.numpy() for pose in training_poses]
    
    # pdb.set_trace()
    cameras_path = str(Path(output_path) / "cameras.json")
    cameras_list_to_json(cameras_path, training_cams)

    # imgs_output_path = str(Path(output_path) / "imgs")

    # pdb.set_trace()
    # for cam in training_cams:
    #     i = training_cams.index(cam)
    #     camera_output_path = str(Path(output_path) / f"camera_{i}")
    #     _render_trajectory_video(
    #         pipeline=self.pipeline,
    #         cameras=cam,
    #         output_filename=camera_output_path,
    #         rendered_output_names=["rgb"],
    #         rendered_resolution_scaling_factor = 1.0,
    #         seconds = 5.0,
    #         output_format = "images",
    #         image_format = "jpeg",
    #         jpeg_quality = 100,
    #         gt_visibility_path=None,
    #         colormap_options=None
    #     )

    render_uncertainty(
        data_dir=data_dir,
        config_path=config_path,
        unc_path=unc_path,
        cameras_path=cameras_path,
        output_path=output_path)
    
    # Read in the rendered images from the npy file stored at output_path
    # def load_images_from_folder(folder):
    folder = output_path
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        # return images

    rendered_images = images

    # Convert list of images to a numpy array for easier manipulation
    rendered_images_array = np.array(rendered_images)

    # Average the values across each image
    averaged_values = rendered_images_array.mean(axis=(1, 2, 3))
    
    # Find the pose with the lowest average uncertainty value
    min_index = np.argmin(averaged_values)
    selected_pose = torch.tensor(training_poses[min_index])

    print(f"Selected pose index: {min_index}, Pose: {selected_pose}")
    return selected_pose, min_index


def evaluate_bayes_uncertainty(
    nerf: NeRF,
    cameras: List[Cameras],
    hessian,
    reduce_mode: Literal["mean", "sum"] = "mean",
    config_path: Path = None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Compute BayesRays Uncertainty Metric
    """
    # init parameters
    filter_out: bool = False
    """Render with filtering"""
    filter_thresh: float = 0.5
    """filter threshold"""
    white_bg: bool = False
    """ Render empty space as white when filtering"""
    black_bg: bool = False
    """ Render empty space as black when filtering"""
    
    # # compute the uncertainty
    # hessian = ComputeUncertainty(load_config=config_path).main(
    #     nerf=nerf,
    #     save_outputs=False
    # )
    
    #dynamically change get_outputs method to include uncertainty
    nerf.pipeline.model.filter_out = filter_out
    nerf.pipeline.model.filter_thresh = filter_thresh
    nerf.pipeline.model.hessian = hessian
    nerf.pipeline.model.lod = np.log2(round(nerf.pipeline.model.hessian.shape[0]**(1/3))-1)
    nerf.pipeline.model.get_uncertainty = types.MethodType(get_uncertainty, nerf.pipeline.model)
    nerf.pipeline.model.white_bg = white_bg
    nerf.pipeline.model.black_bg = black_bg
    nerf.pipeline.model.N =  4096*1000  #approx ray dataset size (train batch size x number of query iterations in uncertainty extraction step)
    new_method = get_output_fn(nerf.pipeline.model)
    nerf.pipeline.model.get_outputs = types.MethodType(new_method, nerf.pipeline.model)
    
    # uncertainty metric per camera
    output_uncertainty = []
    
    # progress indicator
    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    
    # get the outputs at each camera
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            # AABB Box
            aabb_box = None
            
            # cam ray bundle
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, aabb_box=aabb_box)
                
            with torch.no_grad():
                outputs = nerf.pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                
            # compute the desired summary statistic
            if reduce_mode.lower() == "mean":
                output_uncertainty.append(outputs["uncertainty"].mean())
            elif reduce_mode.lower() == "sum":
                output_uncertainty.append(outputs["uncertainty"].sum())
            else:
                raise NotImplementedError(f"reduce_mode for the uncertainty quantification is not implemented!")
            
    # Find the pose with the maximum average uncertainty value
    max_index = np.argmax(output_uncertainty)
    selected_pose = torch.eye(4)
    selected_pose[:3] = torch.tensor(cameras.camera_to_worlds[max_index])

    # print(f"Selected pose index: {max_index}, Pose: {selected_pose}")
    return selected_pose, max_index
  