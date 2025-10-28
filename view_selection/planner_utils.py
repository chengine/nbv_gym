import numpy as np
from typing import List, Optional
import torch
from nerf_utils import *

# =========================================
# Fisher-RF coverage-based candidate scorer
# =========================================
def evaluate_next_view_fisher(
    nerf,                             
    training_poses: List[np.ndarray], 
    rgb_weight: float,
    camera_info: Optional[dict] = None
):
    """
    Filter candidates based on k nearest neighbors
    from the current view, then call fisher_single_view on feasible views.
    """

    width = camera_info["w"].item()
    height = camera_info["h"].item()
    fx = camera_info["fx"].item()
    fy = camera_info["fy"].item()
    cx = camera_info["cx"].item()
    cy = camera_info["cy"].item()

    training_cams,_,_ = nerf.generate_output_cameras(training_poses, w = width, h = height, fx = fx, fy = fy, cx = cx, cy = cy)

    training_poses = [pose.numpy() for pose in training_poses]
    selected_view, acq_scores = nerf.pipeline.fisher_calc_from_views(
        training_cams,
        training_poses,
        rgb_weight,
        camera_info=camera_info
    )

    selected_pose = torch.tensor(training_poses[selected_view])
    return  selected_pose, selected_view, acq_scores

# =========================================
# VISTA coverage-based candidate scorer
# =========================================
def evaluate_next_view_vista(
    vista,                             # VistaCoverage object here
    training_poses: List[np.ndarray], 
    camera_info: Optional[dict] = None,
):
    """
    Filter candidates based on k nearest neighbors
    from the current view, then evaluate VISTA on feasible views.
    """
    # move to the appropriate device
    # Create point cloud
    images, _, _ = vista.camera_voxel_intersection(camera_info["K"], training_poses, camera_info["far_clip"], camera_info["near_clip"])
    # Get the coverage score
    coverage_images = images[..., -1]

    # sum over all values except the first dimension
    coverage_score = coverage_images.sum(dim=tuple(range(1, len(coverage_images.shape))))

    if compute_semantics:
        # Compute the semantic score
        semantic_images = images[..., -2]
        semantic_score = semantic_images.sum(dim=tuple(range(1, len(semantic_images.shape))))
        total_score = coverage_score + semantic_score
    else:
        total_score = coverage_score

    # best index is the one with the highest coverage score
    selected_view = total_score.argmax().item()

    # selected_pose = feasible_candidates[selected_view]
    selected_pose = training_poses[selected_view]
    print(f"Selected view: {selected_view}, Pose: {selected_pose}")
    print("Scores: ", total_score)
    return  selected_pose, selected_view


# =========================================
# 3DGS coverage-based candidate scorer
# =========================================
def evaluate_next_view_3dgs_coverage(
    nerf,                             # NeRF wrapper (your class from nerf_utils.py)
    training_poses: List[np.ndarray] | torch.Tensor,
    num_to_return: int = 1,
    camera_info: Optional[dict] = None
):
    """
    Score each candidate pose by summing the per-pixel 3DGS coverage image rendered at that pose.
    Returns (best_poses, best_indices, scores) to mirror your Fisher selector.
    """
    # Normalize input list -> torch tensor of poses (N,4,4) or (N,3,4)
    if isinstance(training_poses, torch.Tensor):
        poses_t = training_poses
    else:
        poses_t = torch.stack([torch.as_tensor(p) for p in training_poses], dim=0)

    # Pull intrinsics
    width  = int(camera_info["w"].item())
    height = int(camera_info["h"].item())
    fx = float(camera_info["fx"].item())
    fy = float(camera_info["fy"].item())
    cx = float(camera_info["cx"].item())
    cy = float(camera_info["cy"].item())
    cam_type = camera_info.get("cam_camera_type", torch.tensor([[1]]))

    # Build Cameras for candidates (batch object)
    candidate_cams, _, _ = nerf.generate_output_cameras(
        poses_t, w=width, h=height, fx=fx, fy=fy, cx=cx, cy=cy, cam_camera_type=cam_type
    )

    # Nerfstudio Cameras can be iterated or indexed as needed.
    # Sum coverage per candidate
    scores = []
    # If candidate_cams is a batched Cameras, iterate over views using slicing
    for idx in range(poses_t.shape[0]):
        cam_i = candidate_cams[idx:idx+1]
        s = nerf.coverage_score_for_camera(cam_i)  # scalar tensor
        scores.append(s)

    scores = torch.stack(scores).detach().cpu()  # [N]
    k = min(num_to_return, scores.numel())
    top_vals, top_idx = torch.topk(scores, k=k, largest=True, sorted=True)

    # Return in the same shape/type as your Fisher call
    selected_poses = [poses_t[i] for i in top_idx.tolist()]
    selected_views = top_idx.tolist()
    acq_scores = top_vals.tolist()
    return selected_poses, selected_views, acq_scores