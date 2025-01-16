"""Process ReNe data into nerfstudio format

ReNe camera poses are +X right, +Y down, +Z forward

"""

# %%

import os
import yaml
import json
import shutil
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# %%


def pose_trace(pose):
    """Create a plotly trace to visualize a pose

    RGB vectors are used to represent the X, Y, Z axes of the rotation matrix

    Parameters
    ----------
    pose : tuple
        Pose tuple (R, t) where R is a 3x3 rotation matrix and t is a translation

    Returns
    -------
    traces : list
        List of plotly traces for displaying the pose

    """
    # Unpack pose into rotation matrix R and translation vector t
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Define arrow colors for each axis (RGB)
    colors = ["red", "green", "blue"]

    # Define the unit vectors from the columns of R
    axis_vectors = [R[:, 0], R[:, 1], R[:, 2]]

    # Create traces for each axis (X, Y, Z)
    traces = []
    for i, vec in enumerate(axis_vectors):
        arrow_start = t
        arrow_end = t + vec  # Arrow points in the direction of the column of R

        # Create an arrow trace for the axis
        trace = go.Scatter3d(
            x=[arrow_start[0], arrow_end[0]],
            y=[arrow_start[1], arrow_end[1]],
            z=[arrow_start[2], arrow_end[2]],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color=colors[i], width=5),
            showlegend=False,
        )
        traces.append(trace)

    return traces


def pose_traces(pose_list):
    """Create traces for a list of poses

    Parameters
    ----------
    pose_list : list of tuples
        List of poses, where each pose is a tuple (R, t)

    Returns
    -------
    all_traces : list
        List of plotly traces for displaying all poses

    """
    all_traces = []

    for pose in pose_list:
        traces = pose_trace(pose)
        all_traces.extend(traces)

    return all_traces


# %%

dataset_path = data_path = Path(os.path.expanduser("~/Data/rene_dataset"))
scene_name = "apple"

scene_path = dataset_path / scene_name

output_path = Path(os.path.expanduser("~/NeRF/nerfstudio/data/ShadowSplat/ReNe")) / scene_name

# Loop through folders in scene path
for folder in scene_path.iterdir():
    if not folder.is_dir():
        continue

    # Create folder called {scene_name}_{folder_name} in output_path
    output_folder = output_path / f"{scene_name}_{folder.name}/images"
    output_folder.mkdir(parents=True, exist_ok=True)
    transforms_dict = {}

    # Load the camera intrinsics
    with open(folder / "camera.yaml", "r") as f:
        data = yaml.safe_load(f)
        K = data["intrinsics"]["camera_matrix"]
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        transforms_dict["fl_x"] = fx
        transforms_dict["fl_y"] = fy
        transforms_dict["cx"] = cx
        transforms_dict["cy"] = cy
        dist_coeffs = data["intrinsics"]["dist_coeffs"]
        transforms_dict["k1"] = dist_coeffs[0]
        transforms_dict["k2"] = dist_coeffs[1]
        transforms_dict["p1"] = dist_coeffs[2]
        transforms_dict["p2"] = dist_coeffs[3]

    # Load light source pose
    light_pose = np.loadtxt(folder / "light.txt")

    # Load images and camera poses
    num_images = len(list((folder / "data").iterdir())) // 2
    frames = []

    poses = []
    for i in range(num_images):
        image_path = folder / "data" / f"{i:02d}_image.png"
        pose_path = folder / "data" / f"{i:02d}_pose.txt"

        # Load pose
        pose = np.loadtxt(pose_path)
        # Negate Y and Z axis
        pose[:, 1] *= -1
        pose[:, 2] *= -1
        poses.append(pose)

        frame = {
            "file_path": f"images/{i:02d}_image.png",
            "transform_matrix": pose.tolist(),
        }
        frames.append(frame)

        shutil.copy(image_path, output_folder / f"{i:02d}_image.png")

        # Save image, pose, and light pose
        print(image_path, pose, light_pose)

    fig = go.Figure(pose_traces(poses + [light_pose]))
    # fig = go.Figure(pose_traces([light_pose]))
    fig.update_layout(height=900, width=1600, scene=dict(aspectmode="data"))
    fig.show()
    
