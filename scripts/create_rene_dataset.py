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


def get_number_at_end(string):
    number = ""
    for char in reversed(string):
        if char.isnumeric():
            number = char + number
        else:
            break
    return int(number) if number else 0


# %%

light_id_skips = [2, 21, 34]
camera_id_skips = [4, 8, 15, 25, 42, 47]

dataset_path = data_path = Path(os.path.expanduser("~/Datasets/rene_dataset"))

# Process each scene into a nerfstudio dataset
for k, scene in enumerate(dataset_path.iterdir()):
    scene_name = scene.name

    scene_path = dataset_path / scene_name

    transforms_dict = {}
    frames = []

    folders = []
    for folder in scene_path.iterdir():
        folders.append(folder)

    # Sort list
    folders.sort()

    # Loop through folders in scene path
    for j, folder in enumerate(folders):
        if not folder.is_dir():
            continue

        light_id = get_number_at_end(folder.name)

        if light_id in light_id_skips:
            continue

        # NOTE: We assume the entire dataset for that object has the same camera calibration parameters

        # Create folder called {scene_name}_{folder_name} in output_path
        image_path = scene_path / f"{folder.name}/data"

        # Take the first dataset as the reference for camera intrinsics
        if light_id == 0:
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
                transforms_dict["k3"] = dist_coeffs[4]
                transforms_dict["w"], transforms_dict["h"] = data["intrinsics"]["image_size"]
                transforms_dict["camera_model"] = "OPENCV"

            # OPTIONAL. Uncomment if there are point clouds in dataset.
            # transforms_dict["applied_transform"] = ...
            # transforms_dict["ply_file_path"] = ...

        # Load light source pose
        light_pose = np.loadtxt(folder / "light.txt")
        # Negate Y and Z axis
        light_pose[:, 1] *= -1
        light_pose[:, 2] *= -1

        # Load images and camera poses
        num_images = len(list((folder / "data").iterdir())) // 2

        poses = []
        for i in range(num_images):
            if i in camera_id_skips:
                continue

            image_path = os.path.join(folder.name, "data", f"{i:02d}_image.png")
            pose_path = folder / "data" / f"{i:02d}_pose.txt"

            # Load pose
            pose = np.loadtxt(pose_path)
            # Negate Y and Z axis
            pose[:, 1] *= -1
            pose[:, 2] *= -1
            poses.append(pose)

            frame = {
                "file_path": image_path,
                "transform_matrix": pose.tolist(),
                "light_pose": light_pose.tolist(),
                "camera_id": i,
                "light_id": light_id,
            }
            frames.append(frame)

            # shutil.copy(image_path, output_folder / f"{i:02d}_image.png")

            # Save image, pose, and light pose
            # print(image_path, pose, light_pose)
            print("Scene:", scene_name, "Light id:", j, "Camera id:", i)

        # fig = go.Figure(pose_traces(poses + [light_pose]))
        # # fig = go.Figure(pose_traces([light_pose]))
        # fig.update_layout(height=900, width=1600, scene=dict(aspectmode="data"))
        # fig.show()

    transforms_dict["frames"] = frames

    # NOTE: guessing light intrinsics for now
    W, H = 300, 300
    transforms_dict["light_intrinsics"] = {
        "w": W,
        "h": H,
        "fl_x": 450.0,
        "fl_y": 450.0,
        "cx": W / 2.0,
        "cy": H / 2.0,
    }

    with open(scene_path / "transforms.json", "w") as f:
        json.dump(transforms_dict, f, indent=4)
