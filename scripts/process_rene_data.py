"""Process ReNe data into nerfstudio format"""

# %%

import os
import yaml
import json
import numpy as np
from pathlib import Path

dataset_path = data_path = Path(os.path.expanduser("~/Data/rene_dataset"))
scene_name = "apple"

scene_path = dataset_path / scene_name

output_path = Path(os.path.expanduser("~/NeRF/nerfstudio/data/ShadowSplat/ReNe") / scene_name)

# Loop through folders in scene path
for folder in scene_path.iterdir():
    if not folder.is_dir():
        continue

    # Create folder called {scene_name}_{folder_name} in output_path
    output_folder = output_path / f"{scene_name}_{folder.name}"
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
    for i in range(num_images):
        image_path = folder / "data" / f"{i:02d}_image.png"
        pose_path = folder / "data" / f"{i:02d}_pose.txt"

        # Load pose
        pose = np.loadtxt(pose_path)

        # Save image, pose, and light pose
        print(image_path, pose, light_pose)
