import bpy
import os
import numpy as np
import json
from pathlib import Path
import argparse
import time

from shadow_splat.util.general import fibonacci_hemisphere_points
from shadow_splat.util.logger import Logger
from blender_util import (
    look_at_blender,
)
from blender import BlenderScene

# =============== Parameters ===============
GENERATE_PLY = False
NUM_VIEWS = 100
RADIUS = 1.5
MIN_ELEVATION_RAD = np.deg2rad(5.0)
TARGET = (0, 0, 0.3)
RENDER_DEPTH = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "--renderer",
    type=str,
    default="cycles",
    choices=["cycles", "eevee"],
    help="Blender render engine to use",
)
args = parser.parse_args()
RENDERER = args.renderer
BLENDER_FILE = "/home/addai/Blender/master_chief.blend"
# BLENDER_FILE = "/home/addai/Blender/Perseverance_Rover.blend"
OUTPUT_FOLDER = Path("/home/addai/NeRF/shadow_splat/data/spotlight_test")
# OUTPUT_FOLDER = Path("/home/addai/NeRF/shadow_splat/data/perseverance")
IMG_FOLDER = OUTPUT_FOLDER / "images"

# =============== Main ===============
if __name__ == "__main__":
    # Create the output folders if they don't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(IMG_FOLDER):
        os.makedirs(IMG_FOLDER)

    scene = BlenderScene(BLENDER_FILE)

    light = bpy.data.objects["Spot"]
    light_pose = np.array(light.matrix_world)
    W, H = 5000, 5000
    focal_length = 5000.0
    light_intrinsics = {
        "w": W,
        "h": H,
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": W / 2.0,
        "cy": H / 2.0,
    }
    print("light pose: ", light_pose)

    # Rigid offset as a 4x4 homogeneous transform placing the light directly
    # "above" the camera (camera's local +Y)
    LIGHT_CAMERA_OFFSET_M = 0.1  # meters
    LIGHT_CAMERA_OFFSET = np.eye(4)
    LIGHT_CAMERA_OFFSET[:3, 3] = np.array([0.0, LIGHT_CAMERA_OFFSET_M, 0.0])

    intrinsics = scene.get_camera_intrinsics()

    camera_locations = fibonacci_hemisphere_points(
        NUM_VIEWS, radius=RADIUS, min_elevation_rad=MIN_ELEVATION_RAD
    )

    frames = []

    start_time = time.time()

    for i, location in Logger.tqdm(
        enumerate(camera_locations),
        total=NUM_VIEWS,
        desc=f"Rendering {NUM_VIEWS} views",
    ):
        # Set camera
        cam_pose = look_at_blender(location, TARGET)
        scene.set_camera_pose(cam_pose)
        # Compute light pose = camera pose * fixed 4x4 offset
        light_pose = cam_pose @ LIGHT_CAMERA_OFFSET
        scene.set_light_pose(light_pose)

        # Render image
        img_name = f"view_{i:03d}.png"
        file_path = os.path.join(IMG_FOLDER, img_name)
        scene.render(file_path)

        frame_data = {
            "transform_matrix": cam_pose.tolist(),
            "file_path": f"images/{img_name}",
        }
        frame_data["light_pose"] = light_pose.tolist()
        frames.append(frame_data)

    data = {}
    data.update(intrinsics)
    data["light_intrinsics"] = light_intrinsics
    data["ply_file_path"] = "../models/rover_all.ply"
    data["frames"] = frames

    with open(os.path.join(OUTPUT_FOLDER, "transforms.json"), "w") as f:
        json.dump(data, f, indent=4)

    print("Rendering complete. Images and intrinsic matrices saved.")
    print(f"Time taken: {time.time() - start_time} seconds")
