import bpy
import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from shadow_splat.util.general import fibonacci_hemisphere_points
from shadow_splat.util.logger import Logger
from blender_util import set_camera_pose, get_camera_intrinsics

# =============== Parameters ===============
GENERATE_PLY = False
NUM_VIEWS = 100
RADIUS = 2.0
MIN_ELEVATION_RAD = np.deg2rad(5.0)
TARGET = (0, 0, 0.5)

BLENDER_FILE = "/home/addai/Blender/master_chief.blend"
OUTPUT_FOLDER = Path("/home/addai/NeRF/shadow_splat/data/master_chief_eevee")
IMG_FOLDER = OUTPUT_FOLDER / "images"

# =============== Main ===============
if __name__ == "__main__":
    # Create the output folders if they don't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(IMG_FOLDER):
        os.makedirs(IMG_FOLDER)

    bpy.ops.wm.open_mainfile(filepath=BLENDER_FILE)

    # Render settings
    bpy.context.scene.render.engine = "BLENDER_EEVEE"  # 'CYCLES' or 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format

    # Set active camera
    camera = bpy.context.scene.camera
    camera.data.sensor_width = 36
    camera.data.sensor_height = 36
    camera.data.lens = 35  # 35 mm (~52.5 deg FOV)

    camera_locations = fibonacci_hemisphere_points(
        NUM_VIEWS, radius=RADIUS, min_elevation_rad=MIN_ELEVATION_RAD
    )

    frames = []

    # bar = tqdm(total=NUM_VIEWS, position=0, leave=True)
    for i, location in Logger.tqdm(
        enumerate(camera_locations), total=NUM_VIEWS, desc="Rendering views"
    ):
        pose = set_camera_pose(camera, location, TARGET)
        # bpy.context.view_layer.update()

        # Render image
        file_path = os.path.join(IMG_FOLDER, f"view_{i:03d}.png")
        bpy.context.scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)

        frame_data = {"transform_matrix": pose.tolist(), "file_path": f"images/view_{i:03d}.png"}
        frames.append(frame_data)
    #     bar.update(1)

    # bar.close()

    # Compute intrinsic matrix
    intrinsics = get_camera_intrinsics(bpy.context.scene, camera)

    data = {}
    data.update(intrinsics)
    data["ply_file_path"] = "all_points.ply"
    data["frames"] = frames

    with open(os.path.join(OUTPUT_FOLDER, "transforms.json"), "w") as f:
        json.dump(data, f, indent=4)

    print("Rendering complete. Images and intrinsic matrices saved.")
