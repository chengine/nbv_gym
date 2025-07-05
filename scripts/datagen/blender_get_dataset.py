import bpy
import os
import numpy as np
import json
from pathlib import Path
from mathutils import Matrix

from shadow_splat.util.general import fibonacci_hemisphere_points
from shadow_splat.util.logger import Logger
from blender_util import set_camera_pose, get_camera_intrinsics, look_at_blender

# =============== Parameters ===============
GENERATE_PLY = False
NUM_VIEWS = 100
RADIUS = 1.75
MIN_ELEVATION_RAD = np.deg2rad(5.0)
TARGET = (0, 0, 0.3)
RENDER_DEPTH = True

RENDERER = "cycles"
BLENDER_FILE = "/home/addai/Blender/master_chief.blend"
OUTPUT_FOLDER = Path(f"/home/addai/NeRF/shadow_splat/data/master_chief_{RENDERER}")
IMG_FOLDER = OUTPUT_FOLDER / "images"

# =============== Main ===============
if __name__ == "__main__":
    # Create the output folders if they don't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(IMG_FOLDER):
        os.makedirs(IMG_FOLDER)

    # Load the scene
    bpy.ops.wm.open_mainfile(filepath=BLENDER_FILE)
    scene = bpy.context.scene

    if RENDER_DEPTH:
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()

    # Render settings
    if RENDERER == "eevee":
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
    elif RENDERER == "cycles":
        bpy.context.scene.render.engine = "CYCLES"
    else:
        raise ValueError(f"Invalid renderer: {RENDERER}")

    # bpy.context.scene.render.resolution_x = 1024
    # bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.film_transparent = True
    # bpy.context.scene.render.resolution_percentage = 100

    # Hide the plane
    # bpy.data.objects["Plane"].hide_render = True

    # Lights (currently only handle single Sun light)
    light_obj = bpy.data.objects["Sun"]
    light_pose = np.array(light_obj.matrix_world)
    light_intrinsics = {
        "w": 2000,
        "h": 2000,
        "fx": 1650,
        "fy": 1650,
        "cx": 1000.0,
        "cy": 1000.0,
    }

    # Set active camera
    camera = bpy.context.scene.camera
    # camera.data.sensor_width = 36
    # camera.data.sensor_height = 36
    camera.data.lens = 35  # 35 mm (~52.5 deg FOV)

    # Compute intrinsic matrix
    intrinsics = get_camera_intrinsics(bpy.context.scene, camera)

    camera_locations = fibonacci_hemisphere_points(
        NUM_VIEWS, radius=RADIUS, min_elevation_rad=MIN_ELEVATION_RAD
    )

    frames = []

    for i, location in Logger.tqdm(
        enumerate(camera_locations), total=NUM_VIEWS, desc="Rendering views"
    ):
        # pose = set_camera_pose(camera, location, TARGET)
        pose = look_at_blender(location, TARGET)
        camera.matrix_world = Matrix(pose)

        # Render image
        file_path = os.path.join(IMG_FOLDER, f"view_{i:03d}.png")
        bpy.context.scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)

        frame_data = {"transform_matrix": pose.tolist(), "file_path": f"images/view_{i:03d}.png"}
        frame_data["light_pose"] = light_pose.tolist()
        frames.append(frame_data)

    data = {}
    data.update(intrinsics)
    data["ply_file_path"] = "../models/master_chief_all.ply"
    data["frames"] = frames

    with open(os.path.join(OUTPUT_FOLDER, "transforms.json"), "w") as f:
        json.dump(data, f, indent=4)

    print("Rendering complete. Images and intrinsic matrices saved.")
