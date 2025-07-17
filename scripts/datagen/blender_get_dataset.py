import bpy
import os
import numpy as np
import json
from pathlib import Path
from mathutils import Matrix
import argparse

from shadow_splat.util.general import fibonacci_hemisphere_points
from shadow_splat.util.logger import Logger
from blender_util import get_camera_intrinsics, look_at_blender, set_sun_direction

# =============== Parameters ===============
GENERATE_PLY = False
NUM_VIEWS = 100
RADIUS = 1.75
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
OUTPUT_FOLDER = Path("/home/addai/NeRF/shadow_splat/data/master_chief_multi_light")
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
    sun = bpy.data.objects["Sun"]
    light_pose = np.array(sun.matrix_world)
    W, H = 3000, 3000
    focal_length = 1650
    light_intrinsics = {
        "w": W,
        "h": H,
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": W / 2.0,
        "cy": H / 2.0,
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

    # Set lighting
    for azimuth_deg in [45, 135, 225, 315]:
        set_sun_direction(sun, azimuth_deg=azimuth_deg, elevation_deg=-45)
        light_pose = np.array(sun.matrix_world)

        for i, location in Logger.tqdm(
            enumerate(camera_locations),
            total=NUM_VIEWS,
            desc=f"Rendering views, azimuth: {azimuth_deg}°",
        ):
            # Set camera
            pose = look_at_blender(location, TARGET)
            camera.matrix_world = Matrix(pose)

            # Render image
            img_name = f"view_{i:03d}_light_{azimuth_deg}.png"
            file_path = os.path.join(IMG_FOLDER, img_name)
            bpy.context.scene.render.filepath = file_path
            bpy.ops.render.render(write_still=True)

            frame_data = {
                "transform_matrix": pose.tolist(),
                "file_path": f"images/{img_name}",
            }
            frame_data["light_pose"] = light_pose.tolist()
            frames.append(frame_data)

    data = {}
    data.update(intrinsics)
    data["light_intrinsics"] = light_intrinsics
    data["ply_file_path"] = "../models/master_chief_all.ply"
    data["frames"] = frames

    with open(os.path.join(OUTPUT_FOLDER, "transforms.json"), "w") as f:
        json.dump(data, f, indent=4)

    print("Rendering complete. Images and intrinsic matrices saved.")
