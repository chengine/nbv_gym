"""
Generate training images from blender scene

Usage:
    blender <path_to_your_file.blend> --background --python datagen.py

pip
 - bpy
 - numpy

"""

import sys
import os
import bpy

# Add the directory containing datagen.py to sys.path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

from scripts.datagen.blender_util import get_camera_intrinsics  # noqa: E402


if __name__ == "__main__":
    # Get all light sources in the scene
    print("=== Light Sources ===")
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            light = obj.data
            print("Name:", obj.name)
            print("Type:", light.type)
            print("Color:", light.color)
            print("Energy:", light.energy)
            print("Location:", obj.location)
            print("Rotation:", obj.rotation_euler)
            if light.type == "SPOT":
                print("Spot Size:", light.spot_size)
                print("Spot Blend:", light.spot_blend)
            print("---------")

    scene = bpy.context.scene
    camera = scene.camera

    # Get camera extrinsics
    camera_extrinsics = get_camera_intrinsics(scene, camera)
    print(camera_extrinsics)

    # Move the camera
