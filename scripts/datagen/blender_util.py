"""Utilities for blender data generation

References:
[1] https://github.com/maximeraafat/BlenderNeRF

"""

import bpy
import numpy as np
from mathutils import Matrix, Vector
import open3d as o3d


def set_sun_direction(sun, azimuth_deg, elevation_deg):
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    direction = Vector((x, y, z))
    rot_quat = direction.to_track_quat("-Z", "Y")
    sun.rotation_euler = rot_quat.to_euler()


def set_sun_direction_and_location(sun, azimuth_deg, elevation_deg, R):
    # 1. Compute light direction (from sun toward origin)
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    direction = Vector((x, y, z))

    # 2. Set rotation so sun points toward -Z
    rot_quat = direction.to_track_quat("-Z", "Y")
    sun.rotation_euler = rot_quat.to_euler()

    # 3. Set location: put sun "behind" the origin, distance R away, so the light points toward origin
    sun.location = -R * direction.normalized()


def blender_mesh_to_open3d(obj_name: str) -> o3d.geometry.TriangleMesh:
    # Get the Blender object
    obj = bpy.data.objects[obj_name]
    mesh = obj.to_mesh()
    mesh.calc_loop_triangles()

    # Transform vertices to world space
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])

    # Collect triangle indices
    triangles = []
    for tri in mesh.loop_triangles:
        triangles.append([tri.vertices[0], tri.vertices[1], tri.vertices[2]])
    triangles = np.array(triangles)

    # Create Open3D triangle mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.compute_vertex_normals()

    return o3d_mesh


# Function to set camera location and orientation
def set_camera_pose(cam, location, target):
    direction = (target[0] - location[0], target[1] - location[1], target[2] - location[2])
    direction = np.array(direction)  # This is -z
    direction = direction / np.linalg.norm(direction)

    up = np.array([0, 0, 1])

    right = np.cross(up, -direction)
    up = np.cross(-direction, right)

    matrix_world = np.eye(4)
    rotation = np.stack([right, up, -direction], axis=-1)
    translation = location
    matrix_world[:3, :3] = rotation
    matrix_world[:3, -1] = translation
    cam.matrix_world = Matrix(matrix_world)

    return matrix_world


def look_at_blender(camera_pos, target, up=(0, 0, 1)):
    camera_pos = np.array(camera_pos, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up_corrected = np.cross(right, forward)
    up_corrected = up_corrected / np.linalg.norm(up_corrected)

    rot = np.stack([right, up_corrected, -forward], axis=1)
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = camera_pos

    return mat


# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == "VERTICAL":
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == "AUTO":
        if size_x >= size_y:
            return "HORIZONTAL"
        else:
            return "VERTICAL"
    return sensor_fit


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    scene = bpy.context.scene
    f_in_mm = camd.lens
    s_u = scene.render.resolution_x / camd.sensor_width
    s_v = scene.render.resolution_y / camd.sensor_height

    K = np.eye(3)
    K[0, 0] = f_in_mm * s_u
    K[0, 2] = scene.render.resolution_x / 2
    K[1, 2] = scene.render.resolution_y / 2
    K[1, 1] = f_in_mm * s_v

    return K


# Taken from [1]
def get_camera_intrinsics(scene, camera):
    camera_angle_x = camera.data.angle_x
    camera_angle_y = camera.data.angle_y

    # camera properties
    f_in_mm = camera.data.lens  # focal length in mm
    scale = scene.render.resolution_percentage / 100
    width_res_in_px = scene.render.resolution_x * scale  # width
    height_res_in_px = scene.render.resolution_y * scale  # height
    optical_center_x = width_res_in_px / 2
    optical_center_y = height_res_in_px / 2

    # pixel aspect ratios
    size_x = scene.render.pixel_aspect_x * width_res_in_px
    size_y = scene.render.pixel_aspect_y * height_res_in_px
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # sensor fit and sensor size (and camera angle swap in specific cases)
    if camera.data.sensor_fit == "AUTO":
        sensor_size_in_mm = (
            camera.data.sensor_height
            if width_res_in_px < height_res_in_px
            else camera.data.sensor_width
        )
        if width_res_in_px < height_res_in_px:
            sensor_fit = "VERTICAL"
            camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x
        elif width_res_in_px > height_res_in_px:
            sensor_fit = "HORIZONTAL"
        else:
            sensor_fit = "VERTICAL" if size_x <= size_y else "HORIZONTAL"

    else:
        sensor_fit = camera.data.sensor_fit
        if sensor_fit == "VERTICAL":
            sensor_size_in_mm = (
                camera.data.sensor_height
                if width_res_in_px <= height_res_in_px
                else camera.data.sensor_width
            )
            if width_res_in_px <= height_res_in_px:
                camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x

    # focal length for horizontal sensor fit
    if sensor_fit == "HORIZONTAL":
        sensor_size_in_mm = camera.data.sensor_width
        s_u = f_in_mm / sensor_size_in_mm * width_res_in_px
        s_v = f_in_mm / sensor_size_in_mm * width_res_in_px * pixel_aspect_ratio

    # focal length for vertical sensor fit
    if sensor_fit == "VERTICAL":
        s_u = f_in_mm / sensor_size_in_mm * width_res_in_px / pixel_aspect_ratio
        s_v = f_in_mm / sensor_size_in_mm * width_res_in_px

    camera_intr_dict = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": s_u,
        "fl_y": s_v,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": optical_center_x,
        "cy": optical_center_y,
        "w": width_res_in_px,
        "h": height_res_in_px,
    }
    return camera_intr_dict
