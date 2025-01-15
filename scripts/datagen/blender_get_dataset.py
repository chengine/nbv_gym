import bpy
import math
import os
import numpy as np
from mathutils import Vector, Matrix
import json

# Parameters
num_cameras = 10  # Number of equally spaced camera positions
R = 16      # Radius of the sphere
target = (0, 0, 0)  # Point the cameras should look at
output_folder = bpy.path.abspath('//') + "moon/images"  # Path to save rendered images
out_folder = bpy.path.abspath('//') + "moon"  # Path to save intrinsic matrices

z_cutoff = 1.

# Create the output folders if they don't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

def fibonacci_sphere(samples=1000, z_cutoff=None):
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        x, y, z = R*x, R*y, R*z

        if z_cutoff is not None:
            if z >= z_cutoff:
                points.append((x, y, z))
        else:
            points.append((x, y, z))

    return points

# Function to set camera location and orientation
def set_camera_pose(cam, location, target):
    direction = (target[0] - location[0], target[1] - location[1], target[2] - location[2])
    direction = np.array(direction) # This is -z
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

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    scene = bpy.context.scene
#    camd.lens = f_in_mm
#    sensor_width_in_mm = K[1,1]*K[0,2] / (K[0,0]*K[1,2])
#    sensor_height_in_mm = 1  # doesn't matter
#    resolution_x_in_px = K[0,2]*2  # principal point assumed at the center
#    resolution_y_in_px = K[1,2]*2  # principal point assumed at the center

#    s_u = resolution_x_in_px / sensor_width_in_mm
#    s_v = resolution_y_in_px / sensor_height_in_mm

    f_in_mm = camd.lens
    s_u = scene.render.resolution_x / camd.sensor_width
    s_v = scene.render.resolution_y / camd.sensor_height

    K = np.eye(3)
    K[0, 0] = f_in_mm * s_u
    K[0, 2] = scene.render.resolution_x / 2
    K[1, 2] = scene.render.resolution_y / 2
    K[1, 1] = f_in_mm * s_v
    
    return K


# Render settings
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.image_settings.file_format

# Set active camera
camera = bpy.context.scene.camera
# Create and render cameras

# Generate all camera locations distributed on sphere
possible_locations = fibonacci_sphere(num_cameras, z_cutoff = z_cutoff)
    
    
frames = []
    
for i, location in enumerate(possible_locations):
    pose = set_camera_pose(camera, location, target)
    bpy.context.view_layer.update()

    # Render image
    file_path = os.path.join(output_folder, f"camera_{i}.png")
    bpy.context.scene.render.filepath = file_path
    bpy.ops.render.render(write_still=True)
    
    frame_data = {
                "transform_matrix": pose.tolist(),
                "file_path": f"images/camera_{i}.png"
                }
    frames.append(frame_data)
        
# Compute intrinsic matrix
intrinsic_matrix = get_calibration_matrix_K_from_blender(camera.data)

data = {
"w": bpy.context.scene.render.resolution_x,
"h": bpy.context.scene.render.resolution_y,
"fl_x": intrinsic_matrix[0, 0],
"fl_y": intrinsic_matrix[1, 1],
"cx": intrinsic_matrix[0, 2],
"cy": intrinsic_matrix[1, 2],
"camera_model": "OPENCV",
"frames": frames
}

with open(os.path.join(out_folder, "transforms.json"), "w") as f:
    json.dump(data, f, indent=4)

print("Rendering complete. Images and intrinsic matrices saved.")
