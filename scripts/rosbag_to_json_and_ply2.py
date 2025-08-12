from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.image import message_to_cvimage
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import open3d as o3d
import json
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

from itertools import permutations, product

def all_90deg_rotations():
    mats = []
    basis = np.eye(3)
    for perm in permutations([0,1,2]):           # all axis permutations
        for signs in product([-1,1], repeat=3):  # all combinations of axis flips
            mat = np.zeros((3,3))
            for i, p in enumerate(perm):
                mat[i, p] = signs[i]
            if np.linalg.det(mat) > 0.5:         # determinant +1, avoid reflections
                mats.append(mat)
    return mats

rotations = all_90deg_rotations()
print(f"Number of unique 90°-step rotation matrices: {len(rotations)}")

# User-provided camera intrinsics
RGB_INTRINSICS = {
    "height": 720,
    "width": 1280,
    "distortion_model": "plumb_bob",
    "D": [0.0, 0.0, 0.0, 0.0, 0.0],
    "K": [909.1692504882812, 0.0, 655.2548217773438, 0.0, 909.460693359375, 366.4031066894531, 0.0, 0.0, 1.0],
    "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "P": [909.1692504882812, 0.0, 655.2548217773438, 0.0, 0.0, 909.460693359375, 366.4031066894531, 0.0, 0.0, 0.0, 1.0, 0.0]
}

DEPTH_INTRINSICS = {
    "height": 480,
    "width": 848,
    "distortion_model": "Brown Congrady",
    "D": [0.0, 0.0, 0.0, 0.0, 0.0],
    "K": [427.18, 0., 426.96, 0., 427.18, 235.22, 0., 0., 1.]
}

# OpenCV camera to OpenGL camera transform
opencv_to_opengl = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=np.float32)

camera_to_mocap = np.array([
    [-1., 0., 0., 0.],
    [0., 0., -1., 0.],
    [0., -1., 0., 0.],
    [0., 0., 0., 1.]
], dtype=np.float32)

# k = 4

# camera_to_mocap = np.eye(4)
# camera_to_mocap[:3, :3] = rotations[k]
# print("Using camera_to_mocap rotation matrix:")
# print(camera_to_mocap[:3, :3])

# def scale_K(K, orig_shape, new_shape):
#     K = np.array(K).reshape(3, 3).copy()
#     scale_x = new_shape[1] / orig_shape[1]
#     scale_y = new_shape[0] / orig_shape[0]
#     K[0, 0] *= scale_x  # fx
#     K[1, 1] *= scale_y  # fy
#     K[0, 2] *= scale_x  # cx
#     K[1, 2] *= scale_y  # cy
#     return K

# orig_shape = (720, 1280)
# new_shape = (480, 848)

LIGHT_INTRINSICS = {}

# User-provided topic names
COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/depth/image_rect_raw'
CAM_POSE_TOPIC = '/vrpn_mocap/realsense_Tim/pose'
LIGHT_POSE_TOPIC = '/vrpn_mocap/lightsource_Tim/pose'

SAVE_FOLDER = "../realsense_bag"
Path(SAVE_FOLDER + '/images').mkdir(parents=True, exist_ok=True)
Path(SAVE_FOLDER + '/depth').mkdir(parents=True, exist_ok=True)

def pose_msg_to_matrix(msg):
    pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
    rot = R.from_quat(quat).as_matrix()
    tf = np.eye(4)
    tf[:3,:3] = rot
    tf[:3, -1] = pos
    return tf

def depth_to_points(depth_img, K, pose):
    K = np.array(K).reshape(3, 3)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    h, w = depth_img.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    valid = depth_img > 0
    z = depth_img[valid]
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    pts_cam = np.stack((x, y, z), axis=-1)
    pts_hom = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
    world_pts = (pose @ pts_hom.T).T[:, :3]
    return world_pts

# 1. Load messages from bag
typestore = get_typestore(Stores.LATEST)
bagname = SAVE_FOLDER  # Put your bag folder here if different
with Reader(bagname) as reader:
    # Print available topics/types
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # Buffers for messages
    color_msgs, color_ts = [], []
    depth_msgs, depth_ts = [], []
    cam_poses, cam_ts = [], []
    light_poses, light_ts = [], []

    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == COLOR_TOPIC:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            color_msgs.append(msg)
            color_ts.append(timestamp)
        elif connection.topic == DEPTH_TOPIC:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            depth_msgs.append(msg)
            depth_ts.append(timestamp)
        elif connection.topic == CAM_POSE_TOPIC:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            cam_poses.append(msg)
            cam_ts.append(timestamp)
        elif connection.topic == LIGHT_POSE_TOPIC:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            light_poses.append(msg)
            light_ts.append(timestamp)

# Set the minimum time difference (in nanoseconds for ROS bags, which are usually in nanoseconds)
desired_rate_hz = 2
min_dt = int(1e9 / desired_rate_hz)

# Only keep color frames at least min_dt apart
downsampled_indices = []
last_time = None
for i, t in enumerate(color_ts):
    if last_time is None or t - last_time >= min_dt:
        downsampled_indices.append(i)
        last_time = t

print(f"Selected {len(downsampled_indices)} out of {len(color_ts)} color frames at ~{desired_rate_hz} Hz.")

# 2. For each color image, find nearest depth, camera pose, and light pose. Skip frames if the timestamps are too far apart.
frames = []
pcd = o3d.geometry.PointCloud()

min_delta_t = 1e-1 * 1e9

for out_idx, idx in enumerate(tqdm(downsampled_indices, desc="Processing frames")):
    color_msg = color_msgs[idx]
    t_color = color_ts[idx]

    # Nearest depth
    i_depth = np.argmin(np.abs(np.array(depth_ts) - t_color))
    if np.abs(depth_ts[i_depth] - t_color) > min_delta_t:
        print(f"Skipping frame {out_idx} at index {idx} due to large time difference with depth frame.")
        continue
    depth_msg = depth_msgs[i_depth]

    # Nearest camera pose
    i_pose = np.argmin(np.abs(np.array(cam_ts) - t_color))
    if np.abs(cam_ts[i_pose] - t_color) > min_delta_t:
        print(f"Skipping frame {out_idx} at index {idx} due to large time difference with camera pose.")
        continue
    cam_pose = cam_poses[i_pose]

    # Nearest light pose
    i_light = np.argmin(np.abs(np.array(light_ts) - t_color))
    if np.abs(light_ts[i_light] - t_color) > min_delta_t:
        print(f"Skipping frame {out_idx} at index {idx} due to large time difference with light pose.")
        continue
    light_pose = light_poses[i_light]

    # Convert and save images
    color_np = message_to_cvimage(color_msg, 'bgr8')
    depth_np = message_to_cvimage(depth_msg)
    
    # If depth is in uint16, convert to meters
    if depth_np.dtype == np.uint16:
        depth_m = depth_np.astype(np.float32) * 0.001
    else:
        depth_m = depth_np.astype(np.float32)

    color_path = f'{SAVE_FOLDER}/images/rgb_{idx:04d}.png'
    depth_path = f'{SAVE_FOLDER}/depth/depth_{idx:04d}.png'
    cv2.imwrite(color_path, color_np)
    cv2.imwrite(depth_path, (depth_m)) # save as 16UC1

    # Camera and light poses
    cam_matrix_mocap = pose_msg_to_matrix(cam_pose)
    light_matrix_mocap = pose_msg_to_matrix(light_pose)

    cam_matrix = cam_matrix_mocap @ camera_to_mocap # @ opencv_to_opengl
    light_matrix = light_matrix_mocap # @ camera_to_mocap # @ opencv_to_opengl

    # Project depth to point cloud
    # points = depth_to_points(depth_m, RGB_INTRINSICS['K'], cam_matrix)

    # Valid depth mask: >0.5m and <6m, and depth not NaN/Inf
    valid_mask = (depth_m > 0.001) & (depth_m < 1.5) & np.isfinite(depth_m)

    # Get pixel coordinates of valid points
    xx, yy = np.meshgrid(np.arange(depth_m.shape[1]), np.arange(depth_m.shape[0]))
    xx = xx[valid_mask]
    yy = yy[valid_mask]

    z = depth_m[valid_mask]
    K = np.array(DEPTH_INTRINSICS["K"]).reshape(3, 3)
    uv = np.stack((xx, yy, np.ones_like(xx)), axis=-1)  # Shape (N, 2)

    # Backproject only valid points
    pts_cam = np.linalg.inv(K) @ uv.T * z
    world_pts = cam_matrix[:3, :3] @ pts_cam + cam_matrix[:3, -1:]
    world_pts = world_pts.T  # Shape (N, 3)

    # Take world pts and project into RGB image
    K_rgb = np.array(RGB_INTRINSICS['K']).reshape(3, 3)
    uv_rgb = K_rgb @ pts_cam
    uv_rgb = uv_rgb[:2, :] / uv_rgb[2, :]  # Normalize by z to get pixel coordinates
    uv_rgb = np.round(uv_rgb.T).astype(np.int32)  # Shape (N, 2)

    valid_rgb_mask = (uv_rgb[:, 0] >= 0) & (uv_rgb[:, 0] < RGB_INTRINSICS['width']) & (uv_rgb[:, 1] >= 0) & (uv_rgb[:, 1] < RGB_INTRINSICS['height'])
    uv_rgb = uv_rgb[valid_rgb_mask]  # Filter valid pixel coordinates
    pts_pixels = color_np[uv_rgb[:, 1], uv_rgb[:, 0]]  # OpenCV uses (y, x) indexing

    colors = np.zeros_like(pts_cam.T, dtype=np.float32)
    colors[valid_rgb_mask] = pts_pixels / 255.0  # Normalize to [0, 1]

    # Add to point cloud
    if world_pts.shape[0] > 0 and colors.shape[0] == world_pts.shape[0]:
        pcd.points.extend(o3d.utility.Vector3dVector(world_pts))
        pcd.colors.extend(o3d.utility.Vector3dVector(colors))

    # depth_m_normalized = (cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    # fig, ax = plt.subplots(2, figsize=(10, 10))
    # ax[0].imshow(color_for_pcd)
    # ax[0].set_title(f"Color Image {idx}")
    # ax[1].imshow(depth_m_normalized, cmap='gray')
    # ax[1].set_title(f"Depth Image {idx}")
    # plt.pause(1.)

    # # VIsualize point cloud
    pcd_per_frame = o3d.geometry.PointCloud()
    pcd_per_frame.points = o3d.utility.Vector3dVector(world_pts)
    pcd_per_frame.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd_per_frame], window_name=f"Point Cloud {idx}")

    # if out_idx % 10 == 0:
    #     print(f"Processed frame {out_idx} at index {idx}, point cloud size: {len(pcd.points)}")
    #     # Visualize the current point cloud
    #     o3d.visualization.draw_geometries([pcd_per_frame], window_name=f"Fused Point Cloud - Frame {out_idx}")

    frames.append({
        "file_path": f'images/rgb_{idx:04d}.png',
        "depth_path": f'depth/depth_{idx:04d}.png',
        "transform_matrix": cam_matrix.tolist(),
        "light_pose": light_matrix.tolist(),
        "bag_idx": idx
    })

print(f"Processed {len(frames)} frames.")
target_num_points = 300000

if len(pcd.points) > target_num_points:
    # Use random sampling for exact count (Open3D >= 0.15)
    import random
    indices = random.sample(range(len(pcd.points)), target_num_points)
    pcd = pcd.select_by_index(indices)
    print(f"Point cloud randomly downsampled to {target_num_points} points.")
else:
    print(f"Point cloud not downsampled, contains {len(pcd.points)} points.")

# 3. Save the fused point cloud
ply_file_path = f"{SAVE_FOLDER}/all.ply"

# VIsualize point cloud and visualize the camera frustums (OpenCV convention)
vis_objs = [pcd]
for i, frame in enumerate(frames):
    cam_matrix = np.array(frame["transform_matrix"]).reshape(4, 4)
    light_pose = np.array(frame["light_pose"]).reshape(4, 4)

    # Create camera frustum
    cam_frustum = o3d.geometry.LineSet.create_camera_visualization(
        intrinsic=o3d.camera.PinholeCameraIntrinsic(RGB_INTRINSICS['width'], RGB_INTRINSICS['height'], 
                                                     RGB_INTRINSICS['K'][0], RGB_INTRINSICS['K'][4], 
                                                     RGB_INTRINSICS['K'][2], RGB_INTRINSICS['K'][5]),
        extrinsic=cam_matrix, #  @ opencv_to_opengl,  # Convert to OpenGL convention
        scale=1.0
    )
    vis_objs.append(cam_frustum)
o3d.visualization.draw_geometries(vis_objs, window_name="Fused Point Cloud")

# Save point cloud
o3d.io.write_point_cloud(ply_file_path, pcd)

# 4. Write JSON metadata
out_json = {
    "rgb_intrinsics": RGB_INTRINSICS,
    "depth_intrinsics": DEPTH_INTRINSICS,
    "light_intrinsics": LIGHT_INTRINSICS,
    "ply_file_path": ply_file_path,
    "frames": frames
}
with open(f"{SAVE_FOLDER}/transforms.json", "w") as f:
    json.dump(out_json, f, indent=2)

print("Export complete. See:", SAVE_FOLDER)

