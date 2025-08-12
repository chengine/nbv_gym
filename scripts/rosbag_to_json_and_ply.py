import os
import cv2
import numpy as np
import open3d as o3d
import json
from tqdm import tqdm
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# ------ USER CONFIG ------
BAG_PATH = '../realsense_bag'  # e.g. '/path/to/my.bag'
OUT_DIR = '../processed_data'  # Output directory for images, depth maps, PLY, and JSON
PLY_FILE_PATH = os.path.join(OUT_DIR, "all.ply")
JSON_FILE_PATH = os.path.join(OUT_DIR, "transforms.json")

IMAGE_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
IMAGE_POSE_TOPIC = "/vrpn_mocap/realsense_Tim/pose"
LIGHT_POSE_TOPIC = "/vrpn_mocap/lightsource_Tim/pose"

# ---- Intrinsics ----
CAMERA_INTRINSICS = {
    "height": 720,
    "width": 1280,
    "distortion_model": "plumb_bob",
    "D": [0.0, 0.0, 0.0, 0.0, 0.0],
    "K": [909.1692504882812, 0.0, 655.2548217773438, 0.0, 909.460693359375, 366.4031066894531, 0.0, 0.0, 1.0],
    "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "P": [909.1692504882812, 0.0, 655.2548217773438, 0.0, 0.0, 909.460693359375, 366.4031066894531, 0.0, 0.0, 0.0, 1.0, 0.0]
}
DEPTH_INTRINSICS = CAMERA_INTRINSICS.copy()

# ---- Helper Functions ----
def get_pose_matrix(msg):
    # Accepts Pose or PoseStamped message and returns 4x4 numpy matrix
    if hasattr(msg, 'pose'):  # PoseStamped
        msg = msg.pose
    tx, ty, tz = msg.position.x, msg.position.y, msg.position.z
    qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
    R = quaternion_to_matrix([qx, qy, qz, qw])
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = [tx, ty, tz]
    return T.tolist()

def quaternion_to_matrix(q):
    # q = [x, y, z, w]
    x, y, z, w = q
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])
    return R

def depth_to_points(depth, K, T):
    # Backproject depth image (meters) to world points using intrinsics K and camera pose T
    K = np.array(K).reshape(3,3)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    x = (xx - cx) * depth / fx
    y = (yy - cy) * depth / fy
    z = depth
    pts_cam = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    pts_hom = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
    world_pts = (T @ pts_hom.T).T[:, :3]
    return world_pts

def extract_image(msg, encoding='bgr8'):
    # Converts sensor_msgs/msg/Image to numpy array
    import cv_bridge
    bridge = cv_bridge.CvBridge()
    return bridge.imgmsg_to_cv2(msg, encoding)

def extract_depth(msg):
    # Returns depth as float32 in meters if encoding is 16UC1 (usually mm)
    import cv_bridge
    bridge = cv_bridge.CvBridge()
    depth = bridge.imgmsg_to_cv2(msg)
    if msg.encoding == '16UC1':
        depth = depth.astype(np.float32) * 0.001
    return depth

# ---- Main Script ----
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    pcd = o3d.geometry.PointCloud()

    # -- Open bag2 --
    storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_types = {topic: get_message(typ) for topic, typ in topic_types.items()}

    # -- Buffer messages by topic --
    all_msgs = {IMAGE_TOPIC: [], DEPTH_TOPIC: [], IMAGE_POSE_TOPIC: [], LIGHT_POSE_TOPIC: []}
    print("Reading bag...")

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in all_msgs:
            msg = deserialize_message(data, msg_types[topic])
            all_msgs[topic].append((t, msg))

    # -- Timestamp synchronization (greedy nearest for each RGB frame) --
    print("Syncing and writing outputs...")
    for idx, (t_img, img_msg) in enumerate(tqdm(all_msgs[IMAGE_TOPIC])):
        # Find closest depth, cam pose, light pose
        t_depth, depth_msg = min(all_msgs[DEPTH_TOPIC], key=lambda x: abs(x[0] - t_img))
        t_pose, pose_msg = min(all_msgs[IMAGE_POSE_TOPIC], key=lambda x: abs(x[0] - t_img))
        t_light, light_msg = min(all_msgs[LIGHT_POSE_TOPIC], key=lambda x: abs(x[0] - t_img))

        # Extract/save images
        img_np = extract_image(img_msg, encoding='bgr8')
        depth_np = extract_depth(depth_msg)

        img_path = os.path.join(OUT_DIR, f"color_{idx:04d}.png")
        depth_path = os.path.join(OUT_DIR, f"depth_{idx:04d}.png")
        cv2.imwrite(img_path, img_np)
        cv2.imwrite(depth_path, (depth_np * 1000).astype(np.uint16))  # Save as 16UC1 mm

        # Poses
        cam_matrix = get_pose_matrix(pose_msg)
        light_matrix = get_pose_matrix(light_msg)

        # Point cloud fusion
        points = depth_to_points(depth_np, CAMERA_INTRINSICS["K"], np.array(cam_matrix))
        colors = img_np.reshape(-1, 3) / 255.0
        pcd.points.extend(o3d.utility.Vector3dVector(points))
        pcd.colors.extend(o3d.utility.Vector3dVector(colors))

        # Add frame to JSON
        frames.append({
            "file_path": img_path,
            "depth_path": depth_path,
            "transform_matrix": cam_matrix,
            "light_pose": light_matrix
        })

    # Write PLY file
    o3d.io.write_point_cloud(PLY_FILE_PATH, pcd)

    # Write JSON
    out_json = {
        "rgb_intrinsics": CAMERA_INTRINSICS,
        "depth_intrinsics": DEPTH_INTRINSICS,
        "light_intrinsics": {},  # Placeholder as requested
        "ply_file_path": PLY_FILE_PATH,
        "frames": frames
    }
    with open(JSON_FILE_PATH, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"Done. Outputs in {OUT_DIR}")

if __name__ == '__main__':
    main()
