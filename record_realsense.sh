#!/bin/bash

# Usage: ./record_realsense_images.sh <output_bag_name>
# Example: ./record_realsense_images.sh my_realsense_bag

BAG_NAME=${1:-realsense_bag}
IMAGE_TOPIC="/camera/camera/color/image_raw"
DEPTH_TOPIC="/camera/camera/depth/image_rect_raw"
IMAGE_POSE_TOPIC="/vrpn_mocap/realsense_Tim/pose"
LIGHT_POSE_TOPIC="/vrpn_mocap/lightsource_Tim/pose"

echo "Recording topics:"
echo "  $IMAGE_TOPIC"
echo "  $DEPTH_TOPIC"
echo "  $IMAGE_POSE_TOPIC"
echo "  $LIGHT_POSE_TOPIC"
echo "Output bag: $BAG_NAME"

ros2 bag record -o "$BAG_NAME" "$IMAGE_TOPIC" "$DEPTH_TOPIC" "$IMAGE_POSE_TOPIC" "$LIGHT_POSE_TOPIC"
