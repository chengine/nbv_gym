#!/usr/bin/env python3
"""
Camera Optimizer Test Script

This script demonstrates how to use the nerfstudio CameraOptimizer to optimize camera poses
on a simple scene. It creates a synthetic scene with colored 3D points, generates camera
poses with intentional noise, and then optimizes them to match ground truth poses.

Usage:
    python scripts/test_camera_optimizer.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
import torch.optim as optim
import argparse
import os


def create_simple_scene():
    """Create a simple 3D scene with colored points."""
    # Create a cube of points
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    z = np.linspace(-1, 1, 4)

    points = []
    colors = []

    for i in x:
        for j in y:
            for k in z:
                points.append([i, j, k])
                # Color based on position
                colors.append([(i + 1) / 2, (j + 1) / 2, (k + 1) / 2])

    points = torch.tensor(points, dtype=torch.float32)
    colors = torch.tensor(colors, dtype=torch.float32)

    return points, colors


def create_camera_poses(num_cameras=8, radius=3.0, noise_level=0.1):
    """Create camera poses in a circle around the origin with some noise."""

    # Create cameras in a circle
    angles = np.linspace(0, 2 * np.pi, num_cameras, endpoint=False)

    camera_to_worlds = []

    for angle in angles:
        # Position
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.5  # Slightly elevated

        # Look at origin
        forward = np.array([-x, -y, -z])
        forward = forward / np.linalg.norm(forward)

        # Create rotation matrix
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        up = np.cross(right, forward)

        R = np.stack([right, up, forward], axis=1)
        t = np.array([x, y, z])

        # Add noise to pose
        noise_R = np.random.normal(0, noise_level, (3, 3))
        noise_t = np.random.normal(0, noise_level, 3)

        R_noisy = R + noise_R
        t_noisy = t + noise_t

        # Ensure R is still approximately orthogonal
        U, _, Vt = np.linalg.svd(R_noisy)
        R_clean = U @ Vt

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R_clean
        transform[:3, 3] = t_noisy

        camera_to_worlds.append(transform)

    return torch.tensor(np.array(camera_to_worlds), dtype=torch.float32)


def simple_render(camera, points, colors, point_size=3):
    """Simple rendering function that projects 3D points to 2D."""

    # Get camera parameters
    c2w = camera.camera_to_worlds
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy
    width, height = camera.width, camera.height

    # Transform points to camera space
    # Handle both 3x4 and 4x4 camera matrices
    if c2w.shape[-2:] == (3, 4):
        # Convert 3x4 camera-to-world to 4x4 for inverse
        c2w_4x4 = torch.eye(4, device=c2w.device)
        c2w_4x4[:3, :4] = c2w[:3, :4]
        w2c = torch.inverse(c2w_4x4)
    elif c2w.shape[-2:] == (4, 4):
        # Already 4x4, just take inverse
        w2c = torch.inverse(c2w)
    else:
        print(f"Warning: Unexpected camera matrix shape {c2w.shape}")
        return torch.zeros(height, width, 3)

    points_cam = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]

    # Project to 2D
    points_2d = points_cam[:, :2] / points_cam[:, 2:3]
    points_2d = points_2d * torch.tensor([fx, fy]) + torch.tensor([cx, cy])

    # Create image
    image = torch.zeros(height, width, 3)

    # Draw points
    for i, (point_2d, color) in enumerate(zip(points_2d, colors)):
        x, y = int(point_2d[0]), int(point_2d[1])

        # Check if point is in front of camera and within image bounds
        if 0 <= x < width and 0 <= y < height and points_cam[i, 2] > 0:
            # Draw a small square around the point
            for dx in range(-point_size // 2, point_size // 2 + 1):
                for dy in range(-point_size // 2, point_size // 2 + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        image[ny, nx] = color

    return image


def train_camera_optimizer(
    cameras,
    target_images,
    scene_points,
    scene_colors,
    num_iterations=1000,
    learning_rate=1e-3,
    save_plots=True,
):
    """Train the camera optimizer to correct camera poses."""

    # Create camera optimizer configuration
    camera_optimizer_config = CameraOptimizerConfig(
        mode="SO3xR3",  # Optimize rotation and translation separately
        trans_l2_penalty=1e-2,  # L2 penalty on translation
        rot_l2_penalty=1e-3,  # L2 penalty on rotation
    )

    # Create camera optimizer
    camera_optimizer = CameraOptimizer(
        config=camera_optimizer_config,
        num_cameras=len(cameras),
        device="cpu",
    )

    print(f"Created camera optimizer with {camera_optimizer.num_cameras} cameras")
    print(f"Optimization mode: {camera_optimizer.config.mode}")
    print(f"Pose adjustment shape: {camera_optimizer.pose_adjustment.shape}")
    print(f"Pose adjustment requires grad: {camera_optimizer.pose_adjustment.requires_grad}")

    # Set up optimizer
    optimizer = optim.Adam(camera_optimizer.parameters(), lr=learning_rate)

    # Training loop
    losses = []

    print("Starting camera optimization...")

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        total_loss = 0

        # Render from each camera and compare with target
        for i in range(len(cameras)):
            # Get optimized camera pose
            camera = cameras[i : i + 1]  # Keep batch dimension
            # Add metadata with camera index
            camera.metadata = {"cam_idx": i}

            # Try using the forward method directly
            indices = torch.tensor([i], dtype=torch.long)
            correction_matrix = camera_optimizer(indices)

            if iteration == 0:
                print(f"Correction matrix for camera {i}:\n{correction_matrix[0]}")

            # Apply correction manually to maintain gradient flow
            original_pose = camera.camera_to_worlds[0]
            optimized_camera_to_world = torch.cat(
                [
                    torch.bmm(
                        correction_matrix[..., :3, :3], original_pose[..., :3, :3].unsqueeze(0)
                    ),
                    original_pose[..., :3, 3:].unsqueeze(0) + correction_matrix[..., :3, 3:],
                ],
                dim=-1,
            )

            # Debug: Check if optimization is working
            if iteration == 0 and i == 0:
                print(f"Original camera pose:\n{camera.camera_to_worlds[0]}")
                print(f"Optimized camera pose:\n{optimized_camera_to_world[0]}")
                print(f"Pose adjustment for camera {i}: {camera_optimizer.pose_adjustment[i]}")

            # Create temporary camera with optimized pose
            temp_camera = Cameras(
                camera_to_worlds=optimized_camera_to_world,
                fx=cameras.fx,
                fy=cameras.fy,
                cx=cameras.cx,
                cy=cameras.cy,
                width=cameras.width,
                height=cameras.height,
                camera_type=cameras.camera_type,
            )

            # Render image
            rendered_image = simple_render(temp_camera[0], scene_points, scene_colors)

            # Compute loss
            loss = torch.nn.functional.mse_loss(rendered_image, target_images[i])

            # Add a direct loss on the pose adjustment to test gradient flow
            if iteration == 0:
                # Test gradient flow with a simple loss
                test_loss = camera_optimizer.pose_adjustment[i, 0].abs()  # Just use first parameter
                print(f"Test loss for camera {i}: {test_loss.item()}")
                loss = loss + 0.1 * test_loss

            total_loss += loss

        # Add regularization loss
        loss_dict = {}
        camera_optimizer.get_loss_dict(loss_dict)
        if "camera_opt_regularizer" in loss_dict:
            total_loss += loss_dict["camera_opt_regularizer"]

        # Backward pass
        total_loss.backward()

        # Debug: Check gradients
        if iteration % 100 == 0:
            grad_norm = 0
            for param in camera_optimizer.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm**0.5
            print(
                f"Iteration {iteration}, Loss: {total_loss.item():.6f}, Grad norm: {grad_norm:.6f}"
            )

        optimizer.step()
        losses.append(total_loss.item())

    print("Optimization complete!")

    # Plot loss curve
    if save_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig("results/camera_optimizer_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

    return camera_optimizer, losses


def visualize_results(
    cameras,
    optimized_cameras,
    gt_cameras,
    scene_points,
    scene_colors,
    target_images,
    save_plots=True,
):
    """Visualize the optimization results."""

    # Render optimized images
    optimized_images = []
    for i in range(len(cameras)):
        optimized_image = simple_render(optimized_cameras[i], scene_points, scene_colors)
        optimized_images.append(optimized_image)

    optimized_images = torch.stack(optimized_images)

    # Show comparison
    if save_plots:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for i in range(4):
            # Original noisy
            noisy_image = simple_render(cameras[i], scene_points, scene_colors)
            axes[0, i].imshow(noisy_image)
            axes[0, i].set_title(f"Original (Noisy) {i}")
            axes[0, i].axis("off")

            # Optimized
            axes[1, i].imshow(optimized_images[i])
            axes[1, i].set_title(f"Optimized {i}")
            axes[1, i].axis("off")

            # Ground truth
            axes[2, i].imshow(target_images[i])
            axes[2, i].set_title(f"Ground Truth {i}")
            axes[2, i].axis("off")

        plt.tight_layout()
        plt.savefig("results/camera_optimizer_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Compute metrics
    original_images = torch.stack(
        [simple_render(cameras[i], scene_points, scene_colors) for i in range(len(cameras))]
    )
    original_mse = torch.nn.functional.mse_loss(original_images, target_images)
    optimized_mse = torch.nn.functional.mse_loss(optimized_images, target_images)

    print(f"Original MSE: {original_mse.item():.6f}")
    print(f"Optimized MSE: {optimized_mse.item():.6f}")
    print(f"Improvement: {((original_mse - optimized_mse) / original_mse * 100):.2f}%")

    return optimized_images, original_mse, optimized_mse


def analyze_pose_corrections(camera_optimizer, save_plots=True):
    """Analyze how much the camera optimizer corrected the poses."""

    # Get pose adjustments
    pose_adjustments = camera_optimizer.pose_adjustment.detach()
    translation_adjustments = pose_adjustments[:, :3]
    rotation_adjustments = pose_adjustments[:, 3:]

    print("Pose Adjustment Analysis:")
    print(
        f"Translation adjustments - Mean: {translation_adjustments.norm(dim=1).mean():.4f}, "
        f"Max: {translation_adjustments.norm(dim=1).max():.4f}"
    )
    print(
        f"Rotation adjustments - Mean: {rotation_adjustments.norm(dim=1).mean():.4f}, "
        f"Max: {rotation_adjustments.norm(dim=1).max():.4f}"
    )

    # Plot adjustments
    if save_plots:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Translation adjustments
        axes[0, 0].hist(translation_adjustments[:, 0], bins=20, alpha=0.7, label="X")
        axes[0, 0].hist(translation_adjustments[:, 1], bins=20, alpha=0.7, label="Y")
        axes[0, 0].hist(translation_adjustments[:, 2], bins=20, alpha=0.7, label="Z")
        axes[0, 0].set_title("Translation Adjustments")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Rotation adjustments
        axes[0, 1].hist(rotation_adjustments[:, 0], bins=20, alpha=0.7, label="X")
        axes[0, 1].hist(rotation_adjustments[:, 1], bins=20, alpha=0.7, label="Y")
        axes[0, 1].hist(rotation_adjustments[:, 2], bins=20, alpha=0.7, label="Z")
        axes[0, 1].set_title("Rotation Adjustments")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Magnitude of adjustments
        axes[1, 0].plot(translation_adjustments.norm(dim=1))
        axes[1, 0].set_title("Translation Magnitude by Camera")
        axes[1, 0].set_xlabel("Camera Index")
        axes[1, 0].set_ylabel("Magnitude")
        axes[1, 0].grid(True)

        axes[1, 1].plot(rotation_adjustments.norm(dim=1))
        axes[1, 1].set_title("Rotation Magnitude by Camera")
        axes[1, 1].set_xlabel("Camera Index")
        axes[1, 1].set_ylabel("Magnitude")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("results/camera_optimizer_adjustments.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test Camera Optimizer")
    parser.add_argument("--num_cameras", type=int, default=8, help="Number of cameras")
    parser.add_argument(
        "--noise_level", type=float, default=0.1, help="Noise level for camera poses"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1000, help="Number of training iterations"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to results/")
    parser.add_argument("--show_plots", action="store_true", help="Show plots interactively")

    args = parser.parse_args()

    # Create results directory
    if args.save_plots:
        os.makedirs("results", exist_ok=True)

    print("=== Camera Optimizer Test ===")
    print(f"Number of cameras: {args.num_cameras}")
    print(f"Noise level: {args.noise_level}")
    print(f"Training iterations: {args.num_iterations}")

    # 1. Create a simple 3D scene
    print("\n1. Creating 3D scene...")
    scene_points, scene_colors = create_simple_scene()
    print(f"Scene has {len(scene_points)} points")

    # 2. Create camera poses with noise
    print("\n2. Creating camera poses...")
    camera_to_worlds = create_camera_poses(
        args.num_cameras, radius=3.0, noise_level=args.noise_level
    )

    # 3. Create cameras object
    print("\n3. Setting up cameras...")
    width, height = 256, 256
    fov = 60  # degrees
    focal_length = width / (2 * np.tan(np.radians(fov / 2)))

    cameras = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=focal_length,
        fy=focal_length,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
        camera_type=CameraType.PERSPECTIVE,
    )

    # 4. Create ground truth camera poses and target images
    print("\n4. Creating ground truth poses and target images...")
    gt_camera_to_worlds = create_camera_poses(args.num_cameras, radius=3.0, noise_level=0.0)
    gt_cameras = Cameras(
        camera_to_worlds=gt_camera_to_worlds,
        fx=focal_length,
        fy=focal_length,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
        camera_type=CameraType.PERSPECTIVE,
    )

    # Create target images
    target_images = []
    for i in range(args.num_cameras):
        target_image = simple_render(gt_cameras[i], scene_points, scene_colors)
        target_images.append(target_image)

    target_images = torch.stack(target_images)

    # 5. Train camera optimizer
    print("\n5. Training camera optimizer...")
    camera_optimizer, losses = train_camera_optimizer(
        cameras,
        target_images,
        scene_points,
        scene_colors,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        save_plots=args.save_plots,
    )

    # 6. Get optimized camera poses
    print("\n6. Getting optimized camera poses...")
    # Add metadata to cameras for optimization
    for i in range(len(cameras)):
        cameras[i].metadata = {"cam_idx": i}
    optimized_camera_to_worlds = camera_optimizer.apply_to_camera(cameras)
    optimized_cameras = Cameras(
        camera_to_worlds=optimized_camera_to_worlds,
        fx=focal_length,
        fy=focal_length,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
        camera_type=CameraType.PERSPECTIVE,
    )

    # 7. Visualize results
    print("\n7. Visualizing results...")
    optimized_images, original_mse, optimized_mse = visualize_results(
        cameras,
        optimized_cameras,
        gt_cameras,
        scene_points,
        scene_colors,
        target_images,
        save_plots=args.save_plots,
    )

    # 8. Analyze pose corrections
    print("\n8. Analyzing pose corrections...")
    analyze_pose_corrections(camera_optimizer, save_plots=args.save_plots)

    # Show plots if requested
    if args.show_plots:
        plt.show()

    print("\n=== Test Complete ===")
    print(f"Results saved to results/ directory")
    print(f"Final improvement: {((original_mse - optimized_mse) / original_mse * 100):.2f}%")


if __name__ == "__main__":
    main()
