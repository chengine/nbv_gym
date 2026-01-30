"""Debug script to compare coverage metrics between training and external models."""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras

from nbv_gym.util.camera_pose import DifferentiableCameraPose


def test_gradient_flow(model, camera, view_metric="coverage"):
    """Test if gradients flow from coverage score to camera pose.

    Args:
        model: The NBVSplatModel to test with
        camera: A camera object to use as starting point
        view_metric: The view metric to test ("coverage", "fisherrf", etc.)

    Returns:
        bool: True if gradients flow through successfully, False otherwise
    """
    print("\n" + "=" * 60)
    print("GRADIENT FLOW TEST")
    print("=" * 60)

    # Extract initial pose from camera
    c2w = camera.camera_to_worlds[0].clone()

    # Create differentiable pose
    diff_pose = DifferentiableCameraPose(c2w)

    # Get current c2w (with gradients)
    c2w_diff = diff_pose.get_camera_to_world()

    # Create camera with differentiable pose
    test_camera = Cameras(
        camera_to_worlds=c2w_diff,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        width=camera.width,
        height=camera.height,
    ).to(model.device)

    # Compute differentiable coverage score
    print(f"Computing differentiable {view_metric} score...")
    score = model.view_metric_score_for_camera_differentiable(test_camera)

    print(f"\n{view_metric.capitalize()} score: {score.item():.4f}")
    print(f"Score requires_grad: {score.requires_grad}")

    # Check gradients
    print("\nRunning backward pass...")
    try:
        score.backward()

        pos_grad = diff_pose.position.grad
        rot_grad = diff_pose.axis_angle.grad

        print(f"\nPosition gradient: {pos_grad}")
        print(f"Rotation gradient: {rot_grad}")

        if pos_grad is not None:
            print(f"Position grad norm: {pos_grad.norm().item():.6f}")
        else:
            print("Position grad norm: None (no gradient!)")

        if rot_grad is not None:
            print(f"Rotation grad norm: {rot_grad.norm().item():.6f}")
        else:
            print("Rotation grad norm: None (no gradient!)")

        gradients_exist = pos_grad is not None and rot_grad is not None
        gradients_nonzero = gradients_exist and (pos_grad.norm() > 0 or rot_grad.norm() > 0)

        print("\n" + "-" * 40)
        if gradients_nonzero:
            print("RESULT: Gradients flow successfully!")
        elif gradients_exist:
            print("RESULT: Gradients exist but are ZERO")
        else:
            print("RESULT: Gradients are NONE - no gradient flow!")
        print("-" * 40)

        return gradients_nonzero

    except Exception as e:
        print(f"\nERROR during backward pass: {e}")
        return False


def visualize_optimization_renders(model, camera, view_metric="coverage", num_steps=50, lr=0.01, max_pos_delta=None, max_rot_delta=None):
    """Run GD optimization and visualize initial vs optimized renders.

    Creates a comparison figure showing:
    - Row 1: Initial camera (RGB, coverage map, depth)
    - Row 2: Optimized camera (RGB, coverage map, depth)
    - Stats: position delta, score change, etc.

    Args:
        model: The NBVSplatModel to test with
        camera: A camera object to use as starting point
        view_metric: The view metric to optimize
        num_steps: Number of gradient descent steps
        lr: Learning rate for position (rotation uses lr * 0.1)
        max_pos_delta: Maximum position movement from initial pose (epsilon ball constraint)
        max_rot_delta: Maximum rotation movement from initial pose (epsilon ball on axis-angle)

    Returns:
        tuple: (matplotlib figure, stats dict)
    """
    print("\n" + "=" * 60)
    print("VISUALIZE OPTIMIZATION RENDERS")
    print(f"Steps: {num_steps}, LR: {lr}")
    print("=" * 60)

    # Store initial camera info
    initial_c2w = camera.camera_to_worlds[0].clone()
    initial_pos = initial_c2w[:3, 3].clone()

    # Render from initial camera
    print("Rendering from initial camera...")
    with torch.no_grad():
        initial_outputs = model(camera)
    initial_rgb = initial_outputs["rgb"].cpu()
    initial_coverage = initial_outputs["view_metric"].cpu()
    initial_depth = initial_outputs["depth"].cpu()
    initial_score = model.view_metric_score_for_camera(camera).item()

    print(f"Initial score: {initial_score:.4f}")
    print(f"Initial position: {initial_pos.cpu().numpy()}")

    # Run optimization
    constraints = []
    if max_pos_delta:
        constraints.append(f"max_pos_delta={max_pos_delta}")
    if max_rot_delta:
        constraints.append(f"max_rot_delta={max_rot_delta}")
    constraint_str = f", {', '.join(constraints)}" if constraints else ""
    print(f"\nRunning {num_steps} GD steps...{constraint_str}")
    diff_pose = DifferentiableCameraPose(initial_c2w)

    # Store initial position and rotation for epsilon ball constraints
    initial_position = diff_pose.position.detach().clone()
    initial_axis_angle = diff_pose.axis_angle.detach().clone()

    optimizer = torch.optim.Adam([
        {'params': [diff_pose.position], 'lr': lr},
        {'params': [diff_pose.axis_angle], 'lr': lr * 0.1},
    ])

    for step in range(num_steps):
        optimizer.zero_grad()
        c2w_diff = diff_pose.get_camera_to_world()
        opt_camera = Cameras(
            camera_to_worlds=c2w_diff,
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
            width=camera.width,
            height=camera.height,
        ).to(model.device)
        score = model.view_metric_score_for_camera_differentiable(opt_camera)
        score.backward()
        optimizer.step()

        # Project position and rotation back onto epsilon balls if exceeded
        with torch.no_grad():
            if max_pos_delta is not None:
                pos_delta = diff_pose.position - initial_position
                pos_delta_norm = pos_delta.norm()
                if pos_delta_norm > max_pos_delta:
                    diff_pose.position.copy_(
                        initial_position + pos_delta * (max_pos_delta / pos_delta_norm)
                    )

            if max_rot_delta is not None:
                rot_delta = diff_pose.axis_angle - initial_axis_angle
                rot_delta_norm = rot_delta.norm()
                if rot_delta_norm > max_rot_delta:
                    diff_pose.axis_angle.copy_(
                        initial_axis_angle + rot_delta * (max_rot_delta / rot_delta_norm)
                    )

        if step % 10 == 0 or step == num_steps - 1:
            pos_d = (diff_pose.position.detach() - initial_pos.to(diff_pose.position.device)).norm().item()
            rot_d = (diff_pose.axis_angle.detach() - initial_axis_angle).norm().item()
            print(f"  Step {step:3d}: score = {score.item():.4f}, pos_delta = {pos_d:.4f}, rot_delta = {rot_d:.4f}")

    # Get final optimized camera
    with torch.no_grad():
        final_c2w = diff_pose.get_camera_to_world()
    final_pos = final_c2w[0, :3, 3]

    opt_camera = Cameras(
        camera_to_worlds=final_c2w,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        width=camera.width,
        height=camera.height,
    ).to(model.device)

    # Render from optimized camera
    print("\nRendering from optimized camera...")
    with torch.no_grad():
        opt_outputs = model(opt_camera)
    opt_rgb = opt_outputs["rgb"].cpu()
    opt_coverage = opt_outputs["view_metric"].cpu()
    opt_depth = opt_outputs["depth"].cpu()
    opt_score = model.view_metric_score_for_camera(opt_camera).item()

    # Compute stats
    pos_delta = (final_pos - initial_pos.to(final_pos.device)).norm().item()
    rot_delta = (diff_pose.axis_angle.detach() - initial_axis_angle).norm().item()

    print(f"\nFinal score: {opt_score:.4f}")
    print(f"Final position: {final_pos.cpu().numpy()}")
    print(f"Position delta: {pos_delta:.4f}")
    print(f"Rotation delta: {rot_delta:.4f}")
    print(f"Score change: {initial_score:.4f} -> {opt_score:.4f} (delta = {opt_score - initial_score:.4f})")

    # Create visualization (2 rows x 3 cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Initial camera
    axes[0, 0].imshow(initial_rgb.numpy().clip(0, 1))
    axes[0, 0].set_title(f"Initial RGB\nScore: {initial_score:.4f}")
    axes[0, 1].imshow(initial_coverage.squeeze().numpy(), cmap="viridis")
    axes[0, 1].set_title("Initial Coverage")
    im_cov_init = axes[0, 1].images[0]
    plt.colorbar(im_cov_init, ax=axes[0, 1], fraction=0.046)
    axes[0, 2].imshow(initial_depth.squeeze().numpy(), cmap="turbo")
    axes[0, 2].set_title("Initial Depth")
    im_depth_init = axes[0, 2].images[0]
    plt.colorbar(im_depth_init, ax=axes[0, 2], fraction=0.046)

    # Optimized camera
    axes[1, 0].imshow(opt_rgb.numpy().clip(0, 1))
    axes[1, 0].set_title(f"Optimized RGB\nScore: {opt_score:.4f}")
    axes[1, 1].imshow(opt_coverage.squeeze().numpy(), cmap="viridis")
    axes[1, 1].set_title("Optimized Coverage")
    im_cov_opt = axes[1, 1].images[0]
    plt.colorbar(im_cov_opt, ax=axes[1, 1], fraction=0.046)
    axes[1, 2].imshow(opt_depth.squeeze().numpy(), cmap="turbo")
    axes[1, 2].set_title("Optimized Depth")
    im_depth_opt = axes[1, 2].images[0]
    plt.colorbar(im_depth_opt, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        ax.axis("off")

    constraint_parts = []
    if max_pos_delta:
        constraint_parts.append(f"max_pos={max_pos_delta}")
    if max_rot_delta:
        constraint_parts.append(f"max_rot={max_rot_delta}")
    constraint_info = f", {', '.join(constraint_parts)}" if constraint_parts else ""
    plt.suptitle(f"GD Optimization: {num_steps} steps, LR={lr}{constraint_info}\n"
                 f"Pos delta: {pos_delta:.4f}, Rot delta: {rot_delta:.4f}, "
                 f"Score: {initial_score:.4f} -> {opt_score:.4f}")
    plt.tight_layout()

    stats = {
        "initial_score": initial_score,
        "final_score": opt_score,
        "pos_delta": pos_delta,
        "rot_delta": rot_delta,
        "initial_pos": initial_pos.cpu().numpy(),
        "final_pos": final_pos.cpu().numpy(),
    }

    return fig, stats


def test_optimization_trajectory(model, camera, view_metric="coverage", num_steps=10, lr=0.01, max_pos_delta=None, max_rot_delta=None):
    """Run a few GD steps and track the score trajectory.

    Args:
        model: The NBVSplatModel to test with
        camera: A camera object to use as starting point
        view_metric: The view metric to optimize
        num_steps: Number of gradient descent steps
        lr: Learning rate for position (rotation uses lr * 0.1)
        max_pos_delta: Maximum position movement from initial pose (epsilon ball constraint)
        max_rot_delta: Maximum rotation movement from initial pose (epsilon ball on axis-angle)

    Returns:
        tuple: (scores list, positions list)
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION TRAJECTORY TEST")
    constraints = []
    if max_pos_delta:
        constraints.append(f"max_pos_delta={max_pos_delta}")
    if max_rot_delta:
        constraints.append(f"max_rot_delta={max_rot_delta}")
    constraint_str = f", {', '.join(constraints)}" if constraints else ""
    print(f"Steps: {num_steps}, LR: {lr}{constraint_str}")
    print("=" * 60)

    c2w = camera.camera_to_worlds[0].clone()

    diff_pose = DifferentiableCameraPose(c2w)

    # Store initial position and rotation for epsilon ball constraints
    initial_position = diff_pose.position.detach().clone()
    initial_axis_angle = diff_pose.axis_angle.detach().clone()

    optimizer = torch.optim.Adam([
        {'params': [diff_pose.position], 'lr': lr},
        {'params': [diff_pose.axis_angle], 'lr': lr * 0.1},
    ])

    scores = []
    positions = []

    initial_pos = diff_pose.position.detach().clone()
    print(f"\nInitial position: {initial_pos.cpu().numpy()}")
    print("-" * 40)

    for step in range(num_steps):
        optimizer.zero_grad()
        c2w_diff = diff_pose.get_camera_to_world()

        test_camera = Cameras(
            camera_to_worlds=c2w_diff,
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
            width=camera.width,
            height=camera.height,
        ).to(model.device)

        score = model.view_metric_score_for_camera_differentiable(test_camera)

        # We want to MINIMIZE coverage (find areas with LOW coverage)
        # So we do gradient descent on the score directly
        score.backward()
        optimizer.step()

        # Project position and rotation back onto epsilon balls if exceeded
        with torch.no_grad():
            if max_pos_delta is not None:
                pos_delta = diff_pose.position - initial_position
                pos_delta_norm = pos_delta.norm()
                if pos_delta_norm > max_pos_delta:
                    diff_pose.position.copy_(
                        initial_position + pos_delta * (max_pos_delta / pos_delta_norm)
                    )

            if max_rot_delta is not None:
                rot_delta = diff_pose.axis_angle - initial_axis_angle
                rot_delta_norm = rot_delta.norm()
                if rot_delta_norm > max_rot_delta:
                    diff_pose.axis_angle.copy_(
                        initial_axis_angle + rot_delta * (max_rot_delta / rot_delta_norm)
                    )

        scores.append(score.item())
        positions.append(diff_pose.position.detach().clone())

        pos_d = (diff_pose.position.detach() - initial_pos).norm().item()
        rot_d = (diff_pose.axis_angle.detach() - initial_axis_angle).norm().item()

        print(f"Step {step:2d}: score = {score.item():.4f}, "
              f"pos delta = {pos_d:.4f}, rot delta = {rot_d:.4f}, "
              f"pos grad norm = {diff_pose.position.grad.norm().item():.6f}")

    final_pos = diff_pose.position.detach()
    total_movement = (final_pos - initial_pos).norm().item()

    print("-" * 40)
    print(f"Final position: {final_pos.cpu().numpy()}")
    print(f"Total position movement: {total_movement:.4f}")
    print(f"Score change: {scores[0]:.4f} -> {scores[-1]:.4f} "
          f"(delta = {scores[-1] - scores[0]:.4f})")

    if scores[-1] < scores[0]:
        print("RESULT: Score decreased (optimization working!)")
    elif scores[-1] > scores[0]:
        print("RESULT: Score INCREASED (unexpected!)")
    else:
        print("RESULT: Score unchanged (optimization not working)")

    return scores, positions


def main():
    parser = argparse.ArgumentParser(
        description="Debug gradient descent view selector and test gradient flow"
    )
    parser.add_argument("--external-config", type=Path, required=True,
                        help="Path to pretrained 'all' model config.yml")
    parser.add_argument("--training-config", type=Path, default=None,
                        help="Path to partially trained model config.yml (optional)")
    parser.add_argument("--view-metric", type=str, default="coverage")
    parser.add_argument("--num-cameras", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--test-gradients", action="store_true",
                        help="Test if gradients flow through camera pose optimization")
    parser.add_argument("--visualize-renders", action="store_true",
                        help="Visualize initial vs optimized camera renders")
    parser.add_argument("--gd-steps", type=int, default=10,
                        help="Number of gradient descent steps for trajectory test")
    parser.add_argument("--gd-lr", type=float, default=0.01,
                        help="Learning rate for gradient descent trajectory test")
    parser.add_argument("--max-pos-delta", type=float, default=None,
                        help="Maximum position movement from initial pose (epsilon ball constraint)")
    parser.add_argument("--max-rot-delta", type=float, default=None,
                        help="Maximum rotation movement from initial pose (epsilon ball on axis-angle)")
    args = parser.parse_args()

    # Load external model
    print("Loading external model...")
    _, ext_pipeline, _, _ = eval_setup(args.external_config, test_mode="inference")
    ext_model = ext_pipeline.model
    ext_model.eval()
    ext_datamanager = ext_pipeline.datamanager

    # Setup view metric on external model
    ext_model.setup_view_metric(args.view_metric)

    # Get cameras from dataset
    all_cameras = ext_datamanager.train_dataset.cameras
    num_cameras = min(args.num_cameras, len(all_cameras))

    # Run gradient tests if requested
    if args.test_gradients:
        # First update view attributes with subset of cameras (simulating partial training)
        subset_size = num_cameras // 2
        subset_cameras = [all_cameras[i:i+1] for i in range(subset_size)]
        ext_model.update_view_attributes(subset_cameras)

        # Pick a test camera that wasn't used for view attributes
        test_camera_idx = subset_size
        test_camera = all_cameras[test_camera_idx:test_camera_idx+1].to(ext_model.device)

        print(f"\nUsing camera {test_camera_idx} for gradient tests")
        print(f"(View attributes computed from cameras 0-{subset_size-1})")

        # Test 1: Gradient flow
        gradients_ok = test_gradient_flow(ext_model, test_camera, args.view_metric)

        # Test 2: Optimization trajectory (only if gradients flow)
        if gradients_ok:
            scores, positions = test_optimization_trajectory(
                ext_model, test_camera, args.view_metric,
                num_steps=args.gd_steps, lr=args.gd_lr,
                max_pos_delta=args.max_pos_delta,
                max_rot_delta=args.max_rot_delta
            )

            # Plot trajectory
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Score trajectory
            axes[0].plot(scores, 'b-o')
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel(f"{args.view_metric.capitalize()} Score")
            axes[0].set_title("Optimization Trajectory")
            axes[0].grid(True)

            # Position trajectory (3D coords over time)
            positions_np = torch.stack(positions).cpu().numpy()
            axes[1].plot(positions_np[:, 0], label='X')
            axes[1].plot(positions_np[:, 1], label='Y')
            axes[1].plot(positions_np[:, 2], label='Z')
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Position")
            axes[1].set_title("Camera Position Trajectory")
            axes[1].legend()
            axes[1].grid(True)

            plt.suptitle(f"Gradient Descent Optimization ({args.view_metric})\n"
                         f"LR={args.gd_lr}, Steps={args.gd_steps}")
            plt.tight_layout()

            if args.output:
                plt.savefig(args.output, dpi=150, bbox_inches="tight")
                print(f"\nSaved trajectory plot to {args.output}")
            else:
                plt.show()
        else:
            print("\nSkipping optimization trajectory test (gradients not flowing)")

        return  # Exit after gradient tests

    # Run render visualization if requested
    if args.visualize_renders:
        # First update view attributes with subset of cameras (simulating partial training)
        subset_size = num_cameras // 2
        subset_cameras = [all_cameras[i:i+1] for i in range(subset_size)]
        ext_model.update_view_attributes(subset_cameras)

        # Pick a test camera that wasn't used for view attributes
        test_camera_idx = subset_size
        test_camera = all_cameras[test_camera_idx:test_camera_idx+1].to(ext_model.device)

        print(f"\nUsing camera {test_camera_idx} for render visualization")
        print(f"(View attributes computed from cameras 0-{subset_size-1})")

        fig, stats = visualize_optimization_renders(
            ext_model, test_camera, args.view_metric,
            num_steps=args.gd_steps, lr=args.gd_lr,
            max_pos_delta=args.max_pos_delta,
            max_rot_delta=args.max_rot_delta
        )

        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"\nSaved render comparison to {args.output}")
        else:
            plt.show()

        # Print analysis of results
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        score_delta = stats["final_score"] - stats["initial_score"]
        if score_delta < -0.1:
            print("Score decreased significantly - optimization is working")
            print("If training still fails, issue is in training loop integration")
        elif score_delta < 0:
            print("Score decreased slightly - optimization works but may need tuning")
        else:
            print("Score did not decrease - check if coverage landscape is flat")

        if stats["pos_delta"] > 1.0:
            print(f"WARNING: Camera moved {stats['pos_delta']:.2f} units - may be too far")
            print("Consider adding pose regularization or reducing learning rate")
        elif stats["pos_delta"] < 0.01:
            print(f"Camera barely moved ({stats['pos_delta']:.4f} units)")
            print("May need higher learning rate or more steps")

        return  # Exit after visualization

    # Compare coverage scores
    print(f"\nComparing {args.view_metric} scores for {num_cameras} cameras:")
    print("-" * 60)

    # First, update view attributes with subset of cameras (simulating partial training)
    subset_size = num_cameras // 2
    subset_cameras = [all_cameras[i:i+1] for i in range(subset_size)]
    ext_model.update_view_attributes(subset_cameras)

    # Now score remaining cameras
    for idx in range(subset_size, num_cameras):
        camera = all_cameras[idx:idx+1].to(ext_model.device)

        # Get coverage score
        score = ext_model.view_metric_score_for_camera(camera)

        # Also render to see coverage map
        with torch.no_grad():
            outputs = ext_model(camera)

        coverage_map = outputs["view_metric"]
        rgb = outputs["rgb"]

        print(f"Camera {idx}: score = {score.item():.4f}, "
              f"coverage mean = {coverage_map.mean().item():.4f}")

    # Visualize coverage maps for a few cameras
    fig, axes = plt.subplots(2, num_cameras - subset_size, figsize=(4*(num_cameras-subset_size), 8))
    if num_cameras - subset_size == 1:
        axes = axes.reshape(-1, 1)

    for i, idx in enumerate(range(subset_size, num_cameras)):
        camera = all_cameras[idx:idx+1].to(ext_model.device)
        with torch.no_grad():
            outputs = ext_model(camera)

        rgb = outputs["rgb"].cpu().numpy()
        coverage = outputs["view_metric"].cpu().numpy()

        axes[0, i].imshow(rgb.clip(0, 1))
        axes[0, i].set_title(f"RGB (cam {idx})")
        axes[0, i].axis("off")

        im = axes[1, i].imshow(coverage.squeeze(), cmap="viridis")
        axes[1, i].set_title(f"Coverage (cam {idx})")
        axes[1, i].axis("off")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    plt.suptitle(f"External Model Coverage ({args.view_metric})\n"
                 f"Trained on {subset_size} cameras, viewing remaining cameras")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
