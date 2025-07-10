"""
Test timing of update_light_source and rendering
"""

import os

# Set PyTorch CUDA memory allocator to help with fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
import numpy as np
import json
import gc
from datetime import datetime
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox

from shadow_splat.model import ShadowSplatModel, ShadowSplatModelConfig

# from shadow_splat.model_old import ShadowSplatModel, ShadowSplatModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"results/timing_{timestamp}.txt"


# Function to write results to file
def write_result_to_file(result_string):
    with open(output_file, "a") as f:
        f.write(result_string + "\n")


def print_memory_usage():
    """Print current CUDA memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_info = f"CUDA Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        print(memory_info)
        return {"allocated_gb": allocated, "reserved_gb": reserved}
    else:
        print("CUDA not available")
        return {"allocated_gb": 0, "reserved_gb": 0}


def clear_model_memory(model):
    """Clear memory-intensive instance variables from the model"""
    # Clear lighting weights and shadow function
    if hasattr(model, "lighting_weights"):
        del model.lighting_weights
        model.lighting_weights = None

    if hasattr(model, "shadow_fn"):
        del model.shadow_fn
        model.shadow_fn = None

    # Clear any cached meta information
    if hasattr(model, "last_size"):
        del model.last_size

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def clear_shadow_meta(shadow_meta):
    """Explicitly clear all tensors in shadow_meta"""
    if shadow_meta is not None:
        for key, value in shadow_meta.items():
            if isinstance(value, torch.Tensor):
                del value
        shadow_meta.clear()


def get_seed_points():
    data = np.load("data/open3d_env_points.npz")
    points = np.concatenate([data["surface_points"], data["rock_points"]], axis=0)
    points = torch.from_numpy(points).float().to(device)
    points = (1 / 30.0) * points

    return points


if __name__ == "__main__":
    # Load model
    config = ShadowSplatModelConfig()
    config.compute_variance = False
    scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]]))
    model = ShadowSplatModel(config, scene_box, num_train_data=100).to(device)

    # Initialize model points
    points = get_seed_points()

    colors = 127 * torch.ones(points.shape[0], 3)
    model.seed_points = (points, colors)
    model.populate_modules()

    # Increase opacities
    model.gauss_params["opacities"] = torch.logit(0.75 * torch.ones(model.num_points, 1))

    model.training = False
    model = model.to(device)

    print(f"Model loaded with {model.num_points} points")

    # Load transforms
    transforms = json.load(
        open(
            "/home/addai/BlueOrigin/lunar_slam/output/open3d_dataset/solar_progression_6/transforms.json"
        )
    )

    light = Cameras(
        camera_to_worlds=torch.eye(4).unsqueeze(0),
        fx=1650.0,
        fy=1650.0,
        cx=1000.0,
        cy=1000.0,
        width=2000,
        height=2000,
        camera_type=CameraType.ORTHOPHOTO,
    ).to(device)

    # Test timing of update_light_source
    # - Repeat same pose
    light_pose = torch.tensor(transforms["frames"][0]["light_pose"]).to(device)
    light.camera_to_worlds = light_pose.unsqueeze(0)
    times = []
    for i in range(100):
        start_time = time.time()
        shadow_meta = model.compute_irradiance(light)
        times.append(time.time() - start_time)

    # clear_shadow_meta(shadow_meta)
    # clear_model_memory(model)

    # Write result to file
    repeat_result = f"Repeat update_light_source\n  avg: {np.mean(times):.4f}s, max: {np.max(times):.4f}s, min: {np.min(times):.4f}s"
    print(repeat_result)
    write_result_to_file(repeat_result)

    print_memory_usage()

    # - Different pose
    times = []
    for i in range(100):
        idx = np.random.randint(len(transforms["frames"]))
        light_pose = torch.tensor(transforms["frames"][idx]["light_pose"]).to(device)
        light.camera_to_worlds = light_pose.unsqueeze(0)
        start_time = time.time()
        shadow_meta = model.compute_irradiance(light)
        times.append(time.time() - start_time)

        # clear_shadow_meta(shadow_meta)
        # clear_model_memory(model)

    # Write result to file
    different_result = f"Different update_light_source\n  avg: {np.mean(times):.4f}s, max: {np.max(times):.4f}s, min: {np.min(times):.4f}s"
    print(different_result)
    write_result_to_file(different_result)

    print_memory_usage()

    # Test timing of rendering
    # Create a test camera pose for rendering
    test_camera_pose = torch.tensor(transforms["frames"][0]["transform_matrix"]).to(device)
    test_camera = Cameras(
        camera_to_worlds=test_camera_pose.unsqueeze(0),
        fx=1650.0,
        fy=1650.0,
        cx=1000.0,
        cy=1000.0,
        width=2000,
        height=2000,
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)

    times = []
    for i in range(100):
        start_time = time.time()
        outputs = model.get_outputs(test_camera)
        times.append(time.time() - start_time)

        # Clear any outputs to prevent memory accumulation
        del outputs
        torch.cuda.empty_cache()

    # Write result to file
    rendering_result = f"Rendering\n  avg: {np.mean(times):.4f}s, max: {np.max(times):.4f}s, min: {np.min(times):.4f}s"
    print(rendering_result)
    write_result_to_file(rendering_result)

    print(f"Results saved to: {output_file}")
