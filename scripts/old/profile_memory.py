#!/usr/bin/env python3
"""
CUDA Memory Profiling Script for update_light_source method

This script profiles the CUDA memory usage of the update_light_source method
in the ShadowSplatModel class. It tracks memory allocation, deallocation,
and peak usage at different stages of execution.
"""

import os
import sys
import time
import torch
import gc
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import json
from dataclasses import dataclass, asdict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow_splat.model import ShadowSplatModel, ShadowSplatModelConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class MemorySnapshot:
    """Represents a memory snapshot at a specific point in time."""

    stage: str
    allocated: int
    reserved: int
    max_allocated: int
    max_reserved: int
    timestamp: float
    gpu_memory_used: int
    gpu_memory_total: int
    cpu_memory_used: int
    cpu_memory_total: int


class MemoryProfiler:
    """Profiles CUDA memory usage during method execution."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.snapshots: List[MemorySnapshot] = []
        self.start_time = None

    def _get_memory_stats(self) -> Dict[str, int]:
        """Get current memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            max_allocated = torch.cuda.max_memory_allocated(self.device)
            max_reserved = torch.cuda.max_memory_reserved(self.device)
        else:
            allocated = reserved = max_allocated = max_reserved = 0

        # Get GPU memory info
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.get_device_properties(
                self.device
            ).total_memory - torch.cuda.memory_reserved(self.device)
            gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory
        else:
            gpu_memory_used = gpu_memory_total = 0

        # Get CPU memory info
        cpu_memory = psutil.virtual_memory()

        return {
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated,
            "max_reserved": max_reserved,
            "gpu_memory_used": gpu_memory_used,
            "gpu_memory_total": gpu_memory_total,
            "cpu_memory_used": cpu_memory.used,
            "cpu_memory_total": cpu_memory.total,
        }

    def take_snapshot(self, stage: str):
        """Take a memory snapshot at the current stage."""
        if self.start_time is None:
            self.start_time = time.time()

        stats = self._get_memory_stats()
        timestamp = time.time() - self.start_time

        snapshot = MemorySnapshot(stage=stage, timestamp=timestamp, **stats)
        self.snapshots.append(snapshot)

        # Print current memory usage
        print(
            f"[{timestamp:.3f}s] {stage}: "
            f"GPU: {stats['allocated'] / 1024**3:.2f}GB allocated, "
            f"{stats['reserved'] / 1024**3:.2f}GB reserved"
        )

    def reset(self):
        """Reset the profiler state."""
        self.snapshots.clear()
        self.start_time = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def get_summary(self) -> Dict:
        """Get a summary of the profiling results."""
        if not self.snapshots:
            return {}

        # Find peak memory usage
        peak_allocated = max(s.allocated for s in self.snapshots)
        peak_reserved = max(s.reserved for s in self.snapshots)

        # Calculate memory differences between stages
        stage_diffs = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i - 1]
            curr = self.snapshots[i]
            stage_diffs.append(
                {
                    "from": prev.stage,
                    "to": curr.stage,
                    "allocated_diff": curr.allocated - prev.allocated,
                    "reserved_diff": curr.reserved - prev.reserved,
                    "time_diff": curr.timestamp - prev.timestamp,
                }
            )

        return {
            "peak_allocated_gb": peak_allocated / 1024**3,
            "peak_reserved_gb": peak_reserved / 1024**3,
            "total_time": self.snapshots[-1].timestamp if self.snapshots else 0,
            "stage_diffs": stage_diffs,
            "snapshots": [asdict(s) for s in self.snapshots],
        }

    def save_results(self, filename: str):
        """Save profiling results to a JSON file."""
        summary = self.get_summary()
        with open(filename, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Profiling results saved to {filename}")


@contextmanager
def memory_profiling_context(profiler: MemoryProfiler, stage: str):
    """Context manager for profiling a specific stage."""
    profiler.take_snapshot(f"before_{stage}")
    try:
        yield
    finally:
        profiler.take_snapshot(f"after_{stage}")


def create_test_camera(width: int = 800, height: int = 600) -> Cameras:
    """Create a test camera for profiling."""
    # Create a simple perspective camera
    camera_to_worlds = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # [1, 4, 4]
    camera_to_worlds[0, :3, 3] = torch.tensor([0, 0, 5])  # Move camera back

    fx = fy = 1000.0
    cx, cy = width / 2, height / 2

    intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).unsqueeze(
        0
    )  # [1, 3, 3]

    return Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        camera_type=CameraType.PERSPECTIVE,
        times=None,
    )


def create_test_model(num_gaussians: int = 100000) -> ShadowSplatModel:
    """Create a test model with specified number of gaussians."""
    config = ShadowSplatModelConfig(
        num_random=num_gaussians,
        random_init=True,
        random_scale=10.0,
        sh_degree=3,
        strategy="default",
    )

    # Create seed points for the model
    means = (torch.rand(num_gaussians, 3) - 0.5) * 10.0
    colors = torch.rand(num_gaussians, 3) * 255

    model = ShadowSplatModel(
        config=config,
        scene_box=SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]])),
        seed_points=(means, colors),
        num_train_data=1,
    )

    return model


def profile_update_light_source(
    model: ShadowSplatModel,
    light_source: Cameras,
    profiler: MemoryProfiler,
    num_iterations: int = 1,
) -> Dict:
    """Profile the update_light_source method."""

    print(f"Profiling update_light_source with {model.num_points:,} gaussians")
    print(f"Light source resolution: {light_source.width}x{light_source.height}")
    print(f"Number of iterations: {num_iterations}")
    print("-" * 80)

    profiler.reset()

    # Warm up
    print("Warming up...")
    with torch.no_grad():
        for _ in range(2):
            model.update_light_source(light_source)

    profiler.take_snapshot("warmup_complete")

    # Profile iterations
    for i in range(num_iterations):
        print(f"\nIteration {i + 1}/{num_iterations}")

        with memory_profiling_context(profiler, f"iteration_{i + 1}"):
            with torch.no_grad():
                result = model.update_light_source(light_source)

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    profiler.take_snapshot("profiling_complete")

    return profiler.get_summary()


def main():
    """Main profiling function."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot profile GPU memory usage.")
        return

    # Configuration
    num_gaussians_list = [50000, 100000, 200000]
    resolutions = [(400, 300), (800, 600), (1600, 1200)]
    num_iterations = 3

    profiler = MemoryProfiler()
    results = {}

    for num_gaussians in num_gaussians_list:
        results[num_gaussians] = {}

        for width, height in resolutions:
            print(f"\n{'=' * 80}")
            print(f"Profiling: {num_gaussians:,} gaussians, {width}x{height} resolution")
            print(f"{'=' * 80}")

            # Create model and camera
            model = create_test_model(num_gaussians)
            light_source = create_test_camera(width, height)

            # Move to GPU
            model = model.to(device)
            light_source = light_source.to(device)

            # Profile
            summary = profile_update_light_source(model, light_source, profiler, num_iterations)

            results[num_gaussians][f"{width}x{height}"] = summary

            # Clean up
            del model, light_source
            gc.collect()
            torch.cuda.empty_cache()

            # Small delay to ensure cleanup
            time.sleep(1)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"memory_profile_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("PROFILING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: {filename}")

    # Print summary
    print("\nSUMMARY:")
    for num_gaussians, resolutions_data in results.items():
        print(f"\n{num_gaussians:,} gaussians:")
        for resolution, summary in resolutions_data.items():
            print(f"  {resolution}: Peak GPU memory = {summary['peak_allocated_gb']:.2f}GB")


if __name__ == "__main__":
    main()
