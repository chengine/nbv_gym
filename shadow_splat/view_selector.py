"""View selector for progressive view selection in Shadow Splat"""

import random
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import KDTree


class ViewSelector(ABC):
    """Base class for view selection strategies."""

    @abstractmethod
    def select_views(
        self, active_indices: List[int], remaining_indices: List[int], num_to_select: int, **kwargs
    ) -> List[int]:
        """
        Select views to add to the active set.

        Args:
            active_indices: Current active view indices
            remaining_indices: Available view indices not yet in active set
            num_to_select: Number of views to select
            **kwargs: Additional context (e.g., step, model, cameras, etc.)

        Returns:
            List of selected view indices to add
        """
        pass


class RandomViewSelector(ViewSelector):
    """Randomly selects views from remaining indices."""

    def select_views(
        self, active_indices: List[int], remaining_indices: List[int], num_to_select: int, **kwargs
    ) -> List[int]:
        if not remaining_indices:
            return []
        k = min(max(1, num_to_select), len(remaining_indices))
        return random.sample(remaining_indices, k=k)


class AllViewSelector(ViewSelector):
    """Selects all remaining views at once."""

    def select_views(
        self, active_indices: List[int], remaining_indices: List[int], num_to_select: int, **kwargs
    ) -> List[int]:
        return remaining_indices.copy()


class OpticsViewSelector(ViewSelector):
    """Optics-based view selection using coverage scoring."""

    def __init__(
        self,
        coverage_metric: str = "coverage",
        num_nearest_neighbors: int = 5,
        intrinsics_scale: float = 1.0,
        use_kdtree_filter: bool = True,
    ):
        """
        Initialize optics-based view selector.

        Args:
            num_nearest_neighbors: Number of nearest neighbors to consider when using KD-tree filtering
            intrinsics_scale: Scale factor for camera intrinsics during scoring (for efficiency)
            use_kdtree_filter: Whether to use KD-tree filtering to reduce candidate pool
        """
        self.coverage_metric = coverage_metric
        self.num_nearest_neighbors = num_nearest_neighbors
        self.intrinsics_scale = intrinsics_scale
        self.use_kdtree_filter = use_kdtree_filter

    def select_views(
        self, active_indices: List[int], remaining_indices: List[int], num_to_select: int, **kwargs
    ) -> List[int]:
        """
        Optics-based view selection using coverage scoring.

        Args:
            active_indices: Current active view indices
            remaining_indices: Available view indices not yet in active set
            num_to_select: Number of views to select
            **kwargs: Additional context including:
                - model: The ShadowSplatModel instance
                - datamanager: The ViewSelectionDataManager instance
                - step: Current training step
                - pipeline: The pipeline instance (optional)

        Returns:
            List of selected view indices to add (views with lowest coverage scores)
        """
        if not remaining_indices:
            return []

        # Extract required context from kwargs
        model = kwargs.get("model")
        datamanager = kwargs.get("datamanager")

        if model is None or datamanager is None:
            # Fall back to random if required context is missing
            k = min(max(1, num_to_select), len(remaining_indices))
            return random.sample(remaining_indices, k=k)

        # Get candidate cameras from remaining indices
        candidate_indices = remaining_indices.copy()

        # Get all cameras from the dataset
        all_cameras = datamanager.train_dataset.cameras

        # Extract camera origins for KD-tree filtering
        # We'll access cameras individually since Cameras doesn't support list indexing
        candidate_origins_list = []
        for idx in candidate_indices:
            cam = all_cameras[idx : idx + 1]
            origin = cam.camera_to_worlds[0, :3, -1].cpu().numpy()
            candidate_origins_list.append(origin)
        candidate_origins = np.array(candidate_origins_list)

        # Optionally filter candidates using KD-tree nearest neighbors
        if self.use_kdtree_filter and len(active_indices) > 0:
            # Use the last added view as the "root" pose
            root_idx = active_indices[-1]
            root_camera = all_cameras[root_idx : root_idx + 1]
            root_origin = root_camera.camera_to_worlds[0, :3, -1].cpu().numpy()

            # Build KD-tree and query nearest neighbors
            kdtree = KDTree(candidate_origins)
            k = min(self.num_nearest_neighbors, len(candidate_indices))
            _, nearest_indices = kdtree.query(root_origin.reshape(1, 3), k=k)

            # Filter to nearest neighbors
            nearest_indices = nearest_indices.flatten()
            candidate_indices = [candidate_indices[i] for i in nearest_indices]
            # Recompute origins for filtered candidates
            candidate_origins_list = []
            for idx in candidate_indices:
                cam = all_cameras[idx : idx + 1]
                origin = cam.camera_to_worlds[0, :3, -1].cpu().numpy()
                candidate_origins_list.append(origin)
            candidate_origins = np.array(candidate_origins_list)

        # Score each candidate camera using coverage
        scores = []
        for cam_idx in candidate_indices:
            # Access camera using slice notation (Cameras expects tuple/slice, not list)
            camera = all_cameras[cam_idx : cam_idx + 1].to(model.device)
            try:
                score = model.coverage_score_for_camera(
                    camera, intrinsics_scale=self.intrinsics_scale, metric=self.coverage_metric
                )
                scores.append((cam_idx, score.item()))
            except Exception as e:
                # Handle errors gracefully - assign high score
                print(f"Warning: Failed to score camera {cam_idx}: {e}")
                scores.append((cam_idx, float("inf")))

        # Sort by score (ascending) and select top candidates (lowest coverage = most novel views)
        scores.sort(key=lambda x: x[1], reverse=False)
        k = min(num_to_select, len(scores))
        selected_indices = [idx for idx, _ in scores[:k]]

        return selected_indices


def create_view_selector(
    mode: Optional[str] = None,
    num_nearest_neighbors: Optional[int] = None,
    intrinsics_scale: Optional[float] = None,
    use_kdtree_filter: Optional[bool] = None,
    coverage_metric: Optional[str] = None,
) -> ViewSelector:
    """
    Factory function to create a ViewSelector based on mode string.

    Args:
        mode: Selection mode - "random", "all", "optics", or None (defaults to "random")
        num_nearest_neighbors: Number of nearest neighbors for optics selector (optional)
        intrinsics_scale: Intrinsics scale for optics selector (optional)
        use_kdtree_filter: Whether to use KD-tree filtering for optics selector (optional)

    Returns:
        ViewSelector instance
    """
    print(f"Creating view selector for mode: {mode}")
    if mode is None or mode == "random":
        return RandomViewSelector()
    elif mode == "all":
        return AllViewSelector()
    elif mode == "optics":
        # Use provided config values or defaults
        return OpticsViewSelector(
            coverage_metric=coverage_metric if coverage_metric is not None else "coverage",
            num_nearest_neighbors=num_nearest_neighbors if num_nearest_neighbors is not None else 5,
            intrinsics_scale=intrinsics_scale if intrinsics_scale is not None else 1.0,
            use_kdtree_filter=use_kdtree_filter if use_kdtree_filter is not None else True,
        )
    else:
        # Try to import from dotted path
        try:
            from importlib import import_module

            parts = mode.split(".")
            module_path = ".".join(parts[:-1])
            class_name = parts[-1]
            module = import_module(module_path)
            selector_class = getattr(module, class_name)
            return selector_class()
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(
                f"Unknown view selector mode '{mode}'. "
                f"Expected 'random', 'all', 'optics', or a dotted path to a ViewSelector class. "
                f"Error: {e}"
            )
