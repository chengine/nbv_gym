"""View selector for progressive view selection in NBV-Gym"""
import random
from typing import List, Optional, Literal
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import KDTree
import torch
from nbv_gym.model import NBVSplatModel
import time
class BaseViewSelector(ABC):
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

class RandomViewSelector(BaseViewSelector):
    """Randomly selects views from remaining indices."""

    def __init__(
        self,
        num_nearest_neighbors: int = 5,
        use_kdtree_filter: bool = False,
    ):
        """
        Initialize random view selector.

        Args:
            num_nearest_neighbors: Number of nearest neighbors to consider when using KD-tree filtering
            use_kdtree_filter: Whether to use KD-tree filtering to reduce candidate pool before random selection
        """
        self.num_nearest_neighbors = num_nearest_neighbors
        self.use_kdtree_filter = use_kdtree_filter

    def select_views(
        self, active_indices: List[int], remaining_indices: List[int], num_to_select: int, **kwargs
    ) -> List[int]:
        """
        Randomly selects views from remaining indices, optionally filtered by KD-tree.

        Args:
            active_indices: Current active view indices
            remaining_indices: Available view indices not yet in active set
            num_to_select: Number of views to select
            **kwargs: Additional context including:
                - datamanager: The ViewSelectionDataManager instance (required for KD-tree filtering)

        Returns:
            List of selected view indices to add
        """
        if not remaining_indices:
            return []

        # Get candidate indices - start with all remaining indices
        candidate_indices = remaining_indices.copy()

        # Optionally filter candidates using KD-tree nearest neighbors
        if self.use_kdtree_filter and len(active_indices) > 0:
            datamanager = kwargs.get("datamanager")
            if datamanager is not None:
                # Get all cameras from the dataset
                all_cameras = datamanager.train_dataset.cameras

                # Extract camera origins for KD-tree filtering
                candidate_origins_list = []
                for idx in candidate_indices:
                    cam = all_cameras[idx : idx + 1]
                    origin = cam.camera_to_worlds[0, :3, -1].cpu().numpy()
                    candidate_origins_list.append(origin)
                candidate_origins = np.array(candidate_origins_list)

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

        # Randomly select from candidates (filtered or unfiltered)
        k = min(max(1, num_to_select), len(candidate_indices))
        return random.sample(candidate_indices, k=k)


class AllViewSelector(BaseViewSelector):
    """Selects all remaining views at once."""

    def select_views(
        self, active_indices: List[int], remaining_indices: List[int], num_to_select: int, **kwargs
    ) -> List[int]:
        return remaining_indices.copy()


class VanillaViewSelector(BaseViewSelector):
    """Vanilla view selection."""

    def __init__(
        self,
        view_metric: Literal["coverage", "fig", "view_fig", "fig_diag", "view_fig_diag", "fig_color_field"] = "coverage",
        num_nearest_neighbors: int = 5,
        intrinsics_scale: float = 1.0,
        use_kdtree_filter: bool = True,
    ):
        """
        Initialize vanilla view selector.

        Args:
            num_nearest_neighbors: Number of nearest neighbors to consider when using KD-tree filtering
            intrinsics_scale: Scale factor for camera intrinsics during scoring (for efficiency)
            use_kdtree_filter: Whether to use KD-tree filtering to reduce candidate pool
        """
        self.view_metric = view_metric
        self.num_nearest_neighbors = num_nearest_neighbors
        self.intrinsics_scale = intrinsics_scale
        self.use_kdtree_filter = use_kdtree_filter

    # NOTE: OLDER CODE
    def select_views(
        self, active_indices: List[int], remaining_indices: List[int], num_to_select: int, **kwargs
    ) -> List[int]:
        """
        Vanilla view selection using view metric.

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
        if not remaining_indices or len(remaining_indices) == 0:
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

        # TODO: Unify FisherSplat and CoverageSplat models into one.
        if isinstance(model, NBVSplatModel):
            # Feed the "training cameras" to the model to update coverage metrics.
            # Fisher-RF has its own way to use "training cameras".
            # TODO: Need to add a flag to choose a subset of the training cameras to use, or maybe just a sliding window.
            if self.view_metric in ["coverage", "fig", "view_fig", "fig_diag", "view_fig_diag", "fig_color_field"]:
                if len(active_indices) > 0:
                    training_cameras = [all_cameras[idx : idx + 1] for idx in active_indices]
                    model.update_view_attributes(training_cameras)
                else:
                    # If there are no active views, we don't need to do anything
                    pass

            # Score each candidate camera using coverage
            scores = []
            for cam_idx in candidate_indices:
                # Access camera using slice notation (Cameras expects tuple/slice, not list)
                camera = all_cameras[cam_idx : cam_idx + 1].to(model.device)
                score = model.view_metric_score_for_camera(
                    camera,
                    intrinsics_scale=self.intrinsics_scale,
                )
                scores.append((cam_idx, score.item()))

        # elif isinstance(model, FisherSplatModel):
        #     training_cameras = [all_cameras[idx : idx + 1] for idx in active_indices]
        #     candidate_cameras = [all_cameras[idx : idx + 1] for idx in candidate_indices]
        #     scores = model.coverage_score_for_camera(training_cameras, candidate_cameras)

        #     scores = [(candidate_indices[idx], score.item()) for idx, score in enumerate(scores)]
        else:
            raise ValueError(f"Invalid model type: {type(model)}")

        # Sort by score (ascending) and select top candidates (lowest coverage = most novel views)
        scores.sort(key=lambda x: x[1], reverse=False)
        k = min(num_to_select, len(scores))
        selected_indices = [idx for idx, _ in scores[:k]]

        print("Successfully selected views")
        return selected_indices

def create_view_selector(
    mode: Optional[str] = None,
    num_nearest_neighbors: Optional[int] = None,
    intrinsics_scale: Optional[float] = None,
    use_kdtree_filter: Optional[bool] = None,
    view_metric: Optional[str] = None,
) -> BaseViewSelector:
    """
    Factory function to create a BaseViewSelector based on mode string.

    Args:
        mode: Selection mode - "random", "all", "optics", or None (defaults to "random")
        num_nearest_neighbors: Number of nearest neighbors for KD-tree filtering (optional, applies to random and optics selectors)
        intrinsics_scale: Intrinsics scale for optics selector (optional)
        use_kdtree_filter: Whether to use KD-tree filtering (optional, applies to random and optics selectors)
        view_metric: View metric for selection (optional)

    Returns:
        BaseViewSelector instance
    """
    print(f"Creating view selector for mode: {mode}")
    if mode is None or mode == "random":
        # Use provided config values or defaults for random selector
        return RandomViewSelector(
            num_nearest_neighbors=num_nearest_neighbors if num_nearest_neighbors is not None else 5,
            use_kdtree_filter=use_kdtree_filter if use_kdtree_filter is not None else False,
        )
    elif mode == "all":
        return AllViewSelector()
    elif mode == "basic":
        # Use provided config values or defaults
        return VanillaViewSelector(
            view_metric=view_metric if view_metric is not None else "coverage",
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
                f"Expected 'random', 'all', 'basic', or a dotted path to a BaseViewSelector class. "
                f"Error: {e}"
            )
