"""View selector for progressive view selection in Shadow Splat"""

import random
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import KDTree
import torch
from shadow_splat.model import ShadowSplatModel, FisherSplatModel


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

    # NOTE: OLDER CODE
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

        if self.coverage_metric in ["coverage_lit"]:
            light = datamanager.train_dataparser_outputs.lights[0]  # NOTE: assume static light
        else:
            light = None

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

        if isinstance(model, ShadowSplatModel):
            if self.coverage_metric in ["coverage", "fig", "view_fig", "fig_diag", "view_fig_diag", "coverage_lit"]:
                # Feed the "training cameras" to the model to update coverage metrics.
                # Fisher-RF has its own way to use "training cameras".
                # TODO: Need to add a flag to choose a subset of the training cameras to use, or maybe just a sliding window.
                if self.coverage_metric in ["coverage", "coverage_lit"]:
                    model.reset_coverage()

                    if len(active_indices) > 0:
                        training_cameras = [all_cameras[idx : idx + 1] for idx in active_indices]
                        model.update_coverage(training_cameras)
                    else:
                        # If there are no active views, we don't need to do anything
                        pass

                elif self.coverage_metric in ["fig", "view_fig", "fig_diag", "view_fig_diag"]:
                    model.reset_fig()

                    if len(active_indices) > 0:
                        training_cameras = [all_cameras[idx : idx + 1] for idx in active_indices]
                        model.update_fig(training_cameras)
                    else:
                        # If there are no active views, we don't need to do anything
                        pass

                # Score each candidate camera using coverage
                scores = []
                for cam_idx in candidate_indices:
                    # Access camera using slice notation (Cameras expects tuple/slice, not list)
                    camera = all_cameras[cam_idx : cam_idx + 1].to(model.device)
                    score = model.coverage_score_for_camera(
                        camera,
                        light=light,
                        intrinsics_scale=self.intrinsics_scale,
                        metric=self.coverage_metric,
                    )
                    scores.append((cam_idx, score.item()))

            else:
                raise ValueError(f"Invalid coverage metric: {self.coverage_metric}")

        elif isinstance(model, FisherSplatModel):
            training_cameras = [all_cameras[idx : idx + 1] for idx in active_indices]
            candidate_cameras = [all_cameras[idx : idx + 1] for idx in candidate_indices]
            scores = model.coverage_score_for_camera(training_cameras, candidate_cameras)

            scores = [(candidate_indices[idx], score.item()) for idx, score in enumerate(scores)]
        else:
            raise ValueError(f"Invalid model type: {type(model)}")

        # Sort by score (ascending) and select top candidates (lowest coverage = most novel views)
        scores.sort(key=lambda x: x[1], reverse=False)
        k = min(num_to_select, len(scores))
        selected_indices = [idx for idx, _ in scores[:k]]

        return selected_indices

    # NOTE: Newer code that does rollouts
    # def select_views(
    #     self, active_indices: List[int], remaining_indices: List[int], num_to_select: int, **kwargs
    # ) -> List[int]:
    #     """
    #     Optics-based view selection using coverage/FIG scoring with a greedy policy:
    #     iteratively pick the best camera, hypothetically add it to training,
    #     update coverage/fig for just that camera, then rescore.
    #     """
    #     if not remaining_indices or len(remaining_indices) == 0:
    #         return []

    #     # Context
    #     model = kwargs.get("model")
    #     datamanager = kwargs.get("datamanager")
    #     if model is None or datamanager is None:
    #         k = min(max(1, num_to_select), len(remaining_indices))
    #         return random.sample(remaining_indices, k=k)

    #     # Clear any previous render info to avoid leaks
    #     if hasattr(model, "info"):
    #         if isinstance(model.info, dict):
    #             for _, v in list(model.info.items()):
    #                 if isinstance(v, torch.Tensor):
    #                     del v
    #             model.info.clear()
    #         model.info = {}

    #     try:
    #         # Candidate pool (with optional KD-tree filtering around last active)
    #         candidate_indices = remaining_indices.copy()
    #         all_cameras = datamanager.train_dataset.cameras

    #         # Precompute candidate origins for optional KD-tree reduction
    #         cand_origins = []
    #         for idx in candidate_indices:
    #             cam = all_cameras[idx:idx+1]
    #             cand_origins.append(cam.camera_to_worlds[0, :3, -1].cpu().numpy())
    #         cand_origins = np.array(cand_origins)

    #         if self.use_kdtree_filter and len(active_indices) > 0:
    #             root_idx = active_indices[-1]
    #             root_cam = all_cameras[root_idx:root_idx+1]
    #             root_origin = root_cam.camera_to_worlds[0, :3, -1].cpu().numpy()
    #             kdt = KDTree(cand_origins)
    #             k = min(self.num_nearest_neighbors, len(candidate_indices))
    #             _, nn_idx = kdt.query(root_origin.reshape(1, 3), k=k)
    #             nn_idx = nn_idx.flatten()
    #             candidate_indices = [candidate_indices[i] for i in nn_idx]

    #         selected_indices: List[int] = []

    #         # ---- Seed the internal model state with current active views ----
    #         if isinstance(model, ShadowSplatModel):
    #             # Reset and build the current coverage/fig state from active set once
    #             if self.coverage_metric == "coverage":
    #                 model.reset_coverage()
    #                 if len(active_indices) > 0:
    #                     model.update_coverage([all_cameras[i:i+1] for i in active_indices])
    #             elif self.coverage_metric in ["fig", "view_fig"]:
    #                 model.reset_fig()
    #                 if len(active_indices) > 0:
    #                     model.update_fig([all_cameras[i:i+1] for i in active_indices])
    #             else:
    #                 raise ValueError(f"Invalid coverage metric: {self.coverage_metric}")

    #         elif isinstance(model, FisherSplatModel):
    #             # For Fisher, we pass training_cameras each scoring round (no internal reset needed)
    #             pass
    #         else:
    #             raise ValueError(f"Invalid model type: {type(model)}")

    #         # ---- Greedy loop ----
    #         picks = min(num_to_select, len(candidate_indices))
    #         for _ in range(picks):
    #             if len(candidate_indices) == 0:
    #                 break

    #             # Score the current pool
    #             scores: List[tuple[int, float]] = []
    #             if isinstance(model, ShadowSplatModel):
    #                 for cam_idx in candidate_indices:
    #                     cam = all_cameras[cam_idx:cam_idx+1].to(model.device)
    #                     try:
    #                         s = model.coverage_score_for_camera(
    #                             cam, intrinsics_scale=self.intrinsics_scale, metric=self.coverage_metric
    #                         )  # lower is better
    #                         scores.append((cam_idx, float(s.item())))
    #                     except Exception as e:
    #                         print(f"Warning: Failed to score camera {cam_idx}: {e}")
    #                         scores.append((cam_idx, float("inf")))
    #                     finally:
    #                         del cam

    #             else:  # FisherSplatModel
    #                 training_cams = [all_cameras[i:i+1] for i in (active_indices + selected_indices)]
    #                 candidate_cams = [all_cameras[i:i+1] for i in candidate_indices]
    #                 fisher_scores = model.coverage_score_for_camera(training_cams, candidate_cams)
    #                 # fisher_scores is a list/iterable of tensors; lower is better (we negate mean uncertainty upstream)
    #                 scores = [(candidate_indices[i], float(fisher_scores[i].item())) for i in range(len(candidate_indices))]

    #             # Pick the lowest score
    #             scores.sort(key=lambda x: x[1])
    #             best_idx = scores[0][0]
    #             selected_indices.append(best_idx)
    #             candidate_indices.remove(best_idx)

    #             # Update internal state incrementally for ShadowSplatModel
    #             if isinstance(model, ShadowSplatModel):
    #                 best_cam = all_cameras[best_idx:best_idx+1]
    #                 if self.coverage_metric == "coverage":
    #                     model.update_coverage([best_cam])
    #                 else:  # "fig" or "view_fig"
    #                     model.update_fig([best_cam])

    #                 print("Updating hypothetical coverage/fig using camera", best_idx)

    #             # Memory hygiene between iterations
    #             if hasattr(model, "info"):
    #                 if isinstance(model.info, dict):
    #                     for _, v in list(model.info.items()):
    #                         if isinstance(v, torch.Tensor):
    #                             del v
    #                     model.info.clear()
    #                 model.info = {}
    #             gc.collect()
    #             if torch.cuda.is_available():
    #                 torch.cuda.empty_cache()

    #     finally:
    #         # Final cleanup
    #         if hasattr(model, "info"):
    #             if isinstance(model.info, dict):
    #                 for _, v in list(model.info.items()):
    #                     if isinstance(v, torch.Tensor):
    #                         del v
    #                 model.info.clear()
    #             model.info = {}
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    #     return selected_indices


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
        num_nearest_neighbors: Number of nearest neighbors for KD-tree filtering (optional, applies to random and optics selectors)
        intrinsics_scale: Intrinsics scale for optics selector (optional)
        use_kdtree_filter: Whether to use KD-tree filtering (optional, applies to random and optics selectors)
        coverage_metric: Coverage metric for optics selector (optional)

    Returns:
        ViewSelector instance
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
