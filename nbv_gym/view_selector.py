"""View selector for progressive view selection in NBV-Gym"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import KDTree
import torch
from nerfstudio.cameras.cameras import Cameras
from nbv_gym.model import NBVSplatModel, FisherSplatModel
from nbv_gym.util.camera_pose import DifferentiableCameraPose
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
        view_metric: Literal["coverage", "fig", "view_fig", "fig_diag", "view_fig_diag", "fig_color_field", "fisher_rf"] = "coverage",
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

        # Handle FisherSplatModel vs NBVSplatModel
        if isinstance(model, FisherSplatModel):
            # FisherSplatModel uses render_uncertainty_rgb_depth() method
            # Get training and candidate cameras
            training_cameras = [all_cameras[idx : idx + 1] for idx in active_indices]
            candidate_cameras = [all_cameras[idx : idx + 1] for idx in candidate_indices]

            # Render uncertainty maps for all candidate cameras
            uncertainty_maps = model.render_uncertainty_rgb_depth(
                train_cameras=training_cameras,
                test_cameras=candidate_cameras,
                rgb_weight=model.config.rgb_uncertainty_weight,
                depth_weight=model.config.depth_uncertainty_weight,
            )

            # Score each candidate: higher uncertainty = more novel = lower score (we negate)
            scores = []
            for idx, unc_map in enumerate(uncertainty_maps):
                cam_idx = candidate_indices[idx]
                valid_mask = unc_map > 0
                # Negate so that higher uncertainty gives lower score (lower = better for selection)
                score = -((unc_map * valid_mask).sum() / valid_mask.sum().clamp(min=1)).item()
                scores.append((cam_idx, score))

        elif isinstance(model, NBVSplatModel):
            # Feed the "training cameras" to the model to update coverage/view metrics.
            # TODO: Need to add a flag to choose a subset of the training cameras to use, or maybe just a sliding window.
            if self.view_metric in ["coverage", "fig", "view_fig", "fig_diag", "view_fig_diag", "fig_color_field"]:
                if len(active_indices) > 0:
                    training_cameras = [all_cameras[idx : idx + 1] for idx in active_indices]
                    model.update_view_attributes(training_cameras)
                else:
                    # If there are no active views, we don't need to do anything
                    pass

            # Score each candidate camera using view metric
            scores = []
            for cam_idx in candidate_indices:
                # Access camera using slice notation (Cameras expects tuple/slice, not list)
                camera = all_cameras[cam_idx : cam_idx + 1].to(model.device)
                score = model.view_metric_score_for_camera(
                    camera,
                    intrinsics_scale=self.intrinsics_scale,
                )
                scores.append((cam_idx, score.item()))

        else:
            raise ValueError(f"Invalid model type: {type(model)}")

        # Sort by score (ascending) and select top candidates (lowest coverage = most novel views)
        scores.sort(key=lambda x: x[1], reverse=False)
        k = min(num_to_select, len(scores))
        selected_indices = [idx for idx, _ in scores[:k]]

        print("Successfully selected views")
        return selected_indices


@dataclass
class GradientDescentViewSelectorConfig:
    """Configuration for gradient descent view selector."""
    external_3dgs_config_path: Path  # Required: path to "all views" 3DGS config.yml
    num_gradient_steps: int = 50
    learning_rate_position: float = 0.001
    learning_rate_rotation: float = 0.001
    num_nearest_neighbors: int = 5
    intrinsics_scale: float = 1.0
    view_metric: str = "coverage"
    max_position_delta: float = 0.01  # Maximum position movement from initial pose (epsilon ball)
    max_rotation_delta: float = 0.01  # Maximum rotation movement from initial pose (epsilon ball on axis-angle)


class GradientDescentViewSelector(BaseViewSelector):
    """View selector that optimizes camera pose via gradient descent.

    This selector:
    1. Uses kdtree filtering to find the best initial pose from neighbors
    2. Optimizes the camera pose via gradient descent on the coverage metric
    3. Renders RGB from an external pre-trained "all views" 3DGS model
    4. Adds the optimized pose + rendered RGB to the datamanager as synthetic training data
    5. Uses optimized poses (not original dataset poses) for future kdtree queries
    """

    def __init__(self, config: GradientDescentViewSelectorConfig):
        """Initialize gradient descent view selector.

        Args:
            config: Configuration containing external model path and optimization params

        Raises:
            ValueError: If external_3dgs_config_path is not provided
        """
        self.config = config
        self.use_kdtree_filter = True  # Required for this selector

        if config.external_3dgs_config_path is None:
            raise ValueError(
                "GradientDescentViewSelector requires external_3dgs_config_path to be set. "
                "This should point to a pre-trained 3DGS model trained on all views."
            )

        # External model for RGB rendering (lazy loaded)
        self._external_model = None
        self._external_pipeline = None

        # Track optimized poses separately from original dataset
        self.optimized_cameras: List[Cameras] = []
        self.optimized_images: List[torch.Tensor] = []

    def _load_external_model(self):
        """Lazy load the external 3DGS model for RGB rendering."""
        if self._external_model is not None:
            return

        print(f"Loading external 3DGS model from {self.config.external_3dgs_config_path}")
        from nerfstudio.utils.eval_utils import eval_setup

        _, pipeline, _, _ = eval_setup(
            self.config.external_3dgs_config_path,
            test_mode="inference",
        )
        self._external_pipeline = pipeline
        self._external_model = pipeline.model
        self._external_model.eval()
        print("External 3DGS model loaded successfully")

    def _render_from_external_model(self, camera: Cameras) -> torch.Tensor:
        """Render RGB image from external 3DGS model.

        Args:
            camera: Camera to render from

        Returns:
            RGB image tensor [H, W, 3]
        """
        self._load_external_model()

        with torch.no_grad():
            camera = camera.to(self._external_model.device)
            outputs = self._external_model(camera)

        return outputs["rgb"]  # [H, W, 3]

    def _get_kdtree_candidates(
        self,
        active_indices: List[int],
        remaining_indices: List[int],
        datamanager,
    ) -> List[int]:
        """Get candidate indices using kdtree nearest neighbor filtering.

        Uses optimized_cameras if available, otherwise uses last active pose from dataset.

        Args:
            active_indices: Current active view indices
            remaining_indices: Available view indices not yet in active set
            datamanager: The ViewSelectionDataManager instance

        Returns:
            List of candidate indices (subset of remaining_indices)
        """
        all_cameras = datamanager.train_dataset.cameras

        # Build candidate origins from remaining indices (always from original dataset)
        candidate_origins_list = []
        for idx in remaining_indices:
            cam = all_cameras[idx : idx + 1]
            origin = cam.camera_to_worlds[0, :3, 3].cpu().numpy()
            candidate_origins_list.append(origin)
        candidate_origins = np.array(candidate_origins_list)

        # Determine root pose for kdtree query
        if len(self.optimized_cameras) > 0:
            # Use last optimized camera as root
            root_origin = self.optimized_cameras[-1].camera_to_worlds[0, :3, 3].cpu().numpy()
        elif len(active_indices) > 0:
            # Use last active camera from original dataset
            # Filter out negative indices (synthetic views)
            real_active_indices = [idx for idx in active_indices if idx >= 0]
            if real_active_indices:
                root_idx = real_active_indices[-1]
                root_camera = all_cameras[root_idx : root_idx + 1]
                root_origin = root_camera.camera_to_worlds[0, :3, 3].cpu().numpy()
            else:
                # All active indices are synthetic, use last optimized camera
                root_origin = self.optimized_cameras[-1].camera_to_worlds[0, :3, 3].cpu().numpy()
        else:
            # No active views yet, return all candidates
            return remaining_indices

        # Build kdtree and query nearest neighbors
        kdtree = KDTree(candidate_origins)
        k = min(self.config.num_nearest_neighbors, len(remaining_indices))
        _, nearest_indices = kdtree.query(root_origin.reshape(1, 3), k=k)

        # Return filtered candidate indices
        nearest_indices = nearest_indices.flatten()
        return [remaining_indices[i] for i in nearest_indices]

    def _optimize_camera_pose(
        self,
        initial_camera: Cameras,
        model: NBVSplatModel,
    ) -> Cameras:
        """Optimize camera pose via gradient descent on coverage metric.

        Args:
            initial_camera: Initial camera pose to optimize from
            model: NBVSplatModel instance for computing coverage score

        Returns:
            Optimized camera with new pose
        """
        device = model.device

        # Extract initial c2w matrix
        initial_c2w = initial_camera.camera_to_worlds.clone().to(device)

        # Create differentiable pose parameters
        diff_pose = DifferentiableCameraPose(initial_c2w).to(device)

        # Store initial position and rotation for epsilon ball constraints
        initial_position = diff_pose.position.detach().clone()
        initial_axis_angle = diff_pose.axis_angle.detach().clone()

        # Create optimizers with different learning rates for position and rotation
        optimizer = torch.optim.Adam([
            {"params": [diff_pose.position], "lr": self.config.learning_rate_position},
            {"params": [diff_pose.axis_angle], "lr": self.config.learning_rate_rotation},
        ])

        # Store initial score for comparison
        initial_score = None

        # Gradient descent optimization
        for step in range(self.config.num_gradient_steps):
            optimizer.zero_grad()

            # Get current c2w from differentiable pose
            c2w = diff_pose.get_camera_to_world()  # [1, 3, 4]

            # Create camera with current pose
            # We need to create a new Cameras object with the optimized c2w
            optimized_camera = Cameras(
                camera_to_worlds=c2w,
                fx=initial_camera.fx.clone(),
                fy=initial_camera.fy.clone(),
                cx=initial_camera.cx.clone(),
                cy=initial_camera.cy.clone(),
                width=initial_camera.width.clone(),
                height=initial_camera.height.clone(),
                camera_type=initial_camera.camera_type,
            ).to(device)

            # Compute coverage score (differentiable)
            score = model.view_metric_score_for_camera_differentiable(
                optimized_camera,
                intrinsics_scale=self.config.intrinsics_scale,
            )

            if initial_score is None:
                initial_score = score.item()

            # Backward pass (minimize coverage score = find poorly-covered areas)
            score.backward()

            # Optimizer step
            optimizer.step()

            # Project position back onto epsilon ball if exceeded
            with torch.no_grad():
                pos_delta = diff_pose.position - initial_position
                pos_delta_norm = pos_delta.norm()
                if pos_delta_norm > self.config.max_position_delta:
                    # Snap back to surface of epsilon ball
                    diff_pose.position.copy_(
                        initial_position + pos_delta * (self.config.max_position_delta / pos_delta_norm)
                    )

                # Project rotation back onto epsilon ball if exceeded
                rot_delta = diff_pose.axis_angle - initial_axis_angle
                rot_delta_norm = rot_delta.norm()
                if rot_delta_norm > self.config.max_rotation_delta:
                    # Snap back to surface of epsilon ball
                    diff_pose.axis_angle.copy_(
                        initial_axis_angle + rot_delta * (self.config.max_rotation_delta / rot_delta_norm)
                    )

            if step % 10 == 0:
                print(f"  GD step {step}: score = {score.item():.6f}")

        # Get final optimized pose
        with torch.no_grad():
            final_c2w = diff_pose.get_camera_to_world()

        final_score = score.item()
        print(f"  Optimization complete: {initial_score:.6f} -> {final_score:.6f}")

        if final_score >= initial_score:
            print("  Warning: Gradient descent did not improve coverage score")

        # Create final optimized camera
        optimized_camera = Cameras(
            camera_to_worlds=final_c2w.detach(),
            fx=initial_camera.fx.clone(),
            fy=initial_camera.fy.clone(),
            cx=initial_camera.cx.clone(),
            cy=initial_camera.cy.clone(),
            width=initial_camera.width.clone(),
            height=initial_camera.height.clone(),
            camera_type=initial_camera.camera_type,
        )

        return optimized_camera

    def select_views(
        self, active_indices: List[int], remaining_indices: List[int], num_to_select: int, **kwargs
    ) -> List[int]:
        """Select views using gradient descent optimization.

        This method:
        1. Uses kdtree to filter candidates
        2. Scores candidates to find best initial pose
        3. Runs gradient descent to optimize the pose
        4. Renders RGB from external model
        5. Adds synthetic view to datamanager
        6. Returns [-1] sentinel to indicate synthetic view added

        Args:
            active_indices: Current active view indices
            remaining_indices: Available view indices not yet in active set
            num_to_select: Number of views to select (ignored, always adds 1 synthetic view)
            **kwargs: Additional context including:
                - model: The NBVSplatModel instance
                - datamanager: The ViewSelectionDataManager instance
                - step: Current training step

        Returns:
            [-1] sentinel value indicating synthetic view was added
        """
        if not remaining_indices:
            return []

        model = kwargs.get("model")
        datamanager = kwargs.get("datamanager")

        if model is None or datamanager is None:
            print("Warning: model or datamanager not provided, falling back to random selection")
            k = min(max(1, num_to_select), len(remaining_indices))
            return random.sample(remaining_indices, k=k)

        if not isinstance(model, NBVSplatModel):
            raise ValueError(
                f"GradientDescentViewSelector requires NBVSplatModel, got {type(model)}"
            )

        all_cameras = datamanager.train_dataset.cameras

        # Step 1: KDTree filter to get candidates
        candidate_indices = self._get_kdtree_candidates(
            active_indices, remaining_indices, datamanager
        )
        print(f"KDTree filtered to {len(candidate_indices)} candidates")

        # Step 2: Update view attributes for scoring
        # Get real active indices (non-synthetic)
        real_active_indices = [idx for idx in active_indices if idx >= 0]
        if len(real_active_indices) > 0:
            training_cameras = [all_cameras[idx : idx + 1] for idx in real_active_indices]
            # Also include optimized cameras if any
            training_cameras.extend(self.optimized_cameras)
            model.update_view_attributes(training_cameras)

        # Step 3: Score candidates to find best starting point
        scores = []
        for cam_idx in candidate_indices:
            camera = all_cameras[cam_idx : cam_idx + 1].to(model.device)
            score = model.view_metric_score_for_camera(
                camera,
                intrinsics_scale=self.config.intrinsics_scale,
            )
            scores.append((cam_idx, score.item()))

        # Sort by score (ascending) - lowest score = most novel
        scores.sort(key=lambda x: x[1])
        best_idx = scores[0][0]
        best_initial_camera = all_cameras[best_idx : best_idx + 1].to(model.device)
        print(f"Best initial candidate: index {best_idx}, score {scores[0][1]:.6f}")

        # Step 4: Gradient descent optimization
        print("Starting gradient descent optimization...")
        optimized_camera = self._optimize_camera_pose(best_initial_camera, model)

        # Step 5: Render RGB from external model
        print("Rendering RGB from external 3DGS model...")
        rendered_rgb = self._render_from_external_model(optimized_camera)

        # Step 6: Store optimized camera and image
        self.optimized_cameras.append(optimized_camera)
        self.optimized_images.append(rendered_rgb)

        # Step 7: Add synthetic view to datamanager
        synthetic_idx = datamanager.add_synthetic_view(optimized_camera, rendered_rgb)
        datamanager.active_train_indices.append(synthetic_idx)
        datamanager.active_unseen_cameras.append(synthetic_idx)

        print(f"Added synthetic view with index {synthetic_idx}")
        print(f"Total optimized cameras: {len(self.optimized_cameras)}")

        # Return the dataset pose used as starting point so it gets consumed from remaining_indices
        # The synthetic view was already added to active_train_indices above (line 614)
        # By returning best_idx, the datamanager will also add it to active_train_indices,
        # which removes it from remaining_indices so it won't be reused as a starting point
        return [best_idx]


def create_view_selector(
    mode: Optional[str] = None,
    num_nearest_neighbors: Optional[int] = None,
    intrinsics_scale: Optional[float] = None,
    use_kdtree_filter: Optional[bool] = None,
    view_metric: Optional[str] = None,
    external_3dgs_config_path: Optional[Path] = None,
    gd_num_gradient_steps: Optional[int] = None,
    gd_learning_rate_position: Optional[float] = None,
    gd_learning_rate_rotation: Optional[float] = None,
) -> BaseViewSelector:
    """
    Factory function to create a BaseViewSelector based on mode string.

    Args:
        mode: Selection mode - "random", "all", "basic", "gradient_descent", or None (defaults to "random")
        num_nearest_neighbors: Number of nearest neighbors for KD-tree filtering (optional)
        intrinsics_scale: Intrinsics scale for view metric scoring (optional)
        use_kdtree_filter: Whether to use KD-tree filtering (optional)
        view_metric: View metric for selection (optional)
        external_3dgs_config_path: Path to external 3DGS config.yml (required for gradient_descent mode)
        gd_num_gradient_steps: Number of gradient descent steps (optional, for gradient_descent mode)
        gd_learning_rate_position: Learning rate for position (optional, for gradient_descent mode)
        gd_learning_rate_rotation: Learning rate for rotation (optional, for gradient_descent mode)

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
    elif mode == "gradient_descent":
        # Validate required parameters
        if external_3dgs_config_path is None:
            raise ValueError(
                "gradient_descent view selector requires external_3dgs_config_path. "
                "Set --pipeline.external_3dgs_config_path to path of pre-trained 3DGS config.yml"
            )
        if use_kdtree_filter is False:
            raise ValueError(
                "gradient_descent view selector requires use_kdtree_filter=True. "
                "Set --pipeline.view_selection_use_kdtree_filter true"
            )

        config = GradientDescentViewSelectorConfig(
            external_3dgs_config_path=Path(external_3dgs_config_path),
            num_gradient_steps=gd_num_gradient_steps if gd_num_gradient_steps is not None else 50,
            learning_rate_position=gd_learning_rate_position if gd_learning_rate_position is not None else 0.01,
            learning_rate_rotation=gd_learning_rate_rotation if gd_learning_rate_rotation is not None else 0.001,
            num_nearest_neighbors=num_nearest_neighbors if num_nearest_neighbors is not None else 5,
            intrinsics_scale=intrinsics_scale if intrinsics_scale is not None else 1.0,
            view_metric=view_metric if view_metric is not None else "coverage",
        )
        return GradientDescentViewSelector(config)
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
                f"Expected 'random', 'all', 'basic', 'gradient_descent', or a dotted path to a BaseViewSelector class. "
                f"Error: {e}"
            )
