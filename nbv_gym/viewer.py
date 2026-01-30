"""Custom viewer for Shadow Splat"""

from typing import Optional, Literal

import time
import numpy as np
import torch
import viser
import viser.transforms as tf
from nerfstudio.viewer.viewer import Viewer, VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nerfstudio.viewer.render_state_machine import RenderAction
from nerfstudio.data.datasets.base_dataset import InputDataset

from nbv_gym.model import (
    NBVSplatModel,
    NBVSplatModelConfig,
)

class NBVViewer(Viewer):
    """Custom viewer with ability to visualize adding new viewpoints to the scene."""

    def __init__(self, *args, **kwargs):
        # Convert SplatfactoModel to ShadowSplatModel BEFORE calling parent __init__
        pipeline = kwargs["pipeline"]

        # if type(pipeline.model) is SplatfactoModel:
        #     print("model is splatfacto, converting to nbv splat")
        #     self._convert_model(pipeline)
        # else:
        #     raise ValueError(f"Unsupported model type: {type(pipeline.model)}")

        # Initialize the parent Viewer class
        super().__init__(*args, **kwargs)

        print("NBV-Viewer | __init__")
        print(f"pipeline.model: {type(pipeline.model)}")

        # Settings for hiding/showing candidate views
        self._hide_candidates = True
        self._last_active_indices = set()
        self.hide_candidates_checkbox = self.viser_server.gui.add_checkbox(
            label="Hide candidate views", disabled=False, initial_value=True
        )
        self.hide_candidates_checkbox.on_update(lambda _: self._on_hide_candidates_toggle())

    def _convert_model(self, pipeline):
        """Convert the model to a NBVSplatModel"""
        config = NBVSplatModelConfig()
        model = NBVSplatModel(config, pipeline.model.scene_box, pipeline.model.num_train_data)
        model.populate_modules()
        model.seed_points = pipeline.model.seed_points
        model.gauss_params = pipeline.model.gauss_params
        model = model.to(pipeline.device)
        pipeline.model = model
        if hasattr(pipeline, "_model"):
            pipeline._model = model

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            num_rays_per_batch: number of rays per batch, used during training
        """
        self.step = step

        if len(self.render_statemachines) == 0:
            return
        # this stops training while moving to make the response smoother
        while time.time() - self.last_move_time < 0.1:
            time.sleep(0.05)
        if (
            self.trainer is not None
            and self.trainer.training_state == "training"
            and self.train_util != 1
        ):
            if (
                EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
                and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
            ):
                train_s = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                vis_s = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
                train_util = self.train_util
                vis_n = self.control_panel.max_res**2
                train_n = num_rays_per_batch
                train_time = train_n / train_s
                vis_time = vis_n / vis_s

                render_freq = train_util * vis_time / (train_time - train_util * train_time)
            else:
                render_freq = 30
            if step > self.last_step + render_freq:
                self.last_step = step
                clients = self.viser_server.get_clients()
                for id in clients:
                    camera_state = self.get_camera_state(clients[id])
                    if camera_state is not None:
                        self.render_statemachines[id].action(RenderAction("step", camera_state))
                self.update_camera_poses()
                # Update which training cameras are visible based on active subset
                self._update_train_camera_visibility()
                self.update_step(step)

    def _on_hide_candidates_toggle(self):
        self._hide_candidates = bool(self.hide_candidates_checkbox.value)
        self._update_train_camera_visibility(force=True)

    def init_scene(
        self,
        train_dataset: InputDataset,
        train_state: Literal["training", "paused", "completed"],
        eval_dataset: Optional[InputDataset] = None,
    ) -> None:
        """Override init_scene to set camera visibility immediately after creation."""
        # Call parent's init_scene to create the cameras
        super().init_scene(train_dataset, train_state, eval_dataset)

        # Immediately set camera visibility based on active/inactive status
        # This ensures visibility is set right away, not waiting for update_scene()
        self._update_train_camera_visibility(force=True)

    def _get_active_indices(self) -> set:
        """Return the active training indices from the datamanager if present, else all indices."""
        dm = self.pipeline.datamanager
        try:
            if hasattr(dm, "active_train_indices") and dm.active_train_indices is not None:
                return set(int(i) for i in dm.active_train_indices)
        except Exception:
            pass
        # Fallback to all train indices
        try:
            total = len(dm.train_dataset)
            return set(range(total))
        except Exception:
            return set()

    def _add_synthetic_camera_frustum(self, synthetic_idx: int, camera: Cameras) -> None:
        """Add a camera frustum for a synthetic view.

        Args:
            synthetic_idx: Negative index for the synthetic view (-1, -2, etc.)
            camera: The camera object for the synthetic view
        """
        if not hasattr(self, "camera_handles") or self.camera_handles is None:
            return

        # Skip if already exists
        if synthetic_idx in self.camera_handles:
            return

        # Extract camera parameters
        c2w = camera.camera_to_worlds[0].cpu().numpy()  # [3, 4]
        # Add homogeneous row to make it 4x4
        c2w_4x4 = np.eye(4)
        c2w_4x4[:3, :] = c2w

        # Convert to position and quaternion
        position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
        R = c2w[:3, :3]
        # viser uses wxyz quaternion format
        # Apply OpenCV to OpenGL convention (180° X-axis rotation) like parent class does
        R_so3 = tf.SO3.from_matrix(R)
        R_so3 = R_so3 @ tf.SO3.from_x_radians(np.pi)
        wxyz = R_so3.wxyz

        # Get FOV from camera intrinsics
        fx = float(camera.fx[0].cpu())
        fy = float(camera.fy[0].cpu())
        width = int(camera.width[0].cpu())
        height = int(camera.height[0].cpu())
        fov = 2 * np.arctan(height / (2 * fy))
        aspect = width / height

        # Create camera frustum (green for active synthetic view)
        camera_handle = self.viser_server.scene.add_camera_frustum(
            name=f"/cameras/synthetic_{-synthetic_idx}",
            fov=fov,
            aspect=aspect,
            scale=0.3,
            wxyz=wxyz,
            position=position,
            color=(0.0, 1.0, 0.0),  # Green for active
            line_width=3.0,
        )
        self.camera_handles[synthetic_idx] = camera_handle

    def _update_train_camera_visibility(self, force: bool = False) -> None:
        """Highlight active cameras in green and hide/show candidate (inactive) cameras based on checkbox.

        Active cameras are always highlighted in green.
        When "Hide candidate views" is checked, inactive cameras are hidden.
        When unchecked, all cameras are visible.

        Operates only on already-created frustums (Viewer limits number displayed).
        Also creates frustums for synthetic views (negative indices) if needed.
        """
        if not hasattr(self, "camera_handles") or self.camera_handles is None:
            return
        active = self._get_active_indices()

        # Check for new synthetic views (negative indices) and create frustums for them
        dm = self.pipeline.datamanager
        if hasattr(dm, "synthetic_cameras"):
            for i, camera in enumerate(dm.synthetic_cameras):
                synthetic_idx = -(i + 1)  # -1, -2, -3, etc.
                if synthetic_idx not in self.camera_handles:
                    self._add_synthetic_camera_frustum(synthetic_idx, camera)

        # Only skip update if indices haven't changed, hiding is off, and not forcing update
        if not force and active == self._last_active_indices:
            # If indices haven't changed and hiding is off, we can skip
            # (active cameras are already green, inactive ones already visible)
            if not self._hide_candidates:
                return
        self._last_active_indices = active

        # Process each camera: highlight active ones in green, hide/show inactive ones based on checkbox
        for idx, handle in list(self.camera_handles.items()):
            if idx in active:
                if handle.color != (0.0, 1.0, 0.0):
                    # Remove and re-add the handle to apply the green color
                    # Setting the color with handle.color doesn't work
                    handle.remove()
                    camera_handle = self.viser_server.scene.add_camera_frustum(
                        name=handle.name,
                        fov=handle.fov,
                        aspect=handle.aspect,
                        # scale=handle.scale,
                        scale=0.3,
                        image=handle.image,
                        wxyz=handle.wxyz,
                        position=handle.position,
                        color=(0.0, 1.0, 0.0),
                        line_width=3.0,
                    )
                    # Update the handle in the dictionary to point to the new handle
                    self.camera_handles[idx] = camera_handle
                    # camera_handle.on_click(self.create_on_click_callback(idx))
            else:
                # For inactive cameras, hide/show based on checkbox
                handle.visible = not self._hide_candidates

    def update_camera_poses(self):
        """Override to skip synthetic views (negative indices) which aren't in original_c2w.

        The parent class iterates over all camera_handles keys and looks them up in
        original_c2w. Synthetic views have negative indices and aren't in original_c2w,
        causing a KeyError. We temporarily filter them out before calling the parent.
        """
        if not hasattr(self, "camera_handles") or self.camera_handles is None:
            return

        # Filter out synthetic views (negative indices) before calling parent
        original_handles = self.camera_handles
        self.camera_handles = {k: v for k, v in original_handles.items() if k >= 0}

        try:
            super().update_camera_poses()
        finally:
            # Restore all handles (including synthetic views)
            self.camera_handles = original_handles