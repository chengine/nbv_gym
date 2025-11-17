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

from shadow_splat.model import (
    ShadowSplatModel,
    ShadowSplatModelConfig,
    FisherSplatModelConfig,
    FisherSplatModel,
)


class ShadowSplatViewer(Viewer):
    """Custom viewer with an additional slider for adjusting the light source position dynamically."""

    RGB_INTENSITY = False

    def __init__(self, *args, **kwargs):
        # Convert SplatfactoModel to ShadowSplatModel BEFORE calling parent __init__
        pipeline = kwargs["pipeline"]

        if type(pipeline.model) is SplatfactoModel:
            print("model is splatfacto, converting to shadow splat")
            self._convert_model(pipeline)
        elif type(pipeline.model) is FisherSplatModel:
            # Do nothing
            pass
        elif type(pipeline.model) is ShadowSplatModel:
            # Do nothing
            pass
        else:
            raise ValueError(f"Unsupported model type: {type(pipeline.model)}")

        # Initialize the parent Viewer class
        super().__init__(*args, **kwargs)

        print("ShadowSplatViewer | __init__")
        print(f"pipeline.model: {type(pipeline.model)}")

        tabs = self.viser_server.gui.add_tab_group()
        lighting_tab = tabs.add_tab("Light", viser.Icon.SUN)

        with lighting_tab:
            initial_ambient = torch.sigmoid(pipeline.model.light_params["ambient"]).item()
            initial_intensity = (
                torch.exp(pipeline.model.light_params["intensity"]).detach().cpu().numpy()
            ) * np.ones(3)
            self._add_light_source_slider(
                initial_ambient=initial_ambient,
                initial_intensity=initial_intensity,
            )

        # Settings for hiding/showing candidate views
        self._hide_candidates = True
        self._last_active_indices = set()
        self.hide_candidates_checkbox = self.viser_server.gui.add_checkbox(
            label="Hide candidate views", disabled=False, initial_value=True
        )
        self.hide_candidates_checkbox.on_update(lambda _: self._on_hide_candidates_toggle())

    def _convert_model(self, pipeline):
        """Convert the model to a ShadowSplatModel"""
        config = ShadowSplatModelConfig()
        model = ShadowSplatModel(config, pipeline.model.scene_box, pipeline.model.num_train_data)
        model.populate_modules()
        model.seed_points = pipeline.model.seed_points
        model.gauss_params = pipeline.model.gauss_params
        model = model.to(pipeline.device)
        pipeline.model = model
        if hasattr(pipeline, "_model"):
            pipeline._model = model

    def _add_light_source_slider(
        self,
        initial_ambient=0.01,
        initial_intensity=np.ones(3),
    ):
        """Add a slider to the control panel for adjusting the light source position."""
        self.az_slider = self.viser_server.gui.add_slider(
            label="Azimuth", min=-180.0, max=180.0, step=0.1, initial_value=0.0
        )
        self.el_slider = self.viser_server.gui.add_slider(
            label="Elevation", min=-90.0, max=90.0, step=0.1, initial_value=0.0
        )
        self.radius_slider = self.viser_server.gui.add_slider(
            label="Radius", min=0.0, max=10.0, step=0.1, initial_value=1.0
        )
        self.dim_slider = self.viser_server.gui.add_slider(
            label="Dimension", min=0.0, max=5000.0, step=1.0, initial_value=2000
        )
        self.focal_length_slider = self.viser_server.gui.add_slider(
            label="Focal length", min=0.0, max=3000.0, step=1.0, initial_value=1250
        )
        self.ambient_slider = self.viser_server.gui.add_slider(
            label="Ambient", min=0.0, max=1.0, step=0.01, initial_value=initial_ambient
        )
        if self.RGB_INTENSITY:
            self.red_intensity_slider = self.viser_server.gui.add_slider(
                label="Red intensity",
                min=0.0,
                max=10.0,
                step=0.1,
                initial_value=initial_intensity[0],
            )
            self.green_intensity_slider = self.viser_server.gui.add_slider(
                label="Green intensity",
                min=0.0,
                max=10.0,
                step=0.1,
                initial_value=initial_intensity[1],
            )
            self.blue_intensity_slider = self.viser_server.gui.add_slider(
                label="Blue intensity",
                min=0.0,
                max=10.0,
                step=0.1,
                initial_value=initial_intensity[2],
            )
        else:
            self.intensity_slider = self.viser_server.gui.add_slider(
                label="Intensity", min=0.0, max=10.0, step=0.1, initial_value=initial_intensity[0]
            )
        self.origin_input = self.viser_server.gui.add_vector3(
            label="Origin",
            step=0.1,
            initial_value=(0.0, 0.0, 0.0),
        )
        self.camera_type_select = self.viser_server.gui.add_dropdown(
            label="Camera Type",
            options=["Perspective", "Orthographic", "Fisheye"],
            initial_value="Orthographic",
        )
        # Set the callbacks
        self.az_slider.on_update(self.update_light_source_pose)
        self.el_slider.on_update(self.update_light_source_pose)
        self.radius_slider.on_update(self.update_light_source_pose)
        self.dim_slider.on_update(self.update_light_source_pose)
        self.focal_length_slider.on_update(self.update_light_source_pose)
        if self.RGB_INTENSITY:
            self.red_intensity_slider.on_update(self.update_light_source_pose)
            self.green_intensity_slider.on_update(self.update_light_source_pose)
            self.blue_intensity_slider.on_update(self.update_light_source_pose)
        else:
            self.intensity_slider.on_update(self.update_light_source_pose)
        self.ambient_slider.on_update(self.update_light_source_pose)
        self.origin_input.on_update(self.update_light_source_pose)
        self.camera_type_select.on_update(self.update_light_source_pose)

        # Add light source camera
        self.light_source_visualizer = self.viser_server.scene.add_camera_frustum(
            name="/light",
            fov=90.0,
            aspect=1.0,
            scale=1.0,
            color=(1.0, 1.0, 0.0),
            wxyz=tf.SO3.from_x_radians(0.0).wxyz,
            position=(0.0, 0.0, 0.0),
            visible=False,
        )

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
                self.update_training_light_source_frustum()
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

    def _update_train_camera_visibility(self, force: bool = False) -> None:
        """Highlight active cameras in green and hide/show candidate (inactive) cameras based on checkbox.

        Active cameras are always highlighted in green.
        When "Hide candidate views" is checked, inactive cameras are hidden.
        When unchecked, all cameras are visible.

        Operates only on already-created frustums (Viewer limits number displayed).
        """
        if not hasattr(self, "camera_handles") or self.camera_handles is None:
            return
        active = self._get_active_indices()
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

    def update_training_light_source_frustum(self):
        if self.pipeline.datamanager.current_light is None:
            return

        with torch.no_grad():
            light_optimizer = self.pipeline.model.light_optimizer
            c2ws_delta = (
                light_optimizer(torch.tensor([0], device=light_optimizer.device)).cpu().numpy()
            )
        c2w_orig = self.pipeline.datamanager.current_light.camera_to_worlds.squeeze().cpu().numpy()
        c2w_delta = c2ws_delta[0, ...]
        c2w = c2w_orig @ np.concatenate((c2w_delta, np.array([[0, 0, 0, 1]])), axis=0)

        R = tf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
        R = R @ tf.SO3.from_x_radians(np.pi)
        self.light_source_visualizer.position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
        self.light_source_visualizer.wxyz = R.wxyz

        self.light_source_visualizer.visible = True

    def update_light_source_pose(self, event):
        """Update the light source pose based on slider input."""
        # Extract GUI values
        az_rad = torch.deg2rad(torch.tensor(self.az_slider.value))
        el_rad = torch.deg2rad(torch.tensor(self.el_slider.value))
        radius = self.radius_slider.value
        dimension = self.dim_slider.value
        focal_length = self.focal_length_slider.value
        ambient = self.ambient_slider.value
        if self.RGB_INTENSITY:
            red_intensity = self.red_intensity_slider.value
            green_intensity = self.green_intensity_slider.value
            blue_intensity = self.blue_intensity_slider.value
            intensity = [red_intensity, green_intensity, blue_intensity]
        else:
            intensity_val = self.intensity_slider.value
            intensity = [intensity_val, intensity_val, intensity_val]
        origin = torch.tensor(self.origin_input.value)

        if self.camera_type_select.value == "Perspective":
            camera_type = CameraType.PERSPECTIVE
        elif self.camera_type_select.value == "Orthographic":
            camera_type = CameraType.ORTHOPHOTO
        elif self.camera_type_select.value == "Fisheye":
            camera_type = CameraType.FISHEYE

        # Light source pose pointing to the origin
        new_pose = camera_to_world_transform(az_rad, el_rad, origin, radius).to(
            self.pipeline.device
        )
        print("New light source pose:\n", new_pose)

        light_source = Cameras(
            camera_to_worlds=new_pose[None, :3, ...],
            fx=focal_length,
            fy=focal_length,
            cx=dimension / 2.0,
            cy=dimension / 2.0,
            width=int(dimension),
            height=int(dimension),
            camera_type=camera_type,
        )

        with torch.no_grad():
            # self.pipeline.model.last_training_light = light_source
            self.pipeline.model.viewer_light = light_source

        cv_to_gl = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(self.pipeline.device)

        light_source_pose_cv = new_pose @ cv_to_gl

        # Update the light source camera frustum
        self.light_source_visualizer.fov = 2 * np.arctan2(dimension, (2 * focal_length))
        self.light_source_visualizer.position = light_source_pose_cv[:3, 3].cpu().numpy()

        # Convert the opengl light source rotation into the opencv viser format
        self.light_source_visualizer.wxyz = tf.SO3.from_matrix(
            light_source_pose_cv[:3, :3].cpu().numpy()
        ).wxyz
        self.light_source_visualizer.visible = True

        self._trigger_rerender()


def camera_to_world_transform(azimuth_rad, elevation_rad, origin, radius):
    # Compute the camera position in Cartesian coordinates
    x = radius * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = radius * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z = radius * torch.sin(elevation_rad)
    camera_position = torch.tensor([x, y, z]) + origin
    up = torch.tensor([0.0, 0.0, 1.0])
    rotation_matrix = look_at(camera_position, origin, up)

    # Construct the 4x4 camera-to-world transformation matrix
    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = camera_position

    return transform_matrix


def look_at(location, target, up):
    z = location - target
    z /= torch.norm(z)
    x = torch.cross(up, z, dim=0)
    x /= torch.norm(x)
    y = torch.cross(z, x, dim=0)
    y /= torch.norm(y)

    R = torch.stack([x, y, z], dim=1)
    return R
