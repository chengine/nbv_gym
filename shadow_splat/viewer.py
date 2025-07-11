"""Custom viewer for Shadow Splat"""

import numpy as np
import torch
import viser
import viser.transforms as tf
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.models.splatfacto import SplatfactoModel

from shadow_splat.model import ShadowSplatModel, ShadowSplatModelConfig


class ShadowSplatViewer(Viewer):
    """Custom viewer with an additional slider for adjusting the light source position dynamically."""

    def __init__(self, *args, **kwargs):
        # Convert SplatfactoModel to ShadowSplatModel BEFORE calling parent __init__
        pipeline = kwargs["pipeline"]
        if type(pipeline.model) is SplatfactoModel:
            print("model is splatfacto, converting to shadow splat")
            self._convert_model(pipeline)

        # Initialize the parent Viewer class
        super().__init__(*args, **kwargs)

        print("ShadowSplatViewer | __init__")
        print(f"pipeline.model: {type(pipeline.model)}")

        tabs = self.viser_server.gui.add_tab_group()
        lighting_tab = tabs.add_tab("Light", viser.Icon.SUN)

        with lighting_tab:
            # Get initial light parameters from pipeline model
            initial_variance_factor = torch.exp(
                pipeline.model.light_params["variance_factor"]
            ).item()
            initial_cutoff = torch.sigmoid(pipeline.model.light_params["cutoff"]).item()
            initial_intensity = (
                torch.exp(pipeline.model.light_params["intensity"]).detach().cpu().numpy()
            )
            self._add_light_source_slider(
                initial_variance_factor=initial_variance_factor,
                initial_cutoff=initial_cutoff,
                initial_intensity=initial_intensity,
            )

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
        self, initial_variance_factor=0.01, initial_cutoff=0.5, initial_intensity=np.ones(3)
    ):
        """Add a slider to the control panel for adjusting the light source position."""
        self.az_slider = self.viser_server.gui.add_slider(
            label="Azimuth", min=0.0, max=360.0, step=0.1, initial_value=0.0
        )
        self.el_slider = self.viser_server.gui.add_slider(
            label="Elevation", min=0.0, max=90.0, step=0.1, initial_value=45.0
        )
        self.radius_slider = self.viser_server.gui.add_slider(
            label="Radius", min=0.0, max=10.0, step=0.1, initial_value=1.0
        )
        self.dim_slider = self.viser_server.gui.add_slider(
            label="Dimension", min=0.0, max=3000.0, step=1.0, initial_value=2000
        )
        self.focal_length_slider = self.viser_server.gui.add_slider(
            label="Focal length", min=0.0, max=3000.0, step=1.0, initial_value=1650
        )
        self.variance_factor_slider = self.viser_server.gui.add_slider(
            label="Variance factor",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=initial_variance_factor,
        )
        self.cutoff_slider = self.viser_server.gui.add_slider(
            label="Cutoff", min=0.0, max=1.0, step=0.01, initial_value=initial_cutoff
        )
        self.red_intensity_slider = self.viser_server.gui.add_slider(
            label="Red intensity", min=0.0, max=10.0, step=0.1, initial_value=initial_intensity[0]
        )
        self.green_intensity_slider = self.viser_server.gui.add_slider(
            label="Green intensity", min=0.0, max=10.0, step=0.1, initial_value=initial_intensity[1]
        )
        self.blue_intensity_slider = self.viser_server.gui.add_slider(
            label="Blue intensity", min=0.0, max=10.0, step=0.1, initial_value=initial_intensity[2]
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
        self.variance_factor_slider.on_update(self.update_light_source_pose)
        self.red_intensity_slider.on_update(self.update_light_source_pose)
        self.green_intensity_slider.on_update(self.update_light_source_pose)
        self.blue_intensity_slider.on_update(self.update_light_source_pose)
        self.cutoff_slider.on_update(self.update_light_source_pose)
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

    def update_light_source_pose(self, event):
        """Update the light source pose based on slider input."""
        # Extract GUI values
        az_rad = torch.deg2rad(torch.tensor(self.az_slider.value))
        el_rad = torch.deg2rad(torch.tensor(self.el_slider.value))
        radius = self.radius_slider.value
        dimension = self.dim_slider.value
        focal_length = self.focal_length_slider.value
        variance_factor = self.variance_factor_slider.value
        cutoff = self.cutoff_slider.value
        red_intensity = self.red_intensity_slider.value
        green_intensity = self.green_intensity_slider.value
        blue_intensity = self.blue_intensity_slider.value
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
        # print("New light source pose:\n", new_pose)

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
            self.pipeline.model.compute_irradiance(
                light_source,
                variance_factor=variance_factor,
                intensity=[red_intensity, green_intensity, blue_intensity],
                cutoff=cutoff,
            )

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
