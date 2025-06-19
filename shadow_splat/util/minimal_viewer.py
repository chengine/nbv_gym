"""Custom viewer for Shadow Splat"""

import numpy as np
import torch
import viser
import viser.transforms as vtf
import nerfview
from nerfstudio.cameras.cameras import Cameras, CameraType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MinimalViewer:
    """viewer using nerfview."""

    def __init__(self, model):
        self.model = model

        self.viser_server = viser.ViserServer(port=7007)
        self.viser_server.gui.configure_theme(dark_mode=True)

        tabs = self.viser_server.gui.add_tab_group()
        lighting_tab = tabs.add_tab("Light", viser.Icon.SUN)

        with lighting_tab:
            self._add_light_source_slider()

        self.viewer = nerfview.Viewer(
            server=self.viser_server, render_fn=self.render_fn, mode="rendering"
        )

    def render_fn(
        self, camera_state: nerfview.CameraState, render_tab_state: nerfview.RenderTabState
    ) -> np.ndarray:
        # Parse camera state for camera-to-world matrix (c2w) and intrinsic (K) as
        # float64 numpy arrays.
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        c2w = camera_state.c2w
        R = vtf.SO3.from_matrix(c2w[:3, :3])
        R = R @ vtf.SO3.from_x_radians(np.pi)
        c2w[:3, :3] = R.as_matrix()

        K = camera_state.get_K([width, height])

        # Convert camera parameters to Cameras object
        camera_to_worlds = torch.from_numpy(c2w).float().unsqueeze(0)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        cameras = Cameras(
            camera_to_worlds=camera_to_worlds,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
        ).to(device)

        # Render using the model
        outputs = self.model.get_outputs_for_camera(cameras)
        img = outputs["rgb"].cpu().numpy()

        # Convert to uint8
        img = (img * 255).astype(np.uint8)
        return img

    def _add_light_source_slider(self):
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
            label="Dimension", min=0.0, max=2000.0, step=1.0, initial_value=1200
        )
        self.focal_length_slider = self.viser_server.gui.add_slider(
            label="Focal length", min=0.0, max=3000.0, step=1.0, initial_value=1650
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
        self.origin_input.on_update(self.update_light_source_pose)
        self.camera_type_select.on_update(self.update_light_source_pose)

        # Add light source camera
        self.light_source_visualizer = self.viser_server.add_camera_frustum(
            name="/light",
            fov=90.0,
            aspect=1.0,
            scale=1.0,
            color=(1.0, 1.0, 0.0),
            wxyz=vtf.SO3.from_x_radians(0.0).wxyz,
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
        origin = torch.tensor(self.origin_input.value)

        if self.camera_type_select.value == "Perspective":
            camera_type = CameraType.PERSPECTIVE
        elif self.camera_type_select.value == "Orthographic":
            camera_type = CameraType.ORTHOPHOTO
        elif self.camera_type_select.value == "Fisheye":
            camera_type = CameraType.FISHEYE

        # Light source pose pointing to the origin
        new_pose = camera_to_world_transform(az_rad, el_rad, origin, radius).to(self.model.device)
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
            self.model.update_light_source(light_source)

        cv_to_gl = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(self.model.device)

        light_source_pose_cv = new_pose @ cv_to_gl

        # Update the light source camera frustum
        self.light_source_visualizer.fov = 2 * np.arctan2(dimension, (2 * focal_length))
        self.light_source_visualizer.position = light_source_pose_cv[:3, 3].cpu().numpy()

        # Convert the opengl light source rotation into the opencv viser format
        self.light_source_visualizer.wxyz = vtf.SO3.from_matrix(
            light_source_pose_cv[:3, :3].cpu().numpy()
        ).wxyz
        self.light_source_visualizer.visible = True

        # self._trigger_rerender()
        self.viewer.rerender(None)


def camera_to_world_transform(azimuth_rad, elevation_rad, origin, radius):
    # Compute the camera position in Cartesian coordinates
    x = radius * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = radius * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z = radius * torch.sin(elevation_rad)
    camera_position = torch.tensor([x, y, z]) + origin

    # # Compute the forward, right, and up vectors
    # forward = -camera_position / torch.norm(camera_position)  # Normalize
    # up = torch.tensor([0.0, 0.0, 1.0])
    # right = torch.cross(up, forward)
    # right = right / torch.norm(right)  # Normalize
    # up = torch.cross(forward, right)  # Re-compute the up vector to ensure orthogonality

    # # Construct the rotation matrix
    # rotation_matrix = torch.stack([right, up, -forward], dim=1)  # 3x3 rotation matrix

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
    x = torch.cross(up, z)
    x /= torch.norm(x)
    y = torch.cross(z, x)
    y /= torch.norm(y)

    R = torch.stack([x, y, z], dim=1)
    return R
