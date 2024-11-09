"""Custom viewer for Shadow Splat

"""

import torch
import viser
from nerfstudio.viewer.viewer import Viewer  
from nerfstudio.cameras.cameras import Cameras, CameraType


class CustomViewer(Viewer):
    """Custom viewer with an additional slider for adjusting the light source position dynamically."""

    def __init__(self, *args, **kwargs):
        # Initialize the parent Viewer class
        super().__init__(*args, **kwargs)
        
        tabs = self.viser_server.gui.add_tab_group()
        lighting_tab = tabs.add_tab("Light", viser.Icon.SUN)
        
        with lighting_tab:
            self._add_light_source_slider()

    def _add_light_source_slider(self):
        """Add a slider to the control panel for adjusting the light source position."""
        self.az_slider = self.viser_server.gui.add_slider(
            label="Azimuth",
            min=0.0,  
            max=360.0,
            step=0.1,
            initial_value=0.0 
        )
        self.el_slider = self.viser_server.gui.add_slider(
            label="Elevation",
            min=0.0,  
            max=90.0,
            step=0.1,
            initial_value=45.0  
        )
        self.radius_slider = self.viser_server.gui.add_slider(
            label="Radius",
            min=0.0,  
            max=5.0,
            step=0.1,
            initial_value=1.0 
        )
        self.dim_slider = self.viser_server.gui.add_slider(
            label="Dimension",
            min=0.0,  
            max=2000.0,
            step=1.0,
            initial_value=1200 
        )
        self.focal_length_slider = self.viser_server.gui.add_slider(
            label="Focal length",
            min=0.0,  
            max=3000.0,
            step=1.0,
            initial_value=1650 
        )
        self.origin_input = self.viser_server.gui.add_vector3(
            label="Origin",
            min=(-2.0, -2.0, -2.0),  
            max=(2.0, 2.0, 2.0),
            step=1.0,
            initial_value=(0.0, 0.0, 0.0) 
        )
        self.camera_type_select = self.viser_server.gui.add_dropdown(
            label="Camera Type",
            options=["Perspective", "Orthographic", "Fisheye"],
            initial_value="Orthographic"
        )
        # Set the callbacks
        self.az_slider.on_update(self.update_light_source_pose)
        self.el_slider.on_update(self.update_light_source_pose)
        self.radius_slider.on_update(self.update_light_source_pose)
        self.dim_slider.on_update(self.update_light_source_pose)
        self.focal_length_slider.on_update(self.update_light_source_pose)
        self.origin_input.on_update(self.update_light_source_pose)
        self.camera_type_select.on_update(self.update_light_source_pose)

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
        new_pose = camera_to_world_transform(az_rad, el_rad, origin, radius).to(self.pipeline.device)
        print("New light source pose:\n", new_pose)
        
        light_source = Cameras(
                camera_to_worlds=new_pose[None,:3, ...],
                fx=focal_length,
                fy=focal_length,
                cx=dimension / 2.0,
                cy=dimension / 2.0,
                width=int(dimension),
                height=int(dimension),
                camera_type=camera_type,
            )

        self.pipeline.model.update_light_source(light_source)
        self._trigger_rerender()


def camera_to_world_transform(azimuth_rad, elevation_rad, origin, radius):
    # Compute the camera position in Cartesian coordinates
    x = radius * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = radius * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z = radius * torch.sin(elevation_rad)
    camera_position = torch.tensor([x, y, z])

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
    z = (location - target)
    z /= torch.norm(z)
    x = torch.cross(up, z)
    x /= torch.norm(x)
    y = torch.cross(z, x)
    y /= torch.norm(y)

    R = torch.stack([x, y, z], dim=1)
    return R