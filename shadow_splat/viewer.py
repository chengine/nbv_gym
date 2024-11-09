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
        # Add the light source position slider
        with lighting_tab:
            self._add_light_source_slider()
        # self._add_light_source_slider()

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
            max=2.0,
            step=0.1,
            initial_value=1.0 
        )
        # Set the callback for the slider
        self.az_slider.on_update(self.update_light_source_pose)
        self.el_slider.on_update(self.update_light_source_pose)
        self.radius_slider.on_update(self.update_light_source_pose)

    def update_light_source_pose(self, event):
        """Update the light source pose based on slider input."""
        # Extract the actual value from the event
        az_rad = torch.deg2rad(torch.tensor(self.az_slider.value))
        el_rad = torch.deg2rad(torch.tensor(self.el_slider.value))
        radius = self.radius_slider.value

        # Define the new pose matrix based on the slider value
        new_pose = camera_to_world_transform(az_rad, el_rad, radius).to(self.pipeline.device)
        print("New light source pose:\n", new_pose)
        
        light_source = Cameras(
                camera_to_worlds=new_pose[None,:3, ...],
                fx=1650.0,
                fy=1650.0,
                cx=250.0,
                cy=250.0,
                width=500,
                height=500,
                # camera_type=CameraType.PERSPECTIVE,
                camera_type=CameraType.ORTHOPHOTO,
            )

        # Update the light source in the pipeline model
        self.pipeline.model.update_light_source(light_source)
        
        # Trigger a rerender if necessary
        self._trigger_rerender()


def camera_to_world_transform(azimuth_rad, elevation_rad, radius):
    # Compute the camera position in Cartesian coordinates
    x = radius * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = radius * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z = radius * torch.sin(elevation_rad)
    camera_position = torch.tensor([x, y, z])

    # Compute the forward, right, and up vectors
    forward = -camera_position / torch.norm(camera_position)  # Normalize
    up = torch.tensor([0.0, 0.0, 1.0])
    right = torch.cross(up, forward)
    right = right / torch.norm(right)  # Normalize
    up = torch.cross(forward, right)  # Re-compute the up vector to ensure orthogonality

    # Construct the rotation matrix
    rotation_matrix = torch.stack([right, up, -forward], dim=1)  # 3x3 rotation matrix

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