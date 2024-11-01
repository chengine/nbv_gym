"""Custom viewer for Shadow Splat

"""

import torch
from nerfstudio.viewer.viewer import Viewer  
from nerfstudio.cameras.cameras import Cameras, CameraType


class CustomViewer(Viewer):
    """Custom viewer with an additional slider for adjusting the light source position dynamically."""

    def __init__(self, *args, **kwargs):
        # Initialize the parent Viewer class
        super().__init__(*args, **kwargs)
        
        # Add the light source position slider
        self._add_light_source_slider()

    def _add_light_source_slider(self):
        """Add a slider to the control panel for adjusting the light source position."""
        self.light_pose_slider = self.viser_server.gui.add_slider(
            label="Light Source Position",
            min=-10.0,  # Define appropriate range for your scene
            max=10.0,
            step=0.1,
            initial_value=1.0  # or any initial pose value
        )
        # Set the callback for the slider
        self.light_pose_slider.on_update(self.update_light_source_pose)

    def update_light_source_pose(self, event):
        """Update the light source pose based on slider input."""
        # Extract the actual value from the event
        slider_value = self.light_pose_slider.value
        
        # Define the new pose matrix based on the slider value
        new_pose = torch.tensor([
            [0.0, 0.0, 1.0, slider_value],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]).to(self.pipeline.device)
        
        light_source = Cameras(
                camera_to_worlds=new_pose[None,:3, ...],
                fx=1650.0,
                fy=1650.0,
                cx=250.0,
                cy=250.0,
                width=500,
                height=500,
                camera_type=CameraType.ORTHOPHOTO,
            )

        # Update the light source in the pipeline model
        self.pipeline.model.update_light_source(light_source)
        
        # Trigger a rerender if necessary
        self._trigger_rerender()
