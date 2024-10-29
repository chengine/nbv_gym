"""Custom viewer

"""
from nerfstudio.viewer.viewer import Viewer  # Import the original Viewer class
import torch

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
            min_value=-10.0,  # Define appropriate range for your scene
            max_value=10.0,
            step=0.1,
            initial_value=1.0  # or any initial pose value
        )
        # Set the callback for the slider
        self.light_pose_slider.on_change(self.update_light_source_pose)

    def update_light_source_pose(self, slider_value: float):
        """Update the light source pose based on slider input."""
        # Define the new pose matrix based on the slider value
        new_pose = torch.tensor([
            [0.0, 0.0, 1.0, slider_value],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]).to(self.pipeline.device)

        # Update the light source in the pipeline model
        self.pipeline.model.update_light_source(new_pose)
        
        # Trigger a rerender if necessary
        self._trigger_rerender()
