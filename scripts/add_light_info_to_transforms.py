"""
Adding per-frame lighting info to transforms.json

"""

import json
import os
import numpy as np
from pathlib import Path

data_path = Path(os.path.expanduser("~/NeRF/nerfstudio/data/Shadow/rains_chair/"))
transforms_path = data_path / "transforms.json"

with open(transforms_path, "r") as f:
    transforms = json.load(f)

# Add light intrinsics
transforms["light_intrinsics"] = {
    "w": 720,
    "h": 720,
    "fl_x": 1650.0,
    "fl_y": 1650.0,
    "cx": 360.0,
    "cy": 360.0,
}

# Add light pose to each frame
for frame in transforms["frames"]:
    frame["light_pose"] = np.eye(4).tolist()

# Save updated transforms
with open(data_path / "transforms_with_light.json", "w") as f:
    json.dump(transforms, f, indent=4)

print(f"Saved updated transforms to {data_path / 'transforms_with_light.json'}")