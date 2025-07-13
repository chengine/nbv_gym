"""
Adding per-frame lighting info to transforms.json

"""

import json
import os
import numpy as np
from pathlib import Path

data_path = Path(os.path.expanduser("/home/shared/data_nerfstudio/nerfstudio-data/sculpture"))
transforms_path = data_path / "transforms.json"

with open(transforms_path, "r") as f:
    transforms = json.load(f)

# Add light intrinsics
W, H = 3000, 3000
transforms["light_intrinsics"] = {
    "w": W,
    "h": H,
    "fl_x": 1650.0,
    "fl_y": 1650.0,
    "cx": W / 2.0,
    "cy": H / 2.0,
}

# Add light pose to each frame
for frame in transforms["frames"]:
    # frame["light_pose"] = np.eye(4).tolist()
    frame["light_pose"] = [
        [0.6388, 0.4168, -0.6467, -0.6467],
        [-0.7694, 0.3460, -0.5369, -0.5369],
        [0.0000, 0.8406, 0.5417, 0.5417],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ]

# Save updated transforms
with open(data_path / "transforms_with_light.json", "w") as f:
    json.dump(transforms, f, indent=4)

print(f"Saved updated transforms to {data_path / 'transforms_with_light.json'}")
