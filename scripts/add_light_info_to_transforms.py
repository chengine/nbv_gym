"""
Adding per-frame lighting info to transforms.json

"""

import json
import os
import numpy as np
from pathlib import Path

data_path = Path(os.path.expanduser("/home/shared/data_nerfstudio/ShadowSplat/tandt/caterpillar"))
transforms_path = data_path / "transforms_original.json"

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
        [0.9573, -0.1502, 0.2470, 0.2470],
        [0.2890, 0.4973, -0.8180, -0.8180],
        [-0.0000, 0.8545, 0.5195, 0.5195],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ]

# Save updated transforms
with open(data_path / "transforms.json", "w") as f:
    json.dump(transforms, f, indent=4)

print(f"Saved updated transforms to {data_path / 'transforms.json'}")
