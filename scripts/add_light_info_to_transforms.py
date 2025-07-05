"""
Adding per-frame lighting info to transforms.json

"""

import json
import os
import numpy as np
from pathlib import Path

data_path = Path(os.path.expanduser("~/NeRF/shadow_splat/data/master_chief_cycles/"))
transforms_path = data_path / "transforms.json"

with open(transforms_path, "r") as f:
    transforms = json.load(f)

# Add light intrinsics
W, H = 2000, 2000
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
        [-0.27690861, -0.84461385, 0.45820203, 0.50663298],
        [0.72507936, -0.49657366, -0.4771525, -0.32427847],
        [0.63054067, 0.20010521, 0.74991757, 1.25609708],
        [0.0, 0.0, 0.0, 1.0],
    ]

# Save updated transforms
with open(data_path / "transforms_with_light.json", "w") as f:
    json.dump(transforms, f, indent=4)

print(f"Saved updated transforms to {data_path / 'transforms_with_light.json'}")
