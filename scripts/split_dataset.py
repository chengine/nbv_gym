"""
Create a biased train split with majority of frames on one side of the scene, and equally distributed eval frames.
"""

import os
import json
import numpy as np
from pathlib import Path
import shutil

EVAL_FRACTION = 0.1


if __name__ == "__main__":
    data_path = Path(
        os.path.expanduser("/home/shared/data_nerfstudio/ShadowSplat/captures/chair_3pm/")
    )
    transforms_path = data_path / "transforms_original.json"
    image_path = data_path / "images_split"
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    # If images_split exists, delete it
    if image_path.exists():
        answer = input("images_split exists, delete it? (y/n)")
        if answer == "y":
            shutil.rmtree(image_path)
        else:
            print("Aborting...")
            exit()
    os.mkdir(image_path)

    transforms_split = transforms.copy()
    frames_split = []

    for i, frame in enumerate(transforms["frames"]):
        pose = np.asarray(frame["transform_matrix"])
        filename = frame["file_path"].split("/")[-1]

        if i % int(1 / EVAL_FRACTION) == 0:
            # Add to eval
            shutil.copy(data_path / frame["file_path"], image_path / f"eval_{filename}")
            frame["file_path"] = f"images_split/eval_{filename}"
            frames_split.append(frame)
            # transforms_split["frames"][i]["file_path"] = f"images_split/eval_{filename}"

        elif pose[:3, 3][0] > 2.0:
            # Add to train if on biased side
            shutil.copy(data_path / frame["file_path"], image_path / f"train_{filename}")
            frame["file_path"] = f"images_split/train_{filename}"
            frames_split.append(frame)

        else:
            if np.random.rand() < 0.1:
                # Also add to train with some probability
                shutil.copy(data_path / frame["file_path"], image_path / f"train_{filename}")
                frame["file_path"] = f"images_split/train_{filename}"
                frames_split.append(frame)
            # Otherwise, not included

    transforms_split["frames"] = frames_split

    with open(data_path / "transforms_split.json", "w") as f:
        json.dump(transforms_split, f, indent=4)
    print(f"Saved updated transforms to {data_path / 'transforms_split.json'}")
