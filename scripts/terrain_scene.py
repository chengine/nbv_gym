"""
Test relighting a terrain scene
"""

import torch
import time
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox

from shadow_splat.model import ShadowSplatModel, ShadowSplatModelConfig

# from shadow_splat.model_old import ShadowSplatModel, ShadowSplatModelConfig
from shadow_splat.util.minimal_viewer import MinimalViewer
from shadow_splat.util.dem import bilinear_interpolate, upsample_dem_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=str, default="viewer", choices=["viewer", "render"], help="Mode to run in"
)
parser.add_argument(
    "--dataset", type=str, default="open3d", choices=["LAC", "open3d"], help="Dataset to use"
)
args = parser.parse_args()
MODE = args.mode
DATASET = args.dataset


def get_seed_points():
    if DATASET == "LAC":
        dem_path = "/home/addai/Projects/neural_elevation_models/data/Moon_Map_01_0_rep0.dat"
        dem = np.load(dem_path, allow_pickle=True)[:, :, :3]
        dem = torch.from_numpy(dem).float()
        dem = upsample_dem_torch(dem, 4)

        # Randomly sample continuous xy points and interpolate z values
        num_extra_points = 1000000
        H, W = dem.shape[:2]
        i_coords = torch.rand(num_extra_points) * (W - 1)
        j_coords = torch.rand(num_extra_points) * (H - 1)
        z_vals = bilinear_interpolate(dem[:, :, 2], torch.stack([i_coords, j_coords], dim=-1))
        MAX_VAL = 13.5
        # Map from [0,W-1] to [-MAX_VAL,MAX_VAL]
        x_vals = (j_coords / (W - 1)) * (2 * MAX_VAL) - MAX_VAL
        y_vals = (i_coords / (H - 1)) * (2 * MAX_VAL) - MAX_VAL
        extra_points = torch.stack([x_vals, y_vals, z_vals], dim=-1)

        # Reshape to points
        # dem_reshaped = dem.reshape(-1, 3)
        # points = torch.cat([dem_reshaped, extra_points], dim=0).float().to(device)
        points = extra_points.float().to(device)
        points = (1 / MAX_VAL) * points

    elif DATASET == "open3d":
        data = np.load("data/open3d_env_points.npz")
        points = np.concatenate([data["surface_points"], data["rock_points"]], axis=0)
        points = torch.from_numpy(points).float().to(device)
        points = (1 / 30.0) * points

    return points


if __name__ == "__main__":
    config = ShadowSplatModelConfig()
    scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]]))
    model = ShadowSplatModel(config, scene_box, num_train_data=100).to(device)

    # Initialize model points
    points = get_seed_points()

    colors = 127 * torch.ones(points.shape[0], 3)
    model.seed_points = (points, colors)
    model.populate_modules()

    # Increase opacities
    # model.gauss_params["opacities"] = torch.logit(0.75 * torch.ones(model.num_points, 1))

    model.training = False
    model = model.to(device)

    if MODE == "viewer":
        viewer = MinimalViewer(model)
        # viewer.viser_server.scene.add_point_cloud(
        #     "seed points", points.cpu().numpy(), 0.5 * colors.cpu().numpy(), point_size=1e-3
        # )
        while True:
            time.sleep(1.0)

    elif MODE == "render":
        # TODO: get data from transforms instead of hardcoding
        # transforms = json.load(open("/home/addai/BlueOrigin/lunar_slam/output/open3d_dataset/solar_progression_6/transforms.json"))

        # For solar_progression_6/view_0002_light_0004.png
        cam_pose = torch.tensor(
            [
                [
                    0.9961710408648277,
                    -0.009231897826373096,
                    0.0869369277396496,
                    0.08693692773964962,
                ],
                [
                    0.08742572471695988,
                    0.10519271411966656,
                    -0.9906014514192135,
                    -0.9906014514192136,
                ],
                [-0.0, 0.994409002854792, 0.10559704087396808, 0.1055970408739681],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        light_pose = torch.tensor(
            [
                [-0.8090169943749473, -0.34549150281252644, 0.4755282581475768, 0.4755282581475768],
                [0.5877852522924732, -0.4755282581475768, 0.6545084971874736, 0.6545084971874736],
                [0.0, 0.8090169943749475, 0.5877852522924732, 0.5877852522924732],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        camera = Cameras(
            camera_to_worlds=cam_pose.unsqueeze(0),
            fx=1350.7389543325814,
            fy=1350.7389543325814,
            cx=640.0,
            cy=360.0,
            width=1280,
            height=720,
            camera_type=CameraType.PERSPECTIVE,
        ).to(device)
        light = Cameras(
            camera_to_worlds=light_pose.unsqueeze(0),
            fx=1650.0,
            fy=1650.0,
            cx=1000.0,
            cy=1000.0,
            width=2000,
            height=2000,
            camera_type=CameraType.ORTHOPHOTO,
        ).to(device)
        start_time = time.time()
        out = model(camera, light)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        plt.imshow(out["rgb"].detach().cpu().numpy())
        plt.imsave("terrain_render.png", out["rgb"].detach().cpu().numpy())
        plt.show()
