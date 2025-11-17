"""
Test relighting a terrain scene
"""

import torch
import time
import numpy as np
import argparse
import open3d as o3d
import json
import matplotlib.pyplot as plt
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox

from shadow_splat.model import ShadowSplatModel, ShadowSplatModelConfig
from shadow_splat.util.minimal_viewer import MinimalViewer
from shadow_splat.util.dem import bilinear_interpolate, upsample_dem_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=str, default="viewer", choices=["viewer", "render"], help="Mode to run in"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="open3d",
    choices=["LAC", "open3d", "master_chief"],
    help="Dataset to use",
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
        points = extra_points.float().to(device)
        points = (1 / MAX_VAL) * points
        colors = 127 * torch.ones(points.shape[0], 3)

    elif DATASET == "open3d":
        data = np.load("data/open3d_env_points.npz")
        points = np.concatenate([data["surface_points"], data["rock_points"]], axis=0)
        points = torch.from_numpy(points).float().to(device)
        points = (1 / 30.0) * points
        colors = 127 * torch.ones(points.shape[0], 3)

    elif DATASET == "master_chief":
        pcd = o3d.io.read_point_cloud("data/models/all_points_poisson_2000000.ply")
        points = np.asarray(pcd.points)
        points = torch.from_numpy(points).float().to(device)
        colors = np.asarray(pcd.colors)
        colors = torch.from_numpy(colors).float().to(device)
        colors = colors * 255.0
        colors = colors.to(torch.uint8)

    return points, colors


if __name__ == "__main__":
    config = ShadowSplatModelConfig()
    scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]]))
    model = ShadowSplatModel(config, scene_box, num_train_data=100).to(device)

    # Initialize model points
    points, colors = get_seed_points()

    model.seed_points = (points, colors)
    model.populate_modules()

    # Increase opacities
    model.gauss_params["opacities"] = torch.logit(0.75 * torch.ones(model.num_points, 1))
    # model.gauss_params["scales"] = torch.log(1e-2 * torch.ones(model.num_points, 3))

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
        transforms = json.load(open("data/master_chief_multi_light/transforms.json"))
        i = 40
        print(f"Rendering {transforms['frames'][i]['file_path']}")
        cam_pose = torch.tensor(transforms["frames"][i]["transform_matrix"])
        light_pose = torch.tensor(transforms["frames"][i]["light_pose"])

        camera = Cameras(
            camera_to_worlds=cam_pose.unsqueeze(0),
            fx=transforms["fl_x"],
            fy=transforms["fl_y"],
            cx=transforms["cx"],
            cy=transforms["cy"],
            width=int(transforms["w"]),
            height=int(transforms["h"]),
            camera_type=CameraType.PERSPECTIVE,
        ).to(device)
        light = Cameras(
            camera_to_worlds=light_pose.unsqueeze(0),
            fx=transforms["light_intrinsics"]["fl_x"],
            fy=transforms["light_intrinsics"]["fl_y"],
            cx=transforms["light_intrinsics"]["cx"],
            cy=transforms["light_intrinsics"]["cy"],
            width=int(transforms["light_intrinsics"]["w"]),
            height=int(transforms["light_intrinsics"]["h"]),
            camera_type=CameraType.ORTHOPHOTO,
        ).to(device)
        start_time = time.time()
        out = model(camera, light)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        plt.imshow(out["rgb"].detach().cpu().numpy())
        plt.imsave("terrain_render.png", out["rgb"].detach().cpu().numpy())
        plt.show()
