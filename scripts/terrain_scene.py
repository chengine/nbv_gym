import torch
import time
import numpy as np
import torch.nn.functional as F

from nerfstudio.data.scene_box import SceneBox

from shadow_splat.model import ShadowSplatModel, ShadowSplatModelConfig
from shadow_splat.util.minimal_viewer import MinimalViewer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def upsample_dem_torch(dem_xyz: torch.Tensor, scale: int) -> torch.Tensor:
    # dem_xyz: [N, N, 3] → [1, 3, N, N]
    dem = dem_xyz.permute(2, 0, 1).unsqueeze(0)
    upsampled = F.interpolate(dem, scale_factor=scale, mode="bilinear", align_corners=True)
    return upsampled.squeeze(0).permute(1, 2, 0)  # [scale*N, scale*N, 3]


if __name__ == "__main__":
    config = ShadowSplatModelConfig()
    scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]]))
    model = ShadowSplatModel(config, scene_box, num_train_data=100).to(device)

    # Initialize model points
    dem_path = "/home/addai/Projects/neural_elevation_models/data/Moon_Map_01_0_rep0.dat"
    dem = np.load(dem_path, allow_pickle=True)[:, :, :3]
    dem = torch.from_numpy(dem).float()
    dem = upsample_dem_torch(dem, 4)
    print(dem.shape)

    # Reshape to points
    dem_reshaped = dem.reshape(-1, 3)
    # points = torch.from_numpy(dem_reshaped).float().to(device)
    points = dem_reshaped.float().to(device)
    points = 0.1 * points
    print(points.shape)

    colors = 255 * torch.ones(points.shape[0], 3)
    model.seed_points = (points, colors)
    model.populate_modules()

    # Increase opacities
    model.gauss_params["opacities"] = torch.logit(0.75 * torch.ones(model.num_points, 1))

    model.training = False
    model = model.to(device)

    viewer = MinimalViewer(model)

    while True:
        time.sleep(1.0)
