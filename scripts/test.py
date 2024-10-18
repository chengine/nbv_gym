#%%
import numpy as np 
import torch
import time
import open3d as o3d 
from splatloader import GaussianSplat
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_path = Path('outputs/polo/shadow-splat/2024-10-17_170035/config.yml')

splat = GaussianSplat(config_path, dataset_mode='train', device=device)

#%%
# Recovers dataset poses
poses = splat.get_poses()

for i in range(len(poses)):

    # Renders splat from a pose. We call this to get the intermediate variables from the rasterization function.
    tnow = time.time()
    torch.cuda.synchronize()
    splat.update_light_source(poses[i].to(device), mask=None)
    torch.cuda.synchronize()
    print("Time to update light source: ", time.time() - tnow)

    tnow = time.time()
    torch.cuda.synchronize()
    pov_output = splat.render(poses[0])
    torch.cuda.synchronize()
    print("Time to render: ", time.time() - tnow)

    pov_image = pov_output["rgb"].cpu().numpy()

    cv2.imwrite(f'renders/r_{i}.png', cv2.cvtColor(pov_image * 255, cv2.COLOR_BGR2RGB))

#%%
