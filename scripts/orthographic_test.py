#%%
import numpy as np 
import torch
import time
import open3d as o3d 
from splatloader import GaussianSplat
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from nerfstudio.cameras.cameras import Cameras

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def look_at(location, target, up):
    z = (location - target)
    z /= torch.norm(z)
    x = torch.cross(up, z)
    x /= torch.norm(x)
    y = torch.cross(z, x)
    y /= torch.norm(y)

    R = torch.stack([x, y, z], dim=1)
    return R

config_path = Path('outputs/moon_spiral_2/shadow-splat/2024-10-18_135934/config.yml')

splat = GaussianSplat(config_path, dataset_mode='train', device=device, res_factor=0.5)

#%% Define light source

poses = splat.get_poses()

light_source = Cameras

#%%
