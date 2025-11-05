#%%
import os
import torch
import numpy as np
import time
from scipy.stats import uniform_direction
import json
from distributions.vmf import FisherDistribution, FisherKLDivergence
import matplotlib.pyplot as plt

torch.manual_seed(10)

def generate_points_on_sphere(n):
    """Generates n uniformly distributed points on a sphere of radius r."""
    uniform_sphere_dist = uniform_direction(2)
    unit_vectors = uniform_sphere_dist.rvs(n)

    return unit_vectors

# Make font size bigger
plt.rcParams.update({'font.size': 18})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

points = generate_points_on_sphere(100)
points = torch.tensor(points, device=device, dtype=torch.float32)

# Plot for the naive random sampling
data = torch.tensor([[1., 1.]], device=device)
data = data / torch.norm(data, dim=-1, keepdim=True)

datas = [data]
vals = [1.]

for k in range(100):
    best_val = -10
    best_idx = 0
    for i, pt in enumerate(points):
        dot_products = torch.sum(data * pt, dim=-1)
        min_val, min_idx = torch.min(-dot_products, dim=0)
        if min_val.item() > best_val:
            best_val = min_val
            best_idx = i

    best_pt = points[best_idx]
    data = torch.cat([data, best_pt.reshape(1, 2)], dim=0)
    datas.append(data)
    vals.append(best_val.item())

vals = np.exp(np.array(vals))
# normalize
vals = (vals - vals.min()) / (vals.max() - vals.min())
# %%

cmap = plt.get_cmap('jet')

for j, (val, data) in enumerate(zip(vals, datas)):
    data = data.cpu().numpy()
    fig, ax = plt.subplots()
    # Plot a unit circle using patches
    circle = plt.Circle((0, 0), 1, color=cmap(val), fill=False, linewidth=2)
    ax.add_artist(circle)

    # Plot arrows using the data points
    for pt in data:
        ax.arrow(0, 0, pt[0], pt[1], head_width=0.05, head_length=0.03, fc='r', ec='r')

    ax.set_aspect('equal')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.axis('off')
    fig.savefig(f'save_images/vista_coverage/{j}.png', transparent=True)
# %%
