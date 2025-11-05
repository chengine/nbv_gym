#%%
import os
import torch
import numpy as np
import time
import json
import matplotlib.pyplot as plt

from distributions.vmf import FisherDistribution, FisherKLDivergence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

method = 'vista'
json_path = f'save_images/vmf_data_{method}.json'
with open(json_path, 'r') as f:
    datas = json.load(f)

n_clusters = 1
for i, data in enumerate(datas):
    data = torch.tensor(data, device=device, dtype=torch.float32)

    data[:, -1] = 0.
    data = data / torch.norm(data, dim=-1, keepdim=True)

    data_length = data.shape[0]
    index = torch.zeros(data_length, device=device).to(torch.int32)

    fisher = FisherDistribution(device, n_clusters)
    fisher.fit(data, index)

    t = torch.linspace(0., 2*np.pi, 100, device=device)
    data_query = torch.stack([torch.cos(t), torch.sin(t), torch.zeros(100, device=device)], dim=-1)

    probs = []
    for i in range(100):
        probs.append(fisher.evaluate(data_query[i].reshape(1, 3)))
    probs = torch.stack(probs, dim=0)

    fig, ax = plt.subplots()
    # Plot a unit circle using patches
    circle = plt.Circle((0, 0), 1, color='gray', fill=False)
    ax.add_artist(circle)

    data = data.cpu().numpy()
    # Plot arrows using the data points
    for i in range(data_length):
        ax.arrow(0, 0, data[i, 0], data[i, 1], head_width=0.05, head_length=0.03, fc='r', ec='r')

    # Plot surface of distribution
    for p, q in zip(probs, data_query):
        plot_point = torch.stack([q, q * (1. + p.squeeze())], dim=0).cpu().numpy()
        ax.plot(plot_point[:, 0], plot_point[:, 1], 'green')
        ax.scatter(q[0].cpu().numpy(), q[1].cpu().numpy(), 1., c='green')
    surface = data_query * (1. + probs)
    surface = surface.cpu().numpy()
    ax.plot(surface[:, 0], surface[:, 1], 'green')

    ax.set_aspect('equal')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.axis('off')
    fig.savefig(f'save_images/{method}/vmf_{i}.png', transparent=True)