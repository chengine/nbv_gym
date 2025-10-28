#%%
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
import time

from distributions.vmf import FisherDistribution, FisherKLDivergence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 1000000
M = 22500000

fisher = FisherDistribution(device)
fisher_kl = FisherKLDivergence(device)
data = torch.randn((2, M, 3), device=device)
data = data / torch.norm(data, dim=-1, keepdim=True)
index = torch.randint(0, N, (2, M), device=device).to(torch.int32)
n_clusters = N

tnow = time.time()
torch.cuda.synchronize()
fisher_kl.fit(data, index, n_clusters)
kl = fisher_kl.evaluate()
torch.cuda.synchronize()
print(f"Time: {time.time() - tnow}")
print(kl.max(), kl.min())
# %%

fisher = FisherDistribution(device)

N = 1000

data = torch.randn((N, 2), device=device)
# data = torch.zeros((N, 2), device=device)
data = torch.cat([data, torch.zeros((N, 1), device=device)], dim=-1)
data = data / torch.norm(data, dim=-1, keepdim=True)

index = torch.zeros(N, device=device).to(torch.int32)
fisher.fit(data, index, 1)

t = torch.linspace(0., 2*np.pi, 100, device=device)
data_query = torch.stack([torch.cos(t), torch.sin(t), torch.zeros(100, device=device)], dim=-1)

probs = []
for i in range(100):
    probs.append(fisher.evaluate(data_query[i].reshape(1, 3)))
probs = torch.stack(probs, dim=0)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# Plot a unit circle using patches
circle = plt.Circle((0, 0), 1, color='gray', fill=False)
ax.add_artist(circle)

data = data.cpu().numpy()
# Plot arrows using the data points
for i in range(N):
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
plt.savefig(f'vmf_{N}.svg', transparent=True)

# fisher_kl = FisherKLDivergence(device)
# data = torch.randn((2, M, 3), device=device)
# data = data / torch.norm(data, dim=-1, keepdim=True)
# index = torch.randint(0, N, (2, M), device=device).to(torch.int32)
# n_clusters = N

# tnow = time.time()
# torch.cuda.synchronize()
# fisher_kl.fit(data, index, n_clusters)
# kl = fisher_kl.evaluate()
# torch.cuda.synchronize()
# print(f"Time: {time.time() - tnow}")

# %%
