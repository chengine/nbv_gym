#%%
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
    uniform_sphere_dist = uniform_direction(3)
    unit_vectors = uniform_sphere_dist.rvs(n)

    return unit_vectors

fig, ax = plt.subplots()
# Make font size bigger
plt.rcParams.update({'font.size': 18})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

points = generate_points_on_sphere(100)
points = torch.tensor(points, device=device, dtype=torch.float32)

# Plot for the naive random sampling
rand_ints = np.random.choice(100, 100, replace=True)
rand_ints = torch.tensor(rand_ints, device=device).to(torch.int32)
kls = []
kls_0 = []
datas = []
n_clusters = 1

data1 = None
data2 = torch.tensor([[1., 1., 0.]], device=device)
data2 = data2 / torch.norm(data2, dim=-1, keepdim=True)

datas.append(data2)

index1 = None
index2 = torch.zeros(1, device=device).to(torch.int32)

fisher_kl = FisherKLDivergence(device, n_clusters)

fisher_kl.fit([data1, data2], [index1, index2], num_iters=10)

kl = fisher_kl.evaluate()
kls.append(kl)

fisher_kl = FisherKLDivergence(device, n_clusters)
fisher_kl.fit([None, data2], [None, index2], num_iters=10)
kl = fisher_kl.evaluate()
kls_0.append(kl)

for i in range(100):
    data1 = torch.tensor(data2)
    data2 = points[rand_ints[i]].reshape(1, 3)
    data2 = data2 / torch.norm(data2, dim=-1, keepdim=True)
    data2 = torch.cat([data1, data2], dim=0)
    datas.append(data2)

    index1 = torch.tensor(index2)
    index2 = torch.cat([index1, torch.zeros(1, device=device).to(torch.int32)], dim=0)

    fisher_kl = FisherKLDivergence(device, n_clusters)

    fisher_kl.fit([data1, data2], [index1, index2], num_iters=10)

    kl = fisher_kl.evaluate()
    kls.append(kl)

    fisher_kl = FisherKLDivergence(device, n_clusters)
    fisher_kl.fit([None, data2], [None, index2], num_iters=10)
    kl = fisher_kl.evaluate()
    kls_0.append(kl)

kls = torch.stack(kls, dim=0).squeeze().cpu().numpy()       # Incremental
kls_0 = torch.stack(kls_0, dim=0).squeeze().cpu().numpy()   # to uniform

ax.plot(np.arange(len(kls)) + 1, np.exp(kls), color='red', linewidth=2, linestyle='dashed')
ax.plot(np.arange(len(kls_0)) + 1, np.exp(kls_0), color='red', linewidth=2)

# Save datas as a json
datas = [data.cpu().numpy().tolist() for data in datas]
with open('save_images/vmf_data_random.json', 'w') as f:
    json.dump(datas, f, indent=4)

# Plot for minimizing the KL divergence
kls_0 = []
kls = []
datas = []
n_clusters = 1

data1 = None
data2 = torch.tensor([[1., 1., 0.]], device=device)
data2 = data2 / torch.norm(data2, dim=-1, keepdim=True)

datas.append(data2)

index1 = None
index2 = torch.zeros(1, device=device).to(torch.int32)

fisher_kl = FisherKLDivergence(device, n_clusters)
fisher_kl.fit([data1, data2], [index1, index2], num_iters=10)
kl = fisher_kl.evaluate()
kls.append(kl)

fisher_kl = FisherKLDivergence(device, n_clusters)
fisher_kl.fit([None, data2], [None, index2], num_iters=10)
kl = fisher_kl.evaluate()
kls_0.append(kl)

choices = torch.arange(100, device=device).to(torch.int32)

for i in range(100):
    data1 = torch.tensor(data2)
    index1 = torch.tensor(index2)
    index2 = torch.cat([index1, torch.zeros(1, device=device).to(torch.int32)], dim=0)

    # Choose minimizing view direction
    kl_best = 0.
    best_idx = 0
    for j in choices:
        data2_ = points[j:j+1]
        data2_ = data2_ / torch.norm(data2_, dim=-1, keepdim=True)
        data2_ = torch.cat([data1, data2_], dim=0)

        fisher_kl = FisherKLDivergence(device, n_clusters)
        fisher_kl.fit([data1, data2_], [index1, index2], num_iters=10)
        kl_ = fisher_kl.evaluate()
        
        if kl_ >= kl_best:
            kl_best = kl_
            data2 = data2_
            best_idx = j

    datas.append(data2)
    kls.append(kl_best)
    print(best_idx)

    fisher_kl = FisherKLDivergence(device, n_clusters)
    fisher_kl.fit([None, data2], [None, index2], num_iters=10)
    kl = fisher_kl.evaluate()
    kls_0.append(kl)

    #choices = choices[choices != best_idx]

kls = torch.stack(kls, dim=0).squeeze().cpu().numpy()
kls_0 = torch.stack(kls_0, dim=0).squeeze().cpu().numpy()

datas = [data.cpu().numpy().tolist() for data in datas]
with open('save_images/vmf_data_vista.json', 'w') as f:
    json.dump(datas, f, indent=4)

ax.plot(np.arange(len(kls)) + 1, np.exp(kls), color='green', linewidth=2, linestyle='dashed')
ax.plot(np.arange(len(kls_0)) + 1, np.exp(kls_0), color='green', linewidth=2)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([1, len(kls)+ 1])

# Set horizontal line
ax.axhline(y=1, color='gray', alpha=0.5)

# ax.set_xlabel('Number of view directions added')
# ax.set_ylabel('KL divergence (log)')
ax.set_aspect('equal')
plt.savefig('save_images/vmf_kl.svg', transparent=True)

# %%
