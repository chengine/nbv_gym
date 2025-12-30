from nbv_gym.util.coverage import inner_ptilde

import torch

mu1 = torch.randn(1, 3)
mu1 = mu1 / torch.linalg.norm(mu1, dim=-1)
mu2 = torch.randn(1, 3)
mu2 = mu2 / torch.linalg.norm(mu2, dim=-1)
kappa = torch.tensor(1.0)

f = 2 * torch.sinh(kappa * torch.linalg.norm(mu1 + mu2, dim=-1)) / (torch.linalg.norm(mu1 + mu2, dim=-1) * torch.sinh(2 * kappa))

print('mu1: ', mu1)
print('mu2: ', mu2)
print('norm sum: ', torch.linalg.norm(mu1 + mu2, dim=-1))
print('kappa: ', kappa)
print('inner_ptilde: ', inner_ptilde(mu1, mu2, kappa))
print('f: ', f)