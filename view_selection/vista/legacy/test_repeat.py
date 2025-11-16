#%%
import torch

a = torch.randn(10, 3).view(1, 10, 3)
a.expand(5, -1, -1)
    
# %%
