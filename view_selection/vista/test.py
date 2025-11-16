#%%
import numpy as np 
import scipy
import time
import torch

z = np.random.rand(1000000)
tnow = time.time()
out1 = scipy.special.iv(3/2, z)
# scipy.special.i1(z)
print('Elapsed:', time.time() - tnow)

tnow = time.time()
out2 = (np.sqrt(2/(np.pi*z)) * (np.cosh(z) - np.sinh(z)/z))
print('Elapsed:', time.time() - tnow)

print(np.allclose(out1, out2))

z = torch.tensor(z).cuda()
torch.cuda.synchronize()
tnow = time.time()
out3 = (torch.sqrt(2/(np.pi*z)) * (torch.cosh(z) - torch.sinh(z)/z))
torch.cuda.synchronize()
print('Elapsed:', time.time() - tnow)

print(np.allclose(out2, out3.cpu().numpy()))
