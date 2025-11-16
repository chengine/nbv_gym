#%%
# Turn folder of images into a gif

import os
import imageio

method = 'vista_coverage'
images = []

for i in range(1000):
    try:
        images.append(imageio.imread(os.path.join(f'images', f'coverage_{i}.png'))[..., :3])
    except:
        break
    
# Save images as a gif with a frame rate of 1
imageio.mimsave(f'images/{method}.gif', images, fps=100)

# %%
