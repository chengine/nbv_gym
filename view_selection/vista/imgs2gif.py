#%%
# Turn folder of images into a gif

import os
import imageio

method = 'vista_coverage'
images = []

for i in range(101):
    images.append(imageio.imread(os.path.join(f'save_images/{method}', f'{i}.png'))[..., :3])

# Save images as a gif with a frame rate of 1
imageio.mimsave(f'save_images/{method}.gif', images, fps=1)

# %%
