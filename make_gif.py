#%%
from PIL import Image
import glob

def make_gif(frame_folder):
    frames = [Image.open(f"{frame_folder}/r_{i}.png") for i in range(len(glob.glob(f"{frame_folder}/*.png")))]
    frame_one = frames[0]
    frame_one.save(f"{frame_folder}/render.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    
make_gif("renders")
# %%
