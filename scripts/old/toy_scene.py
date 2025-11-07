import torch
import time

from nerfstudio.data.scene_box import SceneBox

from shadow_splat.model import ShadowSplatModel, ShadowSplatModelConfig
from shadow_splat.util.minimal_viewer import MinimalViewer
from shadow_splat.util.general import generate_plane_points, generate_cylinder_points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    config = ShadowSplatModelConfig()
    scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]]))
    model = ShadowSplatModel(config, scene_box, num_train_data=100).to(device)

    # Initialize model points
    plane_points = generate_plane_points(width=10.0, height=10.0, num_points=1000)
    cylinder_points = generate_cylinder_points()
    points = torch.cat([plane_points, cylinder_points], dim=0)
    colors = 255 * torch.ones(points.shape[0], 3)
    model.seed_points = (points, colors)
    model.populate_modules()
    model.training = False
    model = model.to(device)

    viewer = MinimalViewer(model)

    while True:
        time.sleep(1.0)
