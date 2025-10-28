import os
from pathlib import Path
import json
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

import numpy as np

import time
import torch
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import cm

from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.nerfacto import NerfactoModel

from nerfstudio.utils import colormaps, install_checks, colors
from nerfstudio.utils.colormaps import ColormapOptions    
# colormap options
colormap_options = colormaps.ColormapOptions()

import shutil
import subprocess
import warnings

import pdb

from tabulate import tabulate


# # # # #
# # # # # Utils
# # # # #

# apply the colormap
colormap_options1 = ColormapOptions(
    colormap='inferno',
    normalize=colormap_options.normalize,
    colormap_min=colormap_options.colormap_min,
    colormap_max=colormap_options.colormap_max,
    invert=colormap_options.invert,
)

# From INRIA
# Base auxillary coefficient
C0 = 0.28209479177387814


def SH2RGB(sh):
    return sh * C0 + 0.5


class NeRF:
    def __init__(
        self,
        config_path: Path,
        res_factor=None,
        width: int = 640,
        height: int = 360,
        cam_intrinsics:Dict={},
        test_mode: Literal["test", "val", "inference"] = "inference",
        dataset_mode: Literal["train", "val", "test"] = "test",
        device: Union[torch.device, str] = "cuda:0",
    ) -> None:
        # config path
        self.config_path = config_path

        # camera rescale resolution factor
        self.res_factor = res_factor
    
        # camera intrinsics
        cam_fx = cam_intrinsics["fx"]
        cam_fy = cam_intrinsics["fy"]
        cam_cx = cam_intrinsics["cx"]
        cam_cy = cam_intrinsics["cy"]
        cam_height = cam_intrinsics["h"]
        cam_width = cam_intrinsics["w"]
        cam_camera_type = cam_intrinsics["cam_camera_type"]

        # device
        self.device = device

        # initialize pipeline
        self.init_pipeline(test_mode)

        # load dataset
        self.load_dataset(dataset_mode)

        # load cameras
        self.get_cameras()

        # Get reference camera
        self.camera_ref = self.pipeline.datamanager.eval_dataset.cameras[0]
        
        # Render parameters
        self.channels = 3
        self.camera_out, self.width, self.height = self.generate_output_cameras(
        self.camera_ref.camera_to_worlds, w=cam_width, h=cam_height, fx=cam_fx, fy=cam_fy, cx=cam_cx, cy=cam_cy, cam_camera_type=cam_camera_type,
        #self.camera_ref.camera_to_worlds, width, height
        )

    def init_pipeline(self, test_mode: Literal["test", "val", "inference"]):
        # Get config and pipeline
        self.config, self.pipeline, _, _ = eval_setup(
            self.config_path,
            test_mode=test_mode,
        )

    def load_dataset(self, dataset_mode: Literal["train", "val", "test"]):
        # return dataset
        if dataset_mode == "train":
            self.dataset = self.pipeline.datamanager.train_dataset
        elif dataset_mode in ["val", "test"]:
            self.dataset = self.pipeline.datamanager.eval_dataset
        else:
            ValueError(
                'Incorrect value for datset_mode. Accepted values include: dataset_mode: Literal["train", "val", "test"].'
            )

    def get_cameras(self):
        # Camera object contains camera intrinsics and extrinsics
        self.cameras = self.dataset.cameras

        if self.res_factor is not None:
            self.cameras.rescale_output_resolution(self.res_factor)

    def get_poses(self):
        return self.cameras.camera_to_worlds

    def get_images(self):
        # images
        images = [
            self.dataset.get_image_float32(image_idx)
            for image_idx in range(
                len(self.dataset._dataparser_outputs.image_filenames)
            )
        ]

        return images

    def generate_output_cameras(self, poses, w: int, h: int, fx: float, fy: float, cx: float, cy: float, cam_camera_type: int = torch.tensor([[1]])):
        # cameras
        cameras_out = Cameras(
            camera_to_worlds=poses[..., :3, :],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=w,
            height=h,
            camera_type=cam_camera_type,
        )
        
        # cameras_out = []
        # for pose in poses:
        #     camera_to_world = pose[None, :3, ...]

        #     camera_out = Cameras(
        #         camera_to_worlds=1.0 * camera_to_world,
        #         fx=fx,
        #         fy=fy,
        #         cx=cx,
        #         cy=cy,
        #         width=w,
        #         height=h,
        #         camera_type=CameraType.PERSPECTIVE,
        #     )

        #     camera_out = camera_out.to(self.device)
        #     cameras_out.append(camera_out)

        return cameras_out, w, h

    def get_camera_intrinsics(self):
        K = self.cameras[0].get_intrinsics_matrices().squeeze()
        # width and height
        W = int(self.cameras[0].width.item())
        H = int(self.cameras[0].height.item())
        return H, W, K

    def render(
        self,
        pose: torch.Tensor = None,
        cameras: Cameras = None,
        compute_semantics: Optional[bool] = False,
        debug_mode=False,
    ):
        if pose is None and cameras is None:
            raise ValueError(
                "Render requires a valid Pose or Camera Object, but both were set to None!"
            )

        if cameras is None:
            # Render from a single pose
            camera_to_world = pose[None, :3, ...]

            cameras = Cameras(
                camera_to_worlds=camera_to_world,
                fx=self.cameras[0].fx,
                fy=self.cameras[0].fy,
                cx=self.cameras[0].cx,
                cy=self.cameras[0].cy,
                width=self.cameras[0].width,
                height=self.cameras[0].height,
                camera_type=CameraType.PERSPECTIVE,
            )

        cameras = cameras.to(self.device)

        # render outputs
        if isinstance(self.pipeline.model, NerfactoModel):
            aabb_box = None
            camera_ray_bundle = cameras.generate_rays(
                camera_indices=0, aabb_box=aabb_box
            )
            # pdb.set_trace()
            # print(f"type camera_ray_bundle: {type(camera_ray_bundle)}")
            tnow = time.perf_counter()
            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle
                )

            if debug_mode:
                print("Rendering time: ", time.perf_counter() - tnow)

            # insert ray bundles
            outputs["ray_bundle"] = camera_ray_bundle
        # elif isinstance(self.pipeline.model, SplatfactoModel):
        else:
            obb_box = None

            tnow = time.perf_counter()
            with torch.no_grad():
                try:
                    outputs = self.pipeline.model.get_outputs_for_camera(
                        cameras, obb_box=obb_box, compute_semantics=compute_semantics
                    )
                except:
                    outputs = self.pipeline.model.get_outputs_for_camera(
                        cameras, obb_box=obb_box
                    )

            if debug_mode:
                print("Rendering time: ", time.perf_counter() - tnow)

        return outputs

    def generate_point_cloud(self):
        means = self.pipeline.model.means.clone().detach()
        colors = self.pipeline.model.features_dc.clone().detach()

        pointcloud = {
            "points": means,
            "values": colors,
        }

        return pointcloud

    # def xcrpr_render(
    #     self,
    #     xcr: np.ndarray,
    #     xpr: np.ndarray = None,
    #     visual_mode: Literal["static", "dynamic"] = "static",
    # ):

    #     if visual_mode == "static":
    #         image = self.static_render(xcr)
    #     elif visual_mode == "dynamic":
    #         image = self.dynamic_render(xcr, xpr)
    #     else:
    #         raise ValueError(f"Invalid visual mode: {visual_mode}")

    #     # Convert to numpy
    #     image = image.cpu().numpy()

    #     # Convert to uint8
    #     image = (255 * image).astype(np.uint8)

    #     return image

    # def static_render(self, xcr: np.ndarray) -> torch.Tensor:
    #     # Extract the pose
    #     T_c2n = pose2nerf_transform(np.hstack((xcr[0:3], xcr[6:10])))
    #     P_c2n = torch.tensor(T_c2n[0:3, :]).float()

    #     # Render from a single pose
    #     camera_to_world = P_c2n[None, :3, ...]
    #     self.camera_out.camera_to_worlds = camera_to_world

    #     # render outputs
    #     with torch.no_grad():
    #         outputs = self.pipeline.model.get_outputs_for_camera(
    #             self.camera_out, obb_box=None
    #         )

    #     image = outputs["rgb"]

    #     return image


def train_gsplat(
    data_dir, output_dir: str = "outputs", num_train_iterations: int = 30000,
    load_from_checkpoint: bool = False,
    checkpoint_file: str = None,
):  # Run the gsplat generation

    # Get the parent directory of data_dir
    workspace_dir = Path(data_dir).parent

    command = [
        "ns-train",
        "splatfacto",
        "--data",
        data_dir,
        "--vis",
        "viewer",
        # "viewer+wandb",
        "--viewer.quit-on-train-completion",
        "True",
        "--output-dir",
        output_dir,
        "--max-num-iterations",
        str(num_train_iterations),
    ]
    
    if load_from_checkpoint:
        command.extend([
            f"--load-checkpoint",
            str(checkpoint_file), 
        ])
        
    command.extend([
        # "--pipeline.model.camera-optimizer.mode",
        # "SO3xR3",
        "nerfstudio-data",
        "--orientation-method",
        "none",
        "--center-method",
        "none",
        "--auto-scale-poses",
        "False",
        "--eval-mode",
        "all",
    ]
    )

    # Run the command
    result = subprocess.run(
        command, cwd=workspace_dir.as_posix(), capture_output=False, text=True
    )

    # Check the result
    if result.returncode == 0:
        print("Command succeeded.")
        print(result.stdout)  # Output of the command

def train_nerf(
    data_dir, output_dir: str = "outputs", num_train_iterations: int = 30000,
    load_from_checkpoint: bool = False,
    checkpoint_file: str = None,
):  # Run the nerf generation

    # Get the parent directory of data_dir
    workspace_dir = Path(data_dir).parent

    command = [
        "ns-train",
        "nerfacto",
        "--data",
        data_dir,
        "--vis",
        "viewer",
        # "viewer+wandb",
        "--viewer.quit-on-train-completion",
        "True",
        "--output-dir",
        output_dir,
        "--max-num-iterations",
        str(num_train_iterations),
    ]
    
    if load_from_checkpoint:
        command.extend([
            f"--load-checkpoint",
            str(checkpoint_file), 
        ])
        
    command.extend([
        # "--pipeline.datamanager.camera-optimizer.mode",
        # "SO3xR3",
        "nerfstudio-data",
        "--orientation-method",
        "none",
        "--center-method",
        "none",
        "--auto-scale-poses",
        "False",
        "--eval-mode",
        "all",
        # "--eval-mode",
        # "train-split-fraction 1.0",
    ]
    )
    
    # Run the command
    result = subprocess.run(
        command, cwd=workspace_dir.as_posix(), capture_output=False, text=True
    )

    # Check the result
    if result.returncode == 0:
        print("Command succeeded.")
        print(result.stdout)  # Output of the command


def cameras_list_to_json(output_path="camera_path.json",cameras_list=[],  fps=24, duration=16, smoothness=0.5, is_cycle=True):
    """Convert a list of Cameras objects into a JSON file matching BayesRays' expected format."""
    
    if not cameras_list:
        raise ValueError("Cameras list is empty.")

    # Ensure the first item is a Cameras object
    if not isinstance(cameras_list[0], Cameras):
        raise TypeError(f"Expected a list of Cameras objects, but got {type(cameras_list[0])}.")

    # Extract common image resolution from the first camera
    first_camera = cameras_list[0]
    image_width = int(first_camera.width.item())  # Convert tensor to int
    image_height = int(first_camera.height.item())

    # Convert camera type tensor to a readable string
    camera_type_mapping = {
        1: "perspective",
        2: "fisheye",
        3: "equirectangular",
    }
    camera_type = camera_type_mapping[first_camera.camera_type.item()]

    # Initialize JSON structure
    # camera_path_json = {
    #     "render_height": image_height,
    #     "render_width": image_width,
    #     "camera_type": camera_type,
    #     "fps": int(fps),
    #     "seconds": float(duration),
    #     "smoothness_value": float(smoothness),
    #     "is_cycle": bool(is_cycle),
    #     "crop": None,
    #     "keyframes": [],
    #     "camera_path": []
    # }

    camera_path_json = {
        "keyframes": [],
        "camera_type": camera_type,
        "render_height": image_height,
        "render_width": image_width,
        "camera_path": [],
        "fps": int(fps),
        "seconds": float(duration),
        "smoothness_value": float(smoothness),
        "is_cycle": bool(is_cycle),
        "crop": None
    }
    for idx, cam in enumerate(cameras_list):
        # Ensure cam is a Cameras object
        if not isinstance(cam, Cameras):
            raise TypeError(f"List contains a non-Cameras object at index {idx}: {type(cam)}")

        # Extract camera_to_world matrix (ensure 4x4 format)
        c2w = cam.camera_to_worlds.cpu().numpy()[0]  # Shape: (3,4)

        # Append last row [0, 0, 0, 1] to convert to 4x4 homogeneous matrix
        c2w_homogeneous = list(map(lambda row: list(map(float, row)), c2w))  # Convert to Python float
        c2w_homogeneous.append([0.0, 0.0, 0.0, 1.0])  # Append last row

        # Flatten the matrix to match the format in "camera_path"
        c2w_flat = [float(item) for row in c2w_homogeneous for item in row]  # Ensure all elements are Python floats

        # Extract field of view (FOV) from focal length
        fx = float(cam.fx.item())  # Ensure it's a Python float
        fov = float(2 * torch.atan(torch.tensor(image_width / (2 * fx), dtype=torch.float64)) * (180 / torch.pi))  # Convert to degrees

        # Assume equal aspect ratio
        aspect = float(1.0)

        # Estimate the render time if applicable
        render_time = float(idx / len(cameras_list) * duration)

        # Add to "keyframes"
        camera_path_json["keyframes"].append({
            "matrix": str(c2w_flat),  # Convert to string
            "fov": fov,
            "aspect": aspect,
            "properties": f'[[\"FOV\",{fov}],[\"NAME\",\"Camera {idx}\"],[\"TIME\",{render_time}]]'
        })

        camera_path_json["camera_type"] = camera_type
        camera_path_json["render_height"] = image_height
        camera_path_json["render_width"] = image_width

        # Add to "camera_path"
        camera_path_json["camera_path"].append({
            "camera_to_world": c2w_flat,
            "fov": fov,
            "aspect": aspect
        })

    camera_path_json["fps"] = fps
    camera_path_json["seconds"] = duration
    camera_path_json["smoothness_value"] = smoothness
    camera_path_json["is_cycle"] = is_cycle
    camera_path_json["crop"] = None

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(camera_path_json, f, indent=4)

    print(f"Saved camera path to {output_path}")

def extract_uncertainty(config_path, data_dir, output_path: str = "outputs",
):
    # Get the parent directory of data_dir
    workspace_dir = Path(data_dir).parent

    command = [
        "ns-uncertainty",
        "--load-config",
        config_path,
        "--output-path",
        output_path,
    ]

    # Run the command
    result = subprocess.run(
        command, cwd=workspace_dir.as_posix(), capture_output=False, text=True
    )

    # Check the result
    if result.returncode == 0:
        print("Command succeeded.")
        # print(result.stdout)  # Output of the command
    else:
        print("Command Failed.")
        

def render_uncertainty(data_dir,
    config_path,
    unc_path,
    cameras_path,
    output_path: str = "outputs",
):
    # Get the parent directory of data_dir
    '''
    ns-render-u camera-path --load-config={PATH_TO_CONFIG} --unc_path {PATH_TO_UNCERTAINTY} 
    --output-path {PATH_TO_VIDEO} --downscale-factor 2  
    --rendered_output_names rgb depth uncertainty  
    --filter-out True --filter-thresh 1. 
    --white-bg True --black-bg False  --camera_path_filename {PATH_TO_CAMERA_PATH}

    ns-render dataset --load-config <config.yaml> --frame-indices 0 5 10 15 --output-path output.mp4
    '''
    workspace_dir = Path(data_dir).parent

    print(f"config_path: {config_path}")
    print(f"unc_path: {unc_path}")
    print(f"output_path: {output_path}")
    print(f"cameras_path: {cameras_path}")

    command = [
        "ns-render-u",
        "camera-path",
        f"--load-config={config_path}",
        "--unc-path", str(unc_path),
        "--output-path", str(output_path),
        "--rendered_output_names", "uncertainty",
        "--filter-out", "True",
        "--filter-thresh", "1.",
        "--white-bg", "True",
        "--black-bg", "False",
        "--camera-path-filename", str(cameras_path),
        "--output-format", "images",
    ]

    # Run the command
    result = subprocess.run(
        command, cwd=workspace_dir.as_posix(), capture_output=False, text=True
    )

    # Check the result
    if result.returncode == 0:
        print("Command succeeded.")
        # print(result.stdout)  # Output of the command

def read_photometric_scores(output_dir: str):
    """
    Read the PSNR, SSIM, LPIPS scores for the GSplat from the saved file.
    """
    # flag indicating the stats have been found
    found_stats: bool = False

    # pdb.set_trace()
    # find the directory
    for root, subdirs, files in os.walk(output_dir):
        for name in files:
            if "wandb-summary.json" in name:
                if not found_stats:
                    # extract the stats
                    with open(os.path.join(root, name), "r") as fh:
                        stats = json.load(fh)

                        # update the flag
                        found_stats = True
                else:
                    warnings.warn("More than one valid set of stats was found!")

    if not found_stats:
        raise RuntimeError("No stats file was found!")

    # return the PSNR, SSIM, and LPIPS  scores
    psnr = stats["Eval Images Metrics Dict (all images)/psnr"]
    ssim = stats["Eval Images Metrics Dict (all images)/ssim"]
    lpips = stats["Eval Images Metrics Dict (all images)/lpips"]

    return psnr, ssim, lpips


def compute_photometric_scores(
    output_dir: Union[Path, str],
    data_dir: Union[Path, str],
    test_frames_data,
    cam_intrinsics: Dict,
    method_name: str = "default_name",
    run_name: str = "default_run",
    save_rendered_images: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Compute the PSNR, SSIM, LPIPS scores for the GSplat from the image data.
    """

    from pytorch_msssim import SSIM
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    # photometric error metrics
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    # extract the camera transforms
    cam_transforms_data = dict(
        [(frame["file_path"], frame["transform_matrix"]) for frame in test_frames_data]
    )

    # camera intrinsics
    cam_fx = cam_intrinsics["fx"]
    cam_fy = cam_intrinsics["fy"]
    cam_cx = cam_intrinsics["cx"]
    cam_cy = cam_intrinsics["cy"]
    cam_height = cam_intrinsics["h"]
    cam_width = cam_intrinsics["w"]
    cam_camera_type = cam_intrinsics["cam_camera_type"]

    # metrics
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    if save_rendered_images:
        # output directory for rendered images
        img_output_dir = Path(f"{output_dir}/render_results/{method_name}")

        # filename
        img_output_filename = Path(f"{img_output_dir}/{run_name}/rendered_rgb")

        # filename
        unc_img_output_filename = Path(f"{img_output_dir}/{run_name}/rendered_unc")

        # create directory, if necessary
        img_output_filename.mkdir(parents=True, exist_ok=True)

        unc_img_output_filename.mkdir(parents=True, exist_ok=True)

    # identify the configuration file
    # flag indicating the config file has been found
    found_config: bool = False

    # find the directory
    for root, subdirs, files in os.walk(output_dir):
        for name in files:
            if "config.yml" in name:
                if not found_config:
                    # update the config path
                    config_path = Path(os.path.join(root, name))

                    # update the config
                else:
                    warnings.warn("More than one valid config file was found!")

    # load the model
    model = NeRF(config_path, cam_intrinsics=cam_intrinsics, dataset_mode="train")

    # render and compute the photometric error
    for cam_idx, t_key in enumerate(
        tqdm(cam_transforms_data, desc="Computing the Photometric Stats")
    ):
        # ground-truth image
        gt_rgb = Image.open(f"{data_dir}/{t_key}")
        gt_rgb = np.asarray(gt_rgb).astype(float) / 255.0
        gt_rgb = torch.tensor(gt_rgb, device=device).float()
        gt_rgb = gt_rgb.moveaxis(-1, 0).cpu()[None]

        # extract the camera pose
        cam_transforms_pose = torch.tensor(cam_transforms_data[t_key], device=device)

        # create a camera
        cams_render = Cameras(
            fx=cam_fx,
            fy=cam_fy,
            cx=cam_cx,
            cy=cam_cy,
            camera_type=cam_camera_type,
            camera_to_worlds=cam_transforms_pose[:3, :],
            width=cam_width,
            height=cam_height,
        )

        # render an image
        rendered_rgb = model.render(cameras=cams_render, pose=None)["rgb"]
        # print(f"rendered_rgb: {rendered_rgb.shape}")
        rendered_rgb = rendered_rgb.moveaxis(-1, 0).cpu()[None]

        rendered_unc = model.render(cameras=cams_render, pose=None)["uncertainty"]
        # print(f"rendered_unc: {rendered_unc.shape}")
        # print(f"Max rendered_unc: {rendered_unc.max()}")
        # print(f"Min rendered_unc: {rendered_unc.min()}")

        # Apply inferno colormap to uncertainty map
        rendered_unc = rendered_unc.squeeze().cpu().numpy()
        # rendered_unc_colormap = cm.inferno(1 - (rendered_unc / rendered_unc.max()))[:, :, :3]
        rendered_unc_colormap = cm.inferno((rendered_unc - rendered_unc.min())/(rendered_unc.max()-rendered_unc.min()))[:, :, :3]
        rendered_unc_colormap = torch.tensor(rendered_unc_colormap).permute(2, 0, 1).unsqueeze(0).float()

        # rendered_unc = apply_colormap(rendered_unc, ColormapOptions("inferno"))
        # unc_img = colormaps.apply_colormap(
        #     image=rendered_unc,
        #     colormap_options=colormap_options1,
        # ).cpu().numpy()

        # # uncertainty image
        # unc_img = (unc_img * 255).astype(np.uint8)
        
        # rendered_unc = rendered_unc.moveaxis(-1, 0).cpu()[None]

        # compute the metrics
        psnr_vals.append(psnr(gt_rgb, rendered_rgb))
        ssim_vals.append(ssim(gt_rgb, rendered_rgb))
        lpips_vals.append(lpips(gt_rgb, rendered_rgb))

        if save_rendered_images:
            # save the images
            # rendered RGB
            Image.fromarray(
                (
                    rendered_rgb.squeeze().moveaxis(0, -1).detach().cpu().numpy() * 255
                ).astype(np.uint8)
            ).save(Path(f"{img_output_filename}/{t_key.split('/')[-1]}"))

            # Image.fromarray(
            #     unc_img
            #     ).save(Path(f"{unc_img_output_filename}/{t_key.split('/')[-1]}"))
            Image.fromarray(
                (
                    rendered_unc_colormap.squeeze().moveaxis(0, -1).detach().cpu().numpy() * 255
                ).astype(np.uint8)
            ).save(Path(f"{unc_img_output_filename}/{t_key.split('/')[-1]}"))

    # compute the summary statistics of the metrics
    psnr_stats = torch.std_mean(torch.tensor(psnr_vals, device=device))
    ssim_stats = torch.std_mean(torch.tensor(ssim_vals, device=device))
    lpips_stats = torch.std_mean(torch.tensor(lpips_vals, device=device))

    # cast to array
    psnr_stats = [stats.item() for stats in psnr_stats]
    ssim_stats = [stats.item() for stats in ssim_stats]
    lpips_stats = [stats.item() for stats in lpips_stats]

    return psnr_stats, ssim_stats, lpips_stats




def render_from_checkpoint(
    output_dir: Union[Path, str],
    data_dir: Union[Path, str],
    test_frames_data,
    cam_intrinsics: Dict,
    method_name: str = "default_name",
    run_name: str = "default_run",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):

    # extract the camera transforms
    cam_transforms_data = dict(
        [(frame["file_path"], frame["transform_matrix"]) for frame in test_frames_data]
    )

    # camera intrinsics
    cam_fx = cam_intrinsics["fx"]
    cam_fy = cam_intrinsics["fy"]
    cam_cx = cam_intrinsics["cx"]
    cam_cy = cam_intrinsics["cy"]
    cam_height = cam_intrinsics["h"]
    cam_width = cam_intrinsics["w"]
    cam_camera_type = cam_intrinsics["cam_camera_type"]

    # output directory for rendered images
    img_output_dir = Path(f"{output_dir}/render_results/{method_name}")

    # filename
    img_output_filename = Path(f"{img_output_dir}/{run_name}/rendered_rgb")
    # filename
    unc_img_output_filename = Path(f"{img_output_dir}/{run_name}/rendered_unc")

    # create directory, if necessary
    img_output_filename.mkdir(parents=True, exist_ok=True)
    unc_img_output_filename.mkdir(parents=True, exist_ok=True)

    found_config: bool = False

    # find the directory
    for root, subdirs, files in os.walk(output_dir):
        for name in files:
            if "config.yml" in name:
                if not found_config:
                    # update the config path
                    config_path = Path(os.path.join(root, name))

                    # update the config
                else:
                    warnings.warn("More than one valid config file was found!")

    # load the model
    model = NeRF(config_path, cam_intrinsics=cam_intrinsics, dataset_mode="train")

    # render and compute the photometric error
    for cam_idx, t_key in enumerate(
        tqdm(cam_transforms_data, desc="Computing the Photometric Stats")
    ):
        # ground-truth image
        gt_rgb = Image.open(f"{data_dir}/{t_key}")
        gt_rgb = np.asarray(gt_rgb).astype(float) / 255.0
        gt_rgb = torch.tensor(gt_rgb, device=device).float()
        gt_rgb = gt_rgb.moveaxis(-1, 0).cpu()[None]

        # extract the camera pose
        cam_transforms_pose = torch.tensor(cam_transforms_data[t_key], device=device)

        # create a camera
        cams_render = Cameras(
            fx=cam_fx,
            fy=cam_fy,
            cx=cam_cx,
            cy=cam_cy,
            camera_type=cam_camera_type,
            camera_to_worlds=cam_transforms_pose[:3, :],
            width=cam_width,
            height=cam_height,
        )

        # render an image
        rendered_rgb = model.render(cameras=cams_render, pose=None)["rgb"]
        # print(f"rendered_rgb: {rendered_rgb.shape}")
        rendered_rgb = rendered_rgb.moveaxis(-1, 0).cpu()[None]

        rendered_unc = model.render(cameras=cams_render, pose=None)["uncertainty"]
        # print(f"rendered_unc: {rendered_unc.shape}")
        # print(f"Max rendered_unc: {rendered_unc.max()}")
        # print(f"Min rendered_unc: {rendered_unc.min()}")

        # Apply inferno colormap to uncertainty map
        rendered_unc = rendered_unc.squeeze().cpu().numpy()
        # rendered_unc_colormap = cm.inferno(1 - (rendered_unc / rendered_unc.max()))[:, :, :3]
        rendered_unc_colormap = cm.inferno((rendered_unc - rendered_unc.min())/(rendered_unc.max()-rendered_unc.min()))[:, :, :3]
        rendered_unc_colormap = torch.tensor(rendered_unc_colormap).permute(2, 0, 1).unsqueeze(0).float()

        # rendered_unc = apply_colormap(rendered_unc, ColormapOptions("inferno"))
        # unc_img = colormaps.apply_colormap(
        #     image=rendered_unc,
        #     colormap_options=colormap_options1,
        # ).cpu().numpy()

        # # uncertainty image
        # unc_img = (unc_img * 255).astype(np.uint8)
        
        # rendered_unc = rendered_unc.moveaxis(-1, 0).cpu()[None]



        # save the images
        # rendered RGB
        Image.fromarray(
            (
                rendered_rgb.squeeze().moveaxis(0, -1).detach().cpu().numpy() * 255
            ).astype(np.uint8)
        ).save(Path(f"{img_output_filename}/{t_key.split('/')[-1]}"))

        # Image.fromarray(
        #     unc_img
        #     ).save(Path(f"{unc_img_output_filename}/{t_key.split('/')[-1]}"))
        Image.fromarray(
            (
                rendered_unc_colormap.squeeze().moveaxis(0, -1).detach().cpu().numpy() * 255
            ).astype(np.uint8)
        ).save(Path(f"{unc_img_output_filename}/{t_key.split('/')[-1]}"))




def get_path_to_checkpoint(output_dir: str):
    """
    Get the Path to the Checkpoint for the Model.
    """
    # flag indicating the file have been found
    found_path: bool = False

    # pdb.set_trace()
    # find the directory
    for root, subdirs, files in os.walk(output_dir):
        for name in files:
            if ".ckpt" in name:
                if not found_path:
                    # extract the path
                    ckpt_path = os.path.join(root, name)
                
                    # update the flag
                    found_path = True
                else:
                    warnings.warn("More than one valid set of checkpoints was found!")

    if not found_path:
        raise RuntimeError("No checkpoint file was found!")

    return ckpt_path

def train_all(method_name, data_dir, output_dir):
    output_base_dir = Path(f"./outputs/fisher_rf").resolve()
    perf_stats = {}
    # train the GSplat
    train_gsplat(data_dir=data_dir, output_dir=output_dir)

    # pdb.set_trace()
    # retrieve the photometric stats
    psnr, ssim, lpips = read_photometric_scores(output_dir=output_dir)

    # cache the stats
    if not method_name in perf_stats:
        perf_stats[method_name] = {"psnr": [psnr], "ssim": [ssim], "lpips": [lpips]}
    else:
        perf_stats[method_name]["psnr"].append(psnr)
        perf_stats[method_name]["ssim"].append(ssim)
        perf_stats[method_name]["lpips"].append(lpips)

    all_stats = []
    # Sort directories by name (timestamp)
    sorted_subdirs = sorted(
        [subdir for subdir in output_base_dir.iterdir() if subdir.is_dir()],
        key=lambda x: x.name,
    )

    for subdir in sorted_subdirs:
        try:
            psnr, ssim, lpips = read_photometric_scores(output_dir=str(subdir))
            # acq_scores = perf_stats[method_name]['acq_scores'][int(subdir.name.split('_')[-1])]
            all_stats.append(
                {
                    "Directory": subdir.name,
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "LPIPS": lpips,
                }
            )
        except Exception as e:
            print(f"Error reading scores from {subdir}: {e}")

    # Tabulate the data
    print(tabulate(all_stats, headers="keys", tablefmt="grid"))
