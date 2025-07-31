# %%
from pathlib import Path
from load_model import GaussianSplat
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import SSIM
import open3d as o3d
from PIL import Image

from shadow_splat.util.nerfstudio import composite_with_background

USE_GT_POINTS = False
ALBEDO = False

# Model type
model_type = "shadow-splat"  # "shadow-splat" or "splatfacto"
dataset_name = (
    "master_chief_cycles"  # "o3d_el45_az60" or "multi_light" or "single_view_multi_light"
)

if USE_GT_POINTS:
    gt_path = "open3d_dataset/all.ply"  # Path to the point cloud

if __name__ == "__main__":
    # Looks at the directory and looks for the latest checkpoint
    config_path = Path(f"outputs/{dataset_name}/{model_type}/")

    # Get the folder with the most recent timestamp
    latest_folder = max(config_path.glob("*"), key=os.path.getctime)

    config_path = Path(f"{latest_folder}/config.yml")

    print("Loading model from: ", config_path)

    if dataset_name == "single_view_multi_light":
        # Load pipeline
        pipeline = GaussianSplat(config_path, dataset_mode="train", device="cuda")

        dataset_name_eval = "multi_light"
        eval_config_path = Path(f"outputs/{dataset_name_eval}/{model_type}/")

        # Get the folder with the most recent timestamp
        latest_folder_eval = max(eval_config_path.glob("*"), key=os.path.getctime)

        eval_config_path = Path(f"{latest_folder_eval}/config.yml")
        eval_pipeline = GaussianSplat(eval_config_path, dataset_mode="train", device="cuda")

        # Load the training cameras
        cameras = eval_pipeline.get_cameras()

        # Load the light source
        if not ALBEDO:
            light_sources = eval_pipeline.get_light_source()
        else:
            light_sources = None
            model_type = "shadow-splat-albedo"

        # Load images
        images = eval_pipeline.get_images()
    else:
        # Load pipeline
        pipeline = GaussianSplat(config_path, dataset_mode="train", device="cuda")

        # Load the training cameras
        cameras = pipeline.get_cameras()

        # Load the light source
        if not ALBEDO:
            light_sources = pipeline.get_light_source()
        else:
            light_sources = None
            model_type = "shadow-splat-albedo"

        # Load images
        images = pipeline.get_images()

    print("Number of cameras: ", len(cameras))
    print("Number of light sources: ", len(light_sources) if light_sources is not None else 0)
    print("Number of images: ", len(images))

    # Directory to save results
    results_dir = Path(f"results/{dataset_name}/{model_type}/{latest_folder.name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True)

    # Load the ground truth point cloud
    if USE_GT_POINTS:
        gt_pcd = o3d.io.read_point_cloud(gt_path)
        pcd = pipeline.pipeline.model.means.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))

        # Compute chamfer distance between the predicted and ground truth point clouds
        chamfer_dist_gt2pcd = np.asarray(gt_pcd.compute_point_cloud_distance(pcd)).mean()
        chamfer_dist_pcd2gt = np.asarray(pcd.compute_point_cloud_distance(gt_pcd)).mean()
        print(f"Chamfer distance (gt -> pcd): {chamfer_dist_gt2pcd}")
        print(f"Chamfer distance (pcd -> gt): {chamfer_dist_pcd2gt}")

    # Store metrics for each image
    metrics_data = []

    # Render from the training poses
    for idx in range(len(cameras)):
        if model_type == "shadow-splat":
            outputs = pipeline.render(
                cameras[idx : idx + 1],
                light_sources[idx : idx + 1] if light_sources is not None else None,
            )
            rendered_img = outputs["rgb"].squeeze().cpu()
        elif model_type == "splatfacto":
            outputs = pipeline.render(cameras[idx : idx + 1])
            rendered_img = outputs["rgb"].squeeze().cpu()
        elif model_type == "shadow-splat-albedo":
            outputs = pipeline.render(cameras[idx : idx + 1], None)
            rendered_img = outputs["rgb"].squeeze().cpu()
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # Get rendered and ground truth images
        gt_img = images[idx].squeeze().cpu()[..., :3]

        # Ensure images are in the correct format for metrics computation
        # Convert from [H, W, C] to [1, C, H, W] for metrics
        rendered_img_metric = rendered_img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        gt_img_metric = gt_img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        # Compute metrics
        psnr_val = psnr_metric(rendered_img_metric, gt_img_metric).item()
        ssim_val = ssim_metric(rendered_img_metric, gt_img_metric).item()
        lpips_val = lpips_metric(rendered_img_metric, gt_img_metric).item()

        # Store metrics
        metrics_data.append(
            {"image_idx": idx, "psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}
        )

        print(f"Image {idx}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(rendered_img.numpy())
        # ax[1].imshow(gt_img.numpy())
        # ax[0].set_title("Rendered")
        # ax[1].set_title("Ground truth")
        # ax[0].axis("off")
        # ax[1].axis("off")
        # plt.savefig(results_dir / f"render_{idx}.png", bbox_inches="tight", pad_inches=0)
        # plt.close()
        combined_img = np.concatenate([rendered_img.numpy(), gt_img.numpy()], axis=1)
        if combined_img.dtype != np.uint8:
            combined_img = np.clip(combined_img * 255, 0, 255).astype(np.uint8)
        Image.fromarray(combined_img).save(results_dir / f"render_{idx}.png")

        # Create metrics table
        metrics_df = pd.DataFrame(metrics_data)

        # Calculate average metrics
        avg_metrics = {
            "avg_psnr": metrics_df["psnr"].mean(),
            "avg_ssim": metrics_df["ssim"].mean(),
            "avg_lpips": metrics_df["lpips"].mean(),
            "std_psnr": metrics_df["psnr"].std(),
            "std_ssim": metrics_df["ssim"].std(),
            "std_lpips": metrics_df["lpips"].std(),
        }
        if USE_GT_POINTS:
            avg_metrics["chamfer_dist_gt2pcd"] = chamfer_dist_gt2pcd
            avg_metrics["chamfer_dist_pcd2gt"] = chamfer_dist_pcd2gt

        # Save individual metrics to CSV
        metrics_df.to_csv(results_dir / "metrics_per_image.csv", index=False)

        # Save average metrics to CSV
        avg_df = pd.DataFrame([avg_metrics])
        avg_df.to_csv(results_dir / "average_metrics.csv", index=False)

        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Average PSNR: {avg_metrics['avg_psnr']:.4f} ± {avg_metrics['std_psnr']:.4f}")
        print(f"Average SSIM: {avg_metrics['avg_ssim']:.4f} ± {avg_metrics['std_ssim']:.4f}")
        print(f"Average LPIPS: {avg_metrics['avg_lpips']:.4f} ± {avg_metrics['std_lpips']:.4f}")
        if USE_GT_POINTS:
            print(f"Chamfer distance (gt -> pcd): {avg_metrics['chamfer_dist_gt2pcd']:.4f}")
            print(f"Chamfer distance (pcd -> gt): {avg_metrics['chamfer_dist_pcd2gt']:.4f}")
        print(f"Results saved to: {results_dir}")
        print("=" * 50)
        print("\n")
