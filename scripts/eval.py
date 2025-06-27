#%%
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

# Model type
model_type = "splatfacto" # "shadow-splat" or "splatfacto"
dataset_name = "o3d_el45_az60"

# Looks at the directory and looks for the latest checkpoint
config_path = Path(f"outputs/{dataset_name}/{model_type}/")

# Get the folder with the most recent timestamp
latest_folder = max(config_path.glob("*"), key=os.path.getctime)

config_path = Path(f"{latest_folder}/config.yml")

# Directory to save results
results_dir = Path(f"/home/chengine/Research/shadow_splat/results/{dataset_name}/{model_type}/{latest_folder.name}")
results_dir.mkdir(parents=True, exist_ok=True)

# Load pipeline
pipeline = GaussianSplat(config_path, dataset_mode="train", device="cuda")

# Load the training cameras
cameras = pipeline.get_cameras()

# Load the light source
light_sources = pipeline.get_light_source()

# Load images
images = pipeline.get_images()

print("Number of cameras: ", len(cameras))
print("Number of light sources: ", len(light_sources) if light_sources is not None else 0)
print("Number of images: ", len(images))

# Initialize metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3)
lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True)

# Store metrics for each image
metrics_data = []

# Render from the training poses
for idx in range(len(cameras)):
    if model_type == "shadow-splat":
        outputs = pipeline.render(cameras[idx:idx+1], light_sources[idx:idx+1] if light_sources is not None else None)
    elif model_type == "splatfacto":
        outputs = pipeline.render(cameras[idx:idx+1])
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Get rendered and ground truth images
    rendered_img = outputs['rgb'].squeeze().cpu()
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
    metrics_data.append({
        'image_idx': idx,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'lpips': lpips_val
    })
    
    print(f"Image {idx}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rendered_img.numpy())
    ax[1].imshow(gt_img.numpy())
    ax[0].set_title('Rendered')
    ax[1].set_title('Ground truth')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.savefig(results_dir / f'render_{idx}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
        
    # Create metrics table
    metrics_df = pd.DataFrame(metrics_data)
    
    # Calculate average metrics
    avg_metrics = {
        'avg_psnr': metrics_df['psnr'].mean(),
        'avg_ssim': metrics_df['ssim'].mean(),
        'avg_lpips': metrics_df['lpips'].mean(),
        'std_psnr': metrics_df['psnr'].std(),
        'std_ssim': metrics_df['ssim'].std(),
        'std_lpips': metrics_df['lpips'].std()
    }
    
    # Save individual metrics to CSV
    metrics_df.to_csv(results_dir / 'metrics_per_image.csv', index=False)
    
    # Save average metrics to CSV
    avg_df = pd.DataFrame([avg_metrics])
    avg_df.to_csv(results_dir / 'average_metrics.csv', index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Average PSNR: {avg_metrics['avg_psnr']:.4f} ± {avg_metrics['std_psnr']:.4f}")
    print(f"Average SSIM: {avg_metrics['avg_ssim']:.4f} ± {avg_metrics['std_ssim']:.4f}")
    print(f"Average LPIPS: {avg_metrics['avg_lpips']:.4f} ± {avg_metrics['std_lpips']:.4f}")
    print(f"Results saved to: {results_dir}")
    print("="*50)
