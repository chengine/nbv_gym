"""Debug script to compare pretrained 3DGS model renders against ground truth.

Usage:
    python scripts/debug_render_comparison.py --config /path/to/config.yml --num-images 5
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from nerfstudio.utils.eval_utils import eval_setup


def main():
    parser = argparse.ArgumentParser(description="Compare 3DGS renders vs ground truth")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the pretrained model config.yml",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of images to compare (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the comparison figure (optional, will show interactively if not set)",
    )
    parser.add_argument(
        "--use-eval",
        action="store_true",
        help="Use eval dataset instead of train dataset",
    )
    args = parser.parse_args()

    # Load the pretrained model
    print(f"Loading model from: {args.config}")
    config, pipeline, _, step = eval_setup(args.config, test_mode="inference")
    model = pipeline.model
    datamanager = pipeline.datamanager

    model.eval()
    device = model.device
    print(f"Model loaded at step {step}, device: {device}")

    # Choose dataset
    if args.use_eval:
        dataset = datamanager.eval_dataset
        cached_data = datamanager.cached_eval
        dataset_name = "eval"
    else:
        dataset = datamanager.train_dataset
        cached_data = datamanager.cached_train
        dataset_name = "train"

    num_images = min(args.num_images, len(cached_data))
    print(f"Comparing {num_images} images from {dataset_name} dataset (total: {len(cached_data)})")
    print(f"Cache type: {datamanager.config.cache_images_type}")

    # Create figure
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    metrics = []

    for idx in range(num_images):
        # Get ground truth image - use get_image_float32 to ensure proper conversion
        # Cached images may be uint8 and need conversion
        cached_img = cached_data[idx]["image"]
        if cached_img.dtype == torch.uint8:
            gt_image = cached_img.float() / 255.0
        else:
            gt_image = cached_img.float()
        gt_image = gt_image.to(device)  # [H, W, 3]

        # Get corresponding camera
        camera = dataset.cameras[idx:idx+1].to(device)

        # Render from model
        with torch.no_grad():
            outputs = model(camera)

        rendered_rgb = outputs["rgb"]  # [H, W, 3]

        # Compute metrics
        mse = torch.mean((gt_image - rendered_rgb) ** 2).item()
        psnr = -10 * torch.log10(torch.tensor(mse)).item()
        metrics.append({"idx": idx, "mse": mse, "psnr": psnr})

        # Convert to numpy for plotting
        gt_np = gt_image.cpu().numpy().clip(0, 1)
        rendered_np = rendered_rgb.cpu().numpy().clip(0, 1)
        diff_np = (torch.abs(gt_image - rendered_rgb).cpu().numpy() * 3).clip(0, 1)  # Amplified difference

        # Plot
        axes[idx, 0].imshow(gt_np)
        axes[idx, 0].set_title(f"Ground Truth (idx={idx})")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(rendered_np)
        axes[idx, 1].set_title(f"Rendered (PSNR={psnr:.2f}dB)")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(diff_np)
        axes[idx, 2].set_title(f"Difference (3x amplified)")
        axes[idx, 2].axis("off")

        print(f"  Image {idx}: MSE={mse:.6f}, PSNR={psnr:.2f}dB")

    # Summary
    avg_psnr = sum(m["psnr"] for m in metrics) / len(metrics)
    avg_mse = sum(m["mse"] for m in metrics) / len(metrics)
    print(f"\nAverage: MSE={avg_mse:.6f}, PSNR={avg_psnr:.2f}dB")

    fig.suptitle(f"3DGS Render vs Ground Truth Comparison\n{args.config.name}\nAvg PSNR: {avg_psnr:.2f}dB", fontsize=14)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
