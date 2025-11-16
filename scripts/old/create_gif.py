#!/usr/bin/env python3
"""
Script to create a GIF from a directory of images.
Supports various image formats and provides options for duration, loop, and optimization.
"""

import os
import glob
import argparse
from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageSequence
import numpy as np


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from a directory, sorted by name.

    Args:
        directory: Path to the directory containing images
        extensions: List of file extensions to include (e.g., ['.png', '.jpg'])

    Returns:
        List of image file paths, sorted
    """
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

    image_files = []
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(directory, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))

    # Sort files to ensure consistent ordering
    image_files.sort()
    return image_files


def resize_images(
    images: List[Image.Image], scale_factor: Optional[float] = None
) -> List[Image.Image]:
    """
    Resize all images by a scaling factor to maintain aspect ratio.

    Args:
        images: List of PIL Image objects
        scale_factor: Scaling factor (e.g., 0.5 for half size, 2.0 for double size). If None, use the size of the first image.

    Returns:
        List of resized images
    """
    if scale_factor is None:
        return images

    resized_images = []
    for img in images:
        if scale_factor != 1.0:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(img)

    return resized_images


def create_gif(
    input_dir: str,
    output_path: str,
    duration: int = 500,
    loop: int = 0,
    scale_factor: Optional[float] = None,
    optimize: bool = True,
    quality: int = 85,
    extensions: List[str] = None,
) -> None:
    """
    Create a GIF from images in a directory.

    Args:
        input_dir: Directory containing the images
        output_path: Path for the output GIF file
        duration: Duration for each frame in milliseconds
        loop: Number of loops (0 for infinite)
        scale_factor: Scaling factor for resizing (e.g., 0.5 for half size, 2.0 for double size)
        optimize: Whether to optimize the GIF
        quality: Quality setting for optimization (1-100)
        extensions: List of file extensions to include
    """
    # Get image files
    image_files = get_image_files(input_dir, extensions)

    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")

    print(f"Found {len(image_files)} images in {input_dir}")

    # Load images
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            # Convert to RGB if necessary (GIF doesn't support RGBA)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            images.append(img)
            print(f"Loaded: {os.path.basename(file_path)} ({img.size})")
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")

    if not images:
        raise ValueError("No valid images could be loaded")

    # Resize images if requested
    if scale_factor:
        print(f"Resizing images by scale factor: {scale_factor}")
        images = resize_images(images, scale_factor)

    # Save as GIF
    print(f"Creating GIF: {output_path}")
    print(f"Duration: {duration}ms per frame")
    print(f"Loop: {'infinite' if loop == 0 else loop} times")

    # Save the first image and append the rest
    first_image = images[0]
    remaining_images = images[1:]

    first_image.save(
        output_path,
        save_all=True,
        append_images=remaining_images,
        duration=duration,
        loop=loop,
        optimize=optimize,
        quality=quality,
    )

    print(f"GIF created successfully: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Create a GIF from a directory of images")
    parser.add_argument("input_dir", help="Directory containing the images")
    parser.add_argument("-o", "--output", default="output.gif", help="Output GIF file path")
    parser.add_argument(
        "-d", "--duration", type=int, default=500, help="Duration per frame in milliseconds"
    )
    parser.add_argument(
        "-l", "--loop", type=int, default=0, help="Number of loops (0 for infinite)"
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        metavar="FACTOR",
        help="Scale factor for resizing (e.g., 0.5 for half size, 2.0 for double size)",
    )
    parser.add_argument("--no-optimize", action="store_true", help="Disable GIF optimization")
    parser.add_argument("-q", "--quality", type=int, default=85, help="Quality setting (1-100)")
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
        help="File extensions to include",
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return 1

    # Validate quality
    if not 1 <= args.quality <= 100:
        print("Error: Quality must be between 1 and 100")
        return 1

    # Validate scale factor
    if args.scale is not None and args.scale <= 0:
        print("Error: Scale factor must be positive")
        return 1

    try:
        create_gif(
            input_dir=args.input_dir,
            output_path=args.output,
            duration=args.duration,
            loop=args.loop,
            scale_factor=args.scale,
            optimize=not args.no_optimize,
            quality=args.quality,
            extensions=args.extensions,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
