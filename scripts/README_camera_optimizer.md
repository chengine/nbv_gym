# Camera Optimizer Test Script

This script demonstrates how to use the nerfstudio CameraOptimizer to optimize camera poses on a simple scene.

## What it does

The script creates a synthetic 3D scene with colored points, generates camera poses with intentional noise, and then uses the CameraOptimizer to correct those poses to match ground truth poses.

## Features

- **Synthetic Scene**: Creates a 3D cube of colored points
- **Noisy Camera Poses**: Generates camera poses in a circle around the scene with controlled noise
- **Camera Optimizer**: Uses nerfstudio's CameraOptimizer with SO3xR3 mode
- **Simple Rendering**: Projects 3D points to 2D images for comparison
- **Visualization**: Saves comparison plots and analysis
- **Metrics**: Computes MSE improvement and pose adjustment statistics

## Usage

### Basic Usage

```bash
python scripts/test_camera_optimizer.py
```

### With Custom Parameters

```bash
python scripts/test_camera_optimizer.py \
    --num_cameras 12 \
    --noise_level 0.15 \
    --num_iterations 2000 \
    --learning_rate 5e-4 \
    --save_plots
```

### Command Line Arguments

- `--num_cameras`: Number of cameras to create (default: 8)
- `--noise_level`: Amount of noise to add to camera poses (default: 0.1)
- `--num_iterations`: Number of training iterations (default: 1000)
- `--learning_rate`: Learning rate for optimization (default: 1e-3)
- `--save_plots`: Save plots to results/ directory
- `--show_plots`: Show plots interactively

## Output

The script will create several files in the `results/` directory:

1. **`camera_optimizer_loss.png`**: Training loss curve
2. **`camera_optimizer_comparison.png`**: Comparison of original, optimized, and ground truth images
3. **`camera_optimizer_adjustments.png`**: Analysis of pose corrections

## Example Output

```
=== Camera Optimizer Test ===
Number of cameras: 8
Noise level: 0.1
Training iterations: 1000

1. Creating 3D scene...
Scene has 64 points

2. Creating camera poses...

3. Setting up cameras...

4. Creating ground truth poses and target images...

5. Training camera optimizer...
Created camera optimizer with 8 cameras
Optimization mode: SO3xR3
Starting camera optimization...
Iteration 0, Loss: 0.123456
Iteration 100, Loss: 0.045678
...
Optimization complete!

6. Getting optimized camera poses...

7. Visualizing results...
Original MSE: 0.123456
Optimized MSE: 0.012345
Improvement: 90.00%

8. Analyzing pose corrections...
Pose Adjustment Analysis:
Translation adjustments - Mean: 0.0456, Max: 0.1234
Rotation adjustments - Mean: 0.0234, Max: 0.0567

=== Test Complete ===
Results saved to results/ directory
Final improvement: 90.00%
```

## How it Works

1. **Scene Creation**: Creates a 4x4x4 grid of colored 3D points
2. **Camera Generation**: Places cameras in a circle around the scene
3. **Noise Addition**: Adds controlled noise to rotation and translation
4. **Ground Truth**: Creates clean camera poses without noise
5. **Target Images**: Renders images from ground truth poses
6. **Optimization**: Trains CameraOptimizer to correct noisy poses
7. **Evaluation**: Compares original vs optimized rendering quality

## Key Components

### CameraOptimizer Configuration

```python
camera_optimizer_config = CameraOptimizerConfig(
    mode="SO3xR3",  # Optimize rotation and translation separately
    trans_l2_penalty=1e-2,  # L2 penalty on translation
    rot_l2_penalty=1e-3,    # L2 penalty on rotation
)
```

### Training Loop

- Renders images from optimized camera poses
- Compares with ground truth images using MSE loss
- Adds regularization to prevent overfitting
- Updates pose adjustments using gradient descent

### Pose Representation

- **Translation**: 3D vector (x, y, z)
- **Rotation**: 3D axis-angle representation
- **Total**: 6 parameters per camera

## Applications

This script is useful for:

- **Understanding CameraOptimizer**: See how it corrects camera poses
- **Testing Parameters**: Experiment with different noise levels and optimization settings
- **Debugging**: Verify camera optimization is working correctly
- **Education**: Learn about camera pose optimization in NeRF training

## Integration with ShadowSplat

The CameraOptimizer is used in ShadowSplat to:

- Correct camera poses during training
- Improve scene reconstruction quality
- Handle imperfect initial camera calibration from COLMAP
- Potentially optimize light source poses

## Troubleshooting

- **High Loss**: Try reducing noise level or increasing iterations
- **No Improvement**: Check learning rate and regularization parameters
- **Memory Issues**: Reduce number of cameras or image resolution
- **Import Errors**: Ensure nerfstudio is properly installed
