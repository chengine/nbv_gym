import torch
import math
from typing import Union


def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


to_homo = lambda x: torch.cat(
    [x, torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)], dim=-1
)