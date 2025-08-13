import math
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import Literal

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    fully_fused_projection_2dgs,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    rasterize_to_pixels_2dgs,
    spherical_harmonics,
)
from gsplat.distributed import (
    all_gather_int32,
    all_gather_tensor_list,
    all_to_all_int32,
    all_to_all_tensor_list,
)
from gsplat.utils import depth_to_normal
from nerfstudio.cameras.cameras import Cameras, CameraType
import matplotlib.pyplot as plt

def focal_bce(pred, target, alpha_pos=0.9, alpha_neg=0.1, gamma=2.0, eps=1e-6):
    # pred, target in [0,1]
    p = torch.clamp(pred, eps, 1-eps)
    # per-sample alpha blended by target (so positives get alpha_pos)
    alpha = target * alpha_pos + (1 - target) * alpha_neg
    pt = target * p + (1 - target) * (1 - p)
    foc = (1 - pt).pow(gamma)
    bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    return (alpha * foc * bce).mean()

def sg_flux_logI(k: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    log I(k), where I(k) = 2π * (1 - exp(-2k)) / k.
    Stable for all k >= 0.
    """
    k = torch.clamp(k, min=0.0)
    small = (k <= 1e-6)
    out = torch.empty_like(k)

    if small.any():
        ks = k[small]
        # (1 - e^{-2k})/k ≈ 2 - 2k + 4k^2/3 ...  use first term in log-space:
        # log I ≈ log(4π) + log(1 - ks + (2/3)ks^2)  (good enough for tiny ks)
        t = 1.0 - ks + (2.0/3.0)*(ks*ks)
        out[small] = math.log(4*math.pi) + torch.log(torch.clamp(t, min=eps))

    if (~small).any():
        kb = k[~small]
        # log I = log(2π) + log(1 - e^{-2k}) - log k
        out[~small] = (math.log(2*math.pi)
                       + torch.log(-torch.expm1(-2.0*kb))
                       - torch.log(kb))
    return out

def sg_flux_I(k: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # I(k) = 2π * (1 - exp(-2k)) / k, stable
    k = torch.clamp(k, min=0.0)
    small = (k <= 1e-6)
    I = torch.empty_like(k)

    if small.any():
        ks = k[small]
        # Direct series: I ≈ 4π * (1 - ks/1 + (2/3)ks^2)
        I[small] = 4.0*math.pi * (1.0 - ks + (2.0/3.0)*(ks*ks))

    if (~small).any():
        kb = k[~small]
        I[~small] = 2.0*math.pi * (-torch.expm1(-2.0*kb)) / (kb + eps)
    return I

def sg_normalized_value(mu, k, dirs, mode=Literal["batch", "broadcast"]):  # returns exp( κ(μ·ω-1) - log I(κ) )

    if mode == "batch":
        kappa = k
        dot = (mu * dirs).sum(-1).clamp(-1.0, 1.0)
    else:
        kappa = k.unsqueeze(-1)
        dot = torch.einsum('kcm,nm->kcn', mu, dirs).clamp(-1.0, 1.0)                 # [K2,C,N]

    logI = sg_flux_logI(kappa)
    logv = kappa*(dot - 1.0) - logI

    # If you actually need the value (not just logs), exponentiate with care:
    return torch.exp(torch.clamp(logv, max=80.0))  # cap to avoid inf in float32

def sg_mixture_eval(
    mode: Literal["batch", "broadcast"],
    mu: torch.Tensor,        # batch: [N,K,3]      | env: [K2,C,3]
    kappa: torch.Tensor,     # batch: [N,K]        | env: [K2,C]
    weights: torch.Tensor,   # batch: [N,K]        | env: [K2,C]
    s: torch.Tensor,         # batch: [N]          | env: [C]
    dirs: torch.Tensor,      # view directions [N,3] (assumed unit)
    normalize: bool = False,
) -> torch.Tensor:
    """
    Evaluate a mixture of spherical Gaussians (SGs):

        SG(ω; μ, κ) = exp( κ * (μ·ω - 1) )

    mode="batch":
        value_i = s[i] * Σ_k weights[i,k] * SG(dirs[i]; μ[i,k], κ[i,k])
        Returns [N,1]

    mode="broadcast":
        value_{n,c} = s[c] * Σ_k weights[k,c] * SG(dirs[n]; μ[k,c], κ[k,c])
        Returns [N,C]

    No parameter transforms are applied here (no softmax/sigmoid/softplus).
    """
    assert dirs.ndim == 2 and dirs.shape[-1] == 3, "dirs must be [N,3]"
    N = dirs.shape[0]

    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)

    if mode == "batch":
        # Shapes
        assert mu.ndim == 3 and mu.shape[0] == N and mu.shape[-1] == 3, "mu must be [N,K,3]"
        assert kappa.shape == weights.shape and kappa.shape[0] == N, "kappa/weights must be [N,K]"
        assert s.shape[0] == N and s.ndim == 1, "s must be [N]"
        K = mu.shape[1]

        # dot_i,k = μ[i,k] · dirs[i]
        if normalize:
            sg = sg_normalized_value(mu, kappa, dirs[:, None, :], mode="batch")
        else:
            dot = (mu * dirs[:, None, :]).sum(-1).clamp(-1.0, 1.0)
            sg = torch.exp(kappa * (dot - 1.0))

        mix = (weights * sg).sum(dim=1)                             # [N]
        vals = s * mix                                              # [N]
        # print(vals.min(), vals.max())
        return vals                                                 # [N]

    elif mode == "broadcast":
        # Shapes
        assert mu.ndim == 3 and mu.shape[-1] == 3, "mu must be [K2,C,3]"
        K2, C = mu.shape[0], mu.shape[1]
        assert kappa.shape == (K2, C) and weights.shape == (K2, C), "kappa/weights must be [K2,C]"
        assert s.shape == (C,), "s must be [C]"

        # dot_{k,c,n} = μ[k,c] · dirs[n]
        # einsum: (kcm, nm) -> kcn
        if normalize:
            sg = sg_normalized_value(mu, kappa, dirs, mode="broadcast")
        else:
            dot = torch.einsum('kcm,nm->kcn', mu, dirs)                 # [K2,C,N]
            sg  = torch.exp(kappa.unsqueeze(-1) * (dot - 1.0))          # [K2,C,N]
        mix = (weights.unsqueeze(-1) * sg).sum(dim=0)               # [C,N]
        vals = (s.unsqueeze(-1) * mix).transpose(0, 1)              # [N,C]

        return vals
# ---------- Stable SG×SG integral with optional normalization ----------
def sg_pairwise_integral_stable(mu1, k1, mu2, k2,
                                *, normalize1=False, normalize2=False,
                                return_log=False, eps=1e-12):
    dot = (mu1 * mu2).sum(-1).clamp(-1.0, 1.0)
    tau2 = k1*k1 + k2*k2 + 2.0*k1*k2*dot
    tau  = torch.sqrt(torch.clamp(tau2, min=0.0))
    S    = k1 + k2

    small = (tau <= 1e-6)
    logI = torch.empty_like(tau)

    if small.any():
        t = tau[small]; Ss = S[small]
        logI[small] = math.log(4*math.pi) - Ss + torch.log1p((t*t)/6.0)
    if (~small).any():
        t = tau[~small]; Sb = S[~small]
        Sm = Sb - t
        logI[~small] = (math.log(2*math.pi) - Sm
                        + torch.log(-torch.expm1(-2.0*t))
                        - torch.log(t + eps))

    if normalize1:
        logI = logI - sg_flux_logI(k1)
    if normalize2:
        logI = logI - sg_flux_logI(k2)

    return logI if return_log else torch.exp(torch.clamp(logI, max=80.0))

# ---------- Example: batched visibility×env accumulation ----------
def sg_visibility_env_integral_stable(
    mu_v, k_v, w_v, s_v,          # [N,K,3], [N,K], [N,K], [N]
    mu_e, k_e, w_e, s_e,          # [K2,C,3], [K2,C], [K2,C], [C]
    *,
    normalized_env: bool = True,
    normalized_vis: bool = False,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    I[n,c] = ∫ V_n(ω) E_c(ω) dω, with SG mixtures and stable kernel.
    If normalized_env/vis=True, each SG in that term is divided by its flux I(k).

    Returns [N, C].
    """
    N, K, _  = mu_v.shape
    K2, C, _ = mu_e.shape

    mu1 = mu_v[:, :, None, None, :]        # [N,K,1,1,3]
    k1  = k_v[:, :, None, None]            # [N,K,1,1]
    w1  = w_v[:, :, None, None]            # [N,K,1,1]

    mu2 = mu_e[None, None, :, :, :]        # [1,1,K2,C,3]
    k2  = k_e[None, None, :, :]            # [1,1, K2,C]
    w2  = w_e[None, None, :, :]            # [1,1, K2,C]

    Ipair = sg_pairwise_integral_stable(
        mu1, k1, mu2, k2,
        normalize1=normalized_vis,  # both or either → divide both; see note below
        normalize2=normalized_env,
        return_log=False, eps=eps
    )
    # NOTE: If you want to normalize only env or only visibility,
    # replace the boolean above with:
    #   normalized=False
    # and then multiply Ipair by exp(-(logI1 if normalized_vis else 0) - (logI2 if normalized_env else 0))
    # using sg_flux_logI(k1), sg_flux_logI(k2).
 
    contrib = w1 * w2 * Ipair            # [N,K,K2,C]
    summed  = contrib.sum(dim=2).sum(dim=1)  # [N,C]
    return (s_v[:, None] * s_e[None, :]) * summed

# def sg_visibility_env_integral(
#     mu_v: torch.Tensor,       # [N, K, 3]
#     kappa_v: torch.Tensor,    # [N, K]
#     weights_v: torch.Tensor,  # [N, K]
#     s_v: torch.Tensor,        # [N]

#     mu_e: torch.Tensor,       # [K2, C, 3]
#     kappa_e: torch.Tensor,    # [K2, C]
#     weights_e: torch.Tensor,  # [K2, C]
#     s_e: torch.Tensor,        # [C]

#     eps: float = 1e-4,
# ) -> torch.Tensor:
#     """
#     Returns: integrals I of shape [N, C] where
#       I[n,c] = s_v[n] * s_e[c] * sum_k sum_j weights_v[n,k]*weights_e[j,c] * Integral(SG_v(n,k) * SG_e(j,c))
#     """
#     # Shapes
#     N, K, _ = mu_v.shape
#     K2, C, _ = mu_e.shape

#     # cosθ_{n,k,j,c} = μ_v · μ_e
#     # -> [N, K, K2, C]
#     cos_theta = torch.einsum('nkd,jcd->nkjc', mu_v, mu_e)

#     # Broadcast kappa
#     kv = kappa_v[:, :, None, None]         # [N, K, 1, 1]
#     ke = kappa_e[None, None, :, :]         # [1, 1, K2, C]

#     # τ = sqrt(kv^2 + ke^2 + 2 kv ke cosθ)
#     tau2 = kv**2 + ke**2 + 2.0 * kv * ke * cos_theta
#     # Numerical guard: tau >= 0
#     tau = torch.clamp(torch.sqrt(torch.clamp(tau2, min=0.0)) , max=5.)

#     # stable sinhc(x) = sinh(x)/x with limit 1 at x→0
#     small = tau.abs() < 1e-6
#     sinhc = torch.empty_like(tau)
#     # if small.any():
#     #     # series: 1 + x^2/6 for small x
#     #     x = tau[small]
#     #     sinhc[small] = 1.0 + (x **2) / 6.0
#     #     print( (1.0 + (x **2) / 6.0).min(), (1.0 + (x **2) / 6.0).max())
#     # if (~small).any():
#     #     x = tau[~small]
#     #     sinhc[~small] = torch.sinh(x) / (x)
#     #     print( (torch.sinh(x) / x ).min(), (torch.sinh(x) / x).max())

#     sinhc[small] = 1.0 + (tau[small] **2) / 6.0
#     sinhc[~small] = torch.sinh(tau[~small]) / (tau[~small])

#     # Pairwise integral factor (without per-lobe amplitudes/scales)
#     # 4π e^{-(kv+ke)} sinhc(tau)
#     pair = (4.0 * math.pi) * torch.exp(-(kv + ke)) * sinhc  # [N,K,K2,C]

#     # Apply mixture weights (per visibility/env lobe)
#     wv = weights_v[:, :, None, None]   # [N,K,1,1]
#     we = weights_e[None, None, :, :]   # [1,1,K2,C]
#     contrib = wv * we * pair           # [N,K,K2,C]

#     # Sum over lobes
#     summed = contrib.sum(dim=2).sum(dim=1)  # [N, C]

#     # Apply global scales s_v (per N) and s_e (per channel)
#     result = (s_v[:, None] * s_e[None, :]) * summed  # [N, C]

#     return result

# def sh_per_band_params_to_coeffs_batched(
#     Lmax: int,
#     alpha: torch.Tensor,        # [N, C]   unconstrained DC params
#     band_logits: torch.Tensor,  # [N, C, Lmax]  unconstrained logits per band (l=1..Lmax)
#     band_dirs: list,            # list of length Lmax; band_dirs[l-1]: [N, C, 2*l+1] unconstrained
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     """
#     Batched per-band bounded real SH parameterization.

#     Ensures (per channel) the resulting function f(ω) ∈ [0, 1] for all ω by enforcing
#         Σ_{l=1..Lmax} ||a_l|| * w_l ≤ min(c0, 1 - c0),
#     where c0 = sigmoid(alpha) is the DC level and w_l = sqrt((2l+1)/(4π)).

#     Args:
#       Lmax        : max SH degree (>= 0)
#       alpha       : [N, C] unconstrained DC parameters
#       band_logits : [N, C, Lmax] band allocation logits
#       band_dirs   : list length Lmax; band_dirs[l-1] is [N, C, 2*l+1] direction vectors
#       eps         : small constant for numerical stability

#     Returns:
#       coeffs : [N, M, C] real SH coefficients (flattened by l=0..Lmax, m=-l..l),
#                where M = (Lmax+1)^2.
#     """
#     assert Lmax >= 0, "Lmax must be >= 0"
#     N, C = alpha.shape
#     if Lmax == 0:
#         c0 = torch.sigmoid(alpha)                         # [N, C]
#         a00 = (math.sqrt(4*math.pi) * c0)[..., None]      # [N, C, 1]
#         return a00.transpose(-1, -2)                      # [N, 1, C]

#     # shape checks
#     assert band_logits.shape == (N, C, Lmax), "band_logits must be [N, C, Lmax]"
#     assert len(band_dirs) == Lmax, "band_dirs list must have length Lmax"
#     for l in range(1, Lmax+1):
#         assert band_dirs[l-1].shape == (N, C, 2*l+1), f"band_dirs[{l-1}] must be [N, C, {2*l+1}]"

#     # DC and budget B = min(c0, 1 - c0)
#     c0   = torch.sigmoid(alpha)                           # [N, C]
#     a00  = math.sqrt(4*math.pi) * c0                      # [N, C]
#     B    = 0.5 - (c0 - 0.5).abs()                         # [N, C]

#     # per-band weights w_l = sqrt((2l+1)/(4π)), l=1..Lmax
#     wl = torch.tensor([math.sqrt((2*l+1)/(4*math.pi)) for l in range(1, Lmax+1)],
#                       device=alpha.device, dtype=alpha.dtype)  # [Lmax]

#     # allocate budget across bands with softmax
#     tau = torch.softmax(band_logits, dim=-1) * B.unsqueeze(-1)  # [N, C, Lmax], sum_l tau = B

#     # per-band coefficient norms: ||a_l|| = tau_l / w_l
#     a_norm = tau / (wl.view(1, 1, Lmax) + eps)            # [N, C, Lmax]

#     # per-band directions: normalize and scale
#     per_band_coeffs = []
#     for l in range(1, Lmax+1):
#         z_l = band_dirs[l-1]                              # [N, C, 2*l+1]
#         z_norm = z_l.norm(dim=-1, keepdim=True).clamp_min(eps)
#         hat = z_l / z_norm                                # unit
#         al = hat * a_norm[..., l-1].unsqueeze(-1)         # [N, C, 2*l+1]
#         per_band_coeffs.append(al)

#     # pack coefficients in (l, m=-l..l) order; start with DC
#     # We'll assemble as [N, C, M] then transpose to [N, M, C].
#     parts = [a00.unsqueeze(-1)]                           # [N, C, 1]
#     parts.extend(per_band_coeffs)                         # l=1..Lmax
#     coeffs_NCM = torch.cat(parts, dim=-1)                 # [N, C, M], M=(Lmax+1)^2
#     coeffs = coeffs_NCM.transpose(-1, -2).contiguous()    # [N, M, C]
#     return coeffs

@torch.no_grad()
def render_equirect_from_sg(
    mu,                 # (N, N_coeffs, 3) or (1, N_coeffs, 3)
    kappa,
    weights,
    s,
    H: int = 512,
    W: int = 1024,
    phi_offset: float = 0.0,  # yaw (radians), positive rotates to the left
    v_flip: bool = False,     # flip vertically
    convention: str = "y-up", # "y-up" (x=sinθcosφ, y=cosθ, z=sinθsinφ) or "z-up"
    chunk: int = 131072,      # to limit memory
):
    """
    Returns image (H, W, 3), float32 (no tonemapping/clamp).
    Angle convention (equirect):
      u in [0,1) -> φ in [-π, π)
      v in [0,1] -> θ in [0, π] (0 = north pole/top)
    """
    device = mu.device
    # Pixel centers: avoid half-pixel seam misalignment
    u = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    v = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    theta = vv * math.pi                      # [0, π]
    if v_flip:
        theta = math.pi - theta
    phi = (uu * (2.0 * math.pi) - math.pi) + phi_offset  # [-π, π) + offset

    # Unit directions from (θ, φ)
    if convention == "y-up":
        # y is up (north pole at +y)
        sin_t = torch.sin(theta)
        dirs = torch.stack([
            sin_t * torch.cos(phi),  # x
            torch.cos(theta),        # y
            sin_t * torch.sin(phi),  # z
        ], dim=-1)
    elif convention == "z-up":
        # z is up (north pole at +z), common in some vision stacks
        sin_t = torch.sin(theta)
        dirs = torch.stack([
            sin_t * torch.cos(phi),  # x
            sin_t * torch.sin(phi),  # y
            torch.cos(theta),        # z
        ], dim=-1)
    else:
        raise ValueError("convention must be 'y-up' or 'z-up'")

    H_, W_, _ = dirs.shape
    N = H_ * W_
    dirs_flat = dirs.reshape(N, 3)

    # Chunked evaluation to control memory
    out = torch.empty((N, 3), device=device, dtype=torch.float32)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        out[start:end] = sg_mixture_eval(mode="broadcast", mu=mu, kappa=kappa, weights=weights, s=s, dirs=dirs_flat[start:end], normalize=True)

    return torch.clamp(out.reshape(H_, W_, 3), 0.0, 1.0)

@torch.no_grad()
def render_equirect_from_sh(
    coeffs,                 # (N, N_coeffs, 3) or (1, N_coeffs, 3)
    sh_degree,
    H: int = 512,
    W: int = 1024,
    phi_offset: float = 0.0,  # yaw (radians), positive rotates to the left
    v_flip: bool = False,     # flip vertically
    convention: str = "y-up", # "y-up" (x=sinθcosφ, y=cosθ, z=sinθsinφ) or "z-up"
    chunk: int = 131072,      # to limit memory
):
    assert (sh_degree + 1)**2 <= coeffs.shape[1], "sh_degree must be less than or equal to the number of coefficients"
    """
    Returns image (H, W, 3), float32 (no tonemapping/clamp).
    Angle convention (equirect):
      u in [0,1) -> φ in [-π, π)
      v in [0,1] -> θ in [0, π] (0 = north pole/top)
    """
    device = coeffs.device
    # Pixel centers: avoid half-pixel seam misalignment
    u = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    v = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    theta = vv * math.pi                      # [0, π]
    if v_flip:
        theta = math.pi - theta
    phi = (uu * (2.0 * math.pi) - math.pi) + phi_offset  # [-π, π) + offset

    # Unit directions from (θ, φ)
    if convention == "y-up":
        # y is up (north pole at +y)
        sin_t = torch.sin(theta)
        dirs = torch.stack([
            sin_t * torch.cos(phi),  # x
            torch.cos(theta),        # y
            sin_t * torch.sin(phi),  # z
        ], dim=-1)
    elif convention == "z-up":
        # z is up (north pole at +z), common in some vision stacks
        sin_t = torch.sin(theta)
        dirs = torch.stack([
            sin_t * torch.cos(phi),  # x
            sin_t * torch.sin(phi),  # y
            torch.cos(theta),        # z
        ], dim=-1)
    else:
        raise ValueError("convention must be 'y-up' or 'z-up'")

    H_, W_, _ = dirs.shape
    N = H_ * W_
    dirs_flat = dirs.reshape(N, 3)

    # Match coeffs to N
    if coeffs.ndim != 3 or coeffs.shape[-1] != 3:
        raise ValueError("coeffs must have shape (N, N_coeffs, 3) or (1, N_coeffs, 3)")
    if coeffs.shape[0] == 1:
        coeffs_expand = coeffs.expand(N, -1, -1).contiguous()
    elif coeffs.shape[0] == N:
        coeffs_expand = coeffs
    else:
        raise ValueError(f"coeffs batch {coeffs.shape[0]} must be 1 or N={N}")

    # Chunked evaluation to control memory
    out = torch.empty((N, 3), device=device, dtype=torch.float32)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        out[start:end] = spherical_harmonics(sh_degree, dirs_flat[start:end], coeffs_expand[start:end])

    return torch.clamp(out.reshape(H_, W_, 3), 0.0, 1.0)

def logistic_weighting(
    means2d: Tensor,  # [N, 2] where N is the number of gaussians in the frustum
    depths: Tensor,  # [N] where N is the number of gaussians in the frustum
    depth_image: Tensor,  # [H, W]
    variance_image: Tensor,  # [H, W]
    variance_factor: Optional[float] = 1.0,
    cutoff: Optional[float] = 0.3,
    hard_cutoff: Optional[bool] = False,
):
    # NOTE: Only evaluates the logistic distribution for the gaussians in the frustum!
    # Any logic that indexes into the total number of gaussians in the scene should be done outside of this function!

    H, W = depth_image.shape
    N, _ = means2d.shape

    assert N == depths.shape[0], "Number of means and depths must match"
    assert (
        depth_image.shape == variance_image.shape
    ), "Depth and variance images must have the same shape"

    # Check inputs for NaN values
    if torch.isnan(means2d).any():
        raise ValueError("NaN values detected in means2d")
    if torch.isnan(depths).any():
        raise ValueError("NaN values detected in depths")
    if torch.isnan(depth_image).any():
        raise ValueError("NaN values detected in depth_image")
    if torch.isnan(variance_image).any():
        raise ValueError("NaN values detected in variance_image")
    if variance_factor is not None and math.isnan(variance_factor):
        raise ValueError("NaN value detected in variance_factor")
    if cutoff is not None and math.isnan(cutoff):
        raise ValueError("NaN value detected in cutoff")

    # NOTE: These means correspond to Gaussians that are in the frustum!
    pixel_x = means2d[:, 0].long().clamp(0, W - 1)
    pixel_y = means2d[:, 1].long().clamp(0, H - 1)
    projected_pixel_ids = pixel_y * W + pixel_x  # shape [nnz]

    depth_image_flattened = depth_image.reshape(-1)
    variance_image_flattened = variance_image.reshape(-1)

    # Add minimum variance threshold to prevent division by very small numbers
    variance = torch.clamp(variance_image_flattened, min=1e-8)
    # s = torch.sqrt(variance_factor / (math.pi) ** 2 * variance)  # n_pixels
    s = variance_factor * torch.sqrt(3 * variance) / math.pi

    sigmoid_argument = (depths - depth_image_flattened[projected_pixel_ids]) / s[
        projected_pixel_ids
    ]

    sigmoid_weights = 1.0 - torch.sigmoid(sigmoid_argument)

    # Use learnable cutoff parameter for differentiable masking
    if hard_cutoff:
        # During evaluation, use the provided cutoff for consistency
        lit_mask = sigmoid_weights > cutoff
        sigmoid_weights = torch.clamp(sigmoid_weights + lit_mask, 0.0, 1.0)
    else:
        smooth_mask = torch.sigmoid((sigmoid_weights - cutoff) * 10.0)  # 10.0 controls sharpness
        # Apply the smooth mask to create differentiable lighting weights
        sigmoid_weights = smooth_mask + (1.0 - smooth_mask) * sigmoid_weights

    return sigmoid_weights


def chebyshev_weighting(
    means2d: Tensor,  # [N, 2] where N is the number of gaussians in the frustum
    depths: Tensor,  # [N] where N is the number of gaussians in the frustum
    depth_image: Tensor,  # [H, W]
    variance_image: Tensor,  # [H, W]
):
    H, W = depth_image.shape
    N, _ = means2d.shape

    assert N == depths.shape[0], "Number of means and depths must match"
    assert (
        depth_image.shape == variance_image.shape
    ), "Depth and variance images must have the same shape"

    # NOTE: These means correspond to Gaussians that are in the frustum!
    pixel_x = means2d[:, 0].long().clamp(0, W - 1)
    pixel_y = means2d[:, 1].long().clamp(0, H - 1)
    projected_pixel_ids = pixel_y * W + pixel_x  # shape [nnz]

    depth_image_flattened = depth_image.reshape(-1)
    variance_image_flattened = variance_image.reshape(-1)

    # Add minimum variance threshold to prevent division by very small numbers
    variance = torch.clamp(variance_image_flattened, min=1e-6)
    variance_per_gaussian = variance[projected_pixel_ids]

    depth_diff = depths - depth_image_flattened[projected_pixel_ids]
    weights = variance_per_gaussian / (variance_per_gaussian + depth_diff**2)
    weights[depth_diff < 0] = 1.0

    return weights


def calculate_relighting_weights(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]       # NOTE: IMPORTANT! THESE SCALES MUST ALREADY BE POSITIVE
    opacities: Tensor,  # [N]       # NOTE: IMPORTANT! THESE OPACITIES MUST ALREADY BE [0, 1]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    variance_factor: float,
    intensity: List[float],
    ambient: Optional[float] = None,
    background_ambient: Optional[float] = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    tile_size: int = 16,
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    fix_variance: bool = False,
    use_2dgs: bool = False,
    distloss: bool = False,  # 2DGS only
) -> Tuple[Tensor, Tensor, Dict]:
    """Compute the relighting weights for a given light source for the scene."""

    N, _ = means.shape  # Number of gaussians in the scene
    assert N == opacities.shape[0], "Number of means and opacities must match"
    assert N == scales.shape[0], "Number of means and scales must match"
    assert N == quats.shape[0], "Number of means and quaternions must match"

    if not use_2dgs:
        moments, alphas, meta = moment_rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            viewmats,  # [C, 4, 4]
            Ks,  # [C, 3, 3]
            width,
            height,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            eps2d=eps2d,
            packed=True,
            tile_size=tile_size,
            sparse_grad=sparse_grad,
            absgrad=absgrad,
            rasterize_mode=rasterize_mode,
            channel_chunk=channel_chunk,
            distributed=distributed,
            camera_model=camera_model,
        )
    else:
        (moments, alphas, normals, normals_from_depth, distort, median, meta) = (
            moment_rasterization_2dgs(
                means,
                quats,
                scales,
                opacities,
                viewmats,
                Ks,
                width,
                height,
                near_plane=near_plane,
                far_plane=far_plane,
                radius_clip=radius_clip,
                eps2d=eps2d,
                packed=True,
                tile_size=tile_size,
                sparse_grad=sparse_grad,
                absgrad=absgrad,
                distloss=distloss,
            )
        )

    assert not torch.isnan(moments[..., 1]).any(), "Depth squared is nan"
    assert not torch.isnan(moments[..., 0]).any(), "Depth is nan"
    assert not torch.isinf(moments[..., 1]).any(), "Depth squared is inf"
    assert not torch.isinf(moments[..., 0]).any(), "Depth is inf"

    depth_image = moments[..., 0].squeeze()

    if fix_variance:
        # TODO: Implement fix_variance
        variance_image = torch.ones_like(depth_image)
    else:
        depth_sqr_image = moments[..., 1].squeeze()
        variance_image = depth_sqr_image - depth_image**2

    depths = meta["depths"]  # Depths of each gaussian in frustum
    means2d = meta["means2d"]  # 2D means of each gaussian in frustum
    gaussian_ids = meta["gaussian_ids"]  # Indices of the gaussians in the frustum
    conics = meta["conics"]

    assert (meta["radii"] >= 0.0).all(), "Radii must be non-negative"

    # This represents the fraction of light that is received by each gaussian in the frustum
    # weights = logistic_weighting(
    # weights = chebyshev_weighting(
    #     means2d,  # [N, 2] where N is the number of gaussians in the frustum
    #     depths,  # [N] where N is the number of gaussians in the frustum
    #     depth_image,  # [H, W]
    #     variance_image,  # [H, W]
    # )

    # Smoothing across whole Gaussian
    # Sample points in axis directions of the Gaussian
    sample_points = conics_to_semi_major_axis_points(means2d, conics)
    sample_points = sample_points.reshape(-1, 2)
    sample_depths = torch.repeat_interleave(depths, 5)

    weights = chebyshev_weighting(
        sample_points,  # [N, 2] where N is the number of gaussians in the frustum
        sample_depths,  # [N] where N is the number of gaussians in the frustum
        depth_image,  # [H, W]
        variance_image,  # [H, W]
    )
    weights = weights.reshape(means2d.shape[0], 5)
    weights = weights.min(dim=1).values

    assert not torch.isnan(weights).any(), "Weights are nan"
    assert not torch.isinf(weights).any(), "Weights are inf"

    irradiance = torch.ones_like(means)
    irradiance_fraction = torch.ones_like(opacities)

    # The total intensity of the light that is received by each gaussian in the frustum is the product of the intensity of the light source and the fraction of light that is received by the gaussian
    # Additionally, if 2DGS, then a BRDF function is applied.
    if use_2dgs:
        cam2worlds = torch.inverse(viewmats)
        gaussian_normals = meta["normals"]
        gaussian_normals = gaussian_normals / torch.norm(gaussian_normals, dim=-1, keepdim=True)
        incident_dir = cam2worlds[meta["camera_ids"], :3, 3] - means[gaussian_ids, :]
        incident_dir = incident_dir / torch.norm(incident_dir, dim=-1, keepdim=True)
        cosine_angle = torch.sum(gaussian_normals * incident_dir, dim=-1)
        weights = weights * torch.relu(cosine_angle)

    if ambient is not None:
        weights = weights + ambient

    irradiance[gaussian_ids] = torch.stack(
        [weights * intensity[0], weights * intensity[1], weights * intensity[2]], dim=-1
    )
    irradiance_fraction[gaussian_ids] = weights

    # if ambient is not None:
    #     irradiance += ambient  # ambient is in intensity space
    # if background_ambient is not None:
    #     irradiance[~gaussian_ids] += background_ambient
    # irradiance[~gaussian_ids] += intensity[0]

    return irradiance, irradiance_fraction


# Renders the accumulated or expected depth (first moment) and the accumulated
# or expected depth squared (second moment). Will add higher moments as necessary.
def moment_rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    covars: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    # TODO: Rewrite documentation to reflect the moment rendering

    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        **Multi-GPU Distributed Rasterization**: This function can be used in a multi-GPU
        distributed scenario by setting `distributed` to True. When `distributed` is True,
        a subset of total Gaussians could be passed into this function in each rank, and
        the function will collaboratively render a set of images using Gaussians from all ranks. Note
        to achieve balanced computation, it is recommended (not enforced) to have similar number of
        Gaussians in each rank. But we do enforce that the number of cameras to be rendered
        in each rank is the same. The function will return the rendered images
        corresponds to the input cameras in each rank, and allows for gradients to flow back to the
        Gaussians living in other ranks. For the details, please refer to the paper
        `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`_.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of images in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Support N-D Features**: If `sh_degree` is None,
        the `colors` is expected to be with shape [N, D] or [C, N, D], in which D is the channel of
        the features to be rendered. The computation is slow when D > 32 at the moment.
        If `sh_degree` is set, the `colors` is expected to be the SH coefficients with
        shape [N, K, 3] or [C, N, K, 3], where K is the number of SH bases. In this case, it is expected
        that :math:`(\\textit{sh_degree} + 1) ^ 2 \\leq K`, where `sh_degree` controls the
        activated bases in the SH coefficients.

    .. note::
        **Depth Rendering**: This function supports colors or/and depths via `render_mode`.
        The supported modes are "RGB", "D", "ED", "RGB+D", and "RGB+ED". "RGB" renders the
        colored image that respects the `colors` argument. "D" renders the accumulated z-depth
        :math:`\\sum_i w_i z_i`. "ED" renders the expected z-depth
        :math:`\\frac{\\sum_i w_i z_i}{\\sum_i w_i}`. "RGB+D" and "RGB+ED" render both
        the colored image and the depth, in which the depth is the last channel of the output.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [C, N, ...], which is faster but might consume more memory.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in pixel unit) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which is rarely happened in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered image more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        ortho: Whether to use orthographic projection. In such case fx and fy become the scaling
            factors to convert projected coordinates into pixel space and cx, cy become offsets.
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [N, 3, 3], Default is None.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, meta = rasterization(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])

    """
    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert quats.shape == (N, 4), quats.shape
        assert scales.shape == (N, 3), scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    # If in distributed mode, we distribute the projection computation over Gaussians
    # and the rasterize computation over cameras. So first we gather the cameras
    # from all ranks for projection.
    if distributed:
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Gather the number of Gaussians in each rank.
        N_world = all_gather_int32(world_size, N, device=device)

        # Enforce that the number of cameras is the same across all ranks.
        C_world = [C] * world_size
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])

        # Silently change C from local #Cameras to global #Cameras.
        C = len(viewmats)

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
    )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    # TODO: Get rid of colors variable
    if distributed:
        if packed:
            # count how many elements need to be sent to each rank
            cnts = torch.bincount(camera_ids, minlength=C)  # all cameras
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities],
                cnts,
                output_splits=collected_splits,
            )

            # before sending the data, we should turn the camera_ids from global to local.
            # i.e. the camera_ids produced by the projection stage are over all cameras world-wide,
            # so we need to turn them into camera_ids that are local to each rank.
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # and turn gaussian ids from local to global.
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # all to all communication across all ranks.
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

        else:
            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)

    # Rasterize to pixels
    moments = torch.stack((depths, depths**2), dim=-1)
    if backgrounds is not None:
        backgrounds = torch.zeros(C, 2, device=backgrounds.device)

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=C,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_images": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if moments.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (moments.shape[-1] + channel_chunk - 1) // channel_chunk
        render_moments, render_alphas = [], []
        for i in range(n_chunks):
            moments_chunk = moments[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_moments_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                moments_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_moments.append(render_moments_)
            render_alphas.append(render_alphas_)
        render_moments = torch.cat(render_moments, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_moments, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            moments,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )

    # We use expected depth
    render_moments = render_moments / render_alphas.clamp(min=1e-10)

    return render_moments, render_alphas, meta

# Regular rasterization, but allows to simultaneously render additional channels
# TODO: Do we want these additional channels to be view dependent?
def lumen_rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    visibility: List[Tensor],  # [N, K2, 1]
    env_map: List[Tensor],  # [1, K2, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    sh_degree_lumen: int = 3,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    covars: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        **Multi-GPU Distributed Rasterization**: This function can be used in a multi-GPU
        distributed scenario by setting `distributed` to True. When `distributed` is True,
        a subset of total Gaussians could be passed into this function in each rank, and
        the function will collaboratively render a set of images using Gaussians from all ranks. Note
        to achieve balanced computation, it is recommended (not enforced) to have similar number of
        Gaussians in each rank. But we do enforce that the number of cameras to be rendered
        in each rank is the same. The function will return the rendered images
        corresponds to the input cameras in each rank, and allows for gradients to flow back to the
        Gaussians living in other ranks. For the details, please refer to the paper
        `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`_.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of images in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Support N-D Features**: If `sh_degree` is None,
        the `colors` is expected to be with shape [N, D] or [C, N, D], in which D is the channel of
        the features to be rendered. The computation is slow when D > 32 at the moment.
        If `sh_degree` is set, the `colors` is expected to be the SH coefficients with
        shape [N, K, 3] or [C, N, K, 3], where K is the number of SH bases. In this case, it is expected
        that :math:`(\\textit{sh_degree} + 1) ^ 2 \\leq K`, where `sh_degree` controls the
        activated bases in the SH coefficients.

    .. note::
        **Depth Rendering**: This function supports colors or/and depths via `render_mode`.
        The supported modes are "RGB", "D", "ED", "RGB+D", and "RGB+ED". "RGB" renders the
        colored image that respects the `colors` argument. "D" renders the accumulated z-depth
        :math:`\\sum_i w_i z_i`. "ED" renders the expected z-depth
        :math:`\\frac{\\sum_i w_i z_i}{\\sum_i w_i}`. "RGB+D" and "RGB+ED" render both
        the colored image and the depth, in which the depth is the last channel of the output.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [C, N, ...], which is faster but might consume more memory.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in pixel unit) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which is rarely happened in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered image more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        ortho: Whether to use orthographic projection. In such case fx and fy become the scaling
            factors to convert projected coordinates into pixel space and cx, cy become offsets.
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [N, 3, 3], Default is None.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, meta = rasterization(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])
    """
    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert quats.shape == (N, 4), quats.shape
        assert scales.shape == (N, 3), scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == (N,), opacities.shape
    # assert visibility.shape == (N, K2, 1), visibility.shape
    # assert env_map.shape == (1, K2, 3), env_map.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
        if distributed:
            assert colors.dim() == 2, "Distributed mode only supports per-Gaussian colors."
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or (
            colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
        if distributed:
            assert colors.dim() == 3, "Distributed mode only supports per-Gaussian colors."

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    # If in distributed mode, we distribute the projection computation over Gaussians
    # and the rasterize computation over cameras. So first we gather the cameras
    # from all ranks for projection.
    if distributed:
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Gather the number of Gaussians in each rank.
        N_world = all_gather_int32(world_size, N, device=device)

        # Enforce that the number of cameras is the same across all ranks.
        C_world = [C] * world_size
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])

        # Silently change C from local #Cameras to global #Cameras.
        C = len(viewmats)

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
    )

    # TODO: Segment out pixels where there are occluders behind the pixel 

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (   batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    visibility_mu = visibility[0]
    visibility_kappa = visibility[1]
    visibility_logits = visibility[2]
    visibility_scale = visibility[3]

    env_map_mu = env_map[0]
    env_map_kappa = env_map[1]
    env_map_logits = env_map[2]
    env_map_scale = env_map[3]

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [N, D] or [C, N, D]
        if packed:
            if colors.dim() == 2:
                # Turn [N, D] into [nnz, D]
                colors = colors[gaussian_ids]
            else:
                # Turn [C, N, D] into [nnz, D]
                colors = colors[camera_ids, gaussian_ids]
        else:
            if colors.dim() == 2:
                # Turn [N, D] into [C, N, D]
                colors = colors.expand(C, -1, -1)
            else:
                # colors is already [C, N, D]
                pass

    else:
        # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
        camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
        if packed:
            dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz, 3]
            masks = (radii > 0).any(-1)  # [nnz]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [nnz, 3]
                shs = colors[gaussian_ids, :, :]  # [nnz, K, 3]
            else:
                # Turn [C, N, K, 3] into [nnz, 3]
                shs = colors[camera_ids, gaussian_ids, :, :]  # [nnz, K, 3]
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [nnz, 3]

            visibilities_mu = visibility_mu[gaussian_ids, :, :]  # [nnz, K2, 1]
            visibilities_kappa = visibility_kappa[gaussian_ids, :]  # [nnz, K2]
            visibilities_logits = visibility_logits[gaussian_ids, :]  # [nnz, K2]
            visibilities_scale = visibility_scale[gaussian_ids]  # [nnz]

            visibilities = sg_mixture_eval(mode="batch", mu=visibilities_mu, kappa=visibilities_kappa, weights=visibilities_logits, s=visibilities_scale, dirs=dirs, normalize=False)

            # visibility_shs = visibility[gaussian_ids, :, :]  # [nnz, K2, 1]
            # visibilities = spherical_harmonics(sh_degree_lumen, dirs, visibility_shs.repeat(1, 1, 3), masks=masks)  # [nnz, K2, 1]

        else:
            dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
            masks = (radii > 0).any(-1)  # [C, N]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [C, N, K, 3]
                shs = colors.expand(C, -1, -1, -1)  # [C, N, K, 3]
            else:
                # colors is already [C, N, K, 3]
                shs = colors
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]

            # NOTE: THIS MIGHT NOT WORK!
            visibilities_mu = visibility_mu[None, :, :, :]  # [C, N, K2, 1]
            visibilities_kappa = visibility_kappa[None, :, :]  # [C, N, K2]
            visibilities_logits = visibility_logits[None, :, :]  # [C, N, K2]
            visibilities_scale = visibility_scale[None, :]  # [C]

            visibilities = sg_mixture_eval(mode="batch", mu=visibilities_mu, kappa=visibilities_kappa, weights=visibilities_logits, s=visibilities_scale, dirs=dirs, normalize=False)

            # visibility_shs = visibility[None, :, :, :]  # [C, N, K2, 1]
            # visibilities = spherical_harmonics(sh_degree_lumen, dirs, visibility_shs.repeat(1, 1, 3), masks=masks)  # [C, N, K2, 1]

        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)
        # visibilities = torch.clamp_min(visibilities + 0.5, 0.0)

        visibilities = torch.clamp_min(visibilities, 0.0)
        # print(visibilities.min(), visibilities.max())
        # print(visibilities_mu)
        # print(visibilities_kappa)
        # print(visibilities_logits)
        # print(visibilities_scale)
        # raise

    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    if distributed:
        if packed:
            # count how many elements need to be sent to each rank
            cnts = torch.bincount(camera_ids, minlength=C)  # all cameras
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities, colors, visibilities) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities, colors, visibilities],
                cnts,
                output_splits=collected_splits,
            )

            # before sending the data, we should turn the camera_ids from global to local.
            # i.e. the camera_ids produced by the projection stage are over all cameras world-wide,
            # so we need to turn them into camera_ids that are local to each rank.
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # and turn gaussian ids from local to global.
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # all to all communication across all ranks.
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

        else:
            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities, colors, visibilities) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                    colors.flatten(0, 1),
                    visibilities.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)
            colors = reshape_view(C, colors, N_world)
            visibilities = reshape_view(C, visibilities, N_world)

    # Rasterize to pixels
    # if render_mode in ["RGB+D", "RGB+ED", "RGB"]:
        # light_transport = torch.relu(torch.sum(visibility[gaussian_ids, :, :] * env_map, dim=-2))  # [N, 3]

    light_transport = sg_visibility_env_integral_stable(
        visibilities_mu,       # [N, K, 3]
        visibilities_kappa,    # [N, K]
        visibilities_logits,  # [N, K]
        visibilities_scale,        # [N]
        env_map_mu,       # [K2, C, 3]
        env_map_kappa,    # [K2, C]
        env_map_logits,  # [K2, C]
        env_map_scale,      
        normalized_vis = False,
        normalized_env = True,
    )

    colors_lumen = colors * light_transport

    colors = torch.cat((colors, colors_lumen, depths[..., None], depths[..., None]**2), dim=-1)

    if backgrounds is not None:
        backgrounds = torch.cat(
            [
                backgrounds,
                backgrounds,
                torch.zeros(C, 2, device=backgrounds.device),
            ],
            dim=-1,
        )

    # elif render_mode in ["D", "ED"]:
    #     colors = torch.cat((depths[..., None], depths[..., None]**2), dim=-1)
    #     if backgrounds is not None:
    #         backgrounds = torch.zeros(C, 2, device=backgrounds.device)

    # else:
    #     raise ValueError(f"Unsupported render mode: {render_mode}")

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=C,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
            "visibilities": visibilities,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )
    # if render_mode in ["ED", "RGB+ED"]:
    # normalize the accumulated depth to get the expected depth
    render_colors = torch.cat(
        [
            render_colors[..., :-2],
            render_colors[..., -2:] / render_alphas.clamp(min=1e-10),
        ],
        dim=-1,
    )

    return render_colors, render_alphas, meta

# Regular rasterization, but allows to simultaneously render additional channels
# TODO: Do we want these additional channels to be view dependent?
def augmented_rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    additional_channels: Optional[Tensor] = None,  # [(C,) N, D2] or [(C,) N, K, D2]
    color_weights: Optional[Tensor] = None,  # [(C,) N, 3],
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    covars: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        **Multi-GPU Distributed Rasterization**: This function can be used in a multi-GPU
        distributed scenario by setting `distributed` to True. When `distributed` is True,
        a subset of total Gaussians could be passed into this function in each rank, and
        the function will collaboratively render a set of images using Gaussians from all ranks. Note
        to achieve balanced computation, it is recommended (not enforced) to have similar number of
        Gaussians in each rank. But we do enforce that the number of cameras to be rendered
        in each rank is the same. The function will return the rendered images
        corresponds to the input cameras in each rank, and allows for gradients to flow back to the
        Gaussians living in other ranks. For the details, please refer to the paper
        `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`_.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of images in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Support N-D Features**: If `sh_degree` is None,
        the `colors` is expected to be with shape [N, D] or [C, N, D], in which D is the channel of
        the features to be rendered. The computation is slow when D > 32 at the moment.
        If `sh_degree` is set, the `colors` is expected to be the SH coefficients with
        shape [N, K, 3] or [C, N, K, 3], where K is the number of SH bases. In this case, it is expected
        that :math:`(\\textit{sh_degree} + 1) ^ 2 \\leq K`, where `sh_degree` controls the
        activated bases in the SH coefficients.

    .. note::
        **Depth Rendering**: This function supports colors or/and depths via `render_mode`.
        The supported modes are "RGB", "D", "ED", "RGB+D", and "RGB+ED". "RGB" renders the
        colored image that respects the `colors` argument. "D" renders the accumulated z-depth
        :math:`\\sum_i w_i z_i`. "ED" renders the expected z-depth
        :math:`\\frac{\\sum_i w_i z_i}{\\sum_i w_i}`. "RGB+D" and "RGB+ED" render both
        the colored image and the depth, in which the depth is the last channel of the output.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [C, N, ...], which is faster but might consume more memory.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in pixel unit) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which is rarely happened in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered image more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        ortho: Whether to use orthographic projection. In such case fx and fy become the scaling
            factors to convert projected coordinates into pixel space and cx, cy become offsets.
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [N, 3, 3], Default is None.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, meta = rasterization(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])

    """
    # print('colors before', colors.isnan().any(), colors.isinf().any())
    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert quats.shape == (N, 4), quats.shape
        assert scales.shape == (N, 3), scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    if color_weights is not None:
        assert color_weights.shape == (N, 3), color_weights.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    # TODO: We may want to add more checks for the additional channels like with colors
    if additional_channels is not None:
        assert additional_channels.shape[0] == N, additional_channels.shape
        additional_channels = additional_channels[None].expand(C, -1, -1)

    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
        if distributed:
            assert colors.dim() == 2, "Distributed mode only supports per-Gaussian colors."
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or (
            colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
        if distributed:
            assert colors.dim() == 3, "Distributed mode only supports per-Gaussian colors."

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    # If in distributed mode, we distribute the projection computation over Gaussians
    # and the rasterize computation over cameras. So first we gather the cameras
    # from all ranks for projection.
    if distributed:
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Gather the number of Gaussians in each rank.
        N_world = all_gather_int32(world_size, N, device=device)

        # Enforce that the number of cameras is the same across all ranks.
        C_world = [C] * world_size
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])

        # Silently change C from local #Cameras to global #Cameras.
        C = len(viewmats)

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
    )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [N, D] or [C, N, D]
        if packed:
            if colors.dim() == 2:
                # Turn [N, D] into [nnz, D]
                colors = colors[gaussian_ids]
            else:
                # Turn [C, N, D] into [nnz, D]
                colors = colors[camera_ids, gaussian_ids]
        else:
            if colors.dim() == 2:
                # Turn [N, D] into [C, N, D]
                colors = colors.expand(C, -1, -1)
            else:
                # colors is already [C, N, D]
                pass

    else:
        # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
        camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
        if packed:
            dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz, 3]
            dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)
            masks = (radii > 0).any(-1)  # [nnz]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [nnz, 3]
                shs = colors[gaussian_ids, :, :]  # [nnz, K, 3]
            else:
                # Turn [C, N, K, 3] into [nnz, 3]
                shs = colors[camera_ids, gaussian_ids, :, :]  # [nnz, K, 3]
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [nnz, 3]
        else:
            dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
            dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-10)
            masks = (radii > 0).any(-1)  # [C, N]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [C, N, K, 3]
                shs = colors.expand(C, -1, -1, -1)  # [C, N, K, 3]
            else:
                # colors is already [C, N, K, 3]
                shs = colors
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)
    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    if distributed:
        if packed:
            # count how many elements need to be sent to each rank
            cnts = torch.bincount(camera_ids, minlength=C)  # all cameras
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities, colors],
                cnts,
                output_splits=collected_splits,
            )

            # before sending the data, we should turn the camera_ids from global to local.
            # i.e. the camera_ids produced by the projection stage are over all cameras world-wide,
            # so we need to turn them into camera_ids that are local to each rank.
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # and turn gaussian ids from local to global.
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # all to all communication across all ranks.
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

        else:
            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                    colors.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)
            colors = reshape_view(C, colors, N_world)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        if color_weights is not None:
            # The absolute color is the albedo (base color of Gaussian), the fraction of light reflected in each channel, times the intensity of the light incident on the Gaussian
            relit_colors = colors * color_weights
            colors = torch.cat((colors, relit_colors), dim=-1)

        if additional_channels is not None:
            colors = torch.cat((colors, additional_channels, depths[..., None]), dim=-1)
            background_channels_pad_size = additional_channels.shape[-1] + 1
        else:
            colors = torch.cat((colors, depths[..., None]), dim=-1)
            background_channels_pad_size = 1

        if backgrounds is not None and color_weights is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
        elif backgrounds is not None and color_weights is None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )

    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)

    else:  # RGB
        if color_weights is not None:
            # The absolute color is the albedo (base color of Gaussian), the fraction of light reflected in each channel, times the intensity of the light incident on the Gaussian
            relit_colors = colors * color_weights
            colors = torch.cat((colors, relit_colors), dim=-1)

        if additional_channels is not None:
            colors = torch.cat((colors, additional_channels), dim=-1)
            background_channels_pad_size = additional_channels.shape[-1]

        if (
            backgrounds is not None
            and color_weights is not None
            and additional_channels is not None
        ):
            backgrounds = torch.cat(
                [
                    backgrounds,
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
        elif backgrounds is not None and color_weights is not None and additional_channels is None:
            backgrounds = torch.cat([backgrounds, backgrounds], dim=-1)
        elif backgrounds is not None and color_weights is None and additional_channels is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=C,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    return render_colors, render_alphas, meta


# Renders the accumulated or expected depth (first moment) and the accumulated
# or expected variance (second moment) for 2DGS. Will add higher moments as necessary.
def moment_rasterization_2dgs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """Rasterize a set of 2D Gaussians (N) to a batch of image planes (C).

    This function supports a handful of features, similar to the :func:`rasterization` function.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad (Experimental): If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distloss: If true, use distortion regularization to get better geometry detail.
        depth_mode: render depth mode. Choose from expected depth and median depth.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **render_normals**: The rendered normals. [C, height, width, 3].

        **surf_normals**: surface normal from depth. [C, height, width, 3]

        **render_distort**: The rendered distortions. [C, height, width, 1].
        L1 version, different from L2 version in 2DGS paper.

        **render_median**: The rendered median depth. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (normals.shape, surf_normals.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 3])
        >>> print (distort.shape, median_depth.shape)
        torch.Size([1, 200, 300, 1]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'ray_transforms',
        'opacities', 'normals', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras', 'render_distort',
        'gradient_2dgs'])

    """

    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # Compute Ray-Splat intersection transformation.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        packed,
        sparse_grad,
    )

    if packed:
        (
            _,  # batch_ids
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = proj_results
        opacities = opacities[gaussian_ids]
    else:
        radii, means2d, depths, ray_transforms, normals = proj_results
        opacities = opacities.repeat(C, 1)
        camera_ids, gaussian_ids = None, None

    densify = torch.zeros_like(means2d, dtype=means.dtype, requires_grad=True, device="cuda")
    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=C,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # Rasterize to pixels
    colors = torch.stack((depths, depths**2), dim=-1)

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
        distloss=distloss,
    )
    render_normals_from_depth = None

    # normalize the accumulated depth to get the expected depth
    render_colors = render_colors / render_alphas.clamp(min=1e-10)

    if depth_mode == "expected":
        depth_for_normal = render_colors[..., 0:1]

    elif depth_mode == "median":
        depth_for_normal = render_median

    render_normals_from_depth = depth_to_normal(
        depth_for_normal, torch.linalg.inv(viewmats), Ks
    ).squeeze(0)

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "render_distort": render_distort,
        "gradient_2dgs": densify,  # This holds the gradient used for densification for 2dgs
    }

    render_normals = render_normals @ torch.linalg.inv(viewmats)[0, :3, :3].T

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
    )


# Regular rasterization, but allows to simultaneously render additional channels for 2DGS
# TODO: Do we want these additional channels to be view dependent?
def augmented_rasterization_2dgs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    additional_channels: Optional[Tensor] = None,  # [(C,) N, D2] or [(C,) N, K, D2]
    color_weights: Optional[Tensor] = None,  # [(C,) N, 3]
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """Rasterize a set of 2D Gaussians (N) to a batch of image planes (C).

    This function supports a handful of features, similar to the :func:`rasterization` function.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad (Experimental): If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distloss: If true, use distortion regularization to get better geometry detail.
        depth_mode: render depth mode. Choose from expected depth and median depth.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **render_normals**: The rendered normals. [C, height, width, 3].

        **surf_normals**: surface normal from depth. [C, height, width, 3]

        **render_distort**: The rendered distortions. [C, height, width, 1].
        L1 version, different from L2 version in 2DGS paper.

        **render_median**: The rendered median depth. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (normals.shape, surf_normals.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 3])
        >>> print (distort.shape, median_depth.shape)
        torch.Size([1, 200, 300, 1]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'ray_transforms',
        'opacities', 'normals', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras', 'render_distort',
        'gradient_2dgs'])

    """

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    channels = colors.shape[-1]

    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    if color_weights is not None:
        assert color_weights.shape == (N, 3), color_weights.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode
    if distloss:
        assert render_mode in [
            "D",
            "ED",
            "RGB+D",
            "RGB+ED",
        ], f"distloss requires depth rendering, render_mode should be D, ED, RGB+D, RGB+ED, but got {render_mode}"

    if additional_channels is not None:
        assert additional_channels.shape[0] == N, additional_channels.shape
        # additional_channels = additional_channels[None].expand(C, -1, -1)

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (colors.dim() == num_batch_dims + 2 and colors.shape[:-1] == batch_dims + (N,)) or (
            colors.dim() == num_batch_dims + 3 and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    # Compute Ray-Splat intersection transformation.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        packed,
        sparse_grad,
    )

    if packed:
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = proj_results
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]
        image_ids = batch_ids * C + camera_ids
    else:
        radii, means2d, depths, ray_transforms, normals = proj_results
        opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))  # [..., C, N]
        camera_ids, gaussian_ids = None, None
        image_ids = None

    densify = torch.zeros_like(
        means2d, dtype=means.dtype, requires_grad=True, device="cuda"
    )  # Identify intersecting tiles

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=I,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # TODO: SH also suport N-D.
    # Compute the per-view colors
    # if not (colors.dim() == 3 and sh_degree is None):  # silently support [C, N, D] color.
    #     colors = (
    #         colors[gaussian_ids] if packed else colors.expand(C, *([-1] * colors.dim()))
    #     )  # [nnz, D] or [C, N, 3]
    # else:
    #     if packed:
    #         colors = colors[camera_ids, gaussian_ids, :]

    if sh_degree is not None:  # SH coefficients
        camtoworlds = torch.inverse(viewmats)
        if packed:
            dirs = means[..., gaussian_ids, :] - camtoworlds[..., camera_ids, :3, 3]
        else:
            dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]

        if colors.dim() == num_batch_dims + 3:
            # Turn [..., N, K, 3] into [..., C, N, K, 3]
            shs = torch.broadcast_to(
                colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
            )  # [..., C, N, K, 3]
        else:
            # colors is already [..., C, N, K, 3]
            shs = colors
        colors = spherical_harmonics(
            sh_degree, dirs, shs, masks=(radii > 0).all(dim=-1)
        )  # [nnz, D] or [..., C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        if color_weights is not None:
            # The absolute color is the albedo (base color of Gaussian), the fraction of light reflected in each channel, times the intensity of the light incident on the Gaussian
            relit_colors = colors * color_weights
            colors = torch.cat((colors, relit_colors), dim=-1)

        if additional_channels is not None:
            colors = torch.cat((colors, additional_channels, depths[..., None]), dim=-1)
            background_channels_pad_size = additional_channels.shape[-1] + 1
        else:
            colors = torch.cat((colors, depths[..., None]), dim=-1)
            background_channels_pad_size = 1

        if backgrounds is not None and color_weights is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
        elif backgrounds is not None and color_weights is None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )

    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)

    else:  # RGB
        if color_weights is not None:
            # The absolute color is the albedo (base color of Gaussian), the fraction of light reflected in each channel, times the intensity of the light incident on the Gaussian
            relit_colors = colors * color_weights
            colors = torch.cat((colors, relit_colors), dim=-1)

        if additional_channels is not None:
            colors = torch.cat((colors, additional_channels), dim=-1)
            background_channels_pad_size = additional_channels.shape[-1]

        if (
            backgrounds is not None
            and color_weights is not None
            and additional_channels is not None
        ):
            backgrounds = torch.cat(
                [
                    backgrounds,
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
        elif backgrounds is not None and color_weights is not None and additional_channels is None:
            backgrounds = torch.cat([backgrounds, backgrounds], dim=-1)
        elif backgrounds is not None and color_weights is None and additional_channels is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(C, background_channels_pad_size, device=backgrounds.device),
                ],
                dim=-1,
            )
    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
        distloss=distloss,
    )
    render_normals_from_depth = None
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode in ["RGB+ED", "RGB+D"]:
        # render_depths = render_colors[..., -1:]
        if depth_mode == "expected":
            depth_for_normal = render_colors[..., -1:]
        elif depth_mode == "median":
            depth_for_normal = render_median

        render_normals_from_depth = depth_to_normal(
            depth_for_normal, torch.linalg.inv(viewmats), Ks
        ).squeeze(0)

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "render_distort": render_distort,
        "gradient_2dgs": densify,  # This holds the gradient used for densification for 2dgs
    }

    render_normals = torch.einsum(
        "...ij,...hwj->...hwi", torch.linalg.inv(viewmats)[..., :3, :3], render_normals
    )

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
    )


def conics_to_semi_major_axis_points(
    means2d: Tensor,  # [N, 2] - 2D means of Gaussians
    conics: Tensor,   # [N, 3] - conic parameters [a, b, c] for covariance matrix [[a, b], [b, c]]
    scale_factor: float = 1.0,  # Scale factor for the semi-major axis length
) -> Tensor:
    """
    Convert Gaussian conics to sample points on the semi-major axis.
    
    Args:
        means2d: 2D means of Gaussians [N, 2]
        conics: Conic parameters [N, 3] representing covariance matrix [[a, b], [b, c]]
        scale_factor: Scale factor for the semi-major axis length (default: 1.0)
    
    Returns:
        Tensor of shape [N, 4, 2] containing 4 points per Gaussian:
        - points[:, 0, :]: Point in negative direction along semi-major axis
        - points[:, 1, :]: Point in positive direction along semi-major axis  
        - points[:, 2, :]: Point in negative direction along semi-minor axis
        - points[:, 3, :]: Point in positive direction along semi-minor axis
    """
    N = means2d.shape[0]
    device = means2d.device
    
    # Extract covariance matrix parameters
    a = conics[:, 0]  # [N]
    b = conics[:, 1]  # [N] 
    c = conics[:, 2]  # [N]
    
    # Construct covariance matrices
    # [[a, b], [b, c]]
    cov_matrices = torch.stack([
        torch.stack([a, b], dim=1),  # [N, 2]
        torch.stack([b, c], dim=1),  # [N, 2]
    ], dim=2)  # [N, 2, 2]
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = torch.linalg.eigh(cov_matrices)  # [N, 2], [N, 2, 2]
    
    # Sort eigenvalues in descending order (largest first = semi-major axis)
    # eigenvals are already sorted in ascending order from eigh, so reverse
    eigenvals = torch.flip(eigenvals, dims=[1])  # [N, 2] - largest first
    eigenvecs = torch.flip(eigenvecs, dims=[2])  # [N, 2, 2] - corresponding eigenvectors
    
    # Extract semi-major and semi-minor axes
    semi_major_length = torch.sqrt(eigenvals[:, 0]) * scale_factor  # [N]
    semi_minor_length = torch.sqrt(eigenvals[:, 1]) * scale_factor  # [N]
    
    semi_major_dir = eigenvecs[:, :, 0]  # [N, 2] - direction of semi-major axis
    semi_minor_dir = eigenvecs[:, :, 1]  # [N, 2] - direction of semi-minor axis
    
    # Compute sample points
    # Semi-major axis points (positive and negative directions)
    major_pos = means2d + semi_major_length.unsqueeze(1) * semi_major_dir  # [N, 2]
    major_neg = means2d - semi_major_length.unsqueeze(1) * semi_major_dir  # [N, 2]
    
    # Semi-minor axis points (positive and negative directions)  
    minor_pos = means2d + semi_minor_length.unsqueeze(1) * semi_minor_dir  # [N, 2]
    minor_neg = means2d - semi_minor_length.unsqueeze(1) * semi_minor_dir  # [N, 2]
    
    # Stack all points: [major_neg, major_pos, minor_neg, minor_pos]
    sample_points = torch.stack([means2d, major_neg, major_pos, minor_neg, minor_pos], dim=1)  # [N, 5, 2]
    
    return sample_points
