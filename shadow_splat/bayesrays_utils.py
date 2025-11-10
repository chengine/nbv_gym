"""Utilities for BayesRays uncertainty computation in NeRF training."""

from pathlib import Path
from typing import Optional, Tuple
import torch
import numpy as np
from tqdm import tqdm
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.field_components.encodings import HashEncoding
from bayesrays.utils.utils import normalize_point_coords, find_grid_indices


class HessianComputer:
    """Computes Hessian for BayesRays uncertainty from training data."""

    def __init__(
        self,
        lod: int = 8,
        device: torch.device = None,
    ):
        """
        Initialize Hessian computer.

        Args:
            lod: Level of detail (log2 of grid resolution). Default 8 -> 256^3 grid.
            device: Device to compute on.
        """
        self.lod = lod
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hessian = torch.zeros(((2 ** self.lod) + 1) ** 3, device=self.device)

        # Initialize deformation field (HashEncoding) as in BayesRays
        self.deform_field = HashEncoding(
            num_levels=1,
            min_res=2 ** self.lod,
            max_res=2 ** self.lod,
            log2_hashmap_size=self.lod * 3 + 1,  # simple regular grid
            features_per_level=3,
            hash_init_scale=0.0,
            implementation="torch",
            interpolation="Linear",
        )
        self.deform_field.to(self.device)
        self.deform_field.scalings = torch.tensor([2 ** self.lod]).to(self.device)

    def find_uncertainty(
        self,
        points: torch.Tensor,
        deform_points: torch.Tensor,
        rgb: torch.Tensor,
        spatial_distortion,
    ) -> torch.Tensor:
        """
        Compute Hessian contribution from ray batch.

        Mirrors BayesRays ComputeUncertainty.find_uncertainty() logic.

        Args:
            points: Ray sample positions [N_rays, N_samples, 3]
            deform_points: Deformation field outputs [N_rays, N_samples, 3]
            rgb: RGB values [N_rays, N_samples, 3]
            spatial_distortion: Field's spatial distortion

        Returns:
            Hessian tensor [((2^lod)+1)^3]
        """
        inds, coeffs = find_grid_indices(
            points, self.aabb, spatial_distortion, self.lod, self.device
        )

        # Compute gradients for each color channel
        colors = torch.sum(rgb, dim=0)
        colors[0].backward(retain_graph=True)
        r = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()

        colors[1].backward(retain_graph=True)
        g = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()

        colors[2].backward()
        b = deform_points.grad.clone().detach().view(-1, 3)
        deform_points.grad.zero_()

        # Accumulate gradients into grid
        dmy = (torch.arange(points.shape[0])[..., None]).repeat((1, points.shape[1])).flatten().to(self.device)
        first = True
        for corner in range(8):
            if first:
                all_ind = torch.cat((dmy.unsqueeze(-1), inds[corner].unsqueeze(-1)), dim=-1)
                all_r = coeffs[corner].unsqueeze(-1) * r
                all_g = coeffs[corner].unsqueeze(-1) * g
                all_b = coeffs[corner].unsqueeze(-1) * b
                first = False
            else:
                all_ind = torch.cat(
                    (all_ind, torch.cat((dmy.unsqueeze(-1), inds[corner].unsqueeze(-1)), dim=-1)),
                    dim=0,
                )
                all_r = torch.cat((all_r, coeffs[corner].unsqueeze(-1) * r), dim=0)
                all_g = torch.cat((all_g, coeffs[corner].unsqueeze(-1) * g), dim=0)
                all_b = torch.cat((all_b, coeffs[corner].unsqueeze(-1) * b), dim=0)

        keys_all, inds_all = torch.unique(all_ind, dim=0, return_inverse=True)
        grad_r_1 = torch.bincount(inds_all, weights=all_r[..., 0])
        grad_g_1 = torch.bincount(inds_all, weights=all_g[..., 0])
        grad_b_1 = torch.bincount(inds_all, weights=all_b[..., 0])
        grad_r_2 = torch.bincount(inds_all, weights=all_r[..., 1])
        grad_g_2 = torch.bincount(inds_all, weights=all_g[..., 1])
        grad_b_2 = torch.bincount(inds_all, weights=all_b[..., 1])
        grad_r_3 = torch.bincount(inds_all, weights=all_r[..., 2])
        grad_g_3 = torch.bincount(inds_all, weights=all_g[..., 2])
        grad_b_3 = torch.bincount(inds_all, weights=all_b[..., 2])

        grad_1 = grad_r_1 ** 2 + grad_g_1 ** 2 + grad_b_1 ** 2
        grad_2 = grad_r_2 ** 2 + grad_g_2 ** 2 + grad_b_2 ** 2
        grad_3 = grad_r_3 ** 2 + grad_g_3 ** 2 + grad_b_3 ** 2

        grads_all = torch.cat(
            (keys_all[:, 1].unsqueeze(-1), (grad_1 + grad_2 + grad_3).unsqueeze(-1)), dim=-1
        )
        hessian = torch.zeros(((2 ** self.lod) + 1) ** 3, device=self.device)
        hessian = hessian.put((grads_all[:, 0]).long(), grads_all[:, 1], True)

        return hessian

    def compute_hessian_from_datamanager(
        self,
        model: NerfactoModel,
        datamanager,
        max_batches: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute accumulated Hessian from active training views.

        Mirrors BayesRays ComputeUncertainty.main() logic but integrated into training.

        Args:
            model: Nerfacto model instance
            datamanager: DataManager with next_train() method
            max_batches: Max training batches to process (None = all available)

        Returns:
            Accumulated Hessian tensor [((2^lod)+1)^3]
        """
        from nerfstudio.cameras.rays import RayBundle

        # Set AABB for this scene
        self.aabb = model.scene_box.aabb.to(self.device)

        # Reset Hessian
        self.hessian.zero_()

        # Determine number of batches to process
        num_train_images = len(datamanager.train_dataset)
        num_batches = max_batches if max_batches is not None else num_train_images

        # Set model to eval mode
        was_training = model.training
        model.eval()

        try:
            for step in tqdm(
                range(num_batches), desc="BayesRays: Computing Hessian", leave=False
            ):
                # Get next training batch
                # Handle both (ray_bundle, batch) from ray-batched and (cameras, batch, light) from full-image
                result = datamanager.next_train(step)
                if len(result) == 3:
                    cameras, batch, _ = result  # Ignore light (full-image datamanager)
                    # Convert cameras to ray bundle
                    ray_bundle = cameras.generate_rays(
                        camera_indices=torch.arange(
                            cameras.size, device=self.device, dtype=torch.long
                        )
                    )
                else:
                    ray_bundle, batch = result  # Already a RayBundle (ray-batched datamanager)

                # Forward pass to get ray samples and RGB (with gradients enabled for deform_points)
                with torch.enable_grad():
                    outputs, points, offsets = self._get_unc_nerfacto(ray_bundle, model)

                    # Compute Hessian contribution
                    hessian_contrib = self.find_uncertainty(
                        points, offsets, outputs["rgb"], model.field.spatial_distortion
                    )
                    self.hessian += hessian_contrib.clone().detach()

                # Clean up any gradients from this iteration
                torch.cuda.empty_cache()
        finally:
            # Restore training state and clean up
            if was_training:
                model.train()
            # Make sure to clear any leftover gradients
            model.zero_grad()
            torch.cuda.empty_cache()

        return self.hessian

    def _get_unc_nerfacto(self, ray_bundle, model):
        """
        Nerfacto forward pass for uncertainty computation.

        Mirrors BayesRays ComputeUncertainty.get_unc_nerfacto() logic.
        """
        if model.collider is not None:
            ray_bundle = model.collider(ray_bundle)

        ray_samples, weights_list, ray_samples_list = model.proposal_sampler(
            ray_bundle, density_fns=model.density_fns
        )
        points = ray_samples.frustums.get_positions()
        pos, _ = normalize_point_coords(points, self.aabb, model.field.spatial_distortion)

        # Get offsets from deformation field
        offsets = self.deform_field(pos).clone().detach()
        offsets.requires_grad = True

        ray_samples.frustums.set_offsets(offsets)

        from nerfstudio.field_components.field_heads import FieldHeadNames

        field_outputs = model.field(ray_samples, compute_normals=model.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = model.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = model.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = model.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if model.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        return outputs, points, offsets
