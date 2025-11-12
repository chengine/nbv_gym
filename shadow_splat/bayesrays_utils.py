"""Utilities for BayesRays uncertainty computation in NeRF training."""

from pathlib import Path
from typing import Optional, Tuple
import torch
import numpy as np
from tqdm import tqdm
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.models.instant_ngp import NGPModel
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.utils.math import Gaussians, conical_frustum_to_gaussian
# from bayesrays.utils.utils import normalize_point_coords, find_grid_indices
from nerfstudio.utils import colors
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss
)

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


def normalize_point_coords(points, aabb, distortion):
    ''' coordinate normalization process according to density_feild.py in nerfstudio'''
    if distortion is not None:
        pos = distortion(points)
        pos = (pos + 2.0) / 4.0
    else:        
        pos = SceneBox.get_normalized_positions(points, aabb)
    selector = ((pos > 0.0) & (pos < 1.0)).all(dim=-1) #points outside aabb are filtered out
    pos = pos * selector[..., None]
    return pos, selector

def find_grid_indices(points, aabb, distortion, lod, device, zero_out=True):
    pos, selector = normalize_point_coords(points, aabb, distortion)
    pos, selector = pos.view(-1, 3), selector[..., None].view(-1, 1)
    uncertainty_lod = 2 ** lod
    coords = (pos * uncertainty_lod).unsqueeze(0)
    inds = torch.zeros((8, pos.shape[0]), dtype=torch.int32, device=device)
    coefs = torch.zeros((8, pos.shape[0]), device=device)
    corners = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 1], [2, 0, 1, 0], [3, 0, 1, 1], [4, 1, 0, 0], [5, 1, 0, 1], [6, 1, 1, 0],
         [7, 1, 1, 1]], device=device)
    corners = corners.unsqueeze(1)
    inds[corners[:, :, 0].squeeze(1)] = (
            (torch.floor(coords[..., 0]) + corners[:, :, 1]) * uncertainty_lod * uncertainty_lod +
            (torch.floor(coords[..., 1]) + corners[:, :, 2]) * uncertainty_lod +
            (torch.floor(coords[..., 2]) + corners[:, :, 3])).int()
    coefs[corners[:, :, 0].squeeze(1)] = torch.abs(
        coords[..., 0] - (torch.floor(coords[..., 0]) + (1 - corners[:, :, 1]))) * torch.abs(
        coords[..., 1] - (torch.floor(coords[..., 1]) + (1 - corners[:, :, 2]))) * torch.abs(
        coords[..., 2] - (torch.floor(coords[..., 2]) + (1 - corners[:, :, 3])))
    if zero_out:
        coefs[corners[:, :, 0].squeeze(1)] *= selector[..., 0].unsqueeze(0)  # zero out the contribution of points outside aabb box

    return inds, coefs

def get_gaussian_blob_new(self) -> Gaussians: #for mipnerf
    """Calculates guassian approximation of conical frustum.

    Returns:
        Conical frustums approximated by gaussian distribution.
    """
    # Cone radius is set such that the square pixel_area matches the cone area.
    cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
    
    return conical_frustum_to_gaussian(
        origins=self.origins + self.offsets, #deforms Gaussian mean
        directions=self.directions,
        starts=self.starts,
        ends=self.ends,
        radius=cone_radius,
    )


def get_uncertainty(self, points):
    aabb = self.scene_box.aabb.to(points.device)
    ## samples outside aabb will have 0 coeff and hence 0 uncertainty. To avoid problems with these samples we set zero_out=False
    inds, coeffs = find_grid_indices(points, aabb, self.field.spatial_distortion ,self.lod, points.device, zero_out=False)
    cfs_2 = (coeffs**2)/torch.sum((coeffs**2),dim=0, keepdim=True)
    uns = self.un[inds.long()] #[8,N]
    un_points = torch.sqrt(torch.sum((uns*cfs_2),dim=0)).unsqueeze(1)
    
    #for stability in volume rendering we use log uncertainty
    un_points = torch.log10(un_points+1e-12)
    un_points = un_points.view((points.shape[0], points.shape[1],1))
    return un_points

def get_output_nerfacto_new(self, ray_bundle):
    ''' reimplementation of get_output function from models because of lack of proper interface to outputs dict'''
#     original_outputs = self.__class__.get_outputs(self, ray_bundle)  # Call original get_outputs (this is slower than just copying the original method here)
    
    N = self.N
    reg_lambda = 1e-4 /( (2**self.lod)**3)
    H = self.hessian/N + reg_lambda
    self.un = 1/H
            
    max_uncertainty = 6 #approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3 #approximate lower bound of that function (cutting off at hessian = 1000)
    density_fns_new = []
    if self.filter_out:
        for i in self.density_fns:
            density_fns_new.append(lambda x, i=i: i(x) * (self.get_uncertainty(x)<= self.filter_thresh*max_uncertainty))
    else:
        density_fns_new = self.density_fns
    
    if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
    else:
        ray_samples,_, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
    field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
    points = ray_samples.frustums.get_positions()
    un_points = self.get_uncertainty(points)

    #get weights
    if self.filter_out:
        density = field_outputs[FieldHeadNames.DENSITY] * (un_points <= self.filter_thresh*max_uncertainty)
    else:
        density = field_outputs[FieldHeadNames.DENSITY]
    weights = ray_samples.get_weights(density)
    
    uncertainty = torch.sum(weights * un_points, dim=-2) 
    uncertainty += (1-torch.sum(weights,dim=-2)) * min_uncertainty #alpha blending
    
    #normalize into acceptable range for rendering
    uncertainty = torch.clip(uncertainty, min_uncertainty, max_uncertainty)
    uncertainty = (uncertainty-min_uncertainty)/(max_uncertainty-min_uncertainty)
    
    if self.white_bg:
        self.renderer_rgb.background_color=colors.WHITE 
    elif self.black_bg:
        self.renderer_rgb.background_color=colors.BLACK     
    rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
    depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
    accumulation = self.renderer_accumulation(weights=weights)

    original_outputs = {
        "rgb": rgb,
        "accumulation": accumulation,
        "depth": depth,
    }
                                    
    original_outputs['uncertainty'] = uncertainty 
    if self.training:
        original_outputs["weights_list"] = weights_list
        original_outputs["ray_samples_list"] = ray_samples_list
    if self.config.predict_normals:
        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
        original_outputs["normals"] = self.normals_shader(normals)
        original_outputs["pred_normals"] = self.normals_shader(pred_normals)    

    if self.training and self.config.predict_normals:
        original_outputs["rendered_orientation_loss"] = orientation_loss(
            weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
        )

        original_outputs["rendered_pred_normal_loss"] = pred_normal_loss(
            weights.detach(),
            field_outputs[FieldHeadNames.NORMALS].detach(),
            field_outputs[FieldHeadNames.PRED_NORMALS],
        )

    for i in range(self.config.num_proposal_iterations):
        original_outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])


    return original_outputs

def get_output_fn(model):

    if isinstance(model, NerfactoModel):
        return get_output_nerfacto_new
    elif isinstance(model, NGPModel):
        return get_output_ngp_new
    elif isinstance(model, MipNerfModel):
        return get_output_mipnerf_new
    else:
        raise Exception("Sorry, this model is not currently supported.")
