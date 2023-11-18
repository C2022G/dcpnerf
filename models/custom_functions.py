import torch
import vren
from torch.cuda.amp import custom_fwd, custom_bwd
from torch_scatter import segment_csr
from einops import rearrange


class RayAABBIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and axis-aligned voxels.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_voxels, 3) voxel centers
        half_sizes: (N_voxels, 3) voxel half sizes
        max_hits: maximum number of intersected voxels to keep for one ray
                  (for a cubic scene, this is at most 3*N_voxels^(1/3)-2)

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit)
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return vren.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)


class RaySphereIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and spheres.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_spheres, 3) sphere centers
        radii: (N_spheres, 3) radii
        max_hits: maximum number of intersected spheres to keep for one ray

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_sphere_idx: (N_rays, max_hits) hit sphere indices (-1 if no hit)
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, radii, max_hits):
        return vren.ray_sphere_intersect(rays_o, rays_d, center, radii, max_hits)


class RayMarcher(torch.autograd.Function):
    """
    March the rays to get sample point positions and directions.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) normalized ray directions
        hits_t: (N_rays, 2) near and far bounds from aabb intersection
        density_bitfield: (C*G**3//8)
        cascades: int
        scale: float
        exp_step_factor: the exponential factor to scale the steps
        grid_size: int
        max_samples: int

    Outputs:
        rays_a: (N_rays) ray_idx, start_idx, N_samples
        xyzs: (N, 3) sample positions
        dirs: (N, 3) sample view directions
        deltas: (N) dt for integration
        ts: (N) sample ts
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, hits_t,
                density_bitfield, cascades, scale, exp_step_factor,
                grid_size, max_samples):
        # noise to perturb the first sample of each ray
        noise = torch.rand_like(rays_o[:, 0])

        rays_a, xyzs, dirs, deltas, ts, counter = \
            vren.raymarching_train(
                rays_o, rays_d, hits_t,
                density_bitfield, cascades, scale,
                exp_step_factor, noise, grid_size, max_samples)

        total_samples = counter[0]  # total samples for all rays
        # remove redundant output
        xyzs = xyzs[:total_samples]
        dirs = dirs[:total_samples]
        deltas = deltas[:total_samples]
        ts = ts[:total_samples]

        ctx.save_for_backward(rays_a, ts)

        return rays_a, xyzs, dirs, deltas, ts, total_samples

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_drays_a, dL_dxyzs, dL_ddirs,
                 dL_ddeltas, dL_dts, dL_dtotal_samples):
        rays_a, ts = ctx.saved_tensors
        segments = torch.cat([rays_a[:, 1], rays_a[-1:, 1] + rays_a[-1:, 2]])
        dL_drays_o = segment_csr(dL_dxyzs, segments)
        dL_drays_d = \
            segment_csr(dL_dxyzs * rearrange(ts, 'n -> n 1') + dL_ddirs, segments)

        return dL_drays_o, dL_drays_d, None, None, None, None, None, None, None


class HazAllVolumeRenderer(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, c_sigmas, p_sigmas, c_rgbs, p_rgbs, deltas, rays_a, T_threshold):
        total_samples, f_rgb, f_opacity, c_rgb, c_ws, c_opacity = vren.haz_all_composite_train_fw(c_sigmas, p_sigmas,
                                                                                                  c_rgbs, p_rgbs,
                                                                                                  deltas,
                                                                                                  rays_a, T_threshold)
        ctx.save_for_backward(c_sigmas, p_sigmas, c_rgbs, p_rgbs, deltas, rays_a,
                              f_rgb, f_opacity, c_rgb, c_ws, c_opacity)
        ctx.T_threshold = T_threshold
        return total_samples, f_rgb, f_opacity, c_rgb, c_ws, c_opacity

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dtotal_samples, dL_df_rgb, dL_df_opacity, dL_dc_rgb, dL_dc_ws, dL_dc_opacity):
        c_sigmas, p_sigmas, c_rgbs, p_rgbs, \
        deltas, rays_a, \
        f_rgb, f_opacity, c_rgb, c_ws, c_opacity = ctx.saved_tensors
        dL_dc_sigmas, dL_dp_sigmas, dL_dc_rgbs, dL_dp_rgbs = \
            vren.haz_all_composite_train_bw(
                dL_df_rgb.contiguous(), dL_df_opacity,
                dL_dc_rgb.contiguous(), dL_dc_ws, dL_dc_opacity,
                c_sigmas, p_sigmas,
                c_rgbs, p_rgbs,
                deltas, rays_a,
                f_rgb, f_opacity,
                c_rgb, c_ws, c_opacity,
                ctx.T_threshold)
        return dL_dc_sigmas, dL_dp_sigmas, dL_dc_rgbs, dL_dp_rgbs, None, None, None


class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))
