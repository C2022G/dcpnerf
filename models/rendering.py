import torch
from .custom_functions import \
    RayAABBIntersector, RayMarcher, HazAllVolumeRenderer
from einops import rearrange
import vren

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01


@torch.cuda.amp.autocast()
def render(model, rays_o, rays_d, **kwargs):
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0] >= 0) & (hits_t[:, 0, 0] < NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('split', "train") == "train":
        results = __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs)
    else:
        results = __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs)

    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device

    c_opacity = torch.zeros(N_rays, device=device)
    c_depth = torch.zeros(N_rays, device=device)
    c_rgb = torch.zeros(N_rays, 3, device=device)

    f_opacity = torch.zeros(N_rays, device=device)
    f_depth = torch.zeros(N_rays, device=device)
    f_rgb = torch.zeros(N_rays, 3, device=device)
    f_alive_indices = torch.arange(N_rays, device=device)

    samples = 0

    min_samples = 1 if exp_step_factor == 0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(f_alive_indices)
        if N_alive == 0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays // N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], f_alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)

        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs == 0, dim=1)
        if valid_mask.sum() == 0: break

        c_sigmas = torch.zeros(len(xyzs), device=device)
        c_rgbs = torch.zeros(len(xyzs), 3, device=device)

        p_sigmas = torch.zeros(len(xyzs), device=device)
        p_rgbs = torch.zeros(len(xyzs), 3, device=device)

        c_sigmas[valid_mask], _c_rgbs, p_sigmas[valid_mask], _p_rgbs, = model(xyzs[valid_mask], dirs[valid_mask])

        c_rgbs[valid_mask] = _c_rgbs.float()
        c_sigmas = rearrange(c_sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        c_rgbs = rearrange(c_rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        p_rgbs[valid_mask] = _p_rgbs.float()
        p_sigmas = rearrange(p_sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        p_rgbs = rearrange(p_rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        vren.all_composite_test_fw(
            c_sigmas, p_sigmas, c_rgbs, p_rgbs,
            deltas, ts,
            hits_t[:, 0], f_alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, c_rgb, c_depth, c_opacity, f_rgb, f_depth, f_opacity)
        f_alive_indices = f_alive_indices[f_alive_indices >= 0]  # remove converged rays

    results['c_opacity'] = c_opacity
    results['c_depth'] = c_depth
    results['c_rgb'] = c_rgb

    results['f_opacity'] = f_opacity
    results['f_depth'] = f_depth
    results['f_rgb'] = f_rgb

    rgb_bg = torch.ones(3, device=device)
    results['c_rgb'] += rgb_bg * rearrange(1 - c_opacity, 'n -> n 1')
    results['f_rgb'] += rgb_bg * rearrange(1 - f_opacity, 'n -> n 1')

    return results


def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}
    (rays_a, xyzs, dirs,
     results['deltas'], results['ts'], results['rm_samples']) = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, MAX_SAMPLES)

    for k, v in kwargs.items():  # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)

    c_sigmas, c_rgbs, p_sigmas, p_rgbs = model(xyzs, dirs)

    (results['vr_samples'],
     results['f_rgb'], results['f_opacity'], results['c_rgb'], results['c_ws'], results['c_opacity']) = \
        HazAllVolumeRenderer.apply(c_sigmas, p_sigmas, c_rgbs.contiguous(), p_rgbs.contiguous(), results['deltas'],
                                   rays_a, kwargs.get('T_threshold', 1e-4))

    results['c_sigmas'] = c_sigmas
    results['p_sigmas'] = p_sigmas
    results["p_rgbs"] = p_rgbs

    results['rays_a'] = rays_a
    results['hits_t'] = hits_t.squeeze(1)

    rgb_bg = torch.ones(3, device=rays_o.device)

    results['c_rgb'] = results['c_rgb'] + \
                       rgb_bg * rearrange(1 - results['c_opacity'], 'n -> n 1')

    results['f_rgb'] = results['f_rgb'] + \
                       rgb_bg * rearrange(1 - results['f_opacity'], 'n -> n 1')

    return results
