import torch.optim.lr_scheduler
import vren


class Distortion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
         ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class Foggy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, alpha_rat, rays_a, vr_samples):
        loss = vren.foggy_fw(weights, alpha_rat, rays_a, vr_samples)
        ctx.save_for_backward(weights, alpha_rat, rays_a, vr_samples)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        weights, alpha_rat, rays_a, vr_samples = ctx.saved_tensors
        dL_dsigma = vren.foggy_bw(dL_dloss, weights, alpha_rat, rays_a, vr_samples)

        return None, dL_dsigma, None, None, None


class AlphaRat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma, deltas, rays_a, vr_samples):
        alpha_rat = vren.alpha_rat_fw(sigma, deltas, rays_a, vr_samples)
        ctx.save_for_backward(alpha_rat, sigma, deltas, rays_a, vr_samples)
        return alpha_rat

    @staticmethod
    def backward(ctx, dL_dloss):
        alpha_rat, sigma, deltas, rays_a, vr_samples = ctx.saved_tensors
        dL_dsigma = vren.alpha_rat_bw(dL_dloss, alpha_rat, sigma, deltas, rays_a,
                                      vr_samples)

        return dL_dsigma, None, None, None
