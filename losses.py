
import torch
from torch import nn
import vren




class DepthLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ws, ts,deltas,depth,target,rays_a):
        loss = vren.depth_loss_fw(ws,ts,deltas,depth, target, rays_a)

        ctx.save_for_backward(depth,target,rays_a,ts,ws,deltas)

        return loss

    @staticmethod
    def backward(ctx, dL_dloss):

        (depth,target,rays_a,ts,ws,deltas) = ctx.saved_tensors
        dL_dws = vren.depth_loss_bw(dL_dloss, ts,depth,target,rays_a,ws,deltas)

        return dL_dws, None, None, None,None,None,None,None

class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a,rgb_len):

        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a,rgb_len)


        return loss

    @staticmethod
    def backward(ctx, dL_dloss):


        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a,rgb_len) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a,rgb_len)

        return dL_dws, None, None, None,None


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion

    def forward(self, results, target, **kwargs):
        d = {}
        mask = results['mask']>0

        d['rgb'] = (results['rgb'][0:target['rgb'].shape[0]]-target['rgb'])**2

        # DepthLoss.apply() to apply KL depth loss
        d['depth'] = 0.01 * DepthLoss.apply(results['ws'],results['ts'],results['deltas'],results['depth'][target['rgb'].shape[0]:],target['depth_value'],
                                      results['rays_a'])
        # MSE loss
        # d['depth'] = 0.01*((results['depth'][target['rgb'].shape[0]:]-target['depth_value'])**2)

        d['depth_regularization'] = 0.01*(results['ws'][mask])**2

        o = results['opacity'][0:target['rgb'].shape[0]]+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater

        d['opacity'] = self.lambda_opacity * (-o * torch.log(o))


        rgb_len = torch.tensor(target['rgb'].shape[0])[None,...].cuda()
        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion *DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'],rgb_len)


        return d
