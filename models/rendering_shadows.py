import torch
from einops import rearrange, reduce, repeat

__all__ = ['render_rays', 'shadow_mapper', 'shadow_mapper2d']

from models.shadow_mapping import *
import models.shadow_mapping2d as shadow_mapping2d
from models.camera import Camera

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ=='coarse' and test_time and 'fine' in models:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                out_chunks += [model(xyz_embedded, sigma_only=True)]

            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                            # (N_rays*N_samples_, embed_dir_channels)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded_[i:i+chunk]], 1)
                out_chunks += [model(xyzdir_embedded, sigma_only=False)]

            out = torch.cat(out_chunks, 0)
            # out = out.view(N_rays, N_samples_, 4)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=4)
            rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            sigmas = out[..., 3] # (N_rays, N_samples_)
            
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results[f'z_vals_{typ}'] = z_vals
        if test_time and typ == 'coarse' and 'fine' in models:
            return

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')
        depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(1)

        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map

        return

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d)) # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    results = {}
    inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        inference(results, models['fine'], 'fine', xyz_fine, z_vals, test_time, **kwargs)

    return results


def shadow_mapper(cam_results, light_results, cam_pixels, light_pixels, image_shape, cam_ppc, light_ppc, shadow_method_mode='shadow_method_1'):
    """
    1D Shadow mapper, Accepts arbitrary number of pixels, ALL of light pixels and computes if those pixels are in shadow. 
    Assumes all camera pixels belong to the cam_ppc camera!
    """ 
    ret = {}
    cam_ray_termination_dist = cam_results['depth_coarse'] # This is the ray termination distance right? 
    light_ray_termination_dist = light_results['depth_coarse']

    pixels_to_range_cam_coarse = torch.cat([cam_pixels, cam_ray_termination_dist.view(-1,1)], dim=1)
    pixels_to_range_light_coarse = torch.cat([light_pixels, light_ray_termination_dist.view(-1,1)], dim=1)
    pixels_to_w_light = get_normed_w(light_ppc, pixels_to_range_light_coarse, device=light_ray_termination_dist.device)

    # increased delta to 1e-3
    sm_coarse = run_shadow_mapping(image_shape, cam_ppc, light_ppc, pixels_to_range_cam_coarse, pixels_to_w_light, 
                                   cam_ray_termination_dist.device, mode=shadow_method_mode,
                                   delta=1e-3, epsilon=0.0, new_min=0.0, new_max=1.0,
                                   sigmoid=False, use_numpy_meshgrid=True)
    ret['sm_coarse'] = sm_coarse

    if 'depth_fine' in cam_results.keys(): 
        cam_ray_termination_dist = cam_results['depth_fine'] # This is the ray termination distance right? 
        pixels_to_range_cam_fine = torch.cat([cam_pixels, cam_ray_termination_dist.view(-1,1)], dim=1)
        
        if 'depth_fine' in light_results.keys():
            light_ray_termination_dist = light_results['depth_fine']
            pixels_to_range_light_fine = torch.cat([light_pixels, light_ray_termination_dist.view(-1,1)], dim=1)
            pixels_to_w_light = get_normed_w(light_ppc, pixels_to_range_light_fine, device=light_ray_termination_dist.device)

        # increased delta to 1e-3
        sm_fine = run_shadow_mapping(image_shape, cam_ppc, light_ppc, pixels_to_range_cam_fine, pixels_to_w_light, 
                                   cam_ray_termination_dist.device, mode=shadow_method_mode,
                                   delta=1e-3, epsilon=0.0, new_min=0.0, new_max=1.0,
                                   sigmoid=False, use_numpy_meshgrid=True)
        ret['sm_fine'] = sm_fine

    return ret


def shadow_mapper2d(cam_results, light_results, image_shape, cam_ppc, light_ppc, shadow_method_mode='shadow_method_1'):
    """
    2D shadow mapper: accepts cam_results['depth_*'].view(H,W) and light_results['depth_*'].view(H,W). 
    You can change batch_size == H*W and use the flag --shadow_mapper2d to use this function.
    """
    ret = {}
    w, h = image_shape
    N_rays = cam_results['depth_coarse'].shape[0]
    assert N_rays == w*h

    cam_ray_termination_dist = cam_results['depth_coarse'].view(h,w) # This is the ray termination distance right? 
    light_ray_termination_dist = light_results['depth_coarse'].view(h,w)
    # print(light_ray_termination_dist.shape, cam_ray_termination_dist.shape)
    # print(cam_ppc.camera.shape, light_ppc.camera.shape)
    assert N_rays == light_ray_termination_dist.shape[0] * light_ray_termination_dist.shape[1]

    # increased delta to 1e-3
    sm_coarse = shadow_mapping2d.run_shadow_mapping(image_shape, cam_ppc, light_ppc, 
                                    cam_ray_termination_dist, 
                                    light_ray_termination_dist,
                                    device=cam_ray_termination_dist.device, 
                                    mode=shadow_method_mode,
                                    delta=1e-3, epsilon=0.0, new_min=0.0, new_max=1.0, 
                                    sigmoid=False, use_numpy_meshgrid=False)
    ret['sm_coarse'] = sm_coarse.view(-1) # .view(-1,3)

    if 'depth_fine' in cam_results.keys(): 
        cam_ray_termination_dist = cam_results['depth_fine'].view(h,w) # This is the ray termination distance right? 
        
        if 'depth_fine' in light_results.keys():
            light_ray_termination_dist = light_results['depth_fine'].view(h,w)

        # increased delta to 1e-3
        sm_fine = shadow_mapping2d.run_shadow_mapping(image_shape, cam_ppc, light_ppc, 
                                    cam_ray_termination_dist, 
                                    light_ray_termination_dist,
                                    device=cam_ray_termination_dist.device, 
                                    mode=shadow_method_mode,
                                    delta=1e-3, epsilon=0.0, new_min=0.0, new_max=1.0, 
                                    sigmoid=False, use_numpy_meshgrid=False)
        ret['sm_fine'] = sm_fine.view(-1) # .view(-1,3)

    return ret

