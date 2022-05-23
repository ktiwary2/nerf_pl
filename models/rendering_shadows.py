import torch
# from torchsearchsorted import searchsorted
import models.shadow_mapping_utils as shadow_mapping_utils
import models.efficient_shadow_mapping as eff_sm
from models.camera import Camera

__all__ = ['render_rays', 'shadow_mapping', 'efficient_sm']

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""

def sample_pdf(rays, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    from https://github.com/sxyu/pixel-nerf/blob/master/src/render/nerf.py#L120
    Inputs:
        rays: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    # torch.cuda.empty_cache()
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive
                                                               
    u = torch.rand(N_rays, N_importance, device=rays.device) # (N_rays, N_samples)
    # inds = searchsorted(cdf, u, side='right').float() - 1.0 # (N_rays, N_samples)
    inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
    inds = torch.clamp_min(inds, 0.0)

    z_steps = (inds + torch.rand_like(inds)) / N_samples_  # (B, Kf)

    near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
    # if not self.lindisp:  # Use linear sampling in depth space
    z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
    # else:  # Use linear sampling in disparity space
        # z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
    return z_samp

# def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
#     """
#     Sample @N_importance samples from @bins with distribution defined by @weights.
#     Inputs:
#         bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
#         weights: (N_rays, N_samples_)
#         N_importance: the number of samples to draw from the distribution
#         det: deterministic or not
#         eps: a small number to prevent division by zero
#     Outputs:
#         samples: the sampled samples
#     """
#     N_rays, N_samples_ = weights.shape
#     weights = weights + eps # prevent division by zero (don't do inplace op!)
#     pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
#     cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
#     cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
#                                                                # padded to 0~1 inclusive

#     if det:
#         u = torch.linspace(0, 1, N_importance, device=bins.device)
#         u = u.expand(N_rays, N_importance)
#     else:
#         u = torch.rand(N_rays, N_importance, device=bins.device)
#     u = u.contiguous()

#     inds = searchsorted(cdf, u, side='right')
#     below = torch.clamp_min(inds-1, 0)
#     above = torch.clamp_max(inds, N_samples_)

#     inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
#     cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
#     bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

#     denom = cdf_g[...,1]-cdf_g[...,0]
#     denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
#                          # anyway, therefore any value for it is fine (set to 1 here)

#     samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
#     return samples

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
                were_gradients_computed= True
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
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

    def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        # if not weights_only:
            # dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyzdir_embedded = embedding_xyz(xyz_[i:i+chunk])
            # if not weights_only:
            #     xyzdir_embedded = torch.cat([xyz_embedded,
            #                                  dir_embedded[i:i+chunk]], 1)
            # else:
                # xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=True)]

        out = torch.cat(out_chunks, 0)
        sigmas = out.view(N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        # weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        # compute final weighted outputs
        # rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        if weights_only:
            return weights

        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_final), depth_final / torch.sum(weights, -1))

        # if white_back:
        #     rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return depth_final, weights, disp_map

    # Extract models from lists
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    # print("rays: ", rays.shape)
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        # print(near.shape, z_steps.shape, far.shape)
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    if test_time:
        weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        depth_coarse, weights_coarse, disp_map_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {
                #   'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1),
                  'disp_map_coarse': disp_map_coarse,
                 }

    if N_importance > 0: # sample points for fine model
        # z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(rays, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0))
        # if were_gradients_computed: 
        z_vals_ = z_vals_.detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        depth_fine, weights_fine, disp_map_fine = \
            inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)

        # result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)
        result['disp_map_fine'] = disp_map_fine

    return result

def shadow_mapping(cam_results, light_results, rays, ppc, light_ppc, image_shape, batch_size, fine_sampling):
    """
    cam_result: result dictionary with `depth_*`, `opacity_*`
    light_result: result dictionary with `depth_*`, `opacity_*`
    rays: generated rays 
    ppc: [Batch_size] Camera Poses: instance of the Camera() class
    light_ppc: [1] Pose of the Camera at Light position 
    batch_size: batch_size
    fine_sampling: set fine_sampling
    """
    def inference(ppc, light_ppc, image_shape, batch_size, cam_depths, light_depths):
        """
        ppc: [Batch_size] Camera Poses: instance of the Camera() class
        light_ppc: [1] Pose of the Camera at Light position 
        batch_size: batch_size
        cam_depths: [batch_size, H, W] depth from camera viewpoints
        light_depths: [1, H, W] depth from light viewpoint
        """
        shadow_maps = []
        for i in range(batch_size):
            # print('-------------->', ppc[i], ppc[i].keys())
            # eye_pos, camera = ppc[i][0], ppc[i][1]
            cam = Camera.from_camera_eyepos(eye_pos=ppc[i]['eye_pos'].squeeze(0), camera=ppc[i]['camera'].squeeze(0))
            # eye_pos, camera = light_ppc[0], light_ppc[1]
            light_cam = Camera.from_camera_eyepos(eye_pos=light_ppc['eye_pos'].squeeze(0), camera=light_ppc['camera'].squeeze(0))
            sm = eff_sm.run_shadow_mapping(image_shape,
                                                        cam, light_cam, 
                                                        cam_depths.squeeze(0), light_depths.squeeze(0),
                                                        device=cam_depths.device, 
                                                        mode='shadow_method_2',
                                                        delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, 
                                                        sigmoid=False, use_numpy_meshgrid=True)
            shadow_maps += [sm]

        shadow_maps = torch.cat(shadow_maps, 0) # (batch_size, H, W)
        return shadow_maps

    N_rays = rays.shape[0]
    W, H = image_shape

    # Do Shadow Mapping for Coarse Depth 
    cam_depths_coarse = cam_results['depth_coarse'] # (N_rays)
    light_depths_coarse = light_results['depth_coarse'] # (N_rays)
    assert N_rays == cam_depths_coarse.shape[0]
    assert N_rays == light_depths_coarse.shape[0]
    device = cam_depths_coarse.device    

    cam_depths_coarse = cam_depths_coarse.view(batch_size, H, W).to(device)
    light_depths_coarse = light_depths_coarse.view(batch_size, H, W).to(device)
    shadow_maps_coarse = inference(ppc, light_ppc, image_shape, batch_size, cam_depths_coarse, light_depths_coarse)
    shadow_maps_coarse = shadow_maps_coarse.view(-1, 3)
    # print("shadow_maps_coarse.shape", shadow_maps_coarse.shape)

    cam_results['rgb_coarse'] = shadow_maps_coarse

    # Do Shadow Mapping for Fine Depth Maps 
    if fine_sampling: # sample points for fine model
        cam_depths_fine = cam_results['depth_fine'] # (N_rays)
        light_depths_fine = light_results['depth_fine'] # (N_rays)
        device = cam_depths_fine.device    
        assert N_rays == cam_depths_fine.shape[0]
        assert N_rays == light_depths_fine.shape[0]

        cam_depths_fine = cam_depths_fine.view(batch_size, H, W).to(device)
        light_depths_fine = light_depths_fine.view(batch_size, H, W).to(device)
        shadow_maps_fine = inference(ppc, light_ppc, image_shape, batch_size, cam_depths_fine, light_depths_fine)
        shadow_maps_fine = shadow_maps_fine.view(-1, 3)
        # print("shadow_maps_fines.shape", shadow_maps_fine.shape)
        cam_results['rgb_fine'] = shadow_maps_coarse

    return cam_results

######
# Sometimes loss goes to NaN :(
EPSILON = 1e-5
######
def efficient_sm(cam_pixels, light_pixels, cam_results, light_results, 
                 ppc, light_ppc, image_shape, fine_sampling, Light_N_importance, shadow_method):
    """
    cam_pixels: [i,j,1] of size (H,W)
    light_pixels: [i,j,1] of size (H,W)
    cam_result: result dictionary with `depth_*`, `opacity_*`
    light_result: result dictionary with `depth_*`, `opacity_*`
    ppc: [Batch_size] Camera Poses: instance of the Camera() class
    light_ppc: [1] Pose of the Camera at Light position 
    fine_sampling: set fine_sampling (bool)
    image_shape: IMAGE SHAPE OF THE CAMERA AT LIGHT POSITION 
    """

    def inference(ppc, light_camera, image_shape, batched_mesh_range_cam, meshed_normed_light_cam, shadow_method):
        """
        ppc: [Batch_size] Camera Poses: instance of the Camera() class
        light_camera: Instance of class Camera placed at the light position 
        batch_size: batch_size
        batched_mesh_range_cam: [num_rays, 4] (i, j, 1, depth] from camera viewpoints
        meshed_normed_light_cam: [H*W, 4] (i, j, 1, depth] from light viewpoint
        """
        shadow_maps = []
        curr_eye_pos = ppc['eye_pos'][0]
        prev_split_at = 0
        num_splits = 0 
        curr_ppc = Camera.from_camera_eyepos(eye_pos=ppc['eye_pos'][0].squeeze(0), camera=ppc['camera'][0].squeeze(0))
        for i in range(len(ppc['camera'])):
            # each pixel can have a different viewpoint within the batch, therefore we need to split them with the same (depth, pose) 
            # print("PPC: {} {}".format(curr_eye_pos, ppc['eye_pos'][i]))
            if not (torch.equal(curr_eye_pos, ppc['eye_pos'][i])): 
                # means a new ppc is encountered 
                # all pixels from prev_split to i have the same camera pose therefore we can estimate the sm 
                # for those.
                # print("------")
                # print("PPC: {} {}".format(curr_eye_pos, ppc['eye_pos'][i]))
                # print("Found different eye_pos, using shadow method {}, splitting at {}:{}".format(shadow_method, prev_split_at, i))
                sub_batch_mesh_range_cam = batched_mesh_range_cam[prev_split_at:i,:]
                sm = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, sub_batch_mesh_range_cam, 
                                           meshed_normed_light_cam, sub_batch_mesh_range_cam.device, mode=shadow_method, \
                                           delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                           use_numpy_meshgrid=True)
                # print("sm, {}, {}".format(sm.shape, len(shadow_maps)))
                # print("------")
                shadow_maps += [sm]
                prev_split_at = i 
                curr_eye_pos = ppc['eye_pos'][i]
                curr_ppc = Camera.from_camera_eyepos(eye_pos=ppc['eye_pos'][i].squeeze(0), camera=ppc['camera'][i].squeeze(0))
                num_splits += 1

        if prev_split_at == 0: 
            # print("Found No Splits...")
            # means that all the pixels have the same viewpoint! we can batch them together
            shadow_maps = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, batched_mesh_range_cam, 
                                        meshed_normed_light_cam, batched_mesh_range_cam.device, mode=shadow_method, \
                                        delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                        use_numpy_meshgrid=True) # (num_rays, 3)
        else: #prev_split_at < (len(ppc)-1): 
            # do inference on the remaining 
            # print("Doing inference on the last split from [{}:{}]".format(prev_split_at, len(ppc['camera'])))
            sub_batch_mesh_range_cam = batched_mesh_range_cam[prev_split_at:,:]
            sm = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, sub_batch_mesh_range_cam, 
                                        meshed_normed_light_cam, sub_batch_mesh_range_cam.device, mode=shadow_method, \
                                        delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                        use_numpy_meshgrid=True) # (num_rays, 3)
            # print("sm, {}, {}".format(sm.shape, len(shadow_maps)))
            shadow_maps += [sm]
            shadow_maps = torch.cat(shadow_maps, 0) # (num_rays, 3)
            # print("shadow_maps.shape", shadow_maps.shape)

        if num_splits > 5:
            print("Split the batch of rays {} times. Not very efficient...".format(num_splits))
        return shadow_maps


    # should be the same for all the pixels! 
    # print(light_ppc)
    # print(type(light_ppc))
    # light_camera = Camera.from_camera_eyepos(eye_pos=light_ppc['eye_pos'].squeeze(0), camera=light_ppc['camera'].squeeze(0))
    light_camera = light_ppc
    if True: 
        # Do Shadow Mapping for Coarse Depth 
        cam_depths_coarse = cam_results['depth_coarse'] # (N_rays)
        cam_pixels = cam_pixels.to(cam_depths_coarse.device)
        batched_mesh_range_cam_coarse = torch.cat([cam_pixels, cam_depths_coarse.view(-1,1)], dim=1) # [batch_size, 4]
        # print(batched_mesh_range_cam_coarse)
        # assert N_rays == cam_depths_coarse.shape[0]
        # assert N_rays == light_depths_coarse.shape[0]

        light_depths_coarse = light_results['depth_coarse'] # (H*W)
        # This is not Batched, we do full inference on the light! 
        light_pixels = light_pixels.to(light_depths_coarse.device)
        mesh_range_light = torch.cat([light_pixels, light_depths_coarse.view(-1,1)], dim=1) # [H*W, 4]
        ##################### switch off 
        meshed_normed_light_coarse = eff_sm.get_normed_w(light_camera, mesh_range_light, device=light_depths_coarse.device)
        ##################### switch off 
        # meshed_normed_light_coarse = mesh_range_light
        shadow_maps_coarse = inference(ppc, light_camera, image_shape, batched_mesh_range_cam_coarse, meshed_normed_light_coarse, shadow_method)
        # print("shadow_maps_coarse.shape", shadow_maps_coarse.shape)
        shadow_maps_coarse = shadow_maps_coarse.view(-1, 3)

        cam_results['rgb_coarse'] = shadow_maps_coarse + EPSILON * torch.ones_like(shadow_maps_coarse) 

    # Do Shadow Mapping for Fine Depth Maps 
    FINE = True
    if fine_sampling and FINE: # sample points for fine model
        cam_depths_fine = cam_results['depth_fine'] # (N_rays)
        batched_mesh_range_cam_fine = torch.cat([cam_pixels, cam_depths_fine.view(-1,1)], dim=1)
        
        if Light_N_importance: 
            light_depths_fine = light_results['depth_fine'] # (N_rays)
            mesh_range_light = torch.cat([light_pixels, light_depths_fine.view(-1,1)], dim=1)
            meshed_normed_light_fine = eff_sm.get_normed_w(light_camera, mesh_range_light, device=light_depths_fine.device)
            ##################### switch off 
            # meshed_normed_light_fine = mesh_range_light
            ##################### switch off 
            shadow_maps_fine = inference(ppc, light_camera, image_shape, batched_mesh_range_cam_fine, meshed_normed_light_fine, shadow_method)
        else:
            shadow_maps_fine = inference(ppc, light_camera, image_shape, batched_mesh_range_cam_fine, meshed_normed_light_coarse, shadow_method)

        # print("shadow_maps_fine.shape", shadow_maps_fine.shape)
        shadow_maps_fine = shadow_maps_fine.view(-1, 3)
        cam_results['rgb_fine'] = shadow_maps_fine + EPSILON * torch.ones_like(shadow_maps_coarse)

    return cam_results

def get_K(cam_pixels, cam_results, ppc, light_camera, fine_sampling):
    def inference(ppc, light_camera, batched_mesh_range_cam):
        """
        ppc: [Batch_size] Camera Poses: instance of the Camera() class
        light_camera: Instance of the Camera() class 
        batched_mesh_range_cam: [num_rays, 4] (i, j, 1, depth] from camera viewpoints
        """
        proj_maps = []
        curr_eye_pos = ppc['eye_pos'][0]
        prev_split_at = 0
        num_splits = 0 
        curr_ppc = Camera.from_camera_eyepos(eye_pos=ppc['eye_pos'][0].squeeze(0), camera=ppc['camera'][0].squeeze(0))
        for i in range(len(ppc['camera'])):
            # each pixel can have a different viewpoint within the batch, 
            # therefore we need to split them with the same (depth, pose) 
            # print("PPC: {} {}".format(curr_eye_pos, ppc['eye_pos'][i]))
            if not (torch.equal(curr_eye_pos, ppc['eye_pos'][i])):
                sub_batch_mesh_range_cam = batched_mesh_range_cam[prev_split_at:i,:]
                K = eff_sm.get_projections(curr_ppc, light_camera, sub_batch_mesh_range_cam, sub_batch_mesh_range_cam.device)
                proj_maps += [K]
                prev_split_at = i 
                curr_eye_pos = ppc['eye_pos'][i]
                curr_ppc = Camera.from_camera_eyepos(eye_pos=ppc['eye_pos'][i].squeeze(0), camera=ppc['camera'][i].squeeze(0))
                num_splits += 1

        if prev_split_at == 0: 
            # print("Found No Splits...")
            # means that all the pixels have the same viewpoint! we can batch them together
            proj_maps = eff_sm.get_projections(curr_ppc, light_camera, batched_mesh_range_cam, batched_mesh_range_cam.device)
        else: #prev_split_at < (len(ppc)-1): 
            # do inference on the remaining 
            # print("Doing inference on the last split from [{}:{}]".format(prev_split_at, len(ppc['camera'])))
            sub_batch_mesh_range_cam = batched_mesh_range_cam[prev_split_at:,:]
            K = eff_sm.get_projections(curr_ppc, light_camera, sub_batch_mesh_range_cam, sub_batch_mesh_range_cam.device)
            # print("sm, {}, {}".format(sm.shape, len(shadow_maps)))
            proj_maps += [K]
            proj_maps = torch.cat(proj_maps, 0) # (num_rays, 3)
            # print("shadow_maps.shape", shadow_maps.shape)

        if num_splits > 5:
            print("Split the batch of rays {} times. Not very efficient...".format(num_splits))
        return proj_maps

    # Do Shadow Mapping for Coarse Depth 
    cam_depths_coarse = cam_results['depth_coarse'] # (N_rays)
    cam_pixels = cam_pixels.to(cam_depths_coarse.device)
    batched_mesh_range_cam_coarse = torch.cat([cam_pixels, cam_depths_coarse.view(-1,1)], dim=1)
    # print(batched_mesh_range_cam_coarse)
    # assert N_rays == cam_depths_coarse.shape[0]
    # assert N_rays == light_depths_coarse.shape[0]
    proj_maps_coarse = inference(ppc, light_camera, batched_mesh_range_cam_coarse)
    proj_maps_coarse = proj_maps_coarse.view(-1,3)

    FINE = True
    proj_maps_fine = None
    if fine_sampling and FINE: # sample points for fine model
        cam_depths_fine = cam_results['depth_fine'] # (N_rays)
        batched_mesh_range_cam_fine = torch.cat([cam_pixels, cam_depths_fine.view(-1,1)], dim=1)

        proj_maps_fine = inference(ppc, light_camera, batched_mesh_range_cam_fine)
        proj_maps_fine = proj_maps_fine.view(-1,3)

    return proj_maps_coarse, proj_maps_fine
