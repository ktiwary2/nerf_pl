import torch 
from numpy import meshgrid
###
# NOTE: (ktiwary) Currently only accepts Square Images! Will fix this at some point. 
# use_numpy_meshgrid: Since older versions of torch don't have indexing as an arg for torch..
###

EPSILON = 1e-5

def normalize_min_max(tensor, new_max=1.0, new_min=0.0):
     return (tensor - tensor.min())/(tensor.max() - tensor.min() + EPSILON)*(new_max - new_min) + new_min

def get_projections(camera, light_cam, batched_mesh_range_cam, device):
    batched_w_cam = get_normed_w(camera, batched_mesh_range_cam, device=device)
    # print(batched_w_cam)
    K = get_diff_projections(batched_w_cam[:,:3], batched_w_cam[:,3], camera, light_cam, device=device)
    return K
    
def run_shadow_mapping(res, camera, light_cam, batched_mesh_range_cam, meshed_normed_light_cam, device, mode='shadow_method_1', \
                       delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, use_numpy_meshgrid=True):
    """
    Run Shadow Mapping stack from start to finish in a differentiable manner. 
    Args: 
        res: Resolution of the Light View 
        camera: Camera PPC (instance of Camera Class) 
        light_cam: Light Camera PPC (instance of Camera Class)
        range_cam: This can be of arbitrary batched size (1024,)
        meshed_normed_light_cam: Normalized Depth Light must be of size (H*W, 4), range_light cannot be batched! 
            the last (,3) is the normalized depth! call get_normed_w(light_cam, range_light)
        device: 
        mode: see generate_shadow_map
    """
    batched_w_cam = get_normed_w(camera, batched_mesh_range_cam, device=device)
    # print(batched_w_cam)
    K = get_diff_projections(batched_w_cam[:,:3], batched_w_cam[:,3], camera, light_cam, device=device)

    wl_, w_light_bounded_ = get_projected_depths(res, K, meshed_normed_light_cam[:,3], device=device)
    batched_shadow_map = generate_shadow_map(wl_, w_light_bounded_, delta=delta, epsilon=epsilon, 
                                            new_min=new_min, new_max=new_max, 
                                            sigmoid=sigmoid, mode=mode, device=device)

    return batched_shadow_map


def get_normed_w(camera, pixel_depth, device='cpu'):
    """
    pixel_depth:  Must be a of shape (num_rays, 4) array with pixel locations 
    and homonogenous coord (1) in addition to depth in the 4th column. [i, j, 1, depth]
    """
    transformed_pixels_ = pixel_depth[:,:3]# (num_rays, 3)
    transformed_pixels = transformed_pixels_[...,None, :] # (num_rays, 1, 3)
    # print(transformed_pixels.shape, camera.camera.shape)
    coords = transformed_pixels.to(device) * camera.camera.to(device) # (num_rays, 3, 3)
    coords = torch.sum(coords, -1) # (num_rays, 3)
    norm = torch.linalg.norm(coords, dim=1) # (num_rays)
    norm = norm + EPSILON * torch.ones_like(norm)
    normed_depth = pixel_depth[:,3]/norm # (num_rays)
    ret = torch.cat([transformed_pixels_, normed_depth.view(-1,1)], dim=1)
    # print("ret.shape", ret.shape)
    return ret 
    

def get_diff_projections(pixels, w_cam, from_camera, to_camera, device='cpu'):
    """
    pixels: [num_rays, 3] usually just meshed_range_cam[:,:3]
    w_cam: [num_rays] usually just meshed_range_cam[:,3]
    """
#     i, j, k = torch.unbind(pixels, axis=1)
#     print(torch.stack([j,i], axis=1).shape)
#     transformed_pixels = torch.stack([j,i, k], axis=1)[...,None, :] # (num_rays, 1, 3)
    transformed_pixels = pixels[...,None, :] # (num_rays, 1, 3)
    R, Q = from_camera.get_transformation_to(to_camera, device)
    proj = transformed_pixels * R
    proj = torch.sum(proj, -1)
    w_cam_stacked = torch.stack([w_cam, w_cam, w_cam], axis=1)
    coords = w_cam_stacked * proj + Q
    ul, vl, wl = torch.unbind(coords, dim=1)
    ul = torch.div(ul,wl)
    vl = torch.div(vl,wl)
    K = torch.stack([ul,vl,wl], axis=1)
    return K

def get_projected_depths(res, K, w_light, device='cpu'):
    """
    res: (w,h) Resolution of the depth map rendered from the light's perspective! 
    K: [num_rays,3]
    w_light: [w*h] 
    """
    w,h = res
    ul_, vl_, wl = torch.unbind(K, dim=1)
    ul = torch.maximum(torch.tensor(0.).to(device), ul_)
    ul = torch.minimum(torch.tensor(w-1.).to(device), ul) 
    vl = torch.maximum(torch.tensor(0.).to(device), vl_)
    vl = torch.minimum(torch.tensor(h-1.).to(device), vl)

    w_light_bounded = w_light.view(w,h)[vl.to(torch.long),ul.to(torch.long)]

    return wl, w_light_bounded


def generate_shadow_map(wl, w_light_bounded, delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, 
                        sigmoid=False, mode='shadow_method_1', device='cpu'):
    """
    Generates shadow map based on wl&w_light_bounded. 
    mode: 
        `shadow_method_1` generates crisp shadows with no smoothness (w/ aliasing on edges)
            set delta, epsilon
        `shadow_method_2` generates smooth shadows with lower intesities
            set new_min, new_max, sigmoid
            Leads to nans as it is normalizing between 0 and 1, requires small learning_rate
    """
    diff = (wl-w_light_bounded)

    diff = diff.to(device)
    if mode == 'shadow_method_1':
        diff = diff/delta
        diff = torch.max(diff, torch.tensor(epsilon).to(device))
    elif mode == 'shadow_method_2':
        # diff = torch.abs(wl-w_light_bounded) # not sure why but the abs gets rid of the problem below..
        diff = normalize_min_max(diff) # makes it smooth but have > 0 values everywhere  
        if sigmoid: 
            diff = torch.nn.functional.sigmoid(diff)
    else: 
        raise ValueError("{} not found".format(mode))

    differntiable_shadow_img = torch.stack([diff,diff,diff], dim=1)
    differntiable_shadow_img = differntiable_shadow_img.clip(0.0,1.0) 
    return differntiable_shadow_img
###
# END OF DIFFERENTIABLE PIPELINE
###