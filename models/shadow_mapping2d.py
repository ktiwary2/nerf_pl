import torch 
from numpy import meshgrid
###
# NOTE: (ktiwary) Currently only accepts Square Images! Will fix this at some point. 
# use_numpy_meshgrid: Since older versions of torch don't have indexing as an arg for torch..
###

EPSILON = 1e-5

def normalize_min_max(tensor, new_max=1.0, new_min=0.0):
     return (tensor - tensor.min())/(tensor.max() - tensor.min() + EPSILON)*(new_max - new_min) + new_min


def run_shadow_mapping(res, camera, light_cam, range_cam, range_light, device, mode, \
                       delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, use_numpy_meshgrid=True):
    """
    Run Shadow Mapping stack from start to finish in a differentiable manner. 
    Args: 
        res: 
        camera: 
        light_cam:
        range_cam: 
        range_light:
        device: 
        mode:
        etc. 
    """
    w_cam = differentiable_w_mapping(res, camera, range_cam, device=device, use_numpy_meshgrid=use_numpy_meshgrid)
    
    w_light = differentiable_w_mapping(res, light_cam, range_light, device=device, use_numpy_meshgrid=use_numpy_meshgrid)
    
    K = get_differentiable_projections(w_cam, res, camera, light_cam, device=device, use_numpy_meshgrid=use_numpy_meshgrid)
    
    wl, w_light_bounded, _, _ = get_differentiable_depths(res, K, w_light, device=device, 
                                                          use_numpy_meshgrid=use_numpy_meshgrid)
    
    shadow_map = generate_shadow_map(wl, w_light_bounded, device=device, mode=mode, delta=delta, epsilon=epsilon, 
                                     new_min=new_min, new_max=new_max, sigmoid=sigmoid)
    return shadow_map

def differentiable_w_mapping(res, camera, range_cam, device='cpu', use_numpy_meshgrid=False):
    """
    Compute Range -> 'w' values in a differentiable way 
    wrt input range values 
    res: (w,h)
    camera: Camera Class 
    range_cam: Range Values
    """
    w,h = res
    pixels_u = torch.arange(0, w, 1)
    pixels_v = torch.arange(0, h, 1) 
    if use_numpy_meshgrid:
        i, j = meshgrid(pixels_v.numpy(), pixels_u.numpy(), indexing='xy')
        i = torch.tensor(i).to(device)
        j = torch.tensor(j).to(device)
    else:
        i, j = torch.meshgrid(pixels_v, pixels_u, indexing='xy')
    transformed_i = i + 0.5
    transformed_j = j + 0.5
    pixels = torch.stack([transformed_i, transformed_j, torch.ones_like(i)], axis=-1)
    transformed_pixels = pixels[..., None, :] # expand dims 
    transformed_pixels = transformed_pixels.to(device)
    coords = transformed_pixels * camera.camera
    coords = torch.sum(coords, -1)
    # print("coords.shape", coords.shape)
    norm = torch.linalg.norm(coords, dim=2)
    # print("norm.shape", norm.shape)
    norm = norm.to(device)
    # print("range_cam, norm", range_cam.shape, norm.shape)
    w_cam = range_cam/norm
    return w_cam 

def get_differentiable_projections(w_from_camera, res, from_camera, to_camera, device='cpu', use_numpy_meshgrid=False):

    """
    Compute Range -> 'w' values in a differentiable way 
    wrt input range values 
    res: (w,h)
    camera: Camera Class 
    range_cam: Range Values
    """
    w,h = res
    pixels_u = torch.arange(0, w, 1)
    pixels_v = torch.arange(0, h, 1)
    if use_numpy_meshgrid:
        i, j = meshgrid(pixels_v.numpy(), pixels_u.numpy(), indexing='xy')
        i = torch.tensor(i).to(device)
        j = torch.tensor(j).to(device)
    else:
        i, j = torch.meshgrid(pixels_v, pixels_u, indexing='xy')    
    transformed_i = i + 0.5
    transformed_j = j + 0.5
    pixels = torch.stack([transformed_i, transformed_j, torch.ones_like(i)], axis=-1)
    transformed_pixels = pixels[..., None, :] # expand dims 
    transformed_pixels = transformed_pixels.to(device)
    R, Q = from_camera.get_transformation_to(to_camera, device)
    R = R.to(device)
    Q = Q.to(device)

    proj = transformed_pixels * R
    proj = torch.sum(proj, -1)
    w_cam_stacked = torch.stack([w_from_camera, w_from_camera, w_from_camera], axis=2)
    coords = w_cam_stacked * proj + Q
    ul, vl, wl = torch.unbind(coords, dim=2)
    ul = torch.div(ul,wl)
    vl = torch.div(vl,wl)
    K = torch.stack([ul, vl, wl], axis=2)
    return K

def get_differentiable_depths(res, K, w_light, device='cpu', use_numpy_meshgrid=False): 
    """
    Using a K matrix project u,v into light space to compare depths
    Output can be subtracted to get the shadow map. For example: 
        for v in range(h):
            for u in range(w):
                wl_ = wl[v,u]
                if wl[v,u] - w_light_bounded[v,u] > EPSILON: 
                    shadow_img[v,u] = [1.0, 1.0, 1.0]

    """
    w,h = res
    pixels_u = torch.arange(0, w, 1)
    pixels_v = torch.arange(0, h, 1)
    if use_numpy_meshgrid:
        i, j = meshgrid(pixels_v.numpy(), pixels_u.numpy(), indexing='xy')
        i = torch.tensor(i).to(device)
        j = torch.tensor(j).to(device)
    else:
        i, j = torch.meshgrid(pixels_v, pixels_u, indexing='xy')    

    cam_to_light_coords = K[j,i]
    ul_, vl_, wl = torch.unbind(cam_to_light_coords, dim=2)
#     print(ul_, vl_, wl)
#     ul = ul_.clamp(0.,w-1.)
#     vl = vl_.clamp(0.,h-1.) # after clamping this should be the same for all pixels in the image
    ul = torch.maximum(torch.tensor(0.).to(device), ul_)
    ul = torch.minimum(torch.tensor(w-1.).to(device), ul) 
    vl = torch.maximum(torch.tensor(0.).to(device), vl_)
    vl = torch.minimum(torch.tensor(h-1.).to(device), vl)
#     print(ul, vl, wl)

    # since we map everything below 0 to 0 and everything above w,h to w,h, we need to alter the w_light
#     w_light[0,0]= float('inf')  # (Not needed for some reason?)
#     w_light[w-1,h-1]= float('inf')
    
    ul = ul.to(torch.int64)
    vl = vl.to(torch.int64)
    w_light_bounded = w_light[vl, ul]
    
    return wl, w_light_bounded, ul_, vl_,

def generate_shadow_map(wl, w_light_bounded, delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, 
                        sigmoid=False, mode='shadow_method_1', device='cpu'):
    """
    Generates shadow map based on wl&w_light_bounded. 
    mode: 
        `shadow_method_1` generates crisp shadows with no smoothness (w/ aliasing on edges)
            set delta, epsilon
        `shadow_method_2` generates smooth shadows with lower intesities
            set new_min, new_max, sigmoid
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

    # differntiable_shadow_img = torch.stack([diff,diff,diff], dim=2)
    # differntiable_shadow_img = differntiable_shadow_img.clip(0.0,1.0) 
    return diff

###
# END OF DIFFERENTIABLE PIPELINE
###