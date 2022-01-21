"""
from https://github.com/sxyu/pixel-nerf/blob/a5a514224272a91e3ec590f215567032e1f1c260/src/data/SRNDataset.py
"""

import os
import torch
import json
import numpy as np
from PIL import Image
from .ray_utils import get_ray_directions, get_rays
from torchvision import transforms

def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor()]#, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return transforms.Compose(ops)

class PyRednerDataset(torch.utils.data.Dataset):
    """
    Dataset from PyRedner Dataset
    """

    def __init__(
        self, root_dir, split="train", img_wh=(128, 128), hparams=None):
        """
        :param root_dir root director
        :param split train | val | test
        :param image_size result image size (resizes if different)
        """
        super().__init__()
        self.root_dir = root_dir 

        print("Loading PyRedner dataset", self.root_dir)
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        print("Image Size set to: ", img_wh)


        assert os.path.exists(self.root_dir)

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.image_size = img_wh
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )
        # self._coord_trans = torch.ones((4,4), dtype=torch.float32)
        if not (input("cood_trans is set to: {}. PRESS y TO CONTINUE! ".format(self._coord_trans)) == 'y'):
            raise ValueError("cood_trans is set to: {}".format(self._coord_trans))

        self.split = split
        self.z_near = 1
        self.z_far = 200.0
        self.white_back = False
        self.lindisp = False
        self.hparams = hparams
        self.meta()

    def meta(self):
        with open(os.path.join(self.root_dir, "data.json")) as f:
            self.data = json.load(f)
            split = int(0.80 * len(self.data))
            if self.split == 'train':
                self.data = self.data[:split]
            else:
                self.data = self.data[split:]

        if self.split == 'train': # create buffer of all rays and rgb data
            w,h = self.image_size
            self.all_cam_rays = []
            self.all_light_rays = []
            self.all_rgbs = []
            self.cam_poses = []
            self.light_poses = [] 
            self.all_shadow_maps = []
            for dp in self.data: 
                c2w = torch.from_numpy(np.array(dp['c2w'])).to(torch.float32)[:3, :4]
                c2w = c2w @ self._coord_trans # TODO (ktiwary): is this necessary?? 
                self.cam_poses.append(c2w)

                # l2w = torch.from_numpy(np.array(dp['l2w'])).to(torch.float32)[:3, :4]
                # l2w = l2w @ self._coord_trans # TODO (ktiwary): is this necessary?? 
                # self.light_poses.append(l2w)

                # open Image 
                image_path = dp['rgb_cam_path']
                rgb = Image.open(image_path)
                rgb = rgb.resize(self.image_size, Image.LANCZOS)
                rgb = self.image_to_tensor(rgb)
                rgb = rgb.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs.append(rgb)
                # open Shadow Map
                # image_path = dp['shadow_path']
                # self.all_shadow_maps.append(image_path)
                # camera rays 
                f = 0.5 * w / np.tan(dp['camera_hfov']/2)
                directions = get_ray_directions(h, w, f) # (h, w, 3)
                cam_rays_o, cam_rays_d = get_rays(directions, c2w)
                # light rays 
                # f = 0.5 * w / np.tan(dp['light_hfov']/2)
                # directions = get_ray_directions(h, w, f) # (h, w, 3)
                # light_rays_o, light_rays_d = get_rays(directions, c2w)
                self.all_cam_rays += [torch.cat([cam_rays_o, cam_rays_d, 
                                                self.z_near*torch.ones_like(cam_rays_o[:, :1]),
                                                self.z_far*torch.ones_like(cam_rays_o[:, :1])],
                                                1)] # (h*w, 8)

                # self.all_light_rays += [torch.cat([light_rays_o, light_rays_d, 
                #                                  self.z_near*torch.ones_like(light_rays_o[:, :1]),
                #                                  self.z_far*torch.ones_like(light_rays_o[:, :1])],
                #                                  1)] # (h*w, 8)

            self.all_cam_rays = torch.cat(self.all_cam_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

            # self.all_light_rays = torch.stack(self.all_light_rays)
            print("Dataset Shapes cam_rays: {}, all_rgbs {} light_rays: None".format(
                    self.all_cam_rays.shape, self.all_rgbs.shape))


    def __len__(self):
        if self.split == 'train':
            return len(self.all_cam_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)


    def __getitem__(self, index):
        if self.split == 'train': # use data in the buffers
            rgb = self.all_rgbs[index]
            cam_rays = self.all_cam_rays[index]
            # light_rays = self.all_light_rays[index]

        else: # create data for each image separately
            w,h = self.image_size
            dp = self.data[index]
            c2w = torch.from_numpy(np.array(dp['c2w'])).to(torch.float32)[:3, :4]
            c2w = c2w @ self._coord_trans # TODO (ktiwary): is this necessary?? 
            image_path = dp['rgb_cam_path']

            rgb = Image.open(image_path)
            rgb = rgb.resize(self.image_size, Image.LANCZOS)
            rgb = self.image_to_tensor(rgb)
            rgb = rgb.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                
            f = 0.5 * w / np.tan(dp['camera_hfov']/2)
            directions = get_ray_directions(h, w, f) # (h, w, 3)
            cam_rays_o, cam_rays_d = get_rays(directions, c2w)


            cam_rays = torch.cat([cam_rays_o, cam_rays_d, 
                              self.z_near*torch.ones_like(cam_rays_o[:, :1]),
                              self.z_far*torch.ones_like(cam_rays_o[:, :1])],
                              1) # (H*W, 8)


        result = {
            "img_id": index,
            "rgb": rgb,
            "shadow_maps": torch.Tensor(), 
            "cam_ray_bundle": cam_rays,
            "light_ray_bundle": torch.Tensor(),
            "hw": self.image_size
        }

        return result



# python train_shadows.py --dataset_name pyredner --root_dir ../datasets/volumetric/single_box_pyredner/scene_1/ --N_importance 128 --N_samples 64 --num_gpus 2 --img_wh 400 400 --noise_std 0 --num_epochs 100 --batch_size 1024 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 --exp_name pyredner_test_1_coord_trans_true