import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageFilter
from torchvision import transforms as T
from models.camera import Camera
from tqdm import tqdm

from .ray_utils import *

class BlenderEfficientShadows(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800), hparams=None):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        print("Training Image size:", img_wh)
        self.define_transforms()

        self.white_back = True
        # self.white_back = False # Setting it to False (!)
        self.hparams = hparams
        self.black_and_white = False
        if self.hparams is not None and self.hparams.black_and_white_test:
            self.black_and_white = True
        self.read_meta()
        self.hparams.coords_trans = False
        print("------------")
        print("NOTE: self.hparams.coords_trans is set to {} ".format(self.hparams.coords_trans))
        print("------------")

    def read_meta(self):
        # self.split = 'train'
        with open(os.path.join(self.root_dir,
                            #    f"transforms_train.json"), 'r') as f:
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        print("Root Directory: ".format(self.root_dir))
        # if 'bunny' or 'box' or 'vase' in self.root_dir:
        #     res = 200 # these imgs have original size of 200 
        # else:
        #     res = 800

        res = 800
        if 'resolution' in self.meta.keys():
            res = self.meta['resolution']

        print("-------------------------------")
        print("RESOLUTION OF THE ORIGINAL IMAGE IS SET TO {}".format(res))
        print("-------------------------------")

        self.focal = 0.5*res/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=res
        self.focal *= self.img_wh[0]/res # modify focal length to match size self.img_wh

        ################
        self.light_camera_focal = 0.5*res/np.tan(0.5*self.meta['light_camera_angle_x']) # original focal length
        ################
        # if 'bunny' or 'box' or 'vase' in self.root_dir:
        #     self.light_camera_focal = 0.5*res/np.tan(0.5*self.meta['light_angle_x']) # original focal length
        # else:
        #     self.light_camera_focal = 0.5*res/np.tan(0.5*self.meta['light_camera_angle_x']) # original focal length
                                                                    #  when W=res
        self.light_camera_focal *= self.img_wh[0]/res # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 1.0
        self.far = 200.0

        # probably need to change this 
        self.light_near = 1.0
        self.light_far = 200.0

        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
        
        ### Light Camera Matrix 
        ################
        pose = np.array(self.meta['light_camera_transform_matrix'])[:3, :4]
        ################

        # if 'bunny' or 'box' or 'vase' in self.root_dir:
        #     self.meta['light_angle_x'] = 0.5 * self.meta['light_angle_x']
        #     print("Changing the HFOV of Light")
        #     pose = np.array(self.meta['frames'][0]['light_transform'])[:3, :4]
        # else:
        #     pose = np.array(self.meta['light_camera_transform_matrix'])[:3, :4]

        self.l2w = torch.FloatTensor(pose)

        pixels_u = torch.arange(0, w, 1)
        pixels_v = torch.arange(0, h, 1)
        i, j = np.meshgrid(pixels_v.numpy(), pixels_u.numpy(), indexing='xy')
        i = torch.tensor(i) + 0.5 #.unsqueeze(2) 
        j = torch.tensor(j)+ 0.5 #.unsqueeze(2)
        self.light_pixels = torch.stack([i,j, torch.ones_like(i)], axis=-1).view(-1, 3) # (H*W,3)

        light_directions = get_ray_directions(h, w, self.light_camera_focal) # (h, w, 3)
        rays_o, rays_d = get_rays(light_directions, self.l2w) # both (h*w, 3)
        self.light_rays = torch.cat([rays_o, rays_d, 
                                        self.light_near*torch.ones_like(rays_o[:, :1]),
                                        self.light_far*torch.ones_like(rays_o[:, :1])],
                                        1) # (h*w, 8)

        ################
        hfov = self.meta['light_camera_angle_x'] * 180./np.pi
        ################

        # if 'bunny' or 'box' or 'vase' in self.root_dir:
        #     hfov = self.meta['light_angle_x'] * 180./np.pi
        # else:
        #     hfov = self.meta['light_camera_angle_x'] * 180./np.pi

        self.light_ppc = Camera(hfov, (h, w))
        self.light_ppc.set_pose_using_blender_matrix(self.l2w, self.hparams.coords_trans)
        print("LIGHT: c2w: {}\n, camera:{}\n, eye:{}\n".format(self.l2w, self.light_ppc.camera, self.light_ppc.eye_pos))


        ### Light Camera Matrix 

        # new_frames = []
        # # only do on a single image
        # for frame in self.meta['frames']:
        #     if 'r_137' in frame['file_path']:
        #         a = [frame]
        #         new_frames.extend(a * 10)
        #         break
        
        # self.meta['frames'] = new_frames

        if self.split == 'val':
            new_frames = []
            for frame in self.meta['frames']:
                ###### load the RGB+SM Image
                file_path = frame['file_path'].split('/')
                sm_file_path = 'sm_'+ file_path[-1]
                sm_path = os.path.join(self.root_dir, f"{sm_file_path}.png")
                ## Continue if not os.path.exists(shadows)
                if not os.path.exists(sm_path):
                    continue
                else:
                    new_frames.append(frame)
            self.meta['frames']  = new_frames


        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_ppc = []
            self.all_pixels = []

            for frame in tqdm(self.meta['frames']):
                #### change it to load the shadow map
                file_path = frame['file_path'].split('/')
                file_path = 'sm_'+ file_path[-1]
                ################
                image_path = os.path.join(self.root_dir, f"{file_path}.png")
                self.image_paths += [image_path]
                ## Continue if not os.path.exists(shadows)
                if not os.path.exists(image_path):
                    continue
                print("Processing Frame {}".format(image_path))
                ##### 
                # real processing begins 
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)


                hfov = self.meta['camera_angle_x'] * 180./np.pi
                ppc = Camera(hfov, (h, w))
                ppc.set_pose_using_blender_matrix(c2w, self.hparams.coords_trans)
                self.all_ppc.extend([ppc]*h*w)

                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                if not self.hparams.blur == -1:
                    img = img.filter(ImageFilter.GaussianBlur(self.hparams.blur))

                img = self.transform(img) # (4, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 4) RGBA

                # Figure out where the rays originated from 
                pixels_u = torch.arange(0, w, 1)
                pixels_v = torch.arange(0, h, 1)
                i, j = np.meshgrid(pixels_v.numpy(), pixels_u.numpy(), indexing='xy')
                i = torch.tensor(i) + 0.5 #.unsqueeze(2) 
                j = torch.tensor(j)+ 0.5 #.unsqueeze(2)
                pixels = torch.stack([i,j, torch.ones_like(i)], axis=-1).view(-1, 3) # (H*W,3)

                rays_o, rays_d = get_rays(self.directions, c2w)
                rays = torch.cat([rays_o, rays_d, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_o[:, :1])],
                                1) # (H*W, 8)
                print("-------------------------------")
                print("frame: {}\n, c2w: {}\n, camera:{}\n, eye:{}\n".format(file_path, c2w, ppc.camera, ppc.eye_pos))
                print("-------------------------------")

                self.all_rgbs += [img]
                self.all_rays += [rays]
                self.all_pixels += [pixels]

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_pixels = torch.cat(self.all_pixels, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            print("self.all_rgbs.shape, self.all_rays.shape, self.all_pixels.shape, all_ppc.shape", 
                    self.all_rgbs.shape, self.all_rays.shape, self.all_pixels.shape, len(self.all_ppc))
            if not (float(self.hparams.white_pix) == -1):
                print("-------------------------- rgb max {}, min {}".format(self.all_rgbs.max(), self.all_rgbs.min()))
                print("only Training on pixels with shadow map values > 0.")
                all_bw = (self.all_rgbs[:,0] + self.all_rgbs[:,1] + self.all_rgbs[:,2])/3.
                idx = torch.where(all_bw > float(self.hparams.white_pix))
                self.all_rgbs = self.all_rgbs[idx]
                self.all_pixels = self.all_pixels[idx]
                self.all_rays = self.all_rays[idx]
                new_ppc = []
                for i in idx[0]:
                    new_ppc.append(self.all_ppc[i])
                self.all_ppc = new_ppc
                print("self.all_rgbs.shape, self.all_rays.shape, self.all_pixels.shape, all_ppc.shape", 
                        self.all_rgbs.shape, self.all_rays.shape, self.all_pixels.shape, len(self.all_ppc))

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        else:
            return len(self.meta['frames'])

    def __getitem__(self, idx):
        """
        Processes and return rays, rgbs PER image
        instead of on a ray by ray basis. Albeit slower, 
        Implementation of shadow mapping is easier this way.
        """
        if self.split == 'train': # use data in the buffers
            # pose = self.poses[idx]
            # c2w = torch.FloatTensor(pose)

            sample = {'rays': self.all_rays[idx], # (8) Ray originating from pixel (i,j)
                      'pixels': self.all_pixels[idx], # pixel where the ray originated from 
                      'rgbs': self.all_rgbs[idx], # (h*w,3)
                    #   'ppc': [self.all_ppc[idx].eye_pos, self.all_ppc[idx].camera], 
                    #   'light_ppc': [self.light_ppc.eye_pos, self.light_ppc.camera],
                      'ppc': {
                          'eye_pos': self.all_ppc[idx].eye_pos, 
                          'camera': self.all_ppc[idx].camera,
                      },
                      'light_ppc': {
                          'eye_pos': self.light_ppc.eye_pos, 
                          'camera': self.light_ppc.camera,
                      },
                    #   'c2w': pose, # (3,4)
                    # pixel where the light ray originated from  
                      'light_pixels': self.light_pixels, #(h*w, 3)  
                    # light rays 
                      'light_rays': self.light_rays, #(h*w,8)
                    }

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            file_path = frame['file_path'].split('/')
            file_path = 'sm_'+ file_path[-1]

            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            ###########
            w, h = self.img_wh
            hfov = self.meta['camera_angle_x'] * 180./np.pi
            ppc = Camera(hfov, (h, w))
            ppc.set_pose_using_blender_matrix(c2w, self.hparams.coords_trans)
            eye_poses = [ppc.eye_pos]*h*w
            cameras = [ppc.camera]*h*w

            ###########
            img = Image.open(os.path.join(self.root_dir, f"{file_path}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            if not self.hparams.blur == -1:
                img = img.filter(ImageFilter.GaussianBlur(self.hparams.blur))
            img = self.transform(img) # (3, H, W)
            img = img.view(3, -1).permute(1, 0) # (H*W, 3) RGBA
            # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            pixels_u = torch.arange(0, w, 1)
            pixels_v = torch.arange(0, h, 1)
            i, j = np.meshgrid(pixels_v.numpy(), pixels_u.numpy(), indexing='xy')
            i = torch.tensor(i) + 0.5 #.unsqueeze(2) 
            j = torch.tensor(j)+ 0.5 #.unsqueeze(2)
            pixels = torch.stack([i,j, torch.ones_like(i)], axis=-1).view(-1, 3) # (H*W,3)

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            # print("rays.shape", rays.shape)
            # valid_mask = (img[-1]>0).flatten() # (H*W) valid color area

            sample = {'rays': rays,
                      'pixels': pixels, # pixel where rays originated from 
                      'rgbs': img,
                      'ppc': {
                          'eye_pos': eye_poses, 
                          'camera': cameras,
                      },
                      'light_ppc': {
                          'eye_pos': self.light_ppc.eye_pos, 
                          'camera': self.light_ppc.camera,
                      },
                    # pixel where the light ray originated from  
                      'light_pixels': self.light_pixels, #(h*w, 3)  
                    # light rays 
                      'light_rays': self.light_rays, #(h*w,8)
                    }

        return sample