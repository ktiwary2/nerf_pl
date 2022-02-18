import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from models.camera import Camera
from .ray_utils import *
from tqdm import tqdm

class BlenderVariableLightDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800), hparams=None):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        self.hparams = hparams

        self.read_meta()
        self.white_back = False

        if not hparams:
            raise ValueError("hparams cannot be none")

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                            #    f"transforms_train.json"), 'r') as f:
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.cam_hfov_deg = self.meta['camera_angle_x'] * 180./np.pi
        self.light_hfov_deg = self.meta['light_angle_x'] * 180./np.pi
        
        if 'box' or 'bunny' or 'vase' in self.root_dir:
            res = 200 # these imgs have original size of 200 
        else:
            res = 800
        print("-------------------------------")
        print("RESOLUTION OF THE ORIGINAL IMAGE IS SET TO {}".format(res))
        print("-------------------------------")
        
        self.focal = 0.5*res/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=res
        self.light_focal = 0.5*res/np.tan(0.5*self.meta['light_angle_x']) # original focal length
                                                                     # when W=res

        self.focal *= self.img_wh[0]/res # modify focal length to match size self.img_wh
        self.light_focal *= self.img_wh[0]/res # modify focal length to match size self.img_wh
        # bounds, common for all scenes
        self.near = 1.0
        self.far = 100.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
        self.light_directions = get_ray_directions(h, w, self.light_focal) # (h, w, 3)

        ###### Figure out the Pixels rays originated from
        pixels_u = torch.arange(0, w, 1)
        pixels_v = torch.arange(0, h, 1)
        i, j = np.meshgrid(pixels_v.numpy(), pixels_u.numpy(), indexing='xy')
        i = torch.tensor(i) + 0.5 #.unsqueeze(2) 
        j = torch.tensor(j)+ 0.5 #.unsqueeze(2)
        self.imgwh_pixels = torch.stack([i,j, torch.ones_like(i)], axis=-1).view(-1, 3) # (H*W,3)

        ##################
        # new_frames = []
        # # only do on a single image
        # print("Unit testing on a single Image...")
        # for frame in self.meta['frames']:
        #     if 'r_129' in frame['file_path']:
        #         a = [frame, frame, frame, frame, frame]
        #         new_frames.extend(a * 10)
        #         break
        # self.meta['frames']  = new_frames
        # print("--> meta_frames: {}".format(len(self.meta['frames'])))
        ##################
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
            self.ligth_poses = []
            self.all_cam_rays = []
            self.all_light_rays = []
            self.all_rgbs = []
            self.all_cam_ppc = []
            self.all_light_ppc = []
            self.all_sms = []
            self.pixels = []
            print("Creating Dataset...")
            # idx = 0 
            # new_frames = []
            # # only do on a single image
            # for frame in self.meta['frames']:
            #     if 'r_95' in frame['file_path']:
            #         a = [frame, frame, frame, frame, frame]
            #         new_frames.extend(a * 10)
            #         break
            # self.meta['frames'] = new_frames
            for frame in tqdm(self.meta['frames']):
                ###### load the RGB+SM Image
                file_path = frame['file_path'].split('/')
                sm_file_path = 'sm_'+ file_path[-1]
                sm_path = os.path.join(self.root_dir, f"{sm_file_path}.png")
                rgb_path = os.path.join(self.root_dir, f"{file_path[-1]}.png")

                ## Continue if not os.path.exists(shadows)
                if not os.path.exists(sm_path):
                    continue

                self.image_paths += [sm_path]

                print("Processing Frame: {}".format(rgb_path))
                ###### Camera Pose 
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                # Create Camera() from this transform
                cam_ppc = Camera(self.cam_hfov_deg, (w,h))
                cam_ppc.set_pose_using_blender_matrix(c2w)
                # per_pix_cam_ppc = [cam_ppc] * h * w
                self.all_cam_ppc.append(cam_ppc)
                ###### Light Pose
                light_pose = np.array(frame['light_transform'])[:3, :4]
                self.ligth_poses += [light_pose]
                l2w = torch.FloatTensor(light_pose)
                # Create Camera() from this transform
                light_ppc = Camera(self.light_hfov_deg, (w,h))
                light_ppc.set_pose_using_blender_matrix(l2w)
                # per_pix_light_ppc = [light_ppc] * h * w
                self.all_light_ppc.append(light_ppc)
                
                img = Image.open(rgb_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

                sm = Image.open(sm_path)
                sm = sm.resize(self.img_wh, Image.LANCZOS)
                if not (self.hparams.blur == -1):
                    sm = sm.filter(ImageFilter.GaussianBlur(self.hparams.blur))

                sm = self.transform(sm) # (3, h, w)
                sm = sm.view(3, -1).permute(1, 0) # (h*w, 3) RGBA

                self.all_rgbs += [img.unsqueeze(0)] # (1, h*w, 3)
                self.all_sms += [sm.unsqueeze(0)] # (1, h*w, 3)
                
                ###### Camera+Light Directions 
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                light_rays_o, light_rays_d = get_rays(self.light_directions, l2w) # both (h*w, 3)
                self.all_cam_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1).unsqueeze(0)] # (h*w, 8)

                self.all_light_rays += [torch.cat([light_rays_o, light_rays_d, 
                                             self.near*torch.ones_like(light_rays_o[:, :1]),
                                             self.far*torch.ones_like(light_rays_o[:, :1])],
                                             1).unsqueeze(0)] # (h*w, 8)

                ###### Add Pixels to self.pixels 
                self.pixels += [self.imgwh_pixels.unsqueeze(0)]

            self.all_cam_rays = torch.cat(self.all_cam_rays, 0)     # (len(self.meta['frames]), h*w, 8)
            self.all_light_rays = torch.cat(self.all_light_rays, 0) # (len(self.meta['frames]), h*w, 8)
            self.all_sms = torch.cat(self.all_sms, 0)               # (len(self.meta['frames]), h*w, 3)
            self.pixels = torch.cat(self.pixels, 0)                 # (len(self.meta['frames]), h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)             # (len(self.meta['frames]), h*w, 3)

            print("------")
            print("Shapes of Dataset: ")
            print(self.all_cam_rays.shape, self.all_light_rays.shape, self.all_sms.shape, self.pixels.shape, self.all_rgbs.shape)
            print("------")
            if not self.hparams.process_rgb: 
                self.all_rgbs = None

            if self.hparams.batch_size >= h*w :
                print("Note: Setting batch size to h*w {}. Not going to sample rays.".format(h*w))
                self.n_rand = h*w
                self.sub_sample_rays = False
            else:
                self.sub_sample_rays = True
                self.n_rand = self.hparams.batch_size
            
            # assert len(self.meta['frames']) == len(self.all_light_rays)
            # assert len(self.meta['frames']) == len(self.all_cam_rays)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return self.all_light_rays.shape[0]
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        """
        Guarantees that all cam_pixels belong to a single cam_ppc...
        """
        if self.split == 'train': # use data in the buffers

            cam_rays = self.all_cam_rays[idx] # (H*W, 3)
            light_rays = self.all_light_rays[idx] # (h*w, 8)
            sms = self.all_sms[idx] # (H*W, 3)
            # convert to black and white
            # print(sms.shape, sms[:,0].shape)
            sms = sms[:,0] + sms[:,1] + sms[:,2]
            sms = sms/3. # (h*w)
            # print(sms.shape, sms[:,0].shape)
            # raise ValueError('here')
            cam_ppc = self.all_cam_ppc[idx] # (1)
            light_ppc = self.all_light_ppc[idx] # (1)
            cam_pixels = self.pixels[idx] # (h*w, 3)
            
            if self.sub_sample_rays:
                if self.hparams.use_prob_weighting:
                    prob = (sms-torch.min(sms))/(torch.max(sms)-torch.min(sms))
                    random_rays_idx = np.random.choice(cam_rays.shape[0], size=self.n_rand, replace=False, p=prob)  # (batch_size,)
                else:
                    random_rays_idx = np.random.choice(cam_rays.shape[0], size=self.n_rand, replace=False)  # (batch_size,)
                batch_cam_rays = cam_rays[random_rays_idx] # (batch_size, 8)
                batch_sms = sms[random_rays_idx] # (batch_size, 3)
                batch_pixels = cam_pixels[random_rays_idx] # (batch_size, 3)
            else:
                batch_cam_rays = cam_rays
                batch_sms = sms
                batch_pixels = cam_pixels

            if self.hparams.process_rgb: 
                rgbs = self.all_rgbs[idx] # (h*w, 3)
                if self.sub_sample_rays:
                    batch_rgbs = rgbs[random_rays_idx]
                else:
                    batch_rgbs = rgbs

                sample = {'cam_rays': batch_cam_rays, # (batch_size, 3)
                          'light_rays': light_rays, # (h*w, 3)
                          'cam_pixels': batch_pixels, # (batch_size, 3)
                          'light_pixels': self.imgwh_pixels, # (h*w, 3)
                          'rgbs': batch_rgbs, # (batch_size, 3)
                          'sms': batch_sms, # (batch_size, 3)
                          'cam_ppc_eye_pos': cam_ppc.eye_pos, # (1)
                          'cam_ppc_camera': cam_ppc.camera, # (1)
                          'light_ppc_eye_pos': light_ppc.eye_pos, # (1)
                          'light_ppc_camera': light_ppc.camera, # (1)
                          }
            else:
                sample = {'cam_rays': batch_cam_rays, # (batch_size, 3)
                          'light_rays': light_rays, # (h*w, 3)
                          'cam_pixels': batch_pixels, # (batch_size, 3)
                          'light_pixels': self.imgwh_pixels, # (h*w, 3)
                          'sms': batch_sms, # (batch_size, 3)
                          'cam_ppc_eye_pos': cam_ppc.eye_pos, # (1)
                          'cam_ppc_camera': cam_ppc.camera, # (1)
                          'light_ppc_eye_pos': light_ppc.eye_pos, # (1)
                          'light_ppc_camera': light_ppc.camera, # (1)
                          }

        else: # create data for each image separately
            w, h = self.img_wh
            frame = self.meta['frames'][idx]
            pose = np.array(frame['transform_matrix'])[:3, :4]
            c2w = torch.FloatTensor(pose)

            # Create Camera() from this transform
            cam_ppc = Camera(self.cam_hfov_deg, (w,h))
            cam_ppc.set_pose_using_blender_matrix(c2w)
            # cam_ppc_eye_pos = [cam_ppc.eye_pos] * h * w
            # cam_ppc_camera = [cam_ppc.camera] * h * w
            ###### Light Pose
            light_pose = np.array(frame['light_transform'])[:3, :4]
            l2w = torch.FloatTensor(light_pose)
            # Create Camera() from this transform
            light_ppc = Camera(self.light_hfov_deg, (w,h))
            light_ppc.set_pose_using_blender_matrix(l2w)
            # light_ppc_eye_pos = [light_ppc.eye_pos] * h * w
            # light_ppc_camera = [light_ppc.camera] * h * w

            ###### load the RGB+SM Image
            file_path = frame['file_path'].split('/')
            sm_file_path = 'sm_'+ file_path[-1]
            sm_path = os.path.join(self.root_dir, f"{sm_file_path}.png")
            rgb_path = os.path.join(self.root_dir, f"{file_path[-1]}.png")
            print("Validation Processing Frame: {}".format(rgb_path))
            img = Image.open(rgb_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, h, w)
            img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            rgbs = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            sm = Image.open(sm_path)
            sm = sm.resize(self.img_wh, Image.LANCZOS)
            if not (self.hparams.blur == -1):
                sm = sm.filter(ImageFilter.GaussianBlur(self.hparams.blur))

            sm = self.transform(sm) # (3, h, w)
            sms = sm.view(3, -1).permute(1, 0) # (h*w, 3) RGBA
            sms = sms[:,0] + sms[:,1] + sms[:,2]
            sms = sms/3. # (h*w)
            
            ###### Camera+Light Directions 
            rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
            light_rays_o, light_rays_d = get_rays(self.light_directions, l2w) # both (h*w, 3)
            cam_rays = torch.cat([rays_o, rays_d, 
                                  self.near*torch.ones_like(rays_o[:, :1]),
                                  self.far*torch.ones_like(rays_o[:, :1])],
                                  1) # (h*w, 8)

            light_rays = torch.cat([light_rays_o, light_rays_d, 
                                     self.near*torch.ones_like(light_rays_o[:, :1]),
                                     self.far*torch.ones_like(light_rays_o[:, :1])],
                                     1) # (h*w, 8)

            if self.hparams.process_rgb: 
                sample = {'cam_rays': cam_rays,
                          'light_rays': light_rays, 
                          'cam_pixels': self.imgwh_pixels, # (h*w, 3)
                          'light_pixels': self.imgwh_pixels, # (h*w, 3)
                          'rgbs': rgbs, 
                          'sms': sms, 
                          'cam_ppc_eye_pos': cam_ppc.eye_pos,
                          'cam_ppc_camera': cam_ppc.camera,
                          'light_ppc_eye_pos': light_ppc.eye_pos,
                          'light_ppc_camera': light_ppc.camera,
                          }
            else:
                sample = {'cam_rays': cam_rays,
                          'light_rays': light_rays, 
                          'cam_pixels': self.imgwh_pixels, # (h*w, 3)
                          'light_pixels': self.imgwh_pixels, # (h*w, 3)
                          'sms': sms, 
                          'cam_ppc_eye_pos': cam_ppc.eye_pos,
                          'cam_ppc_camera': cam_ppc.camera,
                          'light_ppc_eye_pos': light_ppc.eye_pos,
                          'light_ppc_camera': light_ppc.camera,
                          }

        return sample