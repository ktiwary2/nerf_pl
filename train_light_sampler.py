import os, sys

import imageio
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import ray_utils
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering_shadows import render_rays, efficient_sm, get_K
from models.efficient_shadow_mapping import normalize_min_max, get_normed_w, generate_shadow_map
# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# torch.autograd.set_detect_anomaly(True)

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]
        self.current_light_depth_cnt = 0
        if self.hparams.grad_on_light:
                print("Calculating gradient on light, we will calculate the light depth map every iteration.")
                self.hparams.sample_light_depth_every = 1

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays'].view(-1, 8) # (B, 8)
        rgbs = batch['rgbs'].view(-1, 3) # (B, 3)
        cam_pixels = batch['pixels'].view(-1,3)
        light_rays = batch['light_rays'].view(-1, 8) # (B, 8)
        light_ppc = batch['light_ppc'] # dict
        light_pixels = batch['light_pixels'].view(-1,3)

        ppc = batch['ppc'] # dict
        # print("rays.shape {}, rgb.shape {}".format(rays.shape, rgbs.shape))
        # print("light_rays: {}, light_ppc: {}".format(light_rays.shape, light_ppc))
        # print("ppc: {}".format(ppc))
        return rays, rgbs, cam_pixels, light_rays, light_pixels, light_ppc, ppc

    def forward(self, rays, N_importance, were_gradients_computed=True):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            N_importance=N_importance,
                            chunk=self.hparams.chunk, # chunk size is effective in val mode
                            white_back = self.train_dataset.white_back, 
                            were_gradients_computed=were_gradients_computed)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            # print('start', k, v)
            results[k] = torch.cat(v, 0)
            # print('end', results[k].shape)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        print("Using {} shadow DataLoader (hardcoded)".format(self.hparams.dataset_name))
        if self.hparams.dataset_name not in ['efficient_sm', 'pyredner2']:
            raise ValueError("{} not allowed ".format(self.hparams.dataset_name))
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh), 
                  'hparams': self.hparams
                  }
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)
        ##### Set these since they are constant
        self.l2w = self.train_dataset.l2w
        self.light_focal = self.train_dataset.light_camera_focal
        self.light_pixels = self.train_dataset.light_pixels.view(-1,3)
        self.light_ppc = self.train_dataset.light_ppc
        self.light_near = self.train_dataset.light_near
        self.light_far = self.train_dataset.light_far

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=False, # SET TO False for faster inference !!!
                          num_workers=0,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    # def get_light_depth_map(self, light_pixels, light_rays):


    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs, cam_pixels, _, _, _, ppc = self.decode_batch(batch)

        # everything here should be num_rays big
        assert len(ppc['eye_pos']) == len(ppc['camera'])
        assert len(ppc['eye_pos']) == rgbs.shape[0]

        cam_results = self(rays, N_importance=self.hparams.N_importance)
        proj_maps_coarse, proj_maps_fine = get_K(cam_pixels, cam_results, ppc, self.light_ppc, (self.hparams.N_importance>0))
        w,h = self.hparams.img_wh
        print(proj_maps_coarse.shape)
        device = proj_maps_coarse.device
        if proj_maps_fine is not None: 
            ul_, vl_, wl = torch.unbind(proj_maps_fine, dim=1)
            ul = torch.maximum(torch.tensor(0.).to(device), ul_)
            ul = torch.minimum(torch.tensor(w-1.).to(device), ul).long()
            vl = torch.maximum(torch.tensor(0.).to(device), vl_)
            vl = torch.minimum(torch.tensor(h-1.).to(device), vl).long()
        else:
            ul_, vl_, wl = torch.unbind(proj_maps_coarse, dim=1)
            ul = torch.maximum(torch.tensor(0.).to(device), ul_)
            ul = torch.minimum(torch.tensor(w-1.).to(device), ul).long()
            vl = torch.maximum(torch.tensor(0.).to(device), vl_)
            vl = torch.minimum(torch.tensor(h-1.).to(device), vl).long()

        print("ul, vl", ul, vl, wl)
        self.curr_Light_N_importance = self.hparams.Light_N_importance
        # get light rays from ul,vl
        light_directions = torch.stack([(ul-w/2)/self.light_focal, -(vl-h/2)/self.light_focal, -torch.ones_like(ul)], -1) # (H, W, 3)
        # print("light directions", light_directions.shape)
        rays_o, rays_d = ray_utils.get_rays(light_directions.to(rgbs.device), self.l2w.to(rgbs.device)) # num rays
        light_rays = torch.cat([rays_o, rays_d, 
                                    self.light_near*torch.ones_like(rays_o[:, :1]),
                                    self.light_far*torch.ones_like(rays_o[:, :1])],
                                    1).view(-1, 8).to(rgbs.device)
        # print("light_rays", light_rays.shape)
        i = ul.float() + 0.5 #.unsqueeze(2) 
        j = vl.float()+ 0.5 #.unsqueeze(2)
        light_pixels = torch.stack([i,j, torch.ones_like(i)], axis=-1).view(-1, 3).to(rgbs.device) # (H*W,3)
        # print("light_pixels", light_pixels.shape, light_pixels.shape)
        curr_light_results = self(light_rays.to(rgbs.device), 
                            N_importance=self.curr_Light_N_importance, 
                            were_gradients_computed=False)

        if self.curr_Light_N_importance > 0:
            range_light = curr_light_results['depth_fine']
        else:
            range_light = curr_light_results['depth_coarse']

        mesh_range_light = torch.cat([light_pixels, range_light.view(-1,1)], dim=1).to(rgbs.device)
        w_light = get_normed_w(self.light_ppc, mesh_range_light, device=rgbs.device).to(rgbs.device)
        # print("w_light", w_light.shape, wl.shape)
        sm = generate_shadow_map(wl, w_light[:,3], mode=self.hparams.shadow_method, delta=1e-2, 
                            epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False).to(rgbs.device)
        if self.curr_Light_N_importance > 0 :
            cam_results['rgb_coarse'] = sm
            cam_results['fine'] = sm
        else:
            cam_results['rgb_coarse'] = sm

        # if (self.current_light_depth_cnt % self.hparams.sample_light_depth_every == 0) and (cam_results['rgb_coarse'].shape[0] > 5):
        #     print(shadow_maps_coarse[:5,:]) # only print the first elements 

        log['train/loss'] = loss = self.loss(cam_results, rgbs)
        typ = 'fine' if 'rgb_fine' in cam_results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(cam_results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        # print("---------------Starting Validation---------------")
        rays, rgbs, cam_pixels, _, _, _, ppc = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W,3)
        rgbs = rgbs.squeeze() # (H*W,3)
        # print("---------rgbs.shape", rgbs.shape, rays.shape)
        # raise
        with torch.no_grad():
            cam_results = self(rays, N_importance=self.hparams.N_importance)
            rays = None
            proj_maps_coarse, proj_maps_fine = get_K(cam_pixels, cam_results, ppc, self.light_ppc, (self.hparams.N_importance>0))
            w,h = self.hparams.img_wh
            print(proj_maps_coarse.shape)
            device = proj_maps_coarse.device
            if proj_maps_fine is not None: 
                ul_, vl_, wl = torch.unbind(proj_maps_fine, dim=1)
                ul = torch.maximum(torch.tensor(0.).to(device), ul_)
                ul = torch.minimum(torch.tensor(w-1.).to(device), ul).long()
                vl = torch.maximum(torch.tensor(0.).to(device), vl_)
                vl = torch.minimum(torch.tensor(h-1.).to(device), vl).long()
            else:
                ul_, vl_, wl = torch.unbind(proj_maps_coarse, dim=1)
                ul = torch.maximum(torch.tensor(0.).to(device), ul_)
                ul = torch.minimum(torch.tensor(w-1.).to(device), ul).long()
                vl = torch.maximum(torch.tensor(0.).to(device), vl_)
                vl = torch.minimum(torch.tensor(h-1.).to(device), vl).long()

            print("ul, vl", ul, vl)
            self.curr_Light_N_importance = self.hparams.Light_N_importance
            # get light rays from ul,vl
            light_directions = torch.stack([(ul-w/2)/self.light_focal, -(vl-h/2)/self.light_focal, -torch.ones_like(ul)], -1) # (H, W, 3)
            print("light directions", light_directions.shape)
            rays_o, rays_d = ray_utils.get_rays(light_directions.to(rgbs.device), self.l2w.to(rgbs.device)) # num rays
            light_rays = torch.cat([rays_o, rays_d, 
                                        self.light_near*torch.ones_like(rays_o[:, :1]),
                                        self.light_far*torch.ones_like(rays_o[:, :1])],
                                        1).view(-1, 8).to(rgbs.device)
            print("light_rays", light_rays)
            i = ul.float() + 0.5 #.unsqueeze(2) 
            j = vl.float()+ 0.5 #.unsqueeze(2)
            light_pixels = torch.stack([i,j, torch.ones_like(i)], axis=-1).view(-1, 3).to(rgbs.device) # (H*W,3)
            print("light_pixels", light_pixels, light_pixels.shape)
            curr_light_results = self(light_rays.to(rgbs.device), 
                                N_importance=self.curr_Light_N_importance, 
                                were_gradients_computed=False)

            if self.curr_Light_N_importance > 0:
                range_light = curr_light_results['depth_fine']
            else:
                range_light = curr_light_results['depth_coarse']

            mesh_range_light = torch.cat([light_pixels, range_light.view(-1,1)], dim=1).to(rgbs.device)
            w_light = get_normed_w(self.light_ppc, mesh_range_light, device=rgbs.device).to(rgbs.device)
            print("w_light", w_light.shape, wl.shape)
            sm = generate_shadow_map(wl, w_light[:,3], mode=self.hparams.shadow_method, delta=1e-2, 
                                epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False).to(rgbs.device)
            if self.curr_Light_N_importance > 0 :
                cam_results['rgb_coarse'] = sm
                cam_results['fine'] = sm
            else:
                cam_results['rgb_coarse'] = sm
        log = {'val_loss': self.loss(cam_results, rgbs)}
        typ = 'fine' if 'rgb_fine' in cam_results else 'coarse'
    
        if batch_nb == 0:
            print("---------------Evaluating and saving Images!---------------")
            W, H = self.hparams.img_wh
            img = cam_results[f'rgb_{typ}'].view(H, W, 3).cpu()
            rgb8 = to8b(img.numpy())
            gt8 = to8b(rgbs.view(H, W, 3).cpu().numpy())
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            disp = normalize_min_max(cam_results[f'disp_map_{typ}'].view(H, W))
            disp8 = to8b(disp.cpu().numpy())
            depth8 = visualize_depth(cam_results[f'depth_{typ}'].view(H, W), to_tensor=False) 
            depth = visualize_depth(cam_results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            if not os.path.exists(f'light_sampler_logs/logs/{self.hparams.exp_name}/imgs'):
                os.mkdir(f'light_sampler_logs/logs/{self.hparams.exp_name}/imgs')
            filename = os.path.join(f'light_sampler_logs/logs/{self.hparams.exp_name}/imgs', 'gt_{:03d}.png'.format(self.current_epoch))
            imageio.imwrite(filename, gt8)
            filename = os.path.join(f'light_sampler_logs/logs/{self.hparams.exp_name}/imgs', 'rgb_{:03d}.png'.format(self.current_epoch))
            imageio.imwrite(filename, rgb8)
            filename = os.path.join(f'light_sampler_logs/logs/{self.hparams.exp_name}/imgs', 'depth_{:03d}.png'.format(self.current_epoch))
            imageio.imwrite(filename, depth8)
            # save disp
            filename = os.path.join(f'light_sampler_logs/logs/{self.hparams.exp_name}/imgs', 'disp_{:03d}.png'.format(self.current_epoch))
            imageio.imwrite(filename, disp8)


            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_psnr'] = psnr(cam_results[f'rgb_{typ}'], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'light_sampler_logs/ckpts/{hparams.exp_name}',
                                                                '{epoch:d}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5,)

    logger = TestTubeLogger(
        save_dir="light_sampler_logs/logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if len(hparams.num_gpus)>1 else None,
                      num_sanity_val_steps=hparams.num_sanity_val_steps,
                      benchmark=True,
                      profiler=hparams.num_gpus==1, 
                      auto_scale_batch_size=False)

    trainer.fit(system)
