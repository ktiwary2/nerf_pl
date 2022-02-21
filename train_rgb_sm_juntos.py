import os, sys

import imageio
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering_rgb_sm import render_rays, efficient_sm
from models.efficient_shadow_mapping import normalize_min_max

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
        self.sm_loss = loss_dict['sm']() # another instantiation of rgb_loss

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
        sm = batch['sm'].view(-1, 3) # (B, 3)
        cam_pixels = batch['pixels'].view(-1,3)
        ppc = batch['ppc'] # dict
        # light_rays = batch['light_rays'].view(-1, 8) # (B, 8)
        # light_ppc = batch['light_ppc'] # dict
        # light_pixels = batch['light_pixels'].view(-1,3)

        # print("rays.shape {}, rgb.shape {}".format(rays.shape, rgbs.shape))
        # print("light_rays: {}, light_ppc: {}".format(light_rays.shape, light_ppc))
        # print("ppc: {}".format(ppc))
        return rays, rgbs, sm, cam_pixels, ppc # light_rays, light_pixels, light_ppc

    def forward(self, rays, N_importance):
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
                            white_back = self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            # print('start', k, v)
            results[k] = torch.cat(v, 0)
            # print('end', results[k].shape)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        print("Using {} DataLoader (hardcoded)".format(self.hparams.dataset_name))
        if self.hparams.dataset_name not in ['rgb_sm']:
            raise ValueError("{} not allowed ".format(self.hparams.dataset_name))
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh), 
                  'hparams': self.hparams
                  }
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)
        ##### Set these since they are constant
        self.light_rays = self.train_dataset.light_rays
        self.light_pixels = self.train_dataset.light_pixels.view(-1,3)
        self.light_ppc = self.train_dataset.light_ppc

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


    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs, sm, cam_pixels, ppc = self.decode_batch(batch)

        # everything here should be num_rays big
        assert len(ppc['eye_pos']) == len(ppc['camera'])
        assert len(ppc['eye_pos']) == rgbs.shape[0]

        cam_results = self(rays, N_importance=self.hparams.N_importance)
        # rays = None

        if self.current_light_depth_cnt % self.hparams.sample_light_depth_every == 0:
            print("Updating Light's Depth Map at {}".format(self.current_light_depth_cnt))
            self.current_light_depth_cnt = 1
            if self.hparams.Light_N_importance == -1:
                self.curr_Light_N_importance = int(np.random.choice([0,8,16,32]))        
            else:
                self.curr_Light_N_importance = self.hparams.Light_N_importance

            if self.hparams.grad_on_light:
                print("Using Gradients on Light") 
                self.curr_light_results = self(self.light_rays.to(rgbs.device), 
                                N_importance=self.curr_Light_N_importance)
            else:
                with torch.no_grad():
                    # maybe only use coarse depth for light? no need for fine? 
                    self.curr_light_results = self(self.light_rays.to(rgbs.device), 
                                    N_importance=self.curr_Light_N_importance)
                    # self.curr_light_results['opacity_coarse'] = None
                    # self.curr_light_results['opacity_fine'] = None
        else:
            self.current_light_depth_cnt += 1 
        
        cam_results = efficient_sm(cam_pixels, self.light_pixels.to(rgbs.device),
                        cam_results, self.curr_light_results, 
                        ppc, self.light_ppc, 
                        image_shape=self.hparams.img_wh,  
                        fine_sampling=(self.hparams.N_importance > 0), 
                        Light_N_importance=(self.curr_Light_N_importance>0), 
                        shadow_method=self.hparams.shadow_method)

        # if (self.current_light_depth_cnt % self.hparams.sample_light_depth_every == 0) and (cam_results['rgb_coarse'].shape[0] > 5):
        #     print(shadow_maps_coarse[:5,:]) # only print the first elements 

        log['train/loss'] = rgb_loss = self.loss(cam_results, rgbs)
        log['train/sm_loss'] = sm_loss = self.sm_loss(cam_results, sm)

        loss = self.hparams.rgb_weight * rgb_loss + self.hparams.sm_weight * sm_loss 

        typ = 'fine' if 'rgb_fine' in cam_results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(cam_results[f'rgb_{typ}'], rgbs)
            sm_psnr_ = psnr(cam_results[f'sm_{typ}'], sm)
            log['train/psnr'] = psnr_
            log['train/sm_psnr'] = sm_psnr_

        return {'loss': loss,
                'progress_bar': {'rgb_loss': rgb_loss, 'sm_loss': sm_loss, 
                                  'train_psnr': psnr_, 'train_sm_psnr_': sm_psnr_
                                },
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        # print("---------------Starting Validation---------------")
        rays, rgbs, sm, cam_pixels, ppc = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W,3)
        rgbs = rgbs.squeeze() # (H*W,3)
        print("rgbs.shape", rgbs.shape)
        with torch.no_grad():
            cam_results = self(rays, N_importance=self.hparams.N_importance)
            rays = None
            light_results = self(self.light_rays.to(rgbs.device), N_importance=self.hparams.N_importance)

            cam_results = efficient_sm(cam_pixels, self.light_pixels.to(rgbs.device),
                        cam_results, light_results, 
                        ppc, self.light_ppc, 
                        image_shape=self.hparams.img_wh,  
                        fine_sampling=(self.hparams.N_importance > 0), 
                        Light_N_importance=(self.hparams.N_importance > 0), 
                        shadow_method=self.hparams.shadow_method)

        log = {'val_rgb_loss': self.loss(cam_results, rgbs), 'val_sm_loss': self.sm_loss(cam_results, sm)}
        typ = 'fine' if 'rgb_fine' in cam_results else 'coarse'
    
        if batch_nb == 0:
            print("---------------Evaluating and saving Images!---------------")
            W, H = self.hparams.img_wh
            # get rgb map
            img = cam_results[f'rgb_{typ}'].view(H, W, 3).cpu()
            rgb8 = to8b(img.numpy())
            # get shadow map 
            sm_img = cam_results[f'sm_{typ}'].view(H, W, 3).cpu()
            sm8 = to8b(sm_img.numpy())
            gt8 = to8b(rgbs.view(H, W, 3).cpu().numpy())
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            # print(cam_results[f'disp_map_{typ}'], type(cam_results[f'disp_map_{typ}']))
            disp = visualize_depth(cam_results[f'disp_map_{typ}'].view(H, W), to_tensor=False)
            disp8 = to8b(disp)
            depth8 = visualize_depth(cam_results[f'depth_{typ}'].view(H, W), to_tensor=False) 
            depth = visualize_depth(cam_results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            if not os.path.exists(f'logs_rgb_eff_sm/logs/{self.hparams.exp_name}/imgs'):
                os.mkdir(f'logs_rgb_eff_sm/logs/{self.hparams.exp_name}/imgs')
            # save gt
            filename = os.path.join(f'logs_rgb_eff_sm/logs/{self.hparams.exp_name}/imgs', 'gt_{:03d}.png'.format(self.current_epoch))
            imageio.imwrite(filename, gt8)
            # save rgb
            filename = os.path.join(f'logs_rgb_eff_sm/logs/{self.hparams.exp_name}/imgs', 'rgb_{:03d}.png'.format(self.current_epoch))
            imageio.imwrite(filename, rgb8)
            # save sm
            filename = os.path.join(f'logs_rgb_eff_sm/logs/{self.hparams.exp_name}/imgs', 'sm_{:03d}.png'.format(self.current_epoch))
            imageio.imwrite(filename, sm8)
            # save depth
            filename = os.path.join(f'logs_rgb_eff_sm/logs/{self.hparams.exp_name}/imgs', 'depth_{:03d}.png'.format(self.current_epoch))
            imageio.imwrite(filename, depth8)
            # save disp
            filename = os.path.join(f'logs_rgb_eff_sm/logs/{self.hparams.exp_name}/imgs', 'disp_{:03d}.png'.format(self.current_epoch))
            imageio.imwrite(filename, disp8)

            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_rgb_psnr'] = psnr(cam_results[f'rgb_{typ}'], rgbs)
        log['val_sm_psnr'] = psnr(cam_results[f'sm_{typ}'], sm)
        return log

    def validation_epoch_end(self, outputs):
        print(outputs)
        for x in outputs:
            mean_loss = [x['val_rgb_loss'], x['val_sm_loss']]
            mean_psnr = [x['val_rgb_psnr'], x['val_sm_psnr']]

        mean_loss = torch.stack(mean_loss).mean()
        mean_psnr = torch.stack(mean_psnr).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_rgb_loss': outputs[0]['val_rgb_loss'], 
                                 'val_sm_loss': outputs[0]['val_sm_loss'], 
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'logs_rgb_eff_sm/ckpts/{hparams.exp_name}',
                                                                '{epoch:d}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5,)

    logger = TestTubeLogger(
        save_dir="logs_rgb_eff_sm/logs",
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
