import os
import imageio

from pytorch_lightning.accelerators import accelerator
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering_shadows import *
from models.camera import Camera

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class NeRFSystem(LightningModule):
    def __init__(self, hparams, log_dir=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.rgb_loss = loss_dict['color'](coef=self.hparams.rgb_weight)
        self.sm_loss = loss_dict['shadow'](coef=self.hparams.sm_weight)
        self.opacity_loss = loss_dict['opacity']()


        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3)
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')
        
        self.log_dir = log_dir
        self.img_dir = None

    def forward(self, rays):
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
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh), 
                  'hparams': self.hparams}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus

        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=1,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def decode_batch(self, batch, batch_nb):
        cam_rays = batch['cam_rays'].view(-1,8)
        light_rays = batch['light_rays'].view(-1,8)
        sms = batch['sms'].view(-1) # .view(-1,3)
        cam_ppc_eye_pos = batch['cam_ppc_eye_pos'].squeeze()
        cam_ppc_camera = batch['cam_ppc_camera'].squeeze()
        light_ppc_eye_pos = batch['light_ppc_eye_pos'].squeeze()
        light_ppc_camera = batch['light_ppc_camera'].squeeze()
        cam_pixels = batch['cam_pixels'].view(-1,3)
        light_pixels = batch['light_pixels'].view(-1,3)

        if self.hparams.process_rgb: 
            rgbs = batch['rgbs']
        else: 
            rgbs = None

        # print("sms {}, cam_pixels {}, light_pixels {}, cam_rays {}, light_rays {}".format(sms.shape, cam_pixels.shape,\
        #                                                     light_pixels.shape, cam_rays.shape, light_rays.shape))

        # print("Camera eye pos {} cam {}, light eye_pos {} cam {}".format(cam_ppc_eye_pos.shape, \
        #                     cam_ppc_camera.shape, light_ppc_eye_pos.shape, light_ppc_camera.shape))

        return cam_rays, light_rays, rgbs, sms, cam_ppc_eye_pos, cam_ppc_camera, \
                light_ppc_eye_pos, light_ppc_camera, cam_pixels, light_pixels

    def training_step(self, batch, batch_nb):
        cam_rays, light_rays, rgbs, sms, \
            cam_ppc_eye_pos, cam_ppc_camera, light_ppc_eye_pos,\
            light_ppc_camera, cam_pixels, light_pixels = self.decode_batch(batch, batch_nb)
        
        cam_results = self(cam_rays)
        cam_rays = None
        light_results = self(light_rays)
        light_rays = None

        ### Create PPC for Camera and Light 
        # Assumption is that they all belong to the same camera! 
        cam_ppc = Camera.from_camera_eyepos(cam_ppc_eye_pos, cam_ppc_camera)
        light_ppc = Camera.from_camera_eyepos(light_ppc_eye_pos, light_ppc_camera)

        # call efficient mapping on batch_rays 
        if self.hparams.shadow_mapper2d:
            # clear memory 
            cam_pixels, light_pixels = None, None
            sm_results = shadow_mapper2d(cam_results, light_results, 
                            self.hparams.img_wh, cam_ppc, light_ppc, 
                            shadow_method_mode=self.hparams.shadow_method)

        else:
            sm_results = shadow_mapper(cam_results, light_results, 
                        cam_pixels, light_pixels, self.hparams.img_wh, cam_ppc, 
                        light_ppc, shadow_method_mode=self.hparams.shadow_method)

        sm_loss = self.sm_loss(sm_results, sms)
        # sm_loss = torch.tensor(0.0).to(sms.device)
        # opacity_loss = 0.5 * self.opacity_loss(light_results, sms)
        opacity_loss = 0.0
        # print("----------->opacity_loss", opacity_loss)
        # print("----------->sm_loss", sm_loss)
        if rgbs is not None: 
            rgb_loss = self.rgb_loss(cam_results, rgbs)
            loss = sm_loss + rgb_loss + opacity_loss
        else:
            loss = sm_loss + opacity_loss
            rgb_loss = torch.tensor(0.0).to(sms.device)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in cam_results else 'coarse'
            sm_psnr_ = psnr(sm_results[f'sm_{typ}'], sms)
            if self.hparams.process_rgb: 
                rgb_psnr_ = psnr(cam_results[f'rgb_{typ}'], rgbs)
            else:
                rgb_psnr_ = torch.tensor(0.).to(sms.device)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/rgb_loss', rgb_loss, prog_bar=(rgbs is not None))
        self.log('train/sm_loss', sm_loss, prog_bar=True)
        self.log('train/op_loss', opacity_loss, prog_bar=False)
        self.log('train/rgb_psnr', rgb_psnr_, prog_bar=False)
        self.log('train/sm_psnr', sm_psnr_, prog_bar=True)

        return loss

    def _create_val_imgs_path(self,):
        if self.img_dir is None:
            self.img_dir = os.path.join(self.log_dir+'/val_imgs/')
            print("Saving Validation Images here: {}".format(self.img_dir))
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)

    def validation_step(self, batch, batch_nb):
        self._create_val_imgs_path()

        with torch.no_grad():
            cam_rays, light_rays, rgbs, sms, \
                cam_ppc_eye_pos, cam_ppc_camera, light_ppc_eye_pos,\
                light_ppc_camera, cam_pixels, light_pixels = self.decode_batch(batch, batch_nb)
            
            cam_rays = cam_rays.squeeze() # (H*W, 3)
            light_rays = light_rays.squeeze() # (H*W, 3)
            cam_results = self(cam_rays)
            light_results = self(light_rays)
            cam_rays = None
            light_rays = None

        ### Create PPC for Camera and Light 
        # Assumption is that they all belong to the same camera! 
        cam_ppc = Camera.from_camera_eyepos(cam_ppc_eye_pos, cam_ppc_camera)
        light_ppc = Camera.from_camera_eyepos(light_ppc_eye_pos, light_ppc_camera)

        # call efficient mapping on batch_rays 
        if self.hparams.shadow_mapper2d:
            # clear memory 
            cam_pixels, light_pixels = None, None
            sm_results = shadow_mapper2d(cam_results, light_results, 
                            self.hparams.img_wh, cam_ppc, light_ppc, 
                            shadow_method_mode=self.hparams.shadow_method)

        else:
            sm_results = shadow_mapper(cam_results, light_results, 
                        cam_pixels, light_pixels, self.hparams.img_wh, cam_ppc, 
                        light_ppc, shadow_method_mode=self.hparams.shadow_method)

        # print(sms.shape, sm_results['sm_fine'].shape)
        # raise

        loss = self.sm_loss(sm_results, sms)
        log = {'val_sm_loss': loss}
        if rgbs is not None: 
            rgb_loss = self.rgb_loss(cam_results, rgbs)
            loss = loss + rgb_loss
        else:
            rgb_loss = torch.tensor(0.0).to(sms.device)
            
        log['val_rgb_loss'] = rgb_loss

        typ = 'fine' if 'rgb_fine' in cam_results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = cam_results[f'rgb_{typ}'].view(H, W, 3).cpu()
            rgb8 = to8b(img.numpy())
            img = img.permute(2, 0, 1) # (3, H, W)
            sm_gt = sms.view(H, W).cpu() 
            sm_gt = torch.stack([sm_gt,sm_gt,sm_gt], dim=2)
            sm8 = to8b(sm_gt.numpy())
            sm_gt = sm_gt.permute(2, 0, 1) # (3, H, W)

            if self.hparams.process_rgb: 
                img_gt = rgbs.view(H, W, 3).cpu() 
                gt8 = to8b(img_gt.numpy())
                img_gt = img_gt.permute(2, 0, 1) # (3, H, W)
            depth = visualize_depth(cam_results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            depth8 = visualize_depth(cam_results[f'depth_{typ}'].view(H, W), return_as_numpy=True) # (3, H, W)

            ## Save Images 
            if self.img_dir: 
                if self.hparams.process_rgb: 
                    filename = os.path.join(self.img_dir, 'gt_{:03d}.png'.format(self.current_epoch))
                    imageio.imwrite(filename, gt8)
                filename = os.path.join(self.img_dir, 'rgb_{:03d}.png'.format(self.current_epoch))
                imageio.imwrite(filename, rgb8)
                filename = os.path.join(self.img_dir, 'depth_{:03d}.png'.format(self.current_epoch))
                imageio.imwrite(filename, depth8)
                filename = os.path.join(self.img_dir, 'sm_gt_{:03d}.png'.format(self.current_epoch))
                imageio.imwrite(filename, sm8)

                # save disp
                if f'sm_{typ}' in sm_results.keys():
                    sm = sm_results[f'sm_{typ}'].view(H, W).cpu()
                    sm = torch.stack([sm,sm,sm], dim=2)
                    sm8 = to8b(sm.numpy())
                    sm = sm.permute(2, 0, 1) # (3, H, W)
                    filename = os.path.join(self.img_dir, 'sm_{:03d}.png'.format(self.current_epoch))
                    imageio.imwrite(filename, sm8)

            if not self.hparams.process_rgb: 
                img_gt = torch.zeros_like(img)
            stack = torch.stack([sm_gt, img_gt, img, depth, sm]) # (5, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        if self.hparams.process_rgb: 
            rgb_psnr_ = psnr(cam_results[f'rgb_{typ}'], rgbs)
        else:
            rgb_psnr_ = torch.tensor(0.).to(sms.device)
        sm_psnr_ = psnr(sm_results[f'sm_{typ}'], sms)

        log['val_rgb_psnr'] = rgb_psnr_
        log['val_sm_psnr'] = sm_psnr_

        return log

    def validation_epoch_end(self, outputs):
        for x in outputs:
            mean_loss = [x['val_rgb_loss'], x['val_sm_loss']]
            mean_psnr = [x['val_rgb_psnr'], x['val_sm_psnr']]

        mean_loss = torch.stack(mean_loss).mean()
        mean_psnr = torch.stack(mean_psnr).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=5)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)
    system = NeRFSystem(hparams, log_dir=logger.save_dir+'/{}'.format(hparams.exp_name))

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=hparams.num_sanity_val_steps,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
