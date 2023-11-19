import torch
from config.opt import get_opts
import os
import glob
import numpy as np
import cv2
from einops import rearrange
import argparse

# data
from torch.utils.data import DataLoader
from datasets.nerf import NeRFDataset
from utils.ray_utils import get_rays

# models
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses.regularization import CompositeLoss, DistortionLoss, OpacityLoss, DCPLoss, FoggyLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

import imageio
import warnings
import vren

warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.warmup_steps = 256
        self.update_interval = 16

        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.haz_train_psnr = PeakSignalNoiseRatio(data_range=1)

        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)

        self.haz_val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.haz_val_ssim = StructuralSimilarityIndexMeasure(data_range=1)

        self.model = NGP(scale=self.hparams.scale)

    def forward(self, batch, split):
        poses = batch['pose']
        directions = batch["direction"]
        rays_o, rays_d = get_rays(directions, poses)
        kwargs = {"split": split}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1 / 256
        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = NeRFDataset
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  "haz_dir_name": self.hparams.haz_dir_name
                  }
        if self.hparams.split == "train":
            self.train_dataset = dataset(split="train", **kwargs)
            self.train_dataset.batch_size = self.hparams.batch_size
            self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy
            self.test_dataset = dataset(split='test', **kwargs)

            self.CompositeLoss = CompositeLoss(self.hparams.composite_weight)
            self.DistortionLoss = DistortionLoss(self.hparams.distortion_weight)
            self.OpacityLoss = OpacityLoss(self.hparams.opacity_weight)
            self.DCPLoss = DCPLoss(self.hparams.dcp_weight)
            self.FoggyLoss = FoggyLoss(self.hparams.foggy_weight)
        else:
            self.test_dataset = dataset(split=self.hparams.split, **kwargs)

    def configure_optimizers(self):
        optimizer = FusedAdam(self.parameters(), self.hparams.lr, eps=1e-15)
        scheduler = CosineAnnealingLR(optimizer, self.hparams.num_epochs, self.hparams.lr / 30)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.train_dataset.poses.to(self.device).to(self.device),
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step % self.update_interval == 0:
            self.model.update_density_grid(0.01 * MAX_SAMPLES / 3 ** 0.5,
                                           warmup=self.global_step < self.warmup_steps)

        results = self(batch, split='train')

        loss = 0
        loss += self.CompositeLoss.apply(results, batch)
        loss += self.DistortionLoss.apply(results)
        loss += self.OpacityLoss.apply(results)
        loss += self.DCPLoss.apply(results, batch)
        loss += self.FoggyLoss.apply(results)

        with torch.no_grad():
            self.train_psnr(results['c_rgb'], batch["clear_rgb"])
            self.haz_train_psnr(results['f_rgb'], batch["haz_rgb"])
        self.log('train/loss', loss)
        self.log('train/rm_s', results['rm_samples'] / len(batch['haz_rgb']), True)
        self.log('train/vr_s', results['vr_samples'].sum() / len(batch['haz_rgb']), True)
        self.log('train/psnr', self.train_psnr, True)
        self.log('train/haz_psnr', self.haz_train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        val_dir = f'results/{self.hparams.exp_name}/{"test" if self.hparams.split=="train" else self.hparams.split}'
        self.haz_val_dir = os.path.join(val_dir, "haz")
        self.clear_val_dir = os.path.join(val_dir, "clear")
        os.makedirs(self.haz_val_dir, exist_ok=True)
        os.makedirs(self.clear_val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        results = self(batch, split=self.hparams.split)
        logs = {}

        c_rgb_gt = batch['clear_rgb']
        f_rgb_gt = batch['haz_rgb']

        # compute each metric per image
        self.val_psnr(results['c_rgb'], c_rgb_gt)
        logs['clear_psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.test_dataset.img_wh
        c_rgb_pred = rearrange(results['c_rgb'], '(h w) c -> 1 c h w', h=h)
        c_rgb_gt = rearrange(c_rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(c_rgb_pred, c_rgb_gt)
        logs['clear_ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()

        # compute each metric per image
        self.haz_val_psnr(results['f_rgb'], f_rgb_gt)
        logs['haz_psnr'] = self.haz_val_psnr.compute()
        self.haz_val_psnr.reset()

        f_rgb_pred = rearrange(results['f_rgb'], '(h w) c -> 1 c h w', h=h)
        f_rgb_gt = rearrange(f_rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.haz_val_ssim(f_rgb_pred, f_rgb_gt)
        logs['haz_ssim'] = self.haz_val_ssim.compute()
        self.haz_val_ssim.reset()

        idx = batch['img_idxs']

        c_rgb_pred = rearrange(results['c_rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
        c_rgb_pred = (c_rgb_pred * 255).astype(np.uint8)
        c_depth = depth2img(rearrange(results['c_depth'].cpu().numpy(), '(h w) -> h w', h=h))
        imageio.imsave(os.path.join(self.clear_val_dir, f'{idx:03d}.png'), c_rgb_pred)
        imageio.imsave(os.path.join(self.clear_val_dir, f'{idx:03d}_d.png'), c_depth)

        f_rgb_pred = rearrange(results['f_rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
        f_rgb_pred = (f_rgb_pred * 255).astype(np.uint8)
        f_depth = depth2img(rearrange(results['f_depth'].cpu().numpy(), '(h w) -> h w', h=h))
        imageio.imsave(os.path.join(self.haz_val_dir, f'{idx:03d}.png'), f_rgb_pred)
        imageio.imsave(os.path.join(self.haz_val_dir, f'{idx:03d}_d.png'), f_depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['clear_psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/clear_psnr', mean_psnr, True)

        ssims = torch.stack([x['clear_ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/clear_ssim', mean_ssim, True)

        psnrs = torch.stack([x['haz_psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/haz_psnr', mean_psnr, True)

        ssims = torch.stack([x['haz_ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/haz_ssim', mean_ssim, True)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if (hparams.split != "train") and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=False,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                      if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=1 if hparams.split == "train" else -1,
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    imgs = sorted(glob.glob(os.path.join(system.clear_val_dir, '*.png')))
    imageio.mimsave(os.path.join(system.clear_val_dir, 'rgb.mp4'),
                    [imageio.imread(img) for img in imgs[::2]],
                    fps=30, macro_block_size=1)
    imageio.mimsave(os.path.join(system.clear_val_dir, 'depth.mp4'),
                    [imageio.imread(img) for img in imgs[1::2]],
                    fps=30, macro_block_size=1)

    imgs = sorted(glob.glob(os.path.join(system.haz_val_dir, '*.png')))
    imageio.mimsave(os.path.join(system.haz_val_dir, 'rgb.mp4'),
                    [imageio.imread(img) for img in imgs[::2]],
                    fps=30, macro_block_size=1)
    imageio.mimsave(os.path.join(system.haz_val_dir, 'depth.mp4'),
                    [imageio.imread(img) for img in imgs[1::2]],
                    fps=30, macro_block_size=1)
