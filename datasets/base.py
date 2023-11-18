from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """

    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.batch_size = kwargs.get("batch_size", 8192)
        self.ray_sampling_strategy = "all_images"

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return self.poses.shape[0]

    def __getitem__(self, idx):
        return self.get_pix(idx)

    def get_pix(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in run.py
            if self.ray_sampling_strategy == 'all_images':  # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image':  # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1], self.batch_size)
            rays = self.haz_images[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'haz_rgb': rays[:, :3], "clear_rgb": self.clear_images[img_idxs, pix_idxs]}
            if self.haz_images.shape[-1] == 4:  # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
            sample["pose"] = self.poses[img_idxs]
            sample["direction"] = self.directions[pix_idxs]
            sample["atmospheric_lights_mean"] = self.atmospheric_lights.mean()
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.haz_images) > 0:  # if ground truth available
                rays = self.haz_images[idx]
                sample['haz_rgb'] = rays[:, :3]
                sample['clear_rgb'] = self.clear_images[idx, :, :3]
                if rays.shape[1] == 4:  # HDR-NeRF data
                    sample['exposure'] = rays[0, 3]  # same exposure for all rays
            sample["direction"] = self.directions
        return sample
