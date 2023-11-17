import torch
import json
import numpy as np
import os
from tqdm import tqdm

from utils.ray_utils import get_ray_directions
from utils.color_utils import read_image
from utils.dark_channel import Atomospheric_light_k_means
from einops import rearrange
from .base import BaseDataset


class NeRFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample, **kwargs)

        self.read_intrinsics()
        self.haz_dir_name = kwargs.get("haz_dir_name", "lower")

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "transforms_train.json"), 'r') as f:
            meta = json.load(f)

        w = h = int(800 * self.downsample)
        fx = fy = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x']) * self.downsample

        K = np.float32([[fx, 0, w / 2],
                        [0, fy, h / 2],
                        [0, 0, 1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K, )
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.poses = []
        self.haz_images = []
        self.clear_images = []
        self.atmospheric_lights = []

        with open(os.path.join(self.root_dir, f"transforms_{split}.json"), 'r') as f:
            frames = json.load(f)["frames"]

        print(f'\n Loading {len(frames)} {split} images and a...')
        for frame in tqdm(frames):
            c2w = np.array(frame['transform_matrix'])[:3, :4]
            c2w[:, 1:3] *= -1  # [right up back] to [right down front]
            pose_radius_scale = 1.5
            c2w[:, 3] /= np.linalg.norm(c2w[:, 3]) / pose_radius_scale

            self.poses += [c2w]

            try:
                split_path = os.path.split(frame["file_path"])
                file_path = os.path.join(split_path[0] + "_" + self.haz_dir_name, split_path[1])

                haz_img_path = os.path.join(self.root_dir, f"{file_path}.png")
                haz_img = read_image(haz_img_path, self.img_wh)
                self.haz_images += [haz_img]

                if split == "train":
                    self.atmospheric_lights += [Atomospheric_light_k_means(
                        rearrange(haz_img, "(h w) c->h w c", h=self.img_wh[0]))]

                clear_img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                clear_img = read_image(clear_img_path, self.img_wh)
                self.clear_images += [clear_img]
            except:
                pass

        if len(self.haz_images) > 0:
            self.haz_images = torch.FloatTensor(np.stack(self.haz_images))
        if len(self.clear_images) > 0:
            self.clear_images = torch.FloatTensor(np.stack(self.clear_images))
        if split == "train":
            self.atmospheric_lights = torch.FloatTensor(np.stack(self.atmospheric_lights))
        self.poses = torch.FloatTensor(self.poses)
