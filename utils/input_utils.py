import random
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange


def get_params(img, output_size, n):
    h, w = img.shape[0:-1]
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i_list = [random.randint(0, h - th) for _ in range(n)]
    j_list = [random.randint(0, w - tw) for _ in range(n)]
    return i_list, j_list, th, tw


def n_random_crops(img, x, y, h, w):
    crops = []
    for i in range(len(x)):
        new_crop = img[x[i]:x[i] + h, y[i]:y[i] + w, :]
        crops.append(new_crop)
    # 1 n h w 3
    return torch.stack(crops, dim=0).unsqueeze(0)


def get_images(haz_image, Cc_image=None, patch_n=16, image_size=64):
    i, j, h, w = get_params(haz_image, (image_size, image_size), patch_n)
    haz_image = n_random_crops(haz_image, i, j, h, w)
    if Cc_image != None:
        Cc_image = n_random_crops(Cc_image, i, j, h, w)
        return haz_image, Cc_image
    return haz_image


def get_image_block(img, img_wh, image_size):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if img.ndim == 2:
        w, h = img_wh
        img = rearrange(img, "(h w) c->h w c", h=h, w=w)
    else:
        w, h = img.shape[0:2]
    # 计算块数量
    num_blocks_h = h // image_size
    num_blocks_w = w // image_size

    last_h = h % image_size != 0
    last_w = w % image_size != 0

    # 生成块
    blocks = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = img[i * image_size: (i + 1) * image_size, j * image_size: (j + 1) * image_size, :]
            blocks.append(block)
            if last_h and j == num_blocks_w - 1:
                block = img[i * image_size:(i + 1) * image_size, -image_size - 1:-1, :]
                blocks.append(block)
            if last_w and i == num_blocks_h - 1:
                block = img[-image_size - 1:-1, j * image_size:(j + 1) * image_size, :]
                blocks.append(block)
    if len(blocks) > num_blocks_h * num_blocks_w:
        blocks.pop()
    return torch.stack(blocks, dim=0)
