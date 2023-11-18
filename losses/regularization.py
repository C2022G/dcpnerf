import abc
import os
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch import nn
import torch
import torch.optim.lr_scheduler
from .base_regularization import Distortion, Foggy, AlphaRat
import vren


class CompositeLoss:
    def __init__(self, weight=1):
        self.weight = weight

    def apply(self, results, batch):
        return self.weight * (torch.pow(results['f_rgb'] - batch['haz_rgb'], 2)).mean()


class DistortionLoss:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, results):
        return (self.weight * Distortion.apply(results['c_ws'], results['deltas'],
                                               results['ts'], results['rays_a'])).mean()


class OpacityLoss:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, results):
        o = results['c_opacity'] + 1e-10
        return (self.weight * -(o * torch.log(o))).mean()


class DCPLoss:
    def __init__(self, weight=6e-3):
        self.weight = weight

    def apply(self, results, batch):
        return self.weight * (torch.pow(results['p_rgbs'] - batch["atmospheric_lights_mean"], 2)).mean()


class FoggyLoss:
    def __init__(self, weight=2e-4):
        self.weight = weight

    def apply(self, results):
        with torch.no_grad():
            haz_clear_weight = vren.haz_clear_weight_loss_fw(results['c_sigmas'], results['deltas'], results['ts'],
                                                             results['rays_a'], results['vr_samples'])

        alpha_rat = AlphaRat.apply(results['p_sigmas'], results['deltas'], results['rays_a'],
                                   results['vr_samples'])

        foggy = Foggy.apply((1 + haz_clear_weight), alpha_rat + 1e-10,
                            results['rays_a'],
                            results['vr_samples'])

        return (self.weight * foggy).mean()
