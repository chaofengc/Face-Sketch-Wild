import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image, ImageEnhance
import numpy as np
import random
from matplotlib import pyplot as plt


class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size: tuple, output image size (H, W)
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        for idx, i in enumerate(sample):
            sample[idx] = transforms.functional.resize(i, self.output_size)
        return sample


class ToTensor(object):
    """
    #  Normalize the image and dog to [0, 1], and convert them to tensor.
    Convert numpy array to tensor.
    Swap axis of face: (H, W, C) -> (C, H, W)
    """
    def __call__(self, sample):
        for idx, i in enumerate(sample):
            sample[idx] = transforms.functional.to_tensor(i) * 255.
        return sample 


class CropAndResize(object):
    """
    Random crop an area which is 0.08 ~ 1.0 of original size, and 3/4 ~ 4/3 of original aspect ratio.
    And finally resize the cropped area to given size. 
    """
    def __init__(self, size):
        self.transform_func = transforms.RandomResizedCrop(size)

    def __call__(self, sample):
        seed = np.random.randint(2147483647)
        for idx, i in enumerate(sample):
            random.seed(seed)
            sample[idx] = self.transform_func(i) 

        return sample 


class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, sharp=0.0):
        super(ColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.sharp = sharp

    def __call__(self, sample):
        img = sample[0]
        sharp_factor = np.random.uniform(max(0, 1 - self.sharp), 1 + self.sharp)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharp_factor)

        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        img = transform(img)
        sample[0] = img

        return sample

