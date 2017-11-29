from __future__ import print_function
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import cv2 as cv
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

class Pix2PixDataset(Dataset):
    """
    Photo to photo dataset
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_names = self.get_img_names()
        self.transform = transform

    def get_img_names(self):
        for root, dirs, files in os.walk(self.root_dir):
            return sorted(files)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        img = Image.open(img_path).convert('RGB')

        sample = []
        width, height = img.size
        img1 = img.crop((0, 0, np.floor(width / 2).astype('int'), height))
        img2 = img.crop((np.ceil(width / 2).astype('int'), 0, width, height))
        sample.append(img1)
        sample.append(img2)

        if self.transform:
            sample = self.transform(sample)
        return sample

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
            sample[idx] = transforms.resize(i, self.output_size)
        return sample

class ToTensor(object):
    """
    #  Normalize the image and dog to [0, 1], and convert them to tensor.
    Convert numpy array to tensor.
    Swap axis of face: (H, W, C) -> (C, H, W)
    """
    def __call__(self, sample):
        for idx, i in enumerate(sample):
            sample[idx] = transforms.to_tensor(i) * 255.
        return sample 

class CropAndResize(object):
    """
    Random crop an area which is 0.08 ~ 1.0 of original size, and 3/4 ~ 4/3 of original aspect ratio.
    And finally resize the cropped area to given size. 
    """
    def __init__(self, size):
        self.transform_func = transforms.RandomResizedCrop(size)

    def __call__(self, sample):
        for idx, i in enumerate(sample):
            sample[idx] = self.transform_func(i) 
        return sample 



    


