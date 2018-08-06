import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image, ImageEnhance
import numpy as np
import random
import os


class FaceDataset(Dataset):
    """
    Face dataset.
    Args:
        img_dirs: dir list to read photo from.
    """
    def __init__(self, img_dirs, transform=None):
        self.img_dirs = img_dirs
        self.img_names = self.__get_imgnames__() 
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        for i in self.img_dirs:
            for name in os.listdir(i):
                tmp.append(os.path.join(i, name))
        return tmp

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        face_path   = self.img_names[idx]
        face        = Image.open(face_path).convert('RGB')
        face_origin = Image.open(face_path).convert('RGB')
        sample      = [face, face_origin]

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
            sample[idx] = transforms.functional.resize(i, self.output_size)
        return sample


class ToTensor(object):
    """Convert image to tensor, and normalize the value to [0, 255]
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

if __name__ == '__main__':
    face_sketch_data = FaceSketchDataset('/home/cfchen/face_sketch/Data')
    tran = transforms.Compose([Rescale((248, 200)), 
                               ColorJitter(0.6, 0.5, 0.5, 0.1, 0.5), 
                               ToTensor()
        ])
    test_sample = face_sketch_data[0]
    tran_test = tran(test_sample) 
    print(tran_test['face'].shape, tran_test['sketch'].shape)
    plt.imshow(test_sample['face'])
    plt.waitforbuttonpress()

    #  cv.imshow('face', test_sample['face'])
    #  cv.imshow('sketch', test_sample['sketch'])
    #  cv.waitKey()



