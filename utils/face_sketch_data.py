from __future__ import print_function
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

from img_process import draw_landmark_mask
from models.vgg19 import vgg19
import img_process
from pthutils import tensorToVar

import skimage.transform as sktran
from skimage import color

class FaceSketchDataset(Dataset):
    """
    Face Sketch pair dataset
    """
    def __init__(self, root_dir, sketch_filter=None, transform=None):
        self.root_dir = root_dir
        self.face_dir = os.path.join(root_dir, 'photos')
        self.sketch_dir = os.path.join(root_dir, 'sketches')
        self.sketch_filter = sketch_filter

        self.img_names = self.get_img_names()
        self.sketch_names = self.get_sketch_names()
        self.transform = transform

    def get_img_names(self):
        for root, dirs, files in os.walk(self.face_dir):
            return sorted(files)

    def get_sketch_names(self):
        if self.sketch_filter is None:
            return self.img_names
        else:
            return self.sketch_filter(self.img_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        face_path = os.path.join(self.face_dir, self.img_names[idx % len(self.img_names)])
        sketch_path = os.path.join(self.sketch_dir, self.sketch_names[idx % len(self.sketch_names)])
        photo_of_sketch_path = os.path.join(self.face_dir, self.sketch_names[idx % len(self.sketch_names)])
        face = Image.open(face_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')
        sample = [face, sketch, photo_of_sketch]

        if self.transform:
            sample = self.transform(sample)
        return sample


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
        face_gray   = Image.open(face_path).convert('L')
        sample      = [face, face_origin, face_gray]

        if self.transform:
            sample = self.transform(sample)
        return sample

class AddMask(object):
    """
    Add key facial parts mask.
    """
    def __init__(self, detector_path):
        self.detector_path = detector_path

    def __call__(self, sample):
        mask = draw_landmark_mask(sample[0], self.detector_path)
        sample.append(mask)
        return sample 

class Shift(object):
    """
    Add key facial parts mask.
    """
    def __init__(self, (tx, ty)):
        self.tform = sktran.AffineTransform(translation=(tx, ty))

    def __call__(self, sample):
        sample[0] = sktran.warp(sample[0], self.tform) * 255
        return sample 

class ToGray(object):
    def __call__(self, sample):
        sample[0] = color.rgb2gray(sample[0]) 
        sample[0] = sample[0][..., np.newaxis]
        for idx, i in enumerate(sample[1:]):
            sample[idx + 1] = i.convert('L')
        return sample


#  class FindBestMatch(object):
    #  """
    #  Search the dataset for the most similar image.
    #  """
    #  def __init__(self, dataset_img_list, dataset_path, size=(224, 224), compare_layer=['r51']):
        #  self.vgg19_model = vgg19('/home/cfchen/pytorch_models/vgg_conv.pth')
        #  self.dataset = tensorToVar(torch.load(dataset_path))
        #  self.size = size
        #  self.compare_layer = compare_layer
        #  self.img_name_list = [x.strip() for x in open(dataset_img_list).readlines()]
    
    #  def __call__(self, sample):
        #  assert sample[0].size == self.size, "input image size should be the same with dataset size."
        #  img_var = tensorToVar(transforms.to_tensor(sample[0])) * 255
        #  img_var = img_process.subtract_imagenet_mean_batch(img_var)
        #  img_feat = self.vgg19_model(img_var, self.compare_layer) 

        #  img_feat = F.normalize(img_feat, p=2, dim=1).view(img_feat.size(0), img_feat.size(1), -1)
        #  dataset = F.normalize(self.dataset, p=2, dim=1).view(self.dataset.size(0), self.dataset.size(1), -1)
        #  dist = img_feat * dataset
        #  dist = torch.sum(dist, -1)
        #  dist = torch.sum(dist, -1)
        #  _, best_idx = torch.max(dist, 0)

        #  match_img_path = self.img_name_list[best_idx]
        #  match_sketch_path = self.img_name_list[match_img_path.replace('photos', 'sketches')]
        #  sample[1] = Image.open(match_sketch_path)
        #  sample[2] = Image.open(match_img_path)
        #  print(sample[1].size(), sample[2].size())
        #  return sample


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



