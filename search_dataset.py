from __future__ import print_function
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

import cv2 as cv
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

from models.vgg19 import vgg19
from utils import img_process 
from gpu_manager import GPUManager
from pthutils import tensorToVar

def build_dataset(save_path, dataset_img_list, layer=['r51'], size=(224, 224)):
    vgg19_model = vgg19('/home/cfchen/pytorch_models/vgg_conv.pth')
    img_list = [x.strip() for x in open(dataset_img_list).readlines()]

    feature_dataset = []
    for i in img_list:
        img_var = img_process.read_img_var(i, size=size)
        img_var = img_process.subtract_imagenet_mean_batch(img_var)
        feat = vgg19_model(img_var, layer)
        feature_dataset.append(feat[0].data.cpu())
    feature_dataset = torch.stack(feature_dataset)
    print(feature_dataset.size())
    torch.save(feature_dataset.squeeze(), save_path)


def find_photo_sketch_batch(photo_batch, dataset_path, img_name_list, vgg_model, compare_layer=['r51']):
    """
    Search the dataset to find the best matching image.
    """
    photo_feat = vgg_model(img_process.subtract_imagenet_mean_batch(photo_batch), compare_layer)[0]
    dataset = tensorToVar(torch.load(dataset_path))
    photo_feat = F.normalize(photo_feat, p=2, dim=1).view(photo_feat.size(0), photo_feat.size(1), -1)
    dataset = F.normalize(dataset, p=2, dim=1).view(dataset.size(0), dataset.size(1), -1)
    img_idx = []
    for i in range(photo_feat.size(0)):
        dist = photo_feat[i].unsqueeze(0) * dataset
        dist = torch.sum(dist, -1)
        dist = torch.sum(dist, -1)
        _, best_idx = torch.max(dist, 0)
        img_idx += best_idx.data.cpu().tolist()

    img_name_list = np.array([x.strip() for x in open(img_name_list).readlines()])
    match_img_list = img_name_list[img_idx]
    match_sketch_list = [x.replace('photos', 'sketches') for x in match_img_list]

    match_img_batch = [img_process.read_img_var(x, size=(224, 224)) for x in match_img_list]
    match_sketch_batch = [img_process.read_img_var(x, size=(224, 224)) for x in match_sketch_list]
    return  torch.stack(match_sketch_batch).squeeze(), torch.stack(match_img_batch).squeeze()

    
if __name__ == '__main__':
    gm=GPUManager()
    torch.cuda.set_device(gm.auto_choice())
    build_dataset('./face_sketch_data/feature_dataset.pth', './face_sketch_data/dataset_img_list.txt')
    
