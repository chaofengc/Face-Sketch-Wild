import torch
import torch.nn as nn
import numpy as np
from time import time
from components import *

class SketchNetV5(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, norm='IN'):
        super(SketchNetV5, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(in_channels, 128, kernel_size=5, stride=1)
        self.norm1 = nn.BatchNorm2d(128) if norm=='BN' else torch.nn.InstanceNorm2d(128, affine=True) 
        self.conv2 = ConvLayer(128, 128, kernel_size=5, stride=1)
        self.norm2 = nn.BatchNorm2d(128) if norm=='BN' else torch.nn.InstanceNorm2d(128, affine=True) 
        self.conv3 = ConvLayer(128, 64, kernel_size=1, stride=1)
        self.norm3 = nn.BatchNorm2d(64) if norm=='BN' else torch.nn.InstanceNorm2d(64, affine=True) 
        self.conv4 = ConvLayer(64, 64, kernel_size=1, stride=1)
        self.norm4 = nn.BatchNorm2d(64) if norm=='BN' else torch.nn.InstanceNorm2d(64, affine=True) 
        self.conv5 = ConvLayer(64, 1, kernel_size=3, stride=1)
        self.conv6 = ConvLayer(1, 1, kernel_size=3, stride=1)

        # Upsampling Layers
        self.relu = nn.ReLU()

    def forward(self, X):
        in_X = X
        y = self.relu(self.norm1(self.conv1(in_X)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        y = self.relu(self.norm4(self.conv4(y)))
        y = self.relu(self.conv5(y))
        y = self.relu(self.conv6(y))

        return y


