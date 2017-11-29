import torch
import torch.nn as nn
import numpy as np
from time import time
from components import *

class SketchNetV1(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, norm='IN'):
        super(SketchNetV1, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(in_channels, 32, kernel_size=3, stride=1)
        self.norm1 = nn.BatchNorm2d(32) if norm=='BN' else InstanceNormalization(32) 
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.norm2 = nn.BatchNorm2d(64) if norm=='BN' else InstanceNormalization(64) 
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.norm3 = nn.BatchNorm2d(128) if norm=='BN' else InstanceNormalization(128)

        # Residual layers
        self.res1 = ResidualBlock(128, norm_type='batch')
        self.res2 = ResidualBlock(128, norm_type='batch')
        self.res3 = ResidualBlock(128, norm_type='batch')
        self.res4 = ResidualBlock(128, norm_type='batch')
        self.res5 = ResidualBlock(128, norm_type='batch')

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.norm4 = nn.BatchNorm2d(64) if norm=='BN' else InstanceNormalization(64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.norm5 = nn.BatchNorm2d(32) if norm=='BN' else InstanceNormalization(32)
        self.deconv3 = ConvLayer(32, out_channels, kernel_size=3, stride=1)

        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, X):
        in_X = X
        y = self.relu(self.norm1(self.conv1(in_X)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.norm4(self.deconv1(y)))
        y = self.relu(self.norm5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


