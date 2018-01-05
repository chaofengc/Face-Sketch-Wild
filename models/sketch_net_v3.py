import torch
import torch.nn as nn
import numpy as np
from time import time
from components import *

class SketchNetV3(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, norm='IN'):
        super(SketchNetV3, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(in_channels, 32, kernel_size=3, stride=1)
        self.norm1 = nn.BatchNorm2d(32) if norm=='BN' else torch.nn.InstanceNorm2d(32, affine=True) 
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.norm2 = nn.BatchNorm2d(64) if norm=='BN' else torch.nn.InstanceNorm2d(64, affine=True) 
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.norm3 = nn.BatchNorm2d(128) if norm=='BN' else torch.nn.InstanceNorm2d(128, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(128, norm_type=norm)
        self.res2 = ResidualBlock(128, norm_type=norm)
        self.res3 = ResidualBlock(128, norm_type=norm)
        self.res4 = ResidualBlock(128, norm_type=norm)
        self.res5 = ResidualBlock(128, norm_type=norm)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(256, 64, kernel_size=3, stride=1, upsample=2)
        self.norm4 = nn.BatchNorm2d(64) if norm=='BN' else torch.nn.InstanceNorm2d(64, affine=True) 
        self.deconv2 = UpsampleConvLayer(128, 32, kernel_size=3, stride=1, upsample=2)
        self.norm5 = nn.BatchNorm2d(32) if norm=='BN' else torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(64, out_channels, kernel_size=3, stride=1)

        self.relu = nn.ReLU()

    def forward(self, X):
        y_conv1 = self.relu(self.norm1(self.conv1(X)))
        y_conv2 = self.relu(self.norm2(self.conv2(y_conv1)))
        y_conv3 = self.relu(self.norm3(self.conv3(y_conv2)))
        y = self.res1(y_conv3)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y_deconv0 = self.res5(y)
        y_deconv0 = torch.cat((y_deconv0, y_conv3), 1)
        y_deconv1 = self.relu(self.norm4(self.deconv1(y_deconv0)))
        y_deconv1 = torch.cat((y_deconv1, y_conv2), 1)
        y_deconv2 = self.relu(self.norm5(self.deconv2(y_deconv1)))
        y_deconv2 = torch.cat((y_deconv2, y_conv1), 1)
        y = self.deconv3(y_deconv2)
        return y


