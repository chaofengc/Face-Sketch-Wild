import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from components import *
from pthutils import tensorToVar 

class SketchNetV2(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, norm='IN', time_steps=10):
        super(SketchNetV2, self).__init__()
        self.enc = Encoder(in_channels, out_channels, norm)
        self.dec = Decoder(in_channels, out_channels, norm)
        self.cgru = CGRUCell(128, 128)
        self.time_steps = time_steps
        
    def forward(self, X, tmp_time_steps=None):
        if tmp_time_steps is not None:
            time_steps = tmp_time_steps
        else:
            time_steps = self.time_steps
        enc_feat = self.enc(X)

        y = tensorToVar(torch.zeros_like(X.data)) 
        h_t = tensorToVar(torch.zeros_like(enc_feat.data)) 
        result = []
        for i in range(time_steps):
            h_t = self.cgru(enc_feat, h_t)
            y = y + self.dec(h_t) 
            result.append(y)
        result = torch.stack(result)
        result = result.view(-1, result.size(2), result.size(3), result.size(4))
        return result.contiguous()


class CGRUCell(nn.Module):
    """
    Input: (N, C_in, H, W)
    Output: (N, C_out, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), pad=(1, 1)):
        super(CGRUCell, self).__init__()
        self.r_conv = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size, padding=pad)
        self.z_conv = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size, padding=pad)
        self.h_conv = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size, padding=pad)

    def forward(self, x, h_tm1):
        feat = torch.cat((x, h_tm1), 1)

        z = F.sigmoid(self.z_conv(feat))
        r = F.sigmoid(self.r_conv(feat))
        hh_t = F.relu(self.h_conv(torch.cat((x, r * h_tm1), 1)))
        h_t = (1 - z) * h_tm1 + z * hh_t
        return h_t


class Encoder(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, norm='IN'):
        super(Encoder, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(in_channels, 32, kernel_size=3, stride=1)
        self.norm1 = nn.BatchNorm2d(32) if norm=='BN' else torch.nn.InstanceNorm2d(32) 
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.norm2 = nn.BatchNorm2d(64) if norm=='BN' else torch.nn.InstanceNorm2d(64) 
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.norm3 = nn.BatchNorm2d(128) if norm=='BN' else torch.nn.InstanceNorm2d(128)
        # Residual layers
        self.res1 = ResidualBlock(128, norm_type='batch')
        self.res2 = ResidualBlock(128, norm_type='batch')
        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, X):
        in_X = X
        y = self.relu(self.norm1(self.conv1(in_X)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        return y


class Decoder(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, norm='IN'):
        super(Decoder, self).__init__()
        
        # Residual layers
        self.res4 = ResidualBlock(128, norm_type='batch')
        self.res5 = ResidualBlock(128, norm_type='batch')

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.norm4 = nn.BatchNorm2d(64) if norm=='BN' else torch.nn.InstanceNorm2d(64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.norm5 = nn.BatchNorm2d(32) if norm=='BN' else torch.nn.InstanceNorm2d(32)
        self.deconv3 = ConvLayer(32, out_channels, kernel_size=3, stride=1)

        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, X):
        y = self.res4(X)
        y = self.res5(y)
        y = self.relu(self.norm4(self.deconv1(y)))
        y = self.relu(self.norm5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

