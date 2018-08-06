import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    """VGG19 model.
    ---------------------
    Codes borrowed from: https://github.com/leongatys/PytorchNeuralStyleTransfer
    """
    def __init__(self, pool='max', pool_ks=2, pool_st=2):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)
            self.pool2 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)
            self.pool3 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)
            self.pool4 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)
            self.pool5 = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=pool_ks, stride=pool_st)
            self.pool2 = nn.AvgPool2d(kernel_size=pool_ks, stride=pool_st)
            self.pool3 = nn.AvgPool2d(kernel_size=pool_ks, stride=pool_st)
            self.pool4 = nn.AvgPool2d(kernel_size=pool_ks, stride=pool_st)
            self.pool5 = nn.AvgPool2d(kernel_size=pool_ks, stride=pool_st)

    def forward(self, x, out_keys):
        if len(x.size()) == 3:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        elif x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


def vgg19(model_path):
    vgg = VGG(pool_ks=2, pool_st=2)
    vgg.load_state_dict(torch.load(model_path))
    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.cuda()
    return vgg
