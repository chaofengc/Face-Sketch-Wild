from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from time import time

import mxnet as mx
from mxnet import gluon
from mxnet import nd

class TestConv(nn.Module):
    def __init__(self):
        super(TestConv, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)

    def forward(self, x):
        t1 = time()
        x = self.conv1(x)
        t2 = time()
        x = self.conv2(x)
        t3 = time()
        print('[PyTorch] Conv1 Time: {}\t Conv2 Time: {}'.format(t2 - t1, t3 - t2))
        return x


class TestConvMX(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(TestConvMX, self).__init__(**kwargs)
        self.conv1 = gluon.nn.Conv2D(32, 3)
        self.conv2 = gluon.nn.Conv2D(32, 3)

    def forward(self, x):
        t1 = time()
        x = self.conv1(x)
        t2 = time()
        x = self.conv2(x)
        t3 = time()
        print('[Mxnet] Conv1 Time: {}\t Conv2 Time: {}'.format(t2 - t1, t3 - t2))
        return x


if __name__ == '__main__':
    start = time()
    x = nd.random.uniform(shape=(1, 32, 224, 224), ctx=mx.gpu())
    end = time()
    cnns = TestConvMX()
    cnns.initialize(ctx=mx.gpu())
    print("MXNet Init time", end - start)
    out = cnns(x)

    start = time()
    x = Variable(torch.randn(1, 32, 224, 224).cuda(), requires_grad=False)
    cnns = TestConv()
    cnns.cuda()
    end = time()
    print("PyTorch Init time", end - start)
    out = cnns(x)

    
