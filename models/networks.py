from components import *

class SketchNet(nn.Module):
    """SketchNet, transform input RGB photo to gray sketch.
    ---------------------
    A U-Net architecture similar to: https://arxiv.org/pdf/1603.08155.pdf
    Codes borrowed from: https://github.com/pytorch/examples/tree/master/fast_neural_style
    """
    def __init__(self, in_channels=3, out_channels=1, norm_type='IN'):
        super(SketchNet, self).__init__()
        # Downsample convolution layers
        self.conv1 = ConvLayer(in_channels, 32, kernel_size=3, stride=1, bias=False)
        self.norm1 = NormLayer(32, norm_type)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, bias=False)
        self.norm2 = NormLayer(64, norm_type)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2, bias=False)
        self.norm3 = NormLayer(128, norm_type)

        # Residual layers
        self.res1 = ResidualBlock(128, norm_type)
        self.res2 = ResidualBlock(128, norm_type)
        self.res3 = ResidualBlock(128, norm_type)
        self.res4 = ResidualBlock(128, norm_type)
        self.res5 = ResidualBlock(128, norm_type)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(256, 64, kernel_size=3, stride=1, bias=False, upsample=2)
        self.norm4 = NormLayer(64, norm_type)
        self.deconv2 = UpsampleConvLayer(128, 32, kernel_size=3, stride=1, bias=False, upsample=2)
        self.norm5 = NormLayer(32, norm_type)
        self.deconv3 = ConvLayer(64, out_channels, kernel_size=3, stride=1, bias=True)

        # Non-linear layer
        self.relu = nn.ReLU(True)

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


class DNet(nn.Module):
    """Discriminator network.
    """
    def __init__(self, in_channels=1, norm_type='IN'):
        super(DNet, self).__init__()
        b = True if norm_type == 'none' else False
        self.net = nn.Sequential(
                ConvLayer(in_channels, 32, kernel_size=3, stride=2, bias=True),
                nn.ReLU(inplace=True),
                ConvLayer(32, 64, kernel_size=3, stride=2, bias=b),
                NormLayer(64, norm_type),
                nn.ReLU(inplace=True),
                ConvLayer(64, 128, kernel_size=3, stride=2, bias=b),
                NormLayer(128, norm_type),
                nn.ReLU(inplace=True),
                ConvLayer(128, 256, kernel_size=3, stride=2, bias=b),
                NormLayer(256, norm_type), 
                nn.ReLU(inplace=True),
                ConvLayer(256, 1, kernel_size=3, stride=1, bias=True),
                )

    def forward(self, x):
        return self.net(x) 



