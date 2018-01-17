'''
Fully Convolutional Residual Network
'''
import torch.nn as nn
import resnet


class FCRN(nn.Module):
    '''
    Fully Convolutional Residual Network
    '''
    def __init__(self, up_conv):
        super(FCRN, self).__init__()
        self.res = resnet.resnet50(pretrained=True)
        self.conv_hidden = nn.Conv2d(2048, 1024, 1)

        self.up_conv1 = up_conv(1024, 512)
        self.up_conv2 = up_conv(512, 256)
        self.up_conv3 = up_conv(256, 128)
        self.up_conv4 = up_conv(128, 64)
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.ReLU(inplace=True)
            )
        return

    def forward(self, *inputs):
        inputs = inputs[0]
        inputs = self.res(inputs)
        inputs = self.conv_hidden(inputs)

        inputs = self.up_conv1(inputs)
        inputs = self.up_conv2(inputs)
        inputs = self.up_conv3(inputs)
        inputs = self.up_conv4(inputs)
        inputs = self.conv_final(inputs)
        return inputs
