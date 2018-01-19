'''
Fully Convolutional Residual Network
'''
import torch.nn as nn
import torchvision
import resnet


class FCRN(nn.Module):
    '''
    Fully Convolutional Residual Network
    '''
    def __init__(self, up_conv, use_gpu=False):
        super(FCRN, self).__init__()
        self.res = resnet.resnet50(pretrained=True)
        self.conv_hidden = nn.Conv2d(2048, 1024, 1)

        self.up_conv1 = up_conv(1024, 512, use_gpu)
        self.up_conv2 = up_conv(512, 256, use_gpu)
        self.up_conv3 = up_conv(256, 128, use_gpu)
        self.up_conv4 = up_conv(128, 64, use_gpu)
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.ReLU(inplace=True)
            )

        if use_gpu:
            self.res = self.res.cuda()
            self.conv_hidden = self.conv_hidden.cuda()
            self.up_conv1 = self.up_conv1.cuda()
            self.up_conv2 = self.up_conv2.cuda()
            self.up_conv3 = self.up_conv3.cuda()
            self.up_conv4 = self.up_conv4.cuda()
            self.conv_final = self.conv_final.cuda()
        return

    def forward(self, inputs):
        inputs = self.res(inputs)
        inputs = self.conv_hidden(inputs)

        inputs = self.up_conv1(inputs)
        inputs = self.up_conv2(inputs)
        inputs = self.up_conv3(inputs)
        inputs = self.up_conv4(inputs)
        inputs = self.conv_final(inputs)
        return inputs
