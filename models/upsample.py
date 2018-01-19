'''
Implementation of upsampling blocks:
UnPool2d, UpConv2d, UpProjection2d
'''
import torch
import torch.nn as nn
from torch.autograd import Variable


class UnPool2d(nn.Module):
    '''
    Unpooling Layer implemented following description in
    A. Dosovitskiy, J. Tobias Springberg, and T.Brox.
    <Learning to generate chairs with convolutional neural networks>
    '''
    def __init__(self, use_gpu=False):
        super(UnPool2d, self).__init__()
        kernel = torch.FloatTensor([[1, 0], [0, 0]])
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        if use_gpu:
            self.kernel = Variable(data=kernel).cuda()
        else:
            self.kernel = Variable(data=kernel)
        return

    def forward(self, inputs):
        in_batches = inputs.size()[0]
        in_channels = inputs.size()[1]
        inputs = inputs.view(
            in_batches * in_channels,
            1, inputs.size()[2], inputs.size()[3]
            )
        inputs = torch.nn.functional.conv_transpose2d(
            inputs, self.kernel, stride=2
            )
        inputs = inputs.view(
            in_batches, in_channels, inputs.size()[2], inputs.size()[3]
            )
        return inputs


class UpConv2d(nn.Module):
    '''
    Up-convolution Layer
    '''
    def __init__(self, in_channels, out_channels, use_gpu=False):
        super(UpConv2d, self).__init__()
        self.unpool = UnPool2d(use_gpu)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=2),
            nn.ReLU(inplace=True),
            )
        return

    def forward(self, *inputs):
        inputs = inputs[0]
        inputs = self.unpool(inputs)
        inputs = self.conv(inputs)
        return inputs


class UpProjection2d(nn.Module):
    '''
    Up-Porjection Layer
    '''
    def __init__(self, in_channels, out_channels, use_gpu=False):
        super(UpProjection2d, self).__init__()
        self.unpool = UnPool2d(use_gpu)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
        self.conv_projection = nn.Conv2d(
            in_channels, out_channels, 5, padding=2
            )
        self.relu = nn.ReLU(inplace=True)
        return

    def forward(self, inputs):
        inputs = self.unpool(inputs)
        inputs_0 = self.conv(inputs)
        inputs_1 = self.conv_projection(inputs)
        inputs = self.relu(inputs_0 + inputs_1)
        return inputs
