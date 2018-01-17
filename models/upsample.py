import torch
import torch.nn as nn
from torch.autograd import Variable


class UnPool2d(nn.Module):
    '''
    Unpooling Layer implemented following description in
    A. Dosovitskiy, J. Tobias Springberg, and T.Brox.
    <Learning to generate chairs with convolutional neural networks>
    '''
    def __init__(self):
        super(UnPool2d, self).__init__()
        kernel = torch.FloatTensor([[1, 0], [0, 0]])
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        self.kernel = Variable(data=kernel, requires_grad=False)
        self.requires_grad = False
        return

    def forward(self, *inputs):
        inputs = inputs[0]
        in_batches = inputs.size()[0]
        in_channels = inputs.size()[1]
        inputs = inputs.view(in_batches * in_channels, 1, inputs.size()[2], inputs.size()[3])
        inputs = torch.nn.functional.conv_transpose2d(inputs, self.kernel, stride=2)
        inputs = inputs.view(in_batches, in_channels, inputs.size()[2], inputs.size()[3])
        return inputs


class UpConv2d(nn.Module):
    '''
    Up-convolution Layer
    '''
    def __init__(self, in_channels, out_channels):
        super(UpConv2d, self).__init__()
        self.unconv = UnPool2d()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        return

    def forward(self, *inputs):
        inputs = inputs[0]
        inputs = self.unconv(inputs)
        inputs = self.conv(inputs)
        return inputs


def test():
    '''
    Fuck
    '''
    x = torch.Tensor(1, 3, 12, 12)
    x = Variable(x)

    print('\nTesting UnConv2d Layer:\n')
    print(x.size())
    u = UnPool2d()
    y = u(x)
    print(y.size())

    print('\nTesting UnConv2d Layer:\n')
    print(x.size())
    u = UpConv2d(3, 64)
    y = u(x)
    print(y.size())

    print('\nEnd Test')
    return


if __name__ == '__main__':
    test()
