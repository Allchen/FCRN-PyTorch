'''
Main function
'''
import path
import argparse

import torch
import visdom
import numpy as np
from torch.autograd import Variable

import Berhu
import nyud_dataset
import models.FCRN as FCRN
import models.upsample as upsample


def main(use_gpu=True, batch_size=32, epoch_num=20, start_epoch=0):
    vis = visdom.Visdom()
    win_RGB = vis.image(torch.ones(480, 640), opts=dict(title='GT'), env='FCRN')
    win_Depth = vis.image(torch.ones(256, 256), opts=dict(title='DG'), env='FCRN')
    win_Pred = vis.image(torch.ones(256, 256), opts=dict(title='Predicted'), env='FCRN')
    win_Loss = vis.image(torch.ones(3, 450, 450), opts=dict(title='Refinement loss'), env='FCRN')

    train_data_loader = nyud_dataset.get_nyud_train_set((304, 228), 1)
    fcrn = FCRN.FCRN(up_conv=upsample.UpProjection2d)
    optimizer = torch.optim.SGD(
        fcrn.parameters(),
        lr=1e-2,
        momentum=0.9
        )
    #loss_function = Berhu.BerhuLoss()
    loss_function = torch.nn.L1Loss()

    training_loss = []
    for epoch_i in range(start_epoch, start_epoch+epoch_num):
        train_data_iter = iter(train_data_loader)
        for batch_i in range(len(train_data_iter)):
            data = train_data_iter.next()
            inputs_o, inputs, gts = data
            inputs = Variable(inputs)
            gts = Variable(gts)
            gts = torch.nn.functional.upsample(
                gts, size=(128, 160), mode='bilinear'
                )
            outputs = fcrn(inputs)

            loss = loss_function(outputs, gts)
            loss.backward()
            optimizer.step()
            loss = loss.data[0]
            print('e{0:d}_b{1:d}: loss={2:f}'.format(epoch_i, batch_i, loss))

            training_loss.append(loss)
            vis.image(
                inputs_o[0, :, :, :], opts=dict(title='RGB'),
                env='FCRN', win=win_RGB
                )
            vis.image(
                gts.data[0, :, :, :], opts=dict(title='Depth'),
                env='FCRN', win=win_Depth
                )
            vis.image(
                outputs.data[0, :, :, :], opts=dict(title='Prediction'),
                env='FCRN', win=win_Pred
                )
            vis.line(
                np.array(training_loss), np.arange(len(training_loss)),
                opts=dict(title='Training Loss'), env='FCRN',
                win=win_Loss
                )
    return


if __name__ == '__main__':
    configures = argparse.ArgumentParser()
    configures.add_argument(
        '--cuda', required=False, default=False, action='store_true'
        )
    configures.add_argument(
        '--batch_size', required=False, type=int, default=16
        )
    configures.add_argument(
        '--epoch_num', required=False, type=int, default=20
        )
    configures.add_argument(
        '--start_epoch', required=False, type=int, default=0
        )
    configures = configures.parse_args()

    main(
        use_gpu=configures.cuda, batch_size=configures.batch_size,
        epoch_num=configures.epoch_num,
        start_epoch=configures.start_epoch
        )
