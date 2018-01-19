'''
Main function
'''
import path
import argparse

import torch
import visdom
import numpy as np
from torch.autograd import Variable

import nyud_dataset
import models.FCRN as FCRN
import models.upsample as upsample


def main(use_gpu=True, batch_size=32, epoch_num=20, start_epoch=0, learning_rate=1e-3):
    print('FCRN\nInitializing...')

    vis = visdom.Visdom()
    win_rgb = vis.image(torch.zeros(240, 320), opts=dict(title='GT'), env='FCRN')
    win_Depth = vis.image(torch.zeros(240, 320), opts=dict(title='DG'), env='FCRN')
    win_Pred = vis.image(torch.zeros(240, 320), opts=dict(title='Predicted'), env='FCRN')
    win_Training_Loss = vis.image(torch.zeros(3, 450, 450), opts=dict(title='Refinement loss'), env='FCRN')
    win_Epoch_Loss = vis.image(torch.zeros(3, 450, 450), opts=dict(title='Refinement loss'), env='FCRN')

    train_data_loader = nyud_dataset.get_nyud_train_set((304, 228), 1)
    fcrn = FCRN.FCRN(up_conv=upsample.UpProjection2d, use_gpu=use_gpu)
    optimizer = torch.optim.Adam(
        fcrn.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999)
        )
    loss_function = torch.nn.MSELoss()
    if use_gpu:
        loss_function = loss_function.cuda()

    training_losses = []
    epoch_losses = []
    for epoch_i in range(start_epoch, start_epoch+epoch_num):
        train_data_iter = iter(train_data_loader)
        epoch_loss = 0
        for batch_i in range(len(train_data_iter)):
            data = train_data_iter.next()
            inputs_o, inputs, gts = data
            inputs = Variable(inputs)
            gts = Variable(gts)
            gts = torch.nn.functional.upsample(
                gts, size=(128, 160), mode='bilinear'
                )
            if use_gpu:
                inputs = inputs.cuda()
                gts = gts.cuda()

            outputs = fcrn(inputs)

            loss = loss_function(outputs, gts)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().data[0]
            epoch_loss += loss
            #print('e{0:d}_b{1:d}: loss={2:f}'.format(epoch_i, batch_i, loss))

            training_losses.append(loss)
            if len(training_losses) > 3000:
                training_losses = training_losses[1:]
            vis.image(
                inputs_o[0, :, :, :].cpu(), opts=dict(title='RGB'),
                env='FCRN', win=win_rgb
                )
            vis.image(
                gts.data[0, :, :, :].cpu(), opts=dict(title='Depth'),
                env='FCRN', win=win_Depth
                )
            vis.image(
                outputs.data[0, :, :, :].cpu(), opts=dict(title='Prediction'),
                env='FCRN', win=win_Pred
                )
            vis.line(
                np.array(training_losses), np.arange(len(training_losses)),
                opts=dict(title='Training Loss'), env='FCRN',
                win=win_Training_Loss
                )
        epoch_loss /= len(train_data_iter)
        epoch_losses.append(epoch_loss)
        print('e{0:d} loss={1:f}'.format(epoch_i, epoch_loss))
        vis.line(
            np.array(epoch_losses), np.arange(len(epoch_losses)),
            opts=dict(title='Epoch Loss'), env='FCRN',
            win=win_Epoch_Loss
            )
        torch.save(fcrn.state_dict(), 'parameters/FCRN_epoch{0:d}.pth'.format(epoch_i))
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
    configures.add_argument(
        '--learning_rate', required=False, type=float, default=1e-3
        )
    configures = configures.parse_args()

    main(
        use_gpu=configures.cuda, batch_size=configures.batch_size,
        epoch_num=configures.epoch_num,
        start_epoch=configures.start_epoch,
        learning_rate=configures.learning_rate
        )
