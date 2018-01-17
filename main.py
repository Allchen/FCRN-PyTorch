import torch
import visdom
import nyud_dataset


def main():
    vis = visdom.Visdom()
    win_gt = vis.image(torch.zeros(256, 256), opts=dict(title='GT'), env='FCRN')

    train_data_loader = nyud_dataset.get_nyud_train_set((304, 228), 1)
    train_data_iter = iter(train_data_loader)
    data = train_data_iter.next()
    inputs_o, inputs, gt = data

    print(inputs_o.size())
    print(inputs.size())
    print(gt.size())


    vis.image(inputs[0, :, :, :], win=win_gt, opts=dict(title='Test'), env='FCRN')
    return


if __name__ == '__main__':
    main()
