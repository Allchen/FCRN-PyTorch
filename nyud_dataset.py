'''
Loading NYUD v2 datasets
'''
import torch
import torchvision.transforms as transforms
from folder_s import NYUDFolder

def get_nyud_train_set(image_size=None, batch_size=32):

    if image_size is not None:
        if not isinstance(image_size, tuple):
            print('get_nyud_train_set: invalid size')
            return -1

    if image_size is None:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            ])
    else:
        transformer = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    train_set = NYUDFolder(
        root='data/nyud/train/RGB/',
        root_depth='data/nyud/train/depth_real/',
        transform=transformer
        )
    train_set_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
        )
    return train_set_loader


def get_nyud_test_set(image_size=(256, 256)):

    if image_size is not None:
        if not isinstance(image_size, tuple):
            print('get_nyud_train_set: invalid size')
            return -1

    transformer = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    test_set = NYUDFolder(
        root='data/nyud/test/RGB/',
        root_depth='data/nyud/test/depth_real/',
        transform=transformer
        )
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=4
        )
    return test_set_loader

def get_nyud_unnorm():

    unnorm = transforms.Compose([
        transforms.Normalize((0, 0, 0), (4.3668, 4.4642, 4.4444)),
        transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1))
        ])

    return unnorm

def transform_sullivan(rgb):

    transformer = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    rgb = transformer(rgb)
    processed_rgb = torch.nn.functional.upsample(
                                    rgb,
                                    size=(256, 256),
                                    mode='bilinear'
                                    )

    return processed_rgb
