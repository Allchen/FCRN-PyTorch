import torch
import torchvision.transforms as transforms
import numpy as np
import numpy.random as random
from PIL import Image
from rgbd import RGBDFolder
from transform_mutual import TransformMutual


def get_nyud_train_set(image_size=(480, 640), batch_size=16,
                       root_rgb='data/nyud/train/RGB/',
                       root_depth='data/nyud/train/depth_tensor/',
                       data_augmentation=True
                       ):
    if not data_augmentation:
        transform_rgb = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
            ])
        transform_depth = None
        transform_mutual = None

    else:
        def transform_rgb(rgb, mutual):
            # Scale, Rotation and Translation
            s = mutual.rescale
            transform_data_aug = transforms.Compose([
                transforms.Scale((int(image_size[0]*s), int(image_size[1]*s))),
                transforms.Lambda(lambda img: img.rotate(mutual.rotate)),
                transforms.Lambda(lambda img: img.crop((mutual.translation_x1,
                                                        mutual.translation_y1,
                                                        mutual.translation_x2,
                                                        mutual.translation_y2)))
                ])
            rgb = transform_data_aug(rgb)

            # Flip
            if mutual.flip is True:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)

            # Color Transform
            rgb_factor = np.ones(3) * 0.8 + random.rand(3) * 0.4
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            mean /= rgb_factor
            std /= rgb_factor
            transform_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
                ])
            rgb = transform_tensor(rgb)

            return rgb

        def transform_depth(depth, mutual):
            # Scale, Rotation and Translation
            s = mutual.rescale
            transform_data_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Scale((int(image_size[0]*s), int(image_size[1]*s))),
                transforms.Lambda(lambda img: img.rotate(mutual.rotate)),
                transforms.Lambda(lambda img: img.crop((mutual.translation_x1,
                                                        mutual.translation_y1,
                                                        mutual.translation_x2,
                                                        mutual.translation_y2)))
                ])
            depth = transform_data_aug(depth)

            # Flip
            if mutual.flip is True:
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

            # Convert depth map from PIL to torch.Tensor
            transform_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda img: img/s)
                ])
            depth = transform_tensor(depth)
            depth = depth[0, :, :]

            return depth

        transform_mutual = TransformMutual(
                                img_size=(480, 640),
                                output_size=image_size,
                                rescale_range=(1, 1.5)
                                )

    train_set = RGBDFolder(
        root=root_rgb,
        root_depth=root_depth,
        transform_rgb=transform_rgb,
        transform_depth=transform_depth,
        transform_mutual=transform_mutual
        )
    train_set_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
        )
    return train_set_loader


def get_nyud_test_set(image_size=(256, 256),
                      root_rgb='data/nyud/test/RGB/',
                      root_depth='data/nyud/test/depth_tensor/'
                      ):

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
        root=root_rgb,
        root_depth=root_depth,
        transform=transformer
        )
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=4
        )
    return test_set_loader
