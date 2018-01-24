import numpy as np
import numpy.random as random
import torchvision.transforms as transforms
from transforms_cv import *


def transform_rgb(rgb, mutual):
    if not (len(rgb.shape) == 3 and rgb.shape[2] == 3):
        msg = 'This function only support '
        msg += 'numpy array with 3 channels'
        raise TypeError(msg)
        return rgb

    # Perform scale, rotation and flip transformation on OpenCV image(numpy).
    if mutual.flip:
        rgb = hflip_transform(rgb)
    rgb = rescale_transform(rgb, (mutual.rescale, mutual.rescale))
    rgb = rotation_transform(rgb, mutual.rotate)
    rgb = crop_transform(
                    rgb, mutual.translation_x1,
                    mutual.translation_y1,
                    mutual.translation_x2,
                    mutual.translation_y2,
                    )
    rgb = cv2.resize(
        rgb, (mutual.output_size[0], mutual.output_size[1]),
        rgb, 0, 0,
        interpolation=cv2.INTER_LINEAR
        )

    # Convert OpenCV image into PyTorch Tensor
    rgb = np.transpose(rgb, [2, 0, 1])
    rgb = torch.from_numpy(rgb)

    # Perform random color transformation on PyTorch Tensor.
    color_factor = random.rand(3) * 0.4 + np.ones(3) * 0.8
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean /= color_factor
    std /= color_factor
    transform_final = transforms.Compose([
        transforms.Normalize(mean, std)
        ])
    rgb = transform_final(rgb)

    return rgb


def transform_depth(depth, mutual):
    depth = depth.numpy()

    # Perform scale, rotation and flip transformation on OpenCV image.
    if mutual.flip:
        depth = hflip_transform(depth)
    depth = rescale_transform(depth, (mutual.rescale, mutual.rescale))
    depth /= mutual.rescale
    depth = rotation_transform(depth, mutual.rotate)
    depth = crop_transform(
                    depth, mutual.translation_x1,
                    mutual.translation_y1,
                    mutual.translation_x2,
                    mutual.translation_y2,
                    )
    depth = cv2.resize(
        depth, (mutual.output_size[0], mutual.output_size[1]),
        depth, 0, 0,
        interpolation=cv2.INTER_LINEAR
        )

    # Convert depth map into PyTorch Tensor
    depth = torch.from_numpy(depth)
    depth = depth.unsqueeze(0)
    return depth
