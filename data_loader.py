import os


def pil_loader(path):
    if not os.path.exists(path):
        return None
    from PIL import Image
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def numpy_loader(path):
    if not os.path.exists(path):
        return None
    import numpy as np
    return np.load(path)


def torch_tensor_loader(path):
    if not os.path.exists(path):
        return None
    import torch
    return torch.load(path)
