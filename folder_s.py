'''
Modified Dataset class from PyTorch
'''
import os
import os.path
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [
        d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
        ]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    d = dir

    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderS(data.Dataset):
    """
    A simplified version of torchvision.ImageFolder.
    Args:
    root (string):
        Root directory path.
    transform (callable, optional):
        A function that takes in an PIL imageand returns a transformed version.
    loader (callable, optional):
        A function to load an image given its path.
    """

    def __init__(
            self,
            root,
            root_gt,
            transform=None,
            transform_gt=None,
            loader=default_loader,
            loader_gt=default_loader
            ):

        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: "+root+"\n"
                    "Supported image extensions are: "+",".join(IMG_EXTENSIONS)
                    )
                )
        gt = make_dataset(root_gt)
        if len(gt) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: "+root_gt+"\n"
                    "Supported image extensions are: "+",".join(IMG_EXTENSIONS)
                    )
                )
        if len(gt) != len(imgs):
            raise(
                RuntimeError(
                    "Size of image data and ground truth doesn't match\n"
                    "Found {0:n} images and \
                            {1:n} depth maps".format(len(imgs), len(gt))
                    )
                )

        self.root = root
        self.root_gt = root_gt
        self.imgs = imgs
        self.gt = gt

        self.loader = loader
        self.loader_gt = loader_gt
        self.transform = transform
        self.transform_gt = transform_gt

    def __getitem__(self, index=0):
        # load rgbd images
        path = self.imgs[index]
        img = self.transform(self.loader(path))

        # load ground truth
        path_gt = self.gt[index]
        gt = self.transform_gt(self.loader_gt(path_gt))

        # modify ground truth
        gt = gt[0, :, :]
        gt = gt.view(1, gt.size()[0], gt.size()[1])

        return img, gt

    def __len__(self):
        return len(self.imgs)


class RGBDFolder(ImageFolderS):
    """
    ImageFolderS class specifically defined for rgbd datasets/
    Args:
    root_rgb (string):
        Root directory path for rgb data.
    root_depth (string):
        Root directory path for depth data.
    transform (callable, optional):
        A function that takes in an PIL imageand returns a transformed version.
    loader (callable, optional):
        A function to load an image given its path.
    """
    def __init__(
            self,
            root_rgb,
            root_depth,
            root_gt,
            transform=transforms.ToTensor(),
            loader=default_loader
            ):

        # get file names of rgb images and depth maps
        imgs = make_dataset(root_rgb)
        if len(imgs) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: "+root+"\n"
                    "Supported image extensions are: "+",".join(IMG_EXTENSIONS)
                    )
                )
        depths = make_dataset(root_depth)
        if len(depths) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: "+root+"\n"
                    "Supported image extensions are: "+",".join(IMG_EXTENSIONS)
                    )
                )
        if len(depths) != len(imgs):
            raise(
                RuntimeError(
                    "Size of image data and depth map data doesn't match\n"
                    "Found {0:n} images and \
                            {1:n} depth maps".format(len(imgs), len(depths))
                    )
                )

        # get file names of ground truth
        gt = make_dataset(root_gt)
        if len(depths) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: "+root+"\n"
                    "Supported image extensions are: "+",".join(IMG_EXTENSIONS)
                    )
                )

        self.root_rgb = root_rgb
        self.root_depth = root_depth
        self.root_gt = root_gt
        self.imgs = imgs
        self.depths = depths
        self.gt = gt

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index=0):
        # load rgbd images
        path_img = self.imgs[index]
        rgb = self.transform(self.loader(path_img))

        # load depth maps
        path_depth = self.depths[index]
        d = self.transform(self.loader(path_depth))

        # load ground truth
        path_gt = self.gt[index]
        gt = self.transform(self.loader(path_gt))

        # merge rgb channel and depth channel
        rgbd = torch.Tensor(4, rgb.size()[1], rgb.size()[2])
        rgbd[0:3, :, :] = rgb
        rgbd[3, :, :] = d[0, :, :]

        # modify ground truth
        gt = gt[0, :, :]
        gt = gt.view(1, gt.size()[0], gt.size()[1])

        return rgbd, gt

    def __len__(self):
        return len(self.imgs)


class NYUDFolder(data.Dataset):

    def __init__(
            self,
            root,
            root_depth,
            transform=None,
            loader=default_loader
    ):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(
                RuntimeError(
                    "Found 0 images in subfolders of: "+root+"\n"
                    "Supported image extensions are: "+",".join(IMG_EXTENSIONS)
                    )
                )

        if not root_depth.endswith('/'):
            root_depth += '/'

        self.root = root
        self.root_depth = root_depth
        self.imgs = imgs

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index=0):
        # load rgbd images
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_o = transforms.ToTensor()(img)
            img = self.transform(img)

        # load depth map
        path_depth = self.root_depth + str(int(path[len(path)-9:len(path)-4]))
        path_depth += '.npy'
        depth = np.load(path_depth)

        depth = torch.from_numpy(depth).float()
        depth = depth.mul(0.1)
        depth = depth.unsqueeze(0)

        return img_o, img, depth

    def __len__(self):
        return len(self.imgs)
