"""
Data loader file
"""

import os
import os.path as p
import numpy as np
import torch
import torchvision.transforms as transforms
from natsort import natsorted
from torchvision.datasets.vision import VisionDataset

DEPTH_OPTIONS = torch.tensor([1/2, 1/1.5, 1/1, 1/0.5, 1000])  # depth in meters, where 1000 approximates inf


def cv_loader(path):
    """
    loads .hdr file via cv2, then converts color to rgb
    :param path: path to image file
    :return: img ndarray
    """
    img = cv2.imread(path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    return img


def cvt_monochrome(input_):
    input_ = input_[1, :, :]
    input_ = torch.stack((input_, input_, input_), dim=0)
    return input_


class ImageFolder(VisionDataset):

    def __init__(self, input_dir, img_patch_size=(1024, 1024), input_transform=None, load_all=False, monochrome=False, augment=False):
        """
        :param input_dir: path to input directory
        :param input_transform:
        :param load_all:
        :param monochrome:
        :param augment:
        """
        # super().__init__(root)
        self.input_dir = input_dir
        self.load_all = load_all
        self.img_patch_size = img_patch_size
        self.isMonochrome = monochrome
        self.augment = augment

        self.inputs = natsorted(os.listdir(input_dir))
        if input_transform is not None:
            self.input_transform = input_transform
        else:
            self.input_transform = transforms.Compose([transforms.ToTensor])

        if self.load_all:  # load entire dataset to memory
            raise NotImplementedError

    def __len__(self):
        pass  # TODO

    def __getitem__(self, item_idx):
        if not self.load_all:
            input_sample = cv_loader(p.join(self.input_dir, self.inputs[item_idx]))
        else:
            raise NotImplementedError
        input_sample = self.input_transform(input_sample)
        # TODO: data augmentation
        input_sample = self.random_crop(input_sample)
        input_sample = self.normalize(input_sample)
        if self.isMonochrome:
            input_sample = cvt_monochrome(input_sample)
        return input_sample, self.generate_depth_map()

    def generate_depth_map(self):
        """
        arbitrarily assign a planar depth for a given input image. The target distance is sampled using
        a multinomial distribution on DEPTH_OPTIONS
        :return: planar depth map of shape self.image_patch_size
        """
        rand_depth_idx = np.random.multinomial(1, [1 / 5] * 5)
        rand_depth = DEPTH_OPTIONS[np.argmax(rand_depth_idx)]
        rand_depth_map = torch.ones(self.img_patch_size, dtype=torch.float32) * rand_depth
        return rand_depth_map

    def augment(self):
        raise NotImplementedError  # TODO

    def random_crop(self, input_):
        crop_width, crop_height = self.img_patch_size
        max_h = input_.shape[1] - crop_height
        max_w = input_.shape[2] - crop_width
        h = np.random.randint(0, max_h)
        w = np.random.randint(0, max_w)
        input_crop = input_[:, h: h + crop_height, w: w + crop_width]
        return input_crop

    def normalize(self, input_):
        return input_ / 255.  # as indicated in deep-optics
