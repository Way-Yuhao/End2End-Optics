import cv2
import torch
import numpy as np
import os
from end2end.edof_reader import ImageFolder
from tqdm import tqdm
from matplotlib import pyplot as plt
from config import CUDA_DEVICE

dataset_path = "/mnt/data1/yl241/datasets/Div2K/train/"
version = "v0"


def load_data(dataset_path):
    data_loader = torch.utils.data.DataLoader(
        ImageFolder(input_dir=dataset_path, img_patch_size=(512, 512), input_transform=None, load_all=False,
                    monochrome=False, augment=False), batch_size=16, num_workers=0
    )
    return data_loader


def disp_plt(img, title="", idx=None):
    """
    :param img: image to display
    :param title: title of the figure
    :param idx: index of the file, for print purposes
    :param tone_map: applies tone mapping via cv2 if set to True
    :return: None
    """
    img = img.detach().clone()

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    if img.shape[1] == 3:  # RGB
        img = img.cpu().squeeze().permute(1, 2, 0)
    else:  # monochrome
        img = img.cpu().squeeze()
        img = torch.stack((img, img, img), dim=0).permute(1, 2, 0)
    img = np.float32(img)
    plt.imshow(img)
    # compiling title
    if idx:
        title = "{} (index {})".format(title, idx)
    full_title = "{} / {}".format(version, title)
    plt.title(full_title)

    return


def train_dev():
    train_loader = load_data(dataset_path)

    train_iter = iter(train_loader)
    for i in range(3):
        input_, depth = train_iter.next()
        print(input_.shape)
        print(depth.shape)
        # disp_plt(input_[2, :, :, :], idx=0)
        # plt.show()
        break


def multiply(a, b):
    c = torch.tensor([4]).to(CUDA_DEVICE)
    return a * b + c


def main():
    train_dev()


if __name__ == "__main__":
    main()