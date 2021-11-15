import cv2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import os.path as p

import end2end.optics.optics_utils
from end2end.edof_reader import ImageFolder
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.optim as optim
import optics

"""Global Parameters"""
div2k_dataset_path = "/mnt/data1/yl241/datasets/Div2K/train/"
version = None
num_workers_train = 16
batch_size = 16

"""Hyper Parameters"""
init_lr = 0.01
epoch = 2000


def set_device(devidx=6):
    """
    Sets device to CUDA if available
    :return: CUDA device 0, if available
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(devidx))
        print("CUDA is available. Training on GPU")
    else:
        device = "cpu"
        print("CUDA is unavailable. Training on CPU")
    return device


def load_data(dataset_path):
    data_loader = torch.utils.data.DataLoader(
        ImageFolder(input_dir=dataset_path, img_patch_size=(512, 512), input_transform=None, load_all=False,
                    monochrome=False, augment=False), batch_size=16, num_workers=0)
    return data_loader


def print_params():
    raise NotImplementedError


def load_network_weights(net, path):
    raise NotImplementedError


def save_network_weights(net, ep):
    raise NotImplementedError


def compute_loss(output, target, heightmap, scale):
    """

    :param scale: scalar constant
    :param output:
    :param target:
    :param heightmap:
    :return: mse loss between output image and target image + scaled laplacian l1 regularizer on the heightmap
    """
    mse_criterion = nn.MSELoss()
    mse_loss = mse_criterion(output, target)
    with torch.no_grad():
        laplacian_regularizer = end2end.optics.optics_utils.laplace_l1_regularizer(img_batch=heightmap, scale=scale)
    total_loss = mse_loss + laplacian_regularizer
    return total_loss


def train_dev(net, device, tb, load_weights=False, pre_trained_params_path=None):
    print(device)
    print_params()
    net.to(device)
    net.train()

    if load_weights:
        load_network_weights(net, pre_trained_params_path)

    train_loader = load_data(p.join(div2k_dataset_path, "train"))

    num_mini_batches = len(train_loader)
    # TODO: modify this
    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=.8)

    running_train_loss = 0.0
    # training loop
    for ep in range(epoch):
        print("Epoch ", ep)
        train_iter = iter(train_loader)

        for _ in tqdm(range(num_mini_batches)):
            input_, depth = train_iter.next()
            # TODO: to device
            optimizer.zero_grad()
            # TODO: forward pass
            loss = compute_loss()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        # record loss values after each epoch
        cur_train_loss = running_train_loss / num_mini_batches
        tb.add_scalar('loss/train', cur_train_loss, ep)
        # TODO: dev

        running_train_loss = 0.0
        scheduler.step()
        # TODO: display samples per some epochs

    print("finished training")
    save_network_weights(net, ep="{}_FINAL".format(epoch))
    return


def main():
    version_ = "-v0.0"
    param_to_load = None
    tb = SummaryWriter('./runs/unet' + version_)
    device = set_device()
    net = None  # TODO
    train_dev(net, device, tb, load_weights=False, pre_trained_params_path=param_to_load)

    tb.close()

if __name__ == "__main__":
    main()
