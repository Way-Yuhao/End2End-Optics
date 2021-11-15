import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import os.path as p
from tqdm import tqdm
from matplotlib import pyplot as plt
import end2end.optics.optics_utils
from end2end.edof_reader import ImageFolder
from end2end.model import RGBCollimator

"""Global Parameters"""
div2k_dataset_path = "/mnt/data1/yl241/datasets/Div2K/"
version = None
num_workers_train = 16
batch_size = 16

"""Hyper Parameters"""
init_lr = 0.01
epoch = 2000

"""Simulation Parameters"""
aperture_diameter = 5e-3 # (m)
sensor_distance = 25e-3  # Distance of sensor to aperture (m)
refractive_idcs = np.array([1.4648, 1.4599, 1.4568])  # Refractive idcs of the phaseplate
wave_lengths = np.array([460, 550, 640]) * 1e-9  # Wave lengths to be modeled and optimized for
ckpt_path = None
num_steps = 10001  # Number of SGD steps
patch_size = 1248  # Size of patches to be extracted from images, and resolution of simulated sensor
sample_interval = 2e-6  # Sampling interval (size of one "pixel" in the simulated wavefront)
wave_resolution = 2496, 2496  # Resolution of the simulated wavefront
height_map_noise = 20e-9
hm_reg_scale = 1000.

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
    print("######## Basics ##################")
    print("version: {}".format(version))
    # TODO: add more


def load_network_weights(net, path):
    raise NotImplementedError


def save_network_weights(net, ep):
    raise NotImplementedError


def compute_loss(output, target, heightmap):
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
        laplacian_regularizer = \
            end2end.optics.optics_utils.laplace_l1_regularizer(img_batch=heightmap, scale=hm_reg_scale)
    total_loss = mse_loss + laplacian_regularizer
    return total_loss


def train_dev(net, device, tb, load_weights=False, pre_trained_params_path=None):
    print(device)
    print_params()
    # net.to(device) # TODO: change back
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
            # input_, depth = input_.to(device), depth.to(device)
            # net.to(device) TODO: change back
            optimizer.zero_grad()
            output = net(input_)
            loss = compute_loss(output=output, target=input_, heightmap=net.height_map)
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
    global version
    version = "-v0.0"
    param_to_load = None
    tb = SummaryWriter('./runs/RGBCollimator' + version)
    device = set_device()
    net = RGBCollimator(sensor_distance=sensor_distance, refractive_idcs=refractive_idcs, wave_lengths=wave_lengths,
                        patch_size=patch_size, sample_interval=sample_interval, wave_resolution=wave_resolution,
                        height_map_noise=height_map_noise)  # TODO
    train_dev(net, device, tb, load_weights=False, pre_trained_params_path=param_to_load)

    tb.close()


if __name__ == "__main__":
    main()
