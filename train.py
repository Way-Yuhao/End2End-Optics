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
from config import CUDA_DEVICE
import end2end.optics.optics_utils
from end2end.edof_reader import ImageFolder
from end2end.model import RGBCollimator

"""Global Parameters"""
div2k_dataset_path = "/mnt/data1/yl241/datasets/Div2K/"
version = None
num_workers_train = 8
batch_size = 8

"""Hyper Parameters"""
init_lr = 1e-4
epoch = 2000

"""Simulation Parameters"""
aperture_diameter = 5e-3  # (m)
sensor_distance = 25e-3  # Distance of sensor to aperture (m)
refractive_idcs = np.array([1.4648, 1.4599, 1.4568])  # Refractive idcs of the phaseplate
wave_lengths = np.array([460, 550, 640]) * 1e-9  # Wave lengths to be modeled and optimized for
# wave_lengths = np.array([550, 550, 550]) * 1e-9  # monochrome
ckpt_path = None
num_steps = 10001  # Number of SGD steps
patch_size = 512  # Size of patches to be extracted from images, and resolution of simulated sensor
sample_interval = 2e-6  # Sampling interval (size of one "pixel" in the simulated wavefront)
wave_resolution = 2496, 2496  # Resolution of the simulated wavefront
# wave_resolution = 512, 512  # Resolution of the simulated wavefront FIXME
height_tolerance = 20e-9
hm_reg_scale = 1000.


def load_data(dataset_path):
    data_loader = torch.utils.data.DataLoader(
        ImageFolder(input_dir=dataset_path, img_patch_size=(patch_size, patch_size), input_transform=None, load_all=False,
                    monochrome=False, augment=False), batch_size=batch_size, num_workers=num_workers_train)
    return data_loader


def print_params():
    print("######## Basics ##################")
    print("version: {}".format(version))
    print("Training on {}".format(CUDA_DEVICE))


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


def train_dev(net, device, tb, load_weights=False, pre_trained_params_path=None):
    print_params()
    net.to(CUDA_DEVICE)
    net.train()

    if load_weights:
        load_network_weights(net, pre_trained_params_path)

    train_loader = load_data(p.join(div2k_dataset_path, "small_50"))

    num_mini_batches = len(train_loader)
    # TODO: modify this
    optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=.8)

    running_train_loss = 0.0 # per epoch
    # training loop
    for ep in range(epoch):
        print("Epoch ", ep)
        train_iter = iter(train_loader)

        for i in tqdm(range(num_mini_batches)):
            input_, depth = train_iter.next()
            input_, depth = input_.to(CUDA_DEVICE), depth.to(CUDA_DEVICE)
            optimizer.zero_grad()
            output, psf = net(input_)
            loss = compute_loss(output=output, target=input_, heightmap=net.heightMapElement.height_map)

            tb.add_scalar('loss/train/micro', loss.item(), ep * num_mini_batches + i)
            tb.add_image('normalized_psf', psf[0, :, :, :] / psf.max(), global_step=ep * num_mini_batches + i)

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            torch.cuda.empty_cache()

        # record loss values after each epoch
        print(running_train_loss)
        cur_train_loss = running_train_loss / num_mini_batches
        tb.add_scalar('loss/train', cur_train_loss, ep)
        # TODO: dev

        running_train_loss = 0.0
        # scheduler.step()
        # TODO: display samples per some epochs

    print("finished training")
    save_network_weights(net, ep="{}_FINAL".format(epoch))
    return


def main():
    global version
    version = "-v0.5.2"
    param_to_load = None
    tb = SummaryWriter('./runs/RGBCollimator' + version)
    net = RGBCollimator(sensor_distance=sensor_distance, refractive_idcs=refractive_idcs, wave_lengths=wave_lengths,
                        patch_size=patch_size, sample_interval=sample_interval, wave_resolution=wave_resolution,
                        height_tolerance=height_tolerance)
    with torch.cuda.device(CUDA_DEVICE):
        train_dev(net, CUDA_DEVICE, tb, load_weights=False, pre_trained_params_path=param_to_load)

    tb.close()


if __name__ == "__main__":
    main()
