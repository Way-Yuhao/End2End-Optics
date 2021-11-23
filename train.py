import cv2
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import os.path as p
from tqdm import tqdm
from matplotlib import pyplot as plt
from config import CUDA_DEVICE
import end2end.optics.optics_utils
from end2end.edof_reader import ImageFolder
from end2end.model import RGBCollimator, RGBCollimator_Fourier

"""Global Parameters"""
div2k_dataset_path = "/mnt/data1/yl241/datasets/Div2K/"
version = None
num_workers_train = 8
batch_size = 8

"""Hyper Parameters"""
init_lr = 5e-1
epoch = 2000

"""Simulation Parameters"""
aperture_diameter = 5e-3  # (m)
sensor_distance = 25e-3  # Distance of sensor to aperture (m)
refractive_idcs = np.array([1.4648, 1.4599, 1.4568])  # Refractive idcs of the phaseplate
wave_lengths = np.array([460, 550, 640]) * 1e-9  # Wave lengths to be modeled and optimized for
# wave_lengths = np.array([550, 550, 550]) * 1e-9  # monochrome
ckpt_path = None
num_steps = 10001  # Number of SGD steps FIXME not used
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


def tensorboard_vis(tb, ep, psf=None, height_map=None, train_output=None, plt_1d_psf=True):
    if psf is not None:
        tb.add_image('normalized_psf', psf[0, :, :, :] / psf.max(), global_step=ep)
    if height_map is not None:
        tb.add_image('normalized_height_map', height_map[0, :, ::4, ::4] / height_map.max(), global_step=ep)
    if train_output is not None:
        output_img_grid = torchvision.utils.make_grid(train_output)
        tb.add_image("train_outputs", output_img_grid, global_step=ep)
    if plt_1d_psf:
        psf_plot = torch.sum(psf, dim=2)
        psf_plot = psf_plot.cpu().detach().numpy()
        fig, ax = plt.subplots()
        ax.plot(psf_plot[0, 0, :], c='r')
        ax.plot(psf_plot[0, 1, :], c='g')
        ax.plot(psf_plot[0, 2, :], c='b')
        tb.add_figure(tag="1D_psf", figure=fig, global_step=ep)
    return


def train_dev(net, tb, load_weights=False, pre_trained_params_path=None):
    print_params()
    net.to(CUDA_DEVICE)
    net.train()

    if load_weights:
        load_network_weights(net, pre_trained_params_path)

    train_loader = load_data(p.join(div2k_dataset_path, "small_50"))

    num_mini_batches = len(train_loader)
    optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=.8)

    running_train_loss = 0.0  # per epoch
    psf, height_map, output = None, None, None
    # training loop
    for ep in range(epoch):
        print("Epoch ", ep)
        train_iter = iter(train_loader)
        for i in tqdm(range(num_mini_batches)):
            input_, depth = train_iter.next()
            input_, depth = input_.to(CUDA_DEVICE), depth.to(CUDA_DEVICE)
            optimizer.zero_grad()
            output, psf, height_map = net(input_)
            loss = compute_loss(output=output, target=input_, heightmap=net.heightMapElement.height_map)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            # torch.cuda.empty_cache()

        # record loss values after each epoch
        print(running_train_loss)
        cur_train_loss = running_train_loss / num_mini_batches
        tb.add_scalar('loss/train', cur_train_loss, ep)
        if ep % 10 == 9:
            tensorboard_vis(tb, ep, psf=psf, height_map=height_map, train_output=output, plt_1d_psf=True)
            raise Exception()

        # TODO: dev
        running_train_loss = 0.0
        # scheduler.step()

    print("finished training")
    save_network_weights(net, ep="{}_FINAL".format(epoch))
    return


def main():
    global version
    version = "-v1.0.5-test"
    param_to_load = None
    tb = SummaryWriter('./runs/RGBCollimator' + version)  # TODO rename this
    # simple lens
    # net = RGBCollimator(sensor_distance=sensor_distance, refractive_idcs=refractive_idcs, wave_lengths=wave_lengths,
    #                     patch_size=patch_size, sample_interval=sample_interval, wave_resolution=wave_resolution,
    #                     height_tolerance=height_tolerance)

    # Fourier system
    net = RGBCollimator_Fourier(sensor_distance=sensor_distance, refractive_idcs=refractive_idcs, wave_lengths=wave_lengths,
                        patch_size=patch_size, sample_interval=sample_interval, wave_resolution=wave_resolution,
                        height_tolerance=height_tolerance)

    train_dev(net, tb, load_weights=False, pre_trained_params_path=param_to_load)
    tb.close()


if __name__ == "__main__":
    main()
