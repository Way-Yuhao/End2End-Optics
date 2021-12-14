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
from end2end.config import CUDA_DEVICE
import end2end.optics.optics_utils
from end2end.edof_reader import ImageFolder
from end2end.model import RGBCollimator, RGBCollimator_Fourier, AchromaticEdofFourier
from end2end.edof_reader import DEPTH_OPTIONS

"""Global Parameters"""
div2k_dataset_path = "/mnt/data1/yl241/datasets/Div2K/"
network_weight_path = "./weight/"
model_name = None
version = None
num_workers_train = 4
batch_size = 4

"""Hyper Parameters"""
init_lr = 5e-1
epoch = 2000

"""Simulation Parameters"""
aperture_diameter = 5e-3  # (m)
sensor_distance = 35.5e-3  # Distance of sensor to aperture (m)
refractive_idcs = np.array([1.4648, 1.4599, 1.4568])  # Refractive idcs of the phase plate
wave_lengths = np.array([460, 550, 640]) * 1e-9  # Wave lengths to be modeled and optimized for
# wave_lengths = np.array([550, 550, 550]) * 1e-9  # monochrome
ckpt_path = None
num_steps = 10001  # Number of SGD steps FIXME not used
patch_size = 512  # Size of patches to be extracted from images, and resolution of simulated sensor
# FIXME: REMOVE *2 !!!!!!!!!!!
sample_interval = 2e-6 * 2  # Sampling interval (size of one "pixel" in the simulated wavefront)
# wave_resolution = 2496, 2496  # Resolution of the simulated wavefront
wave_resolution = 1248, 1248
height_tolerance = 20e-9
hm_reg_scale = 1000.


def load_data(dataset_path, mode="train"):
    data_loader = torch.utils.data.DataLoader(
        ImageFolder(input_dir=dataset_path, img_patch_size=(patch_size, patch_size),
                    depth_map_resolution=wave_resolution, input_transform=None, load_all=False, monochrome=False,
                    augment=False, mode=mode), batch_size=batch_size, num_workers=num_workers_train)
    return data_loader


def print_params():
    print("######## Basics ##################")
    print("version: {}".format(version))
    print("Training on {}".format(CUDA_DEVICE))


def load_network_weights(net, path):
    if path is None:
        print("================\nWARNING: param path is None")
    else:
        print("loading pre-trained weights from {}".format(path))
        net.load_state_dict(torch.load(path))


def save_network_weights(net, ep=None):
    filename = network_weight_path + "{}{}_epoch_{}.pth".format(model_name, version, ep)
    torch.save(net.state_dict(), filename)
    print("network weights saved to ", filename)
    return


def compute_loss(output, target, heightmap):
    """
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


def compute_psnr(output, target):
    """
    https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
    modified to incoporate pixel values ranging from [0, 1]
    :param output:
    :param target:
    :return:
    """
    mse = torch.mean((output - target) ** 2)
    return 20 * torch.log10(1. / torch.sqrt(mse))


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


def tensorboard_vis(tb, ep, mode="train", psf=None, height_map=None, output=None, plt_1d_psf=True, target=None):
    if psf is not None:
        transform = torchvision.transforms.Compose([
            lambda x: x / x.max(),
            torchvision.transforms.CenterCrop(50)
        ])
        # crop_transform = torchvision.transforms.CenterCrop(50)
        cropped_psf = transform(psf)
        psf_img_grid = torchvision.utils.make_grid(cropped_psf)
        tb.add_image("{}/psf".format(mode), psf_img_grid, global_step=ep)
        # tb.add_image('{}/normalized_psf'.format(mode), cropped_psf[0, :, :, :] / psf.max(), global_step=ep)
    if height_map is not None:
        tb.add_image('{}/normalized_height_map'.format(mode), height_map[0, :, ::4, ::4] / height_map.max(), global_step=ep)
    if output is not None:
        output_img_grid = torchvision.utils.make_grid(output)
        tb.add_image("{}/outputs".format(mode), output_img_grid, global_step=ep)
    if target is not None:
        target_img_grid = torchvision.utils.make_grid(target)
        tb.add_image("{}/target".format(mode), target_img_grid, global_step=ep)
    if plt_1d_psf:
        psf_plot = torch.sum(psf, dim=2)
        psf_plot = psf_plot.cpu().detach().numpy()
        fig, ax = plt.subplots()
        ax.plot(psf_plot[0, 0, 256-25:256+25], c='r', label="R, 0.5m")
        ax.plot(psf_plot[0, 1, 256-25:256+25], c='g', label="G, 0.5m")
        ax.plot(psf_plot[0, 2, 256-25:256+25], c='b', label="B, 0.5m")

        ax.plot(psf_plot[2, 0, 256 - 25:256 + 25], c='r', linestyle="dotted",  label="R, 1m")
        ax.plot(psf_plot[2, 1, 256 - 25:256 + 25], c='g', linestyle="dotted", label="G, 1m")
        ax.plot(psf_plot[2, 2, 256 - 25:256 + 25], c='b', linestyle="dotted", label="B, 1m")

        ax.plot(psf_plot[4, 0, 256 - 25:256 + 25], c='r', linestyle="dashed", label="R, inf")
        ax.plot(psf_plot[4, 1, 256 - 25:256 + 25], c='g', linestyle="dashed", label="G, inf")
        ax.plot(psf_plot[4, 2, 256 - 25:256 + 25], c='b', linestyle="dashed", label="B, inf")
        ax.legend()
        tb.add_figure(tag="{}/1D_psf".format(mode), figure=fig, global_step=ep)
    return


def tensorboard_vis_all_depths(tb, output=None, target=None, ep=0):
    if output is not None:
        output_img_grid = torchvision.utils.make_grid(output)
        tb.add_image("test_all_depths/outputs", output_img_grid, global_step=ep)
    if target is not None:
        target_img_grid = torchvision.utils.make_grid(target)
        tb.add_image("test_all_depths/target", target_img_grid, global_step=ep)
    return


def train_dev(net, tb, load_weights=False, pre_trained_params_path=None):
    print_params()
    net.to(CUDA_DEVICE)
    net.train()

    if load_weights:
        load_network_weights(net, pre_trained_params_path)

    train_loader = load_data(p.join(div2k_dataset_path, "train"))
    dev_loader = load_data(p.join(div2k_dataset_path, "valid"))
    train_num_mini_batches = len(train_loader)
    dev_num_mini_batches = len(dev_loader)
    optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=.8)

    running_train_loss, running_dev_loss = 0.0, 0.0  # per epoch
    train_psf, train_height_map, train_output, dev_output, input_ = None, None, None, None, None
    lowest_dev_score = 100000
    # training & validation loop
    for ep in range(epoch):
        print("Epoch ", ep)
        train_iter, dev_iter = iter(train_loader), iter(dev_loader)
        # TRAIN
        for _ in tqdm(range(train_num_mini_batches)):
            input_, depth = train_iter.next()
            input_, depth = input_.to(CUDA_DEVICE), depth.to(CUDA_DEVICE)
            optimizer.zero_grad()
            train_output, _, train_height_map = net(input_, depth)
            train_loss = compute_loss(output=train_output, target=input_, heightmap=net.heightMapElement.height_map)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        # DEV
        with torch.no_grad():
            for _ in range(dev_num_mini_batches):
                input_, depth = dev_iter.next()
                input_, depth = input_.to(CUDA_DEVICE), depth.to(CUDA_DEVICE)
                dev_output, _, dev_height_map = net(input_, depth)
                dev_loss = compute_loss(output=dev_output, target=input_, heightmap=net.heightMapElement.height_map)
                running_dev_loss += dev_loss.item()

        # record loss values after each epoch
        cur_train_loss = running_train_loss / train_num_mini_batches
        cur_dev_loss = running_dev_loss / dev_num_mini_batches
        print("train loss = {:.4} | val loss = {:.4}".format(cur_train_loss, cur_dev_loss))
        tb.add_scalar('loss/train', cur_train_loss, ep)
        tb.add_scalar('loss/dev', cur_dev_loss, ep)
        tb.add_scalar('loss/lr', scheduler._last_lr[0], ep)
        scheduler.step()
        if ep == 0:
            tensorboard_vis(tb, ep, mode="dev", target=input_, plt_1d_psf=False)
        if ep % 5 == 0:
            # tensorboard_vis(tb, ep, mode="train", psf=None, height_map=train_height_map,
            #                 output=train_output, plt_1d_psf=False)
            tensorboard_vis(tb, ep, mode="dev", psf=net.sample_psfs(), height_map=dev_height_map,
                            output=dev_output, plt_1d_psf=True)
            save_network_weights(net, ep)
        if cur_dev_loss <= lowest_dev_score and cur_dev_loss <= 0.027:
            save_network_weights(net, ep="{}_lowest={:.4f}".format(ep, cur_dev_loss))
            lowest_dev_score = cur_dev_loss
        running_train_loss, running_dev_loss = 0.0, 0.0
        # scheduler.step()

    print("finished training")
    save_network_weights(net, ep="{}_FINAL".format(epoch))
    return


def predict(net, tb, param_path):
    global batch_size
    batch_size = len(DEPTH_OPTIONS)
    net.to(CUDA_DEVICE)
    load_network_weights(net, param_path)
    dev_loader = load_data(p.join(div2k_dataset_path, "valid"), mode="test")
    dev_num_mini_batches = len(dev_loader)
    running_dev_loss, running_psnr_error = 0, 0
    with torch.no_grad():
        dev_iter = iter(dev_loader)
        # select a few images to generate outputs at all depths
        input_, depth = dev_iter.next()
        input_, depth = input_.to(CUDA_DEVICE), depth.to(CUDA_DEVICE)
        dev_output, _, dev_height_map = net(input_, depth)
        tensorboard_vis(tb, 0, mode="test", psf=net.sample_psfs(), height_map=dev_height_map,
                        output=dev_output, plt_1d_psf=True)
        print("rendered psfs to tenserboard")
        print("rendering sample images at all depths")
        for idx in tqdm(range(4)):  # select 4 samples to display at all depths
            input_, _ = dev_iter.next()
            input_ = input_.to(CUDA_DEVICE)
            single_output_all_depths = net.sample_output_multi_depths(input_)
            tensorboard_vis_all_depths(tb, output=single_output_all_depths, target=input_[0, :, :, :].unsqueeze(dim=0),
                                       ep=idx)
        del dev_iter  # reset

        print("evaluating on the entire dataset")
        # compute losses across entire test set
        dev_iter = iter(dev_loader)
        for _ in tqdm(range(dev_num_mini_batches)):
            input_, depth = dev_iter.next()
            input_, depth = input_.to(CUDA_DEVICE), depth.to(CUDA_DEVICE)
            dev_output, _, dev_height_map = net(input_, depth)
            dev_loss = compute_loss(output=dev_output, target=input_, heightmap=net.heightMapElement.height_map)
            psnr = compute_psnr(dev_output, input_)
            running_dev_loss += dev_loss.item()
            running_psnr_error += psnr
    total_dev_loss = running_dev_loss / dev_num_mini_batches
    average_psnr = running_psnr_error / dev_num_mini_batches
    print("test MSE Error = {:.4} | PSNR = {:.4}".format(total_dev_loss, average_psnr))

    print("finished evaluation")


def main():
    global version, model_name
    # model_name = "AchromaticEdofFourier"
    # version = "-v3.1.2"
    model_name = "CubicPhasePlate"
    version = "-v4.0.0"
    param_to_load = "./weight/AchromaticEdofFourier-v3.1.1_epoch_6_lowest=0.0228.pth"
    param_to_load = None
    tb = SummaryWriter('./runs/' + model_name + version)
    # simple lens
    # net = RGBCollimator(sensor_distance=sensor_distance, refractive_idcs=refractive_idcs, wave_lengths=wave_lengths,
    #                     patch_size=patch_size, sample_interval=sample_interval, wave_resolution=wave_resolution,
    #                     height_tolerance=height_tolerance)

    # Fourier system
    # net = RGBCollimator_Fourier(sensor_distance=sensor_distance, refractive_idcs=refractive_idcs, wave_lengths=wave_lengths,
    #                     patch_size=patch_size, sample_interval=sample_interval, wave_resolution=wave_resolution,
    #                     height_tolerance=height_tolerance)

    net = AchromaticEdofFourier(sensor_distance=sensor_distance, refractive_idcs=refractive_idcs, wave_lengths=wave_lengths,
                        patch_size=patch_size, sample_interval=sample_interval, wave_resolution=wave_resolution,
                        height_tolerance=height_tolerance, frequency_range=.625)
    # train_dev(net, tb, load_weights=False, pre_trained_params_path=param_to_load)
    predict(net, tb, param_to_load)
    tb.close()


if __name__ == "__main__":
    main()
