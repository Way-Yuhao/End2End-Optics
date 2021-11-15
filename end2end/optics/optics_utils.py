"""implementation of optics functions and light propogation functions"""
__author__ = "Yuhao Liu", "Krish Kabra"

import torch
import numpy as np


def get_intensities(input_field):
    """
    Extract 2D intensity data from a given wave field
    :param input_field:
    :return:
    """
    return torch.square(torch.abs(input_field))


def area_down_sampling(input_image, target_size_length):
    input_shape = input_image.shape.as_list()
    input_image = torch

def fspecial_gaussian(shape, sigma):
    """
    returns a 2D gaussian mask
    should behave the same as MATLAB's fspecial('gaussian',[shape],[sigma])
    :param shape:
    :param sigma:
    :return:
    """
    # TODO
    pass


def zoom(image_batch, zoom_fraction):
    """
    Perform center crop on a batch of images
    :param image_batch:
    :param zoom_fraction:float (0, 1], fraction of size to crop
    :return:
    """
    # TODO: possibly unused
    pass


def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs):
    """
    Calculates the phase shifts created by a height map with certain refractive index for light with specific wave
    length. See Equation (1) in End2End paper
    """
    # refractive index difference
    delta_N = refractive_idcs.reshape([1, 1, 1, -1]) - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1, 1, 1, -1])
    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = torch.exp(phi)
    return phase_shifts

def get_intensities(input_field):
    return torch.square(torch.abs(input_field))

def laplace_l1_regularizer(scale):  # TODO
    pass


def laplacian_filter_pytorch(img_batch): # TODO

    pass

def get_zernike_volume():
    # TODO: function unknown
    pass
