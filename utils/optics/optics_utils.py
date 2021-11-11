"""implementation of optics functions and light propogation functions"""
__author__ = "Yuhao Liu", "Krish Kabra"

import torch

def get_zernike_volume():
    # TODO: function unknown
    pass


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


def comp_exp_tf(phase, dtype=torch.complex64):
    pass
    # TODO. Required in Fresnel Propagation


