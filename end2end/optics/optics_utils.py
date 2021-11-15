"""implementation of optics functions and light propogation functions"""
__author__ = "Yuhao Liu", "Krish Kabra"

import torch
import torch.nn.functional as F

import numpy as np
import math
import poppy


def get_zernike_volume(resolution, n_terms, scale_factor=1e-6):
    """

    :param resolution:
    :param n_terms:
    :param scale_factor:
    """
    zernike_volume = poppy.zernike.zernike_basis(nterms=n_terms, npix=resolution, outside=0.0)
    return zernike_volume * scale_factor

def get_intensities(input_field):
    """
    Extract 2D intensity data from a given wave field
    :param input_field
    :return:
    """
    return torch.square(torch.abs(input_field))


def least_common_multiple(a, b):
    return abs(a * b) / math.gcd(a, b) if a and b else 0


def area_down_sampling(input_image, target_side_length):
    """

    :param input_image Tensor[batch_size, num_wavelengths, height, width]
    """
    input_shape = list(input_image.shape)
    input_image = input_image.type(torch.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1] / target_side_length)
        output_img = F.avg_pool2d(input_image,
                                  (factor, factor),
                                  strides=(factor, factor),
                                  )
    else:
        # We upsample the image and then average pool
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor > 10:
            print(
                "Warning: area downsampling is very expensive and not precise if source and target wave length have a large least common multiple")
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = F.interpolate(input_image,
                                      size=2 * [upsample_factor * target_side_length],
                                      mode='nearest'
                                      )
        output_img = F.avg_pool2d(img_upsampled,
                                  (upsample_factor, upsample_factor)
                                  strides=[upsample_factor, upsample_factor],
                                  )

    return output_img


def fspecial_gaussian(shape, sigma):
    """
    returns a 2D gaussian mask
    should behave the same as MATLAB's fspecial('gaussian',[shape],[sigma])
    :param shape:
    :param sigma:
    :return:
    """
    # TODO
    raise NotImplementedError


def zoom(image_batch, zoom_fraction):
    """
    Perform center crop on a batch of images
    :param image_batch:
    :param zoom_fraction:float (0, 1], fraction of size to crop
    :return:
    """
    # TODO: possibly unused
    raise NotImplementedError


def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs):
    """
    Calculates the phase shifts created by a height map with certain refractive index for light with specific wave
    length. See Equation (1) in End2End paper
    """
    # refractive index difference
    delta_N = refractive_idcs.reshape([1, -1, 1, 1]) - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1, -1, 1, 1])
    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = torch.exp(phi)
    return phase_shifts


def laplace_l1_regularizer(img_batch, scale):  # FIXME: call signature differs from tf
    """
    :param img_batch:
    :param scale: scalar constant
    :return:
    """
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")  # what is this

    laplace_filtered = laplacian_filter_pytorch(img_batch)
    # laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
    laplace_filtered = laplace_filtered[:, :, 1:-1, 1:-1]  # pytorch specific
    laplacian_regularizer = scale * torch.mean(torch.abs(laplace_filtered))
    return laplacian_regularizer


def laplacian_filter_pytorch(img_batch):
    laplacian_filter = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
    laplacian_filter = torch.reshape(laplacian_filter, (1, 1, 3, 3))  # pytorch specific

    filter_input = img_batch.type(torch.float32)
    # may heave to make sure require_grad is false
    filtered_batch = F.conv2d(filter_input, laplacian_filter, padding="SAME")
    return filtered_batch


def get_zernike_volume():
    # TODO: function unknown
    raise NotImplementedError
