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
            print("Warning: area downsampling is very expensive and not precise "
                  "if source and target wave length have a large least common multiple")
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = F.interpolate(input_image,
                                      size=2 * [upsample_factor * target_side_length],
                                      mode='nearest')
        output_img = F.avg_pool2d(img_upsampled,
                                  (upsample_factor, upsample_factor),
                                  strides=[upsample_factor, upsample_factor])

        return output_img


def psf2otf(psf, output_size):
    """
    apply Fourier Transform to psf to obtain its optical transfer function
    :param psf: point spread function of shape  (m, c, h, w)
    :param output_size: size of otf
    :return: otf of shape (m, c, h', w')
    """
    _, _, fh, fw = psf.shape  # filter height and width
    if output_size[2] != fh:  # requires padding
        pad = (output_size[2] - fh) / 2
        if (output_size[2] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1
        padded = F.pad(psf, [pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0])
    else:
        padded = psf
    padded = torch.fft.ifft2(padded)
    otf = torch.fft.fft2(torch.complex(padded, torch.tensor(0.)))
    return otf


def img_psf_conv(img, psf, otf=None, adjoint=False, circular=False):
    """
    convolve input image with a given psf
    :param img: input image of shape (m, c, h, w)
    :param psf: point spread function of shape (c, h, w)
    :param otf: optical transfer function, or ft of psf
    :param adjoint: whether to perform an adjoin convolution or not. Legacy problem
    :param circular: whether to perform a circular convolution or not. Legacy problem
    :return: convolved image of shape (??)
    """
    if adjoint is False or circular is False:
        raise NotImplementedError  # not used in the scope of this paper
    assert (torch.is_tensor(img))  # ensure the dim of img follow pytorch tensor convention
    m, c, h, w = img.shape  # used to be [m, h, w, c] for tf
    assert (h == w)  # legacy problem. Previous code requires height = width
    target_side_length = 2 * h
    height_pad = (target_side_length - h) / 2
    width_pad = (target_side_length - w) / 2
    pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.ceil(height_pad))
    pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

    padded_img = F.pad(img, [pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0], mode="constant")
    padded_img_shape = padded_img.shape
    img_fft = torch.fft.fft2(padded_img)

    if otf is None:
        otf = psf2otf(psf, output_size=padded_img_shape)  # pytorch specific, should be (m, c, padded_h, padded_w)

    otf = otf.astype(torch.complex64)
    img_fft = img_fft.astype(torch.complex64)
    convolved_img = torch.fft.ifft2(img_fft, otf).astype(torch.float32)
    assert(convolved_img.shape == img.shape)  # is this right?
    return convolved_img


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
    filtered_batch = F.conv2d(filter_input, laplacian_filter, padding="same")
    return filtered_batch
