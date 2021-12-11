import torch
import torch.nn as nn
import numpy as np
import end2end.optics.elements_pytorch as elements
import end2end.optics.propagations_pytorch as propagations
import end2end.optics.optics_utils as optics_utils
import end2end.decoder.deconv as deconv
from config import CUDA_DEVICE


class RGBCollimator(nn.Module):
    """Section 3.2 simple lens check"""

    def __init__(self, sensor_distance, refractive_idcs, wave_lengths, patch_size, sample_interval,
                 wave_resolution, height_tolerance, block_size=1):
        super(RGBCollimator, self).__init__()

        self.wave_res = torch.tensor(wave_resolution).to(CUDA_DEVICE)
        self.wave_lengths = torch.tensor(wave_lengths).to(CUDA_DEVICE)
        self.sensor_distance = torch.tensor(sensor_distance).to(CUDA_DEVICE)
        self.sample_interval = torch.tensor(sample_interval).to(CUDA_DEVICE)
        self.patch_size = torch.tensor(patch_size).to(CUDA_DEVICE)
        self.refractive_idcs = torch.tensor(refractive_idcs).to(CUDA_DEVICE)
        self.height_tolerance = torch.tensor(height_tolerance).to(CUDA_DEVICE)
        self.block_size = torch.tensor(block_size).to(CUDA_DEVICE)

        # trainable height map
        height_map_shape = [1, 1, self.wave_res[0] // block_size, self.wave_res[1] // block_size]
        # self.height_map = self.height_map_initializer()

        # Input field is a planar wave.
        self.input_field = torch.ones((1, len(self.wave_lengths), self.wave_res[0], self.wave_res[1])).to(CUDA_DEVICE)

        # Planar wave hits aperture: phase is shifted by phase plate
        self.heightMapElement = \
            elements.HeightMapElement(height_map_shape=height_map_shape, wave_lengths=self.wave_lengths,
                                      height_tolerance=self.height_tolerance, refractive_idcs=self.refractive_idcs)

        self.circularAperture = elements.CircularAperture()

        # Propagate field from aperture to sensor
        self.fresnelPropagation = \
            propagations.FresnelPropagation(input_shape=self.input_field.shape, distance=self.sensor_distance,
                                            discretization_size=self.sample_interval, wave_lengths=self.wave_lengths)

        # TODO: verify this
        self.circularAperture.requires_grad_(False)
        self.fresnelPropagation.requires_grad_(False)

    def forward(self, x):
        """
        :param x: input image
        :return: output image [m, c, x, y], psf [1, c, x, y]
        """
        field = self.heightMapElement(self.input_field)
        field = self.circularAperture(field)
        field = self.fresnelPropagation(field)

        # The psf is the intensities of the propagated field.
        psfs = optics_utils.get_intensities(field)

        # Downsample psf to image resolution & normalize to sum to 1
        psfs = optics_utils.area_down_sampling(psfs, self.patch_size)
        psfs = torch.div(psfs, torch.sum(psfs, dim=(2, 3), keepdim=True))

        # Image formation: PSF is convolved with input image
        output_image = optics_utils.img_psf_conv(x, psfs).type(torch.float32)

        # add sensor noise
        # FIXME
        # standard deviation drawn from uni dist
        rand_sigma = torch.tensor((.02 - .001) * torch.rand(1) + 0.001).to(CUDA_DEVICE)
        # add gaussian noise
        output_image += torch.normal(mean=torch.zeros_like(output_image),
                                     std=torch.ones_like(output_image) * rand_sigma)

        return output_image, psfs, self.heightMapElement.height_map


class RGBCollimator_Fourier(nn.Module):
    """Section 3.2 simple lens check"""

    def __init__(self, sensor_distance, refractive_idcs, wave_lengths, patch_size, sample_interval,
                 wave_resolution, height_tolerance, frequency_range=0.5, block_size=1):
        super(RGBCollimator_Fourier, self).__init__()

        self.wave_res = torch.tensor(wave_resolution).to(CUDA_DEVICE)
        self.wave_lengths = torch.tensor(wave_lengths).to(CUDA_DEVICE)
        self.sensor_distance = torch.tensor(sensor_distance).to(CUDA_DEVICE)
        self.sample_interval = torch.tensor(sample_interval).to(CUDA_DEVICE)
        self.patch_size = torch.tensor(patch_size).to(CUDA_DEVICE)
        self.refractive_idcs = torch.tensor(refractive_idcs).to(CUDA_DEVICE)
        self.height_tolerance = torch.tensor(height_tolerance).to(CUDA_DEVICE)
        self.block_size = torch.tensor(block_size).to(CUDA_DEVICE)

        # trainable height map
        height_map_shape = [1, 1, self.wave_res[0] // block_size, self.wave_res[1] // block_size]
        # self.height_map = self.height_map_initializer()

        # Input field is a planar wave.
        self.input_field = torch.ones((1, len(self.wave_lengths), self.wave_res[0], self.wave_res[1])).to(
            CUDA_DEVICE)

        # Planar wave hits aperture: phase is shifted by phase plate
        self.heightMapElement = \
            elements.FourierElement(height_map_shape=height_map_shape, frequency_range=frequency_range,
                                    wave_lengths=self.wave_lengths, height_tolerance=self.height_tolerance,
                                    refractive_idcs=self.refractive_idcs)

        self.circularAperture = elements.CircularAperture()

        # Propagate field from aperture to sensor
        self.fresnelPropagation = \
            propagations.FresnelPropagation(input_shape=self.input_field.shape, distance=self.sensor_distance,
                                            discretization_size=self.sample_interval,
                                            wave_lengths=self.wave_lengths)

        # TODO: verify this
        self.circularAperture.requires_grad_(False)
        self.fresnelPropagation.requires_grad_(False)

    # def height_map_initializer(self):
    #     height_map = torch.full(self.height_map_shape, 1e-4, dtype=torch.float64)
    #     return nn.parameter.Parameter(height_map, requires_grad=True)

    def forward(self, x):
        """
        :param x: input image
        :return: output image [m, c, x, y], psf [1, c, x, y]
        """
        field = self.heightMapElement(self.input_field)
        field = self.circularAperture(field)
        field = self.fresnelPropagation(field)

        # The psf is the intensities of the propagated field.
        psfs = optics_utils.get_intensities(field)

        # Downsample psf to image resolution & normalize to sum to 1
        psfs = optics_utils.area_down_sampling(psfs, self.patch_size)
        psfs = torch.div(psfs, torch.sum(psfs, dim=(2, 3), keepdim=True))
        # optics.attach_summaries('PSF', psfs, image=True, log_image=True) TODO

        # Image formation: PSF is convolved with input image
        output_image = optics_utils.img_psf_conv(x, psfs).type(torch.float32)
        # optics.attach_summaries('output_image', output_image, image=True, log_image=False) TODO

        # add sensor noise
        # FIXME
        rand_sigma = torch.tensor((.02 - .001) * torch.rand(1) + 0.001).to(
            CUDA_DEVICE)  # standard deviation drawn from uni dist
        # add gaussian noise
        output_image += torch.normal(mean=torch.zeros_like(output_image),
                                     std=torch.ones_like(output_image) * rand_sigma)

        return output_image, psfs, self.heightMapElement.height_map


class AchromaticEdofFourier(nn.Module):
    def __init__(self, sensor_distance, refractive_idcs, wave_lengths, patch_size, sample_interval,
                 wave_resolution, height_tolerance, frequency_range=0.5, block_size=1, init_gamma=1.5):
        super(AchromaticEdofFourier, self).__init__()

        self.wave_res = torch.tensor(wave_resolution).to(CUDA_DEVICE)
        self.wave_lengths = torch.tensor(wave_lengths).to(CUDA_DEVICE)
        self.sensor_distance = torch.tensor(sensor_distance).to(CUDA_DEVICE)
        self.sample_interval = torch.tensor(sample_interval).to(CUDA_DEVICE)
        self.patch_size = torch.tensor(patch_size).to(CUDA_DEVICE)
        self.refractive_idcs = torch.tensor(refractive_idcs).to(CUDA_DEVICE)
        self.height_tolerance = torch.tensor(height_tolerance).to(CUDA_DEVICE)
        self.block_size = torch.tensor(block_size).to(CUDA_DEVICE)
        self.physical_size = self.wave_res[0] * self.sample_interval
        self.pixel_size = self.sample_interval * self.wave_res[0] / self.patch_size
        self.init_gamma = init_gamma

        print("Physical size is %0.2e.\nWave resolution is %d." % (self.physical_size, self.wave_res[0]))

        # trainable height map
        height_map_shape = [1, 1, self.wave_res[0] // block_size, self.wave_res[1] // block_size]
        # self.height_map = self.height_map_initializer()

        # Planar wave hits aperture: phase is shifted by phase plate
        self.heightMapElement = \
            elements.FourierElement(height_map_shape=height_map_shape, frequency_range=frequency_range,
                                    wave_lengths=self.wave_lengths, height_tolerance=self.height_tolerance,
                                    refractive_idcs=self.refractive_idcs)

        self.circularAperture = elements.CircularAperture()

        # Propagate field from aperture to sensor
        self.fresnelPropagation = \
            propagations.FresnelPropagation(input_shape=self.input_field.shape, distance=self.sensor_distance,
                                            discretization_size=self.sample_interval,
                                            wave_lengths=self.wave_lengths)

        # Deconvolve image using inverse filter
        self.inverseFilter = deconv.InverseFilter(init_gamma=self.init_gamma)

        # TODO: verify this
        self.circularAperture.requires_grad_(False)
        self.fresnelPropagation.requires_grad_(False)

    # def height_map_initializer(self):
    #     height_map = torch.full(self.height_map_shape, 1e-4, dtype=torch.float64)
    #     return nn.parameter.Parameter(height_map, requires_grad=True)

    def forward(self, input_img, depth_map):
        """
        :param input_img: [m, c, x, y]
        :param depth_map: [m, c, x, y]
        :return: output image [m, c, x, y], psf [m, c, x, y]
        """

        # spherical wavefront based on target depth
        xx = torch.linspace(-self.wave_res[0] // 2, self.wave_res[0] // 2, self.wave_res[0])
        yy = torch.linspace(-self.wave_res[1] // 2, self.wave_res[1] // 2, self.wave_res[1])
        grid_x, grid_y = torch.meshgrid(xx, yy)

        grid_x = grid_x / self.wave_res[0] * self.physical_size
        grid_y = grid_y / self.wave_res[1] * self.physical_size

        squared_sum = torch.unsqueeze(torch.unsqueeze(grid_x ** 2 + grid_y ** 2, 0), 0)  # FIXME: is this best way of ading dims?
        curvature = torch.sqrt(squared_sum + depth_map ** 2)
        input_field = torch.exp(torch.complex(torch.zeros_like(curvature), curvature))

        field = self.heightMapElement(input_field)
        field = self.circularAperture(field)
        field = self.fresnelPropagation(field)

        # The psf is the intensities of the propagated field.
        psfs = optics_utils.get_intensities(field)

        # Downsample psf to image resolution & normalize to sum to 1
        psfs = optics_utils.area_down_sampling(psfs, self.patch_size)
        psfs = torch.div(psfs, torch.sum(psfs, dim=(2, 3), keepdim=True))
        # optics.attach_summaries('PSF', psfs, image=True, log_image=True) TODO

        # Image formation: PSF is convolved with input image
        output_image = optics_utils.img_psf_conv(input_img, psfs).type(torch.float32)
        # optics.attach_summaries('output_image', output_image, image=True, log_image=False) TODO

        # add sensor noise
        # FIXME
        rand_sigma = torch.tensor((.02 - .001) * torch.rand(1) + 0.001).to(
            CUDA_DEVICE)  # standard deviation drawn from uni dist
        # add gaussian noise
        output_image += torch.normal(mean=torch.zeros_like(output_image),
                                     std=torch.ones_like(output_image) * rand_sigma)

        # deconvolve noisy and blurry image
        output_image = self.inverseFilter(output_image, output_image, psfs)

        return output_image, psfs
