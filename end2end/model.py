import torch
import torch.nn as nn
import numpy as np
import end2end.optics.elements_pytorch as elements
import end2end.optics.propagations_pytorch as propagations
import end2end.optics.optics_utils as optics_utils


class RGBCollimator(nn.Module):
    """Section 3.2 simple lens check"""

    def __init__(self, sensor_distance, refractive_idcs, wave_lengths, patch_size, sample_interval,
                 wave_resolution, height_tolerance, block_size=1):
        super(RGBCollimator, self).__init__()

        self.wave_res = wave_resolution
        self.wave_lengths = wave_lengths
        self.sensor_distance = sensor_distance
        self.sample_interval = sample_interval
        self.patch_size = patch_size
        self.refractive_idcs = refractive_idcs
        self.height_tolerance = height_tolerance
        self.block_size = block_size

        # trainable height map
        height_map_shape = [1, 1, self.wave_res[0] // block_size, self.wave_res[1] // block_size]
        # self.height_map = self.height_map_initializer()

        # Input field is a planar wave.
        self.input_field = nn.Parameter(torch.ones((1, len(self.wave_lengths), self.wave_res[0], self.wave_res[1])))

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

    # def height_map_initializer(self):
    #     height_map = torch.full(self.height_map_shape, 1e-4, dtype=torch.float64)
    #     return nn.parameter.Parameter(height_map, requires_grad=True)

    def forward(self, x):
        """
        :param x: input image
        :return:
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
        output_image = optics_utils.img_psf_conv(x, psfs).astype(torch.float32)
        # optics.attach_summaries('output_image', output_image, image=True, log_image=False) TODO

        # add sensor noise
        rand_sigma = (.02 - .001) * torch.rand() + 0.001  # standard deviation drawn from uni dist
        # add gaussian noise
        output_image += torch.normal(mean=torch.zeros_like(output_image),
                                     std=torch.ones_like(output_image) * rand_sigma)

        return output_image
