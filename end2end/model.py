import torch
import torch.nn as nn
import numpy as np
import end2end.optics as optics


class RGBCollimator(nn.Module):  # TODO
    """Section 3.2 simple lens check"""

    def __init__(self, sensor_distance, refractive_idcs, wave_lengths, patch_size, sample_interval,
                 wave_resolution, height_map_noise):
        super(RGBCollimator, self).__init__()

        self.wave_res = wave_resolution
        self.wave_lengths = wave_lengths
        self.sensor_distance = sensor_distance
        self.sample_interval = sample_interval
        self.patch_size = patch_size
        self.refractive_idcs = refractive_idcs
        self.height_map_noise = height_map_noise

        # trainable height map
        self.height_map = self._height_map_initializer()

    def forward(self, x):
        # Input field is a planar wave.
        input_field = torch.ones((1, self.wave_res[0], self.wave_res[1], len(self.wave_lengths)))

        # Planar wave hits aperture: phase is shifted by phase plate
        field = optics.elements.height_map_element(input_field,
                                                   wave_lengths=self.wave_lengths,
                                                   height_tolerance=self.height_map_noise,
                                                   refractive_idcs=self.refractive_idcs)
        field = optics.elements.circular_aperture(field)

        # Propagate field from aperture to sensor
        field = optics.propagations.propagate_fresnel(field,
                                                      distance=self.sensor_distance,
                                                      sampling_interval=self.sample_interval,
                                                      wave_lengths=self.wave_lengths)

        # The psf is the intensities of the propagated field.
        psfs = optics.optics_utils.get_intensities(field)

        # Downsample psf to image resolution & normalize to sum to 1
        psfs = optics.optics_utils.area_down_sampling(psfs, self.patch_size)
        psfs = torch.div(psfs, torch.sum(psfs, dim=(2, 3), keepdim=True))
        # optics.attach_summaries('PSF', psfs, image=True, log_image=True) TODO

        # Image formation: PSF is convolved with input image

        output_image = optics.img_psf_conv(x, psfs).astype(torch.float32)
        # optics.attach_summaries('output_image', output_image, image=True, log_image=False) TODO

        # add sensor noise
        # output_image += tf.random_uniform(minval=0.001, maxval=0.02, shape=[])
        rand_sigma = (.02 - .001) * torch.rand() + 0.001  # standard deviation drawn from uni dist
        # add gaussian noise
        output_image += torch.normal(mean=torch.zeros_like(output_image),
                                     std=torch.ones_like(output_image) * rand_sigma)

        return output_image

    def _height_map_initializer(self):
        height_map = torch.full(self.height_map_shape, 1e-4, dtype=torch.float64)
        return nn.parameter.Parameter(height_map, requires_grad=True)
