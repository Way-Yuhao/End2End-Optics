import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import end2end.optics.optics_utils as optics_utils
from config import CUDA_DEVICE


class CircularAperture(nn.Module):

    def __init__(self):
        super(CircularAperture, self).__init__()

    def forward(self, input_field):
        """
        Propogate input field through circular aperture
        :param input_field:
        :return:
        """
        input_shape = list(input_field.shape)
        [x, y] = np.mgrid[-input_shape[2] // 2: input_shape[2] // 2,
                          -input_shape[3] // 2: input_shape[3] // 2].astype(np.float64)

        max_val = np.amax(x)

        r = np.sqrt(x ** 2 + y ** 2)[None, None, :, :]
        # aperture = (r < max_val).astype(np.float64)  # Why cast like this?
        aperture = (r < max_val)
        aperture = np.concatenate((aperture, aperture, aperture), axis=1)
        return torch.tensor(aperture).to(CUDA_DEVICE) * input_field
        # return input_field[aperture == 0 ] =


class HeightMapElement(nn.Module):
    """
    Propogate wavefront through a phase modulating element with a given height map
    """

    def __init__(self, height_map_shape, wave_lengths, refractive_idcs, height_tolerance=None, lateral_tolerance=None):
        """
        :param wave_lengths (np.ndarray[num_wavelengths,]): list of wavelengths to be modeled
        :param height_map (Tensor[1, height, width, 1]): spatial thickness map of the phase plate
        :param refractive_idcs (np.ndarray[num_wavelengths,]): list of refractive indicies of the phase plate
        :param height_tolerance: range of uniform noise added to height map
        :param lateral_tolerance: ?? (not needed)
        """
        super(HeightMapElement, self).__init__()
        self.height_map_shape = height_map_shape
        self.wave_lengths = wave_lengths
        self.refractive_idcs = refractive_idcs
        self.height_tolerance = height_tolerance
        self.lateral_tolerance = lateral_tolerance
        self.height_map = self.height_map_initializer()
        self.phase_shifts = None
        self.height_map_noise = None

        if self.height_tolerance is not None:
            print("Phase plate with manufacturing tolerance {:0.2e}".format(self.height_tolerance))

    def height_map_initializer(self):
        height_map = torch.full(self.height_map_shape, 1e-4, dtype=torch.float64)
        return nn.parameter.Parameter(height_map, requires_grad=True)

    def forward(self, x):
        """
        :param x: input_field (Tensor[batch_size, num_wavelengths, height, width]), complex valued wavefront
        :return: input field shifted by phase plate
        """
        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            pass
            # TODO: require_grad? Add noise later
            # height_map_noise = -2 * self.height_tolerance * torch.rand(self.height_map_shape, requires_grad=False) \
                                    # + self.height_tolerance
            # self.height_map_noise = -2 * self.height_tolerance * torch.rand(self.height_map_shape, requires_grad=False) \
            #                   + self.height_tolerance
            # height_map_noise = -2 * self.height_tolerance * torch.rand(self.height_map_shape) \
            #                         + self.height_tolerance
            # self.height_map = self.height_map + self.height_map_noise.to("cuda:6")

        self.phase_shifts = optics_utils.phaseshifts_from_height_map(self.height_map, self.wave_lengths,
                                                                     self.refractive_idcs)
        input_field = x.type(torch.complex64)
        return torch.multiply(input_field, self.phase_shifts)


# class HeightMapElement(PhasePlate):
#     """
#     Propogate wavefront through a phase modulating element with a given height map
#     """
#
#     def __init__(self, height_map_shape, wave_lengths, refractive_idcs, height_tolerance=None):
#         """
#         :param input_field (Tensor[batch_size, height, width, num_wavelengths]): complex valued wavefront
#         :param wave_lengths (np.ndarray[num_wavelengths,]): list of wavelengths to be modeled
#         :param refractive_idcs (np.ndarray[num_wavelengths,]): list of refractive indicies of the phase plate
#         # :param block_size: ??
#         :param height_map_initializer (Tensor[1, height, width, 1]): custom initialization for height map of phase plate
#         # :param height_map_regularizer: ?? NOT NEEDED
#         :param height_tolerance: range of uniform noise added to height map (default height tolerance is 2 nm)
#         :return: Phase plate element
#         """
#         super(HeightMapElement, self).__init__(height_map_shape=height_map_shape, wave_lengths=wave_lengths,
#                                                refractive_idcs=refractive_idcs, height_tolerance=height_tolerance)
#         self.height_map = self.height_map_initializer()
#
#     def height_map_initializer(self):
#         height_map = torch.full(self.height_map_shape, 1e-4, dtype=torch.float64)
#         return nn.parameter.Parameter(height_map, requires_grad=True)
#
#     def forward(self, input_field):
#         return super(HeightMapElement, self).forward(input_field)


class FourierElement(nn.Module):
    """
     Propogate wavefront through a phase modulating element with a given height map
     """

    def __init__(self, height_map_shape, frequency_range, wave_lengths, refractive_idcs, height_tolerance=None, lateral_tolerance=None):
        """
        :param wave_lengths (np.ndarray[num_wavelengths,]): list of wavelengths to be modeled
        :param refractive_idcs (np.ndarray[num_wavelengths,]): list of refractive indicies of the phase plate
        # :param block_size: ??
        :param height_map_initializer (Tensor[1, height, width, 1]): custom initialization for height map of phase plate
        # :param height_map_regularizer: ?? NOT NEEDED
        :param height_tolerance: range of uniform noise added to height map (default height tolerance is 2 nm)
        :return: Phase plate element
        """

        super(FourierElement, self).__init__()
        self.height_map_shape = height_map_shape
        self.fourier_coeff_shape = [height_map_shape[0], height_map_shape[1], int(height_map_shape[2]*frequency_range), int(height_map_shape[3]*frequency_range)]
        self.frequency_range = frequency_range
        self.wave_lengths = wave_lengths
        self.refractive_idcs = refractive_idcs
        self.height_tolerance = height_tolerance
        self.lateral_tolerance = lateral_tolerance
        self.fourier_coeffs_reals, self.fourier_coeffs_complex = self.fourier_coeffs_initializer()
        self.padding_width = int((1 - self.frequency_range) * self.height_map_shape[2]) // 2
        self.height_map = None
        self.phase_shifts = None
        self.height_map_noise = None
        self.fourier_coeffs = None

        if self.height_tolerance is not None:
            print("Phase plate with manufacturing tolerance {:0.2e}".format(self.height_tolerance))

    def fourier_coeffs_initializer(self):
        fourier_coeffs_reals = torch.zeros(self.fourier_coeff_shape, dtype=torch.float32)
        fourier_coeffs_complex = torch.zeros(self.fourier_coeff_shape, dtype=torch.float32)
        return nn.parameter.Parameter(fourier_coeffs_reals, requires_grad=True), \
            nn.parameter.Parameter(fourier_coeffs_complex, requires_grad=True)

    def forward(self, x):
        """
        :param x: input_field (Tensor[batch_size, num_wavelengths, height, width]), complex valued wavefront
        :return: input field shifted by phase plate
        """
        _, _, height, width = x.shape

        self.fourier_coeffs = torch.complex(self.fourier_coeffs_reals, self.fourier_coeffs_complex)
        fourier_coeffs_padded = F.pad(self.fourier_coeffs,
                                      [self.padding_width, self.padding_width, self.padding_width, self.padding_width, 0, 0, 0, 0],
                                      mode='constant', value=0.0)

        self.height_map = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fourier_coeffs_padded, dim=[2, 3])))

        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            pass
            # TODO: require_grad? Add noise later
            # height_map_noise = -2 * self.height_tolerance * torch.rand(self.height_map_shape, requires_grad=False) \
            # + self.height_tolerance
            # self.height_map_noise = -2 * self.height_tolerance * torch.rand(self.height_map_shape, requires_grad=False) \
            #                   + self.height_tolerance
            # height_map_noise = -2 * self.height_tolerance * torch.rand(self.height_map_shape) \
            #                         + self.height_tolerance
            # self.height_map = self.height_map + self.height_map_noise.to("cuda:6")

        self.phase_shifts = optics_utils.phaseshifts_from_height_map(self.height_map, self.wave_lengths,
                                                                     self.refractive_idcs)
        input_field = x.type(torch.complex64)
        return torch.multiply(input_field, self.phase_shifts)


def height_map_element():  # TODO
    pass  # see non-pytorch version


def fourier_element():  # TODO
    pass


def zernike_element():  # TODO
    pass


def get_vanilla_height_map():  # TODO
    pass


def get_fourier_height_map():  # TODO
    pass


class SingleLensSetup():  # TODO
    """
    """

    def __init__(self):
        pass


class ZernikeSystem():  # TODO
    """

    """

    def __init__(self):
        pass
