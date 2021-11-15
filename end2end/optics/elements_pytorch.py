import torch
import numpy as np
import end2end.optics.optics_utils as optics_utils


class PhasePlate(torch.nn.Module):
    """
    Propogate input field through circular aperture
    """
    def __init__(self, wave_lengths, height_map, refractive_idcs, height_tolerance=None, lateral_tolerance=None):
        """
        :param wave_lengths (np.ndarray[num_wavelengths,]): list of wavelengths to be modeled
        :param height_map (Tensor[1, height, width, 1]): spatial thickness map of the phase plate
        :param refractive_idcs (np.ndarray[num_wavelengths,]): list of refractive indicies of the phase plate
        :param height_tolerance: range of uniform noise added to height map
        :param lateral_tolerance: ?? (not needed)
        """
        super(PhasePlate, self).__init__()
        self.wave_lengths = wave_lengths
        self.height_map = height_map
        self.refractive_idcs = refractive_idcs
        self.height_tolerance = height_tolerance
        self.lateral_tolerance = lateral_tolerance

        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            height_map_noise = -2 * self.height_tolerance * torch.rand(self.height_map.shape, requires_grad=False) \
                               + self.height_tolerance
            self.height_map = self.height_map + height_map_noise
            print("Phase plate with manufacturing tolerance {:0.2e}".format(self.height_tolerance))

        self.phase_shifts = optics_utils.phaseshifts_from_height_map(self.height_map, self.wave_lengths,
                                                                     self.refractive_idcs)

    def forward(self, x):
        """
        :param x: input_field (Tensor[batch_size, height, width, num_wavelengths]), complex valued wavefront
        :return: input field shifted by phase plate
        """
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