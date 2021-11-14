"""Optical elements"""

import torch
from torch.nn.parameter import Parameter
import numpy as np

from numpy.fft import ifftshift
import fractions
import poppy

import optics_utils as optics

class PhasePlate():
    """
    Model for optical element that modulates wavefront via phase shifts
    """
    def __init__(self,
                 wave_lengths,
                 height_map,
                 refractive_idcs,
                 height_tolerance=None,
                 lateral_tolerance=None):
        """
        :param wave_lengths (np.ndarray[num_wavelengths,]): list of wavelengths to be modeled
        :param height_map (Tensor[1, height, width, 1]): spatial thickness map of the phase plate
        :param refractive_idcs (np.ndarray[num_wavelengths,]): list of refractive indicies of the phase plate
        :param height_tolerance: range of uniform noise added to height map
        :param lateral_tolerance: ?? (not needed)
        """
        self.wave_lengths = wave_lengths
        self.height_map = height_map
        self.refractive_idcs = refractive_idcs
        self.height_tolerance = height_tolerance
        self.lateral_tolerance = lateral_tolerance

        self._build()

    def _build(self): #TODO
        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            self.height_map += -2 * self.height_tolerance * torch.rand(self.height_map.shape) + self.height_tolerance
            print("Phase plate with manufacturing tolerance {:0.2e}".format(self.height_tolerance))

        self.phase_shifts = optics.phaseshifts_from_height_map(self.height_map,
                                                        self.wave_lengths,
                                                        self.refractive_idcs)
    def __call__(self, input_field):
        """
        :param input_field (Tensor[batch_size, height, width, num_wavelengths]): complex valued wavefront
        :return: input field shifted by phase plate
        """
        input_field = input_field.type(torch.complex64)
        return torch.multiply(input_field, self.phase_shifts)

def height_map_element(input_field,
                       # name,
                       wave_lengths,
                       refractive_idcs,
                       block_size=1,
                       # height_map_initializer=None,
                       # height_map_regularizer=None, # NOT NEEDED
                       height_tolerance=None,  # Default height tolerance is 2 nm.
                       ): # TODO
    """
    Propogate wavefront through a phase modulating element with a given height map

    :param input_field (Tensor[batch_size, height, width, num_wavelengths]): complex valued wavefront
    :param name:
    :param wave_lengths (np.ndarray[num_wavelengths,]): list of wavelengths to be modeled
    :param refractive_idcs (np.ndarray[num_wavelengths,]): list of refractive indicies of the phase plate
    :param block_size: ??
    :param height_map_initializer (Tensor[1, height, width, 1]): custom initialization for height map of phase plate
    :param height_map_regularizer: ?? NOT NEEDED
    :param height_tolerance: range of uniform noise added to height map
    :return: Phase plate element
    """
    _, height, width, _ = input_field.shape.tolist()

    height_map_shape = [1, height // block_size, width // block_size, 1]

    # if height_map_initializer is None:
    #     height_map_initializer = torch.full(height_map_shape, 1e-4, dtype=torch.float64)

    # height_map_var = Parameter(height_map_initializer, requires_grad=True)

    height_map_full = torch.

    # height_map = torch.square(height_map_full)

    element = PhasePlate(wave_lengths=wave_lengths,
                         height_map=height_map,
                         refractive_idcs=refractive_idcs,
                         height_tolerance=height_tolerance)

    return element(input_field)

def fourier_element(): #TODO
    pass

def zernike_element(): #TODO
    pass

def get_vanilla_height_map(): #TODO
    pass

def get_fourier_height_map(): #TODO
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