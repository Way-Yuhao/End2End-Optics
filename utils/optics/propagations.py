"""Optical elements & propagations"""

import abc  # abstract class

import numpy as np
import torch
import torch.nn.functional as F


class Propagation(abc.ABC):
    def __init__(self, input_shape, distance, discretization_size, lamda):
        """

        :param input_shape:
        :param distance: z
        :param discretization_size:
        :param lamda: wavelengths
        """
        # TODO: what is the structure of input_shape?
        self.input_shape = input_shape
        self.distance = distance
        self.wave_lengths = lamda
        self.wave_nos = 2. * np.pi / lamda  # wavenumber, Eq3
        self.discretization_size = discretization_size  # TODO: of sensor?

    @abc.abstractmethod
    def _propagate(self, input_field):
        """ propogates an input field through a medium """

    def __call__(self, input_field):
        return self._propagate(input_field)


class FresnelPropagation(Propagation):
    def _propagate(self, input_field):
        _, M_orig, N_orig, _ = self.input_shape  # TODO: M=height, N=width?
        # zero padding
        Mpad, Npad = M_orig // 4, N_orig // 4
        M = M_orig + 2 * Mpad
        N = N_orig + 2 * Npad
        # padded_input_field = tf.pad(input_field, [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]])
        padded_input_field = F.pad(input_field, [0, 0, Mpad, Mpad, Npad, Npad, 0, 0], mode='constant', value=0)
        [x, y] = np.mgrid[-N // 2: N // 2, -M // 2: M // 2]

        # spatial frequency; max frequency = 1 / ( 2 * pixel_size) TODO: why?
        fx = x / (self.discretization_size * N)
        fy = y / (self.discretization_size * M)

        # rearranges a zero-frequency-shifted Fourier transform Y back to the original transform output
        fx = torch.fft.ifftshift(fx)
        fy = torch.fft.ifftshift(fy)

        fx = fx[None, :, :, None]         # TODO: why
        fy = fy[None, :, :, None]

        squared_sum = torch.square(fx) + torch.square(fy)

        # create an un-trainable variable so that this computation can be reused from call to call
        if torch.is_tensor(self.distance):
            tmp = np.float64(self.wave_lengths * np.pi * -1. * squared_sum)
            constant_exponent_part = torch.empty(size=padded_input_field.shape, dtype=torch.float64,
                                                 requires_grad=False)
            # H = compl_exp_tf() TODO

