import torch
import numpy as np
import torch.nn.functional as F

# class Propagation(torch.nn.Module):
#     def __init__(self, input_shape, distance, discretization_size, wave_lengths):
#         """
#         :param input_shape:
#         :param distance: z
#         :param discretization_size:
#         :param wave_lengths: wavelengths
#         """
#         super(Propagation, self).__init__()
#         # TODO: what is the structure of input_shape?
#         self.input_shape = input_shape
#         self.distance = torch.tensor(distance)
#         self.wave_lengths = torch.tensor(wave_lengths)
#         # self.wave_nos = 2. * np.pi / wave_lengths  # wavenumber, Eq3
#         self.discretization_size = discretization_size  # TODO: of sensor?
#
#     def forward(self, x):
#         """
#         Propagation
#         :param x: input field
#         :return:
#         """


class FresnelPropagation(torch.nn.Module):
    def __init__(self, input_shape, distance, discretization_size, wave_lengths):
        """
        :param input_shape:
        :param distance: z
        :param discretization_size:
        :param wave_lengths: wavelengths
        """
        super(FresnelPropagation, self).__init__()
        self.input_shape = input_shape
        self.distance = torch.tensor(distance)
        self.wave_lengths = torch.tensor(wave_lengths)
        # self.wave_nos = 2. * np.pi / wave_lengths  # wavenumber, Eq3
        self.discretization_size = discretization_size  # TODO: of sensor?

    def forward(self, input_field):
        """
        :param input_field: complex valued wavefront (Tensor[batch_size, num_wavelengths, height, width])
        :return:
        """

        _, _, M_orig, N_orig = self.input_shape
        # zero padding
        Mpad, Npad = M_orig // 4, N_orig // 4
        M = M_orig + 2 * Mpad
        N = N_orig + 2 * Npad

        padded_input_field = F.pad(input_field, [0, 0, Mpad, Mpad, Npad, Npad, 0, 0], mode='constant', value=0.0)
        [x, y] = np.mgrid[-N // 2: N // 2, -M // 2: M // 2]

        # spatial frequency; max frequency = 1 / ( 2 * pixel_size)
        fx = torch.tensor(x) / (self.discretization_size * N)
        fy = torch.tensor(y) / (self.discretization_size * M)

        # rearranges a zero-frequency-shifted Fourier transform Y back to the original transform output
        fx = torch.fft.ifftshift(fx)
        fy = torch.fft.ifftshift(fy)

        fx = fx[None, None, :, :]
        fy = fy[None, None, :, :]

        # Transfer function for Fresnel propagation
        # (see derivation at https://www.cis.rit.edu/class/simg738/Handouts/Derivation_of_Fresnel_Diffraction.pdf)
        squared_sum = torch.square(fx) + torch.square(fy)
        complex_exponent_part = -1. * np.pi * self.wave_lengths * squared_sum * self.distance
        # complex_exponent_part = 2 * np.pi * self.distance * (1/self.wave_lengths) - 1. * np.pi * self.wave_lengths * squared_sum * self.distance # should it be this????

        H = torch.exp(torch.complex(torch.zeros_like(complex_exponent_part), complex_exponent_part))

        objFT = torch.fft.fft2(padded_input_field)
        # Convolution
        out_field = torch.fft.ifft2(objFT * H)

        return out_field
