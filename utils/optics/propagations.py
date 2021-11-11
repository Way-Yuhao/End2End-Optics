"""Optical elements & propagations"""

import abc  # abstract class

import numpy as np


class Propagation(abc.ABC):
    def __init__(self, input_shape, distance, discretization_size, lamda):
        """

        :param input_shape:
        :param distance:
        :param discretization_size:
        :param lamda: wavelengths
        """
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
        pass

