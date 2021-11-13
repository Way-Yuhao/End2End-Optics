"""Optical elements"""


class PhasePlate():
    """

    """
    def __init__(self,
                 wave_lengths,
                 height_map,
                 refractive_idcs,
                 height_tolerance=None,
                 lateral_tolerance=None):
        """

        :param wave_lengths:
        :param height_map:
        :param refractive_idcs:
        :param height_tolerance:
        :param lateral_tolerance:
        """
        self.wave_lengths = wave_lengths
        self.height_map = height_map
        self.refractive_idcs = refractive_idcs
        self.height_tolerance = height_tolerance
        self.lateral_tolerance = lateral_tolerance

        self._build()

    def _build(self): #TODO:

        pass

    def __call__(self, input_field):
        """

        :param input_field:
        :return:
        """
        pass

def height_map_element(): # TODO:
    pass

def fourier_element(): #TODO:
    pass

def zernike_element(): #TODO:
    pass

def get_vanilla_height_map(): #TODO:
    pass

def get_fourier_height_map(): #TODO:
    pass


class SingleLensSetup():  # TODO:
    """

    """

    def __init__(self):
        pass


class ZernikeSystem():  # TODO:
    """

    """

    def __init__(self):
        pass