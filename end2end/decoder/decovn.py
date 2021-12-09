import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import end2end.optics.optics_utils as optics_utils
from config import CUDA_DEVICE

class inverse_filter(nn.Module):
    """
    Inverse filtering in the frequency domain.
    """
    def __init__(self, gamma=None, init_gamma=2.):
        super(inverse_filter, self).__init__()

        self.init_gamma = init_gamma
        if gamma is None:
            # Gamma (the regularization parameter) is also a trainable parameter.
            self.gamma = self.gamma_initializer()
        else:
            self.gamma = gamma

    def gamma_initializer(self):
        gamma = torch.Tensor(self.init_gamma)
        gamma = torch.square(gamma) # Enforces positivity of gamma.
        return nn.parameter.Parameter(gamma, requires_grad=True)

    def forward(self, blurred_img, psfs):
        """
        :param blurred_img: [m, c, x, y]
        :param psfs: [m, c, x, y]
        :return: deconvolved output image
        """

        # img_fft = torch.fft.fft2(blurred_img)
        # otf = optics_utils.psf2otf(psfs, output_size=blurred_img.shape[1:3])

        return NotImplementedError