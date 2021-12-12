import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import end2end.optics.optics_utils as optics_utils
from end2end.config import CUDA_DEVICE


class InverseFilter(nn.Module):
    """
    Inverse filtering in the frequency domain.
    """
    def __init__(self, gamma=None, init_gamma=2.):
        super(InverseFilter, self).__init__()

        self.init_gamma = init_gamma.to(CUDA_DEVICE)
        if gamma is None:
            # Gamma (the regularization parameter) is also a trainable parameter.
            self.gamma = self.gamma_initializer()
        else:
            self.gamma = gamma

    def gamma_initializer(self):
        # gamma = torch.Tensor(self.init_gamma)
        gamma = torch.square(self.init_gamma)  # Enforces positivity of gamma.
        return nn.parameter.Parameter(gamma, requires_grad=True)

    def forward(self, blurred_img, estimated_img, psfs):
        """
        :param blurred_img: [m, c, x, y]
        :param estimated_img: [m, c, x, y]
        :param psfs: [m, c, x, y]
        :return: deconvolved output image
        """

        img_fft = torch.fft.fft2(blurred_img)
        otf = optics_utils.psf2otf(psfs, output_size=blurred_img.shape)  # 0, 1, 2, 3

        # This is a slight modification to standard inverse filtering - gamma not only regularizes the inverse filtering,
        # but also trades off between the regularized inverse filter and the unfiltered estimate_transp.
        numerator = torch.conj(otf) * img_fft + torch.fft.fft2(torch.complex(self.gamma*estimated_img,
                                                            torch.zeros_like(estimated_img)))
        denominator = torch.square(torch.abs(otf)) + self.gamma
        filtered = torch.div(numerator, denominator)

        result = torch.fft.ifft2(filtered)

        return torch.real(result)