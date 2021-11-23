"""playground python file to test behavior of functions"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from yuhao_testing2 import multiply
from config import CUDA_DEVICE
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # return x * torch.tensor([2])
        wtf = self.a + torch.nn.Parameter(torch.tensor([2.]).to(CUDA_DEVICE))
        return multiply(x, self.a)


def main():
    tb = SummaryWriter('../runs/test' + "-v0.0")  # TODO rename this
    print(os.getcwd())
    print(os.path.exists("./sample_psf.pt"))
    psf = torch.load("./sample_psf.pt")  # [1, 3, 512, 512]
    print(psf.shape)
    psf_plot = torch.sum(psf, dim=2)
    print("sum of psf = {}".format(torch.sum(psf)))
    print(psf_plot.shape)  # shape [1, 3, 512]

    psf_plot = psf_plot.cpu().detach().numpy()
    print(psf_plot.shape)
    fig, ax = plt.subplots()
    ax.plot(psf_plot[0, 0, :], c='r')
    ax.plot(psf_plot[0, 1, :], c='g')
    ax.plot(psf_plot[0, 2, :], c='b')
    # plt.show()
    tb.add_figure(tag="1D_psf", figure=fig, global_step=1)
    tb.add_figure(tag="1D_psf", figure=fig, global_step=2)
    tb.close()


if __name__ == "__main__":
    main()
