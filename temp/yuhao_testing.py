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
    a = 1.12345
    print("{:.2}".format(a))

if __name__ == "__main__":
    main()
