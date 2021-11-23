"""playground python file to test behavior of functions"""

import numpy as np
import torch
import torch.nn.functional as F
from yuhao_testing2 import multiply
from config import CUDA_DEVICE

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # return x * torch.tensor([2])
        wtf = self.a + torch.nn.Parameter(torch.tensor([2.]).to(CUDA_DEVICE))
        return multiply(x, self.a)


def main():
    print(torch.cuda.device_count())
    print(CUDA_DEVICE)


if __name__ == "__main__":
    main()
