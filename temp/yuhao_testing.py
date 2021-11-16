"""playground python file to test behavior of functions"""

import numpy as np
import torch
import torch.nn.functional as F


class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # return x * torch.tensor([2])
        wtf = self.a + torch.nn.Parameter(torch.tensor([2.]).to("cuda:6"))
        return multiply(x, self.a)


def multiply(a, b):
    return a * b


def main():
    m = M()
    m.to("cuda:6")
    input = torch.tensor(4).to("cuda:6")
    out = m(input)
    print(out)


if __name__ == "__main__":
    main()
