"""playground python file to test behavior of functions"""

import os
import numpy as np
import torch
import optics


def main():
    a = torch.tensor([1, 1], dtype=torch.int8)
    print(a)
    print(a.dtype)
    b = torch.tensor(a, dtype=torch.float32)
    print(b)
    print(b.dtype)


if __name__ == "__main__":
    main()
