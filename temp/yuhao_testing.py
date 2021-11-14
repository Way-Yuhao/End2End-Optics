"""playground python file to test behavior of functions"""

import numpy as np
import torch
from utils.edof_reader import DEPTH_OPTIONS

# a = torch.ones((10, 10, 3))
# a = torch.unsqueeze(a, dim=0)
# print(a.shape)
# print(torch.log(torch.tensor([5 * [1/5]])))

img_patch_size = (1024, 1024)


def generate_depth_map():
    """
    arbitrarily assign a planar depth for
    :return:
    """
    rand_depth_idx = np.random.multinomial(1, [1 / 5] * 5)
    rand_depth = DEPTH_OPTIONS[np.argmax(rand_depth_idx)]
    rand_depth_map = torch.ones(img_patch_size, dtype=torch.float32) * rand_depth
    return rand_depth_map


def main():
    for i in range(10):
        print(generate_depth_map()[0][0])


if __name__ == "__main__":
    main()
