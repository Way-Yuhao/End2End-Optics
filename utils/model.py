
import torch
import torch.nn as nn


class RGBCollimator(nn.Module):  # TODO
    """Section 3.2 simple lens check"""
    def __init__(self):
        super().__init__()
        # trainable height map

        self.height_map = self._height_map_initializer()

    def forward(self, x):

        pass

    def _height_map_initializer(self):
        height_map = torch.full(self.height_map_shape, 1e-4, dtype=torch.float64)
        return nn.parameter.Parameter(height_map, requires_grad=True)


