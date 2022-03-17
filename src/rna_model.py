import torch.nn as nn

from types_ import *
from utils import *


class RNAEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: List):
        super(RNAEncoder, self).__init__()

        self.in_channels = in_channels

        modules = [
        nn.Sequential(nn.Dropout())]
        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)