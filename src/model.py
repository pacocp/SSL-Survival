import torch.nn as nn
from types_ import *
import numpy as np


class SSLModel(nn.Module):
    def __init__(self,
                 rna_encoder,
                 wsi_encoder,
                 distance: str,
                 in_channels: int,
                 out_channels: int,
                 hidden_dims: List):
        super(SSLModel,self).__init__()
        self.rna_encoder = rna_encoder
        self.wsi_encoder = wsi_encoder

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.fc = nn.Sequential(*modules)
        self.out_layer = nn.Linear(in_channels, out_channels)
        self.distance = distance

    def forward(self, x1, x2):
        embd1 = self.rna_encoder(x1)
        embd2 = self.wsi_encoder.forward_extract(x2)

        if self.distance == 'euclidean':
            dist = (embd1 - embd2).pow(2).sum(1).sqrt()
        dist = dist.view(-1, 1)
        out = self.fc(dist)
        out = self.out_layer(out)
        return out

