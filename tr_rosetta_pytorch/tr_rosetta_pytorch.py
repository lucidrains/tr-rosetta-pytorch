import torch
from torch import nn, einsum
import torch.nn.functional as F

class trDesign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class trRosetta(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x