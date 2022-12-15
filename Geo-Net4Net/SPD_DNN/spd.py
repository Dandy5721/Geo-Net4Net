import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.autograd import Function
import numpy as np
import time
import math

from spdnet.utils import *
from spdnet import StiefelParameter, SPDParameter
from spd.parallel_transport import riemannian_mean, sqrtm, geodesic, inv_sqrtm

"""
Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning. In AAAI (Vol. 1, No. 2, p. 3).
"""

# BiMap layer
class SPDTransform(nn.Module):

    def __init__(self, input_size, output_size, in_channels=1):
        super(SPDTransform, self).__init__()

        if in_channels > 1:
            self.weight = StiefelParameter(torch.FloatTensor(in_channels, input_size, output_size), requires_grad=True)
        else:
            self.weight = StiefelParameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        weight = self.weight

        output = weight.transpose(-2, -1) @ input @ weight
        return output

class SPDVectorize(nn.Module):

    def __init__(self, vectorize_all=True):
        super(SPDVectorize, self).__init__()
        self.register_buffer('vectorize_all', torch.tensor(vectorize_all))

    def forward(self, input):
        row_idx, col_idx = np.triu_indices(input.shape[-1])
        output = input[..., row_idx, col_idx]

        if self.vectorize_all:
            output = torch.flatten(output, 1)
        return output

class SPDTangentSpace(nn.Module):

    def __init__(self, vectorize=True, vectorize_all=True):
        super(SPDTangentSpace, self).__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize(vectorize_all=vectorize_all)

    def forward(self, input):

        try:
            s, u = input.symeig(eigenvectors=True)
        except:
            print(input)
            torch.save(input, 'error.pt')

        s = s.log().diag_embed()
        output = u @ s @ u.transpose(-2, -1)

        if self.vectorize:
            output = self.vec(output)

        return output

class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.FloatTensor([epsilon]))

    def forward(self, input):
        try:
            s, u = input.symeig(eigenvectors=True)
        except:
            print(input)
            torch.save(input, 'error.pt')
        s = s.clamp(min=self.epsilon[0])
        s = s.diag_embed()
        output = u @ s @ u.transpose(-2, -1)

        return output

class SPDNormalization(nn.Module):
    def __init__(self, input_size, momentum=0.1):
        super(SPDNormalization, self).__init__()
        self.momentum = momentum
        # temp = torch.randn(input_size, input_size)
        self.register_buffer('running_mean', torch.eye(input_size))
        self.G = SPDParameter(torch.eye(input_size), requires_grad=True)

    def forward(self, input):
        if self.training:
            center = riemannian_mean(input, num_iter=1)
            with torch.no_grad():
                self.running_mean = geodesic(self.running_mean, center, self.momentum)
        else:
            center = self.running_mean
        center_sqrt_inv = inv_sqrtm(center)
        center_norm = center_sqrt_inv @ input @ center_sqrt_inv

        G_sqrt = sqrtm(self.G)
        output = G_sqrt @ center_norm @ G_sqrt

        return output