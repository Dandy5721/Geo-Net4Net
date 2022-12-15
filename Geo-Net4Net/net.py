from torch import nn
import torch

from spdnet.spd import SPDTransform, SPDRectified, SPDNormalization
from mean_shift import SPD_GBMS_RNN

class MSNet(nn.Module):
    def __init__(self, num_classes):
        super(MSNet, self).__init__()
        self.layers = nn.Sequential(

            SPDNormalization(268),
            SPDTransform(268, 128, 1),
            SPDNormalization(128),
            SPDRectified(),

            SPDTransform(128, 64, 1),
            SPDNormalization(64),
            SPDRectified(),

            SPDTransform(64, 32, 1),
            SPDNormalization(32),
            SPDRectified(),

            SPDTransform(32, 16, 1),
            SPDNormalization(16),

            SPD_GBMS_RNN(),
            SPD_GBMS_RNN(),
            SPD_GBMS_RNN(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
