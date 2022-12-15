import torch
from torch import nn
import numpy as np
from math import inf

class SPD_GBMS_RNN(nn.Module):
    def __init__(self, bandwidth=0.5):
        super(SPD_GBMS_RNN, self).__init__()
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth))

    def log(self, X):
        S, U = X.symeig(eigenvectors=True)
        S = S.log().diag_embed()
        return U @ S @ U.transpose(-2, -1)

    def exp(self, X):
        S, U = X.symeig(eigenvectors=True)
        S = S.exp().diag_embed()
        return U @ S @ U.transpose(-2, -1)

    def logm(self, X, Y):
        return self.log(Y) - self.log(X)

    def expm(self, X, Y):
        return self.exp(self.log(X) + Y)

    def forward(self, X):
        bandwidth = self.bandwidth
        try:
            log_X = self.log(X)
        except RuntimeError:
            print(X)
            exit(-1)

        pair_dis = torch.norm(log_X.unsqueeze(-4) - log_X.unsqueeze(-3) + 1e-7, p='fro', dim=(-2, -1))
        log_Y_X = log_X.unsqueeze(-4) - log_X.unsqueeze(-3)
        pair_dis_square = pair_dis ** 2
        W = torch.exp(-0.5 * pair_dis_square / (bandwidth * bandwidth))
        D = W.sum(dim=-1).diag_embed()
        D_inv = D.inverse()

        M = ((log_Y_X.permute(2, 3, 0, 1) @ W).diagonal(dim1=-2, dim2=-1) @ D_inv).permute(2, 0, 1)
        output = self.expm(X, M)

        return output

def cosine_similarity(input):
    output = input @ input.transpose(-2, -1) * 0.5 + 0.5
    return output

def distance_similarity(input):
    pair_dis = torch.norm(input.unsqueeze(-3) - input.unsqueeze(-2), dim=-1)
    return 1 / (1 + pair_dis)

def spd_distance_similarity(input):
    S, U = input.symeig(eigenvectors=True)
    S = S.log().diag_embed()
    log_X = U @ S @ U.transpose(-2, -1)
    pair_dis = torch.norm(log_X.unsqueeze(-4) - log_X.unsqueeze(-3) + 1e-7, p='fro', dim=(-2, -1))
    return 1 / (1 + pair_dis)

def similarity_loss(input, targets, similarity='cosine', alpha=0):
    sim_fun = {'cosine': cosine_similarity, 'eucl_dist': distance_similarity, 'spd_dist': spd_distance_similarity}
    assert similarity in sim_fun
    similarity = sim_fun[similarity](input)
    identity_matrix = targets.unsqueeze(0) == targets.unsqueeze(0).T
    loss = (1 - similarity) * identity_matrix + torch.clamp(similarity - alpha, min=0) * (~identity_matrix)
    loss = torch.mean(loss)
    return loss