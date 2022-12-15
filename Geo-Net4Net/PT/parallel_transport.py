import torch
import numpy as np

def log(X):
    S, U = X.symeig(eigenvectors=True)
    mask = (S <= 0).any(dim=-1)
    if mask.any():
        S_min, _ = S.min(dim=-1)
        S = S + ((1e-5 + abs(S_min)) * mask).unsqueeze(-1)

    S = S.log().diag_embed()
    return U @ S @ U.transpose(-2, -1)

def exp(X):
    S, U = X.symeig(eigenvectors=True)
    S = S.exp().diag_embed()
    return U @ S @ U.transpose(-2, -1)

def sqrtm(X):
    S, U = X.symeig(eigenvectors=True)
    S = S.sqrt().diag_embed()
    return U @ S @ U.transpose(-2, -1)

def inv_sqrtm(X):
    S, U = X.symeig(eigenvectors=True)
    S = S.sqrt().reciprocal().diag_embed()
    return U @ S @ U.transpose(-2, -1)

def power(X, exponent):
    S, U = X.symeig(eigenvectors=True)
    S = S.pow(exponent).diag_embed()
    return U @ S @ U.transpose(-2, -1)

def geodesic(x, y, t):
    x_sqrt = x.cholesky()
    x_sqrt_inv = x_sqrt.inverse()
    return x_sqrt @ power(x_sqrt_inv @ y @ x_sqrt_inv.transpose(-2, -1), t) @ x_sqrt.transpose(-2, -1)

def logm(x, y):
    c = x.cholesky()
    c_inv = c.inverse()
    return c @ log(c_inv @ y @ c_inv.transpose(-2, -1)) @ c.transpose(-2, -1)

def expm(x, y):
    c = x.cholesky()
    c_inv = c.inverse()
    return c @ exp(c_inv @ y @ c_inv.transpose(-2, -1)) @ c.transpose(-2, -1)

def riemannian_mean(spds, num_iter=20, eps_thresh=1e-4):
    mean = spds.mean(dim=-3).unsqueeze(-3)
    for iter in range(num_iter):
        tangent_mean = logm(mean, spds).mean(dim=-3).unsqueeze(-3)
        mean = expm(mean, tangent_mean)
        eps = tangent_mean.norm(p='fro', dim=(-2, -1)).mean()
        if eps < eps_thresh:
            break
    return mean.squeeze(-3)