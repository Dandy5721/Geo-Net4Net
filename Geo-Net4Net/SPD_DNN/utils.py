import torch
import numpy as np

def symmetric(A):
    return 0.5 * (A + A.transpose(-2, -1))

def orthogonal_projection(A, B):
    out = A - B @ A.transpose(-2, -1) @ B
    return out

def retraction(A, ref=None):
    if ref == None:
        data = A
    else:
        data = A + ref
    Q, R = data.qr()
    # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
    sign = (R.diagonal(dim1=-2, dim2=-1).sign() + 0.5).sign().diag_embed()
    out = Q @ sign
    return out
