import torch
from sklearn.cluster import SpectralClustering

class GeneratePseudoLabels:
    def __init__(self, n_clusters, bandwidth, n_jobs=-1):
        self.n_clusters = n_clusters
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs

    def fit(self, X):
        S, U = X.detach().symeig(eigenvectors=True)
        S = S.log().diag_embed()
        log_X = U @ S @ U.transpose(-2, -1)
        pair_dis = torch.norm(log_X.unsqueeze(-4) - log_X.unsqueeze(-3), p='fro', dim=(-2, -1))
        pair_dis_square = pair_dis ** 2
        affinity = torch.exp(-0.5 * pair_dis_square / (self.bandwidth * self.bandwidth))
        clustering = SpectralClustering(affinity='precomputed', n_clusters=self.n_clusters, n_jobs=self.n_jobs)
        self.labels_ = clustering.fit(affinity).labels_
        return self
