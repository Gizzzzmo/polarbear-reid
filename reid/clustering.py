from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import torch

import hdbscan

def k_means(features, k):
    labels = KMeans(k).fit_predict(features)
    return labels

def dbscan(features, rho, min_samples):
    dist = torch.cdist(features, features).detach().cpu().numpy()
    print(dist.mean())
    tri_mat = np.triu(dist,1)       # tri_mat.dim=2
    tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
    tri_mat = np.sort(tri_mat,axis=None)
    top_num = np.round(rho*tri_mat.size).astype(int)
    print(f'rho={rho}, top_num={top_num}, tri_mat.size={tri_mat.size}, {tri_mat.shape}')
    eps = tri_mat[:top_num].mean()
    labels = DBSCAN(eps, min_samples=min_samples, metric='precomputed', n_jobs=8).fit_predict(dist)
    return labels

def _hdbscan(distances, min_samples, min_cluster_size):
    clusterer = hdbscan.HDBSCAN(min_cluster_size, min_samples, metric='precomputed')
    labels = clusterer.fit_predict(distances)
    return clusterer, labels
