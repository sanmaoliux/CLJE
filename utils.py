import numpy as np
import math
from sklearn.metrics import normalized_mutual_info_score, v_measure_score, adjusted_rand_score, accuracy_score
from sklearn import cluster
from sklearn.preprocessing import Normalizer
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

def normalize_multiview_data(data_views, row_normalized=True):
    '''The rows or columns of a matrix normalized '''
    norm2 = Normalizer(norm='l2')
    num_views = len(data_views)
    for idx in range(num_views):
        if row_normalized:
            data_views[idx] = norm2.fit_transform(data_views[idx])
        else:
            data_views[idx] = norm2.fit_transform(data_views[idx].T).T

    return data_views

def spectral_clustering(W, num_clusters):
    """
    Apply spectral clustering on W.
    # Arguments
    :param W: an affinity matrix
    :param num_clusters: the number of clusters
    :return: cluster labels.
    """
    assign_labels = 'kmeans'
    spectral = cluster.SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack', affinity='precomputed')
    spectral.fit(W)
    labels = spectral.fit_predict(W)

    return labels

def cal_spectral_embedding(W, num_clusters):
    D = np.diag(1 / np.sqrt(np.sum(W, axis=1) + math.e))
    Z = np.dot(np.dot(D, W), D)
    U, _, _ = np.linalg.svd(Z)
    eigenvectors = U[:, 0 : num_clusters]

    return eigenvectors

def cal_spectral_embedding_1(W, num_clusters):
    D = np.diag(np.power((np.sum(W, axis=1) + math.e), -0.5))
    L = np.eye(len(W)) - np.dot(np.dot(D, W), D)
    eigvals, eigvecs = np.linalg.eig(L)
    x_val = []
    x_vec = np.zeros((len(eigvecs[:, 0]), len(eigvecs[0])))
    for i in range(len(eigvecs[:, 0])):
        for j in range(len(eigvecs[0])):
            x_vec[i][j] = eigvecs[i][j].real
    for i in range(len(eigvals)):
        x_val.append(eigvals[i].real)
    indices = np.argsort(x_val)[: num_clusters]
    eigenvectors = x_vec[:, indices[: num_clusters]]

    return eigenvectors

def cal_l2_distances(data_view):
    '''
    calculate Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        dists[i] = np.sqrt(np.sum(np.square(data_view - data_view[i]), axis=1)).T
    return dists

def cal_l2_distances_1(data_view):
    '''
    calculate Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            dists[i][j] = np.sqrt(np.sum(np.square(data_view[i] - data_view[j])))

    return dists

def cal_squared_l2_distances(data_view):
    '''
    calculate squared Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        dists[i] = np.sum(np.square(data_view - data_view[i]), axis=1).T
    return dists

def cal_squared_l2_distances_1(data_view):
    '''
    calculate squared Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            dists[i][j] = np.sum(np.square(data_view[i] - data_view[j]))

    return dists

def cal_similiarity_matrix(data_view, k):
    '''
    calculate similiarity matrix
    '''
    num_samples = data_view.shape[0]
    dist = cal_squared_l2_distances(data_view)

    W = np.zeros((num_samples, num_samples), dtype=float)

    idx_set = dist.argsort()[::1]
    for i in range(num_samples):
        idx_sub_set = idx_set[i, 1:(k + 2)]
        di = dist[i, idx_sub_set]
        W[i, idx_sub_set] = (di[k] - di) / (di[k] - np.mean(di[0:(k - 1)]) + math.e)

    W = (W + W.T) / 2

    return W

def calculate_joint_entropy(z_i, z_j, bins=30):
    z_i = z_i.cpu().detach().numpy().ravel()
    z_j = z_j.cpu().detach().numpy().ravel()

    # 移除 NaN 和 Inf 值
    valid_mask = np.isfinite(z_i) & np.isfinite(z_j)
    z_i = z_i[valid_mask]
    z_j = z_j[valid_mask]

    if z_i.size == 0 or z_j.size == 0:
        return 0.0  # 防止全为 NaN 的情况

    hist_2d, _, _ = np.histogram2d(z_i, z_j, bins=bins)
    p_ij = hist_2d / np.sum(hist_2d)
    p_ij = p_ij[np.nonzero(p_ij)]

    if p_ij.size == 0:
        return 0.0  # 防止分布全为 0 的情况

    return -np.sum(p_ij * np.log(p_ij))
