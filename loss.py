import time
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from metrics import *
from torch.nn.functional import normalize
from dataprocessing import *

class DeepMVCLoss(nn.Module):
    def __init__(self, num_samples, num_clusters, lambda_, beta, gamma):
        super(DeepMVCLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.lambda_ = lambda_
        self.beta = beta
        self.gamma = gamma

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        return mask.bool()

    def forward_prob(self, q_i, q_j):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = (p_i * torch.log(p_i + 1e-10)).sum()

        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = (p_j * torch.log(p_j + 1e-10)).sum()

        entropy = ne_i + ne_j

        return entropy

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)

        if normalized:
            sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
        else:
            sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

        sim_i_j = torch.diag(sim, self.num_clusters)
        sim_j_i = torch.diag(sim, -self.num_clusters)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    # def reconstruction_loss(self, x, x_recon):
    #     return F.mse_loss(x, x_recon, reduction='mean')

    def joint_entropy_loss(self, features_list):
        num_views = len(features_list)
        total_joint_entropy = 0.0
        num_pairs = 0

        for i in range(num_views):
            for j in range(i + 1, num_views):
                joint_entropy = calculate_joint_entropy(features_list[i], features_list[j])
                total_joint_entropy += joint_entropy
                num_pairs += 1

        if num_pairs > 0:
            average_joint_entropy = total_joint_entropy / num_pairs
        else:
            average_joint_entropy = 0.0

        return average_joint_entropy

def calculate_joint_entropy(z_i, z_j, bins=30):
    z_i = z_i.cpu().detach().numpy().ravel()
    z_j = z_j.cpu().detach().numpy().ravel()

    # 移除 NaN 和 Inf 值
    valid_mask = np.isfinite(z_i) & np.isfinite(z_j)
    z_i = z_i[valid_mask]
    z_j = z_j[valid_mask]

    if z_i.size == 0 or z_j.size == 0:
        return 0.0

    hist_2d, _, _ = np.histogram2d(z_i, z_j, bins=bins)
    p_ij = hist_2d / np.sum(hist_2d)
    p_ij = p_ij[np.nonzero(p_ij)]

    if p_ij.size == 0:
        return 0.0

    return -np.sum(p_ij * np.log(p_ij))
