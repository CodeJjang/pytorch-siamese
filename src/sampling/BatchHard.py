import torch
import numpy as np


class BatchHard:
    def __init__(self, margin):
        self.margin = margin


    def __call__(self, embeddings, labels):
        # Unroll embeddings and labels
        embeddings = torch.stack([embed for embed in embeddings]).reshape(-1, 2)
        labels = labels.reshape(-1)

        batch_size = labels.shape[0]
        distances = self._calc_dist(embeddings, embeddings).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))

        diag_indices = np.diag_indices(batch_size)
        label_eq_mask = labels == labels.T

        # Calc all positive indices
        positive_indices = self._get_positive_indices(label_eq_mask, diag_indices)

        # Calc hard negative indices
        negative_indices = self._get_semi_hard_negative_indices(distances, label_eq_mask, diag_indices)

        pos = embeddings[positive_indices].contiguous().view(batch_size, -1)
        neg = embeddings[negative_indices].contiguous().view(batch_size, -1)
        return embeddings, pos, neg

    def _calc_dist(self, emb1, emb2):
        '''
        compute the eucilidean distance matrix between embeddings1 and embeddings2
        using gpu
        '''
        m, n = emb1.shape[0], emb2.shape[0]
        emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
        emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distances = emb1_pow + emb2_pow
        distances = distances.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
        distances = distances.clamp(min=1e-12).sqrt()
        return distances

    def _get_hard_negative_indices(self, distances, label_eq_mask, diag_indices):
        dist_diff = distances.copy()
        label_eq_mask[diag_indices] = True
        dist_diff[label_eq_mask == True] = np.inf
        negative_indices = np.argmin(dist_diff, axis=1)
        return negative_indices

    def _get_semi_hard_negative_indices(self, distances, label_eq_mask, diag_indices):
        positive_distances = distances.copy()
        label_eq_mask[diag_indices] = False
        positive_distances[label_eq_mask == False] = -np.inf

        negative_distances = distances.copy()
        label_eq_mask[diag_indices] = True
        negative_distances[label_eq_mask == True] = np.inf

        negative_distances[positive_distances >= negative_distances] = np.inf
        negative_distances[negative_distances >= positive_distances + self.margin] = np.inf
        negative_indices = np.argmin(negative_distances, axis=1)
        return negative_indices

    def _get_hard_positive_indices(self, distances, label_eq_mask, diag_indices):
        label_eq_mask[diag_indices] = False
        dist_same = distances.copy()
        dist_same[label_eq_mask == False] = -np.inf
        positive_indices = np.argmax(dist_same, axis=1)
        return positive_indices

    def _get_positive_indices(self, label_eq_mask, diag_indices):
        label_eq_mask[diag_indices] = False
        positive_indices_rows, positive_indices_cols = np.where(label_eq_mask == True)
        unique_positive_row_indices, unique_positive_col_indices = np.unique(positive_indices_rows, return_index=True)
        positive_indices = positive_indices_cols[unique_positive_col_indices]
        return positive_indices
