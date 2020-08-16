import torch
import numpy as np


class BatchHard:

    def __call__(self, embeddings, labels):
        # Unroll embeddings and labels
        embeddings = torch.stack([embed for embed in embeddings]).reshape(-1, 2)
        labels = labels.reshape(-1)

        batch_size = labels.shape[0]
        distances = self._calc_dist(embeddings, embeddings).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))

        diag_indices = np.diag_indices(batch_size)

        # Calc positive indices
        label_eq_mask = labels == labels.T
        label_eq_mask[diag_indices] = False
        dist_same = distances.copy()
        dist_same[label_eq_mask == False] = -np.inf
        positive_indices = np.argmax(dist_same, axis=1)

        # Calc negative indices
        dist_diff = distances.copy()
        label_eq_mask[diag_indices] = True
        dist_diff[label_eq_mask == True] = np.inf
        negative_indices = np.argmin(dist_diff, axis=1)

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
