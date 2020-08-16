import torch
import numpy as np

class BatchHard:

    def __call__(self, embeds, labels):
        embeds = torch.stack([embed for embed in embeds]).reshape(-1, 2)
        labels = labels.reshape(-1)

        dist_mtx = self._pdist_torch(embeds, embeds).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis=1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis=1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg

    def _pdist_torch(self, emb1, emb2):
        '''
        compute the eucilidean distance matrix between embeddings1 and embeddings2
        using gpu
        '''
        m, n = emb1.shape[0], emb2.shape[0]
        emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
        emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_mtx = emb1_pow + emb2_pow
        dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
        dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
        return dist_mtx
