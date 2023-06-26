import torch
from torch import nn
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from torch.nn import functional as F

from IPython import embed


def pdist_cos(emb1, emb2):
    B1, D = emb1.size()
    B2, D = emb2.size()
    dot = emb1 @ emb2.t()
    norm1 = emb1.norm(dim=1).view(B1, 1)
    norm2 = emb2.norm(dim=1).view(1, B2)
    dot /= norm1 * norm2
    return dot


def pdist_cos_np(emb1, emb2):
    B1, D = emb1.shape[0], emb1.shape[1]
    B2, D = emb2.shape[0], emb2.shape[1]
    dot = emb1 @ emb2.T
    norm1 = np.resize(np.linalg.norm(emb1, axis=1), (B1, 1))
    norm2 = np.resize(np.linalg.norm(emb2, axis=1), (1, B2))
    dot /= np.matmul(norm1, norm2)
    return dot


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



colors = ['gray', 'orangered', 'gold', 'darkorange', 'limegreen', 'deepskyblue', 'darkviolet']

c, h, w = 256, 72, 36
part = 6

# f = open('figs/histogram_method.pkl','rb')
# dic = pickle.load(f)
# f.close()

f = open('figs/histogram_baseline.pkl','rb')
dic2 = pickle.load(f)
f.close()

query_feat_high = dic2["query_feat_high"]
gall_feat_high = dic2["gall_feat_high"]
query_feat_low = dic2["query_feat_low"]
gall_feat_low = dic2["gall_feat_low"]
query_label = dic2["query_label"]
gall_label = dic2["gall_label"]

# for cosine similarity
query_feat_high = F.normalize(query_feat_high, dim=1)
gall_feat_high = F.normalize(gall_feat_high, dim=1)
query_feat_low = F.normalize(query_feat_low, dim=1)
gall_feat_low = F.normalize(gall_feat_low, dim=1)

# l2norm = Normalize(2)
# query_feat_high = l2norm(query_feat_high)
# gall_feat_high = l2norm(gall_feat_high)
# query_feat_low = l2norm(query_feat_low)
# gall_feat_low = l2norm(gall_feat_low)


distmat_high_cross = torch.mm(query_feat_high, gall_feat_high.permute(1, 0))
distmat_low_cross = torch.mm(query_feat_low, gall_feat_low.permute(1, 0))
distmat_high_intra_q = torch.mm(query_feat_high, query_feat_high.permute(1, 0))
distmat_high_intra_p = torch.mm(gall_feat_high, gall_feat_high.permute(1, 0))
distmat_low_intra_q = torch.mm(query_feat_low, query_feat_low.permute(1, 0))
distmat_low_intra_p = torch.mm(gall_feat_low, gall_feat_low.permute(1, 0))

qnum, pnum = query_label.shape[0], gall_label.shape[0]
mask = query_label.unsqueeze(-1).expand(-1, pnum).eq(gall_label.unsqueeze(-2).expand(qnum, -1))
maskq = query_label.unsqueeze(-1).expand(-1, qnum).eq(query_label.unsqueeze(-2).expand(qnum, -1)) & ~(torch.eye(qnum).to(torch.bool))
maskp = gall_label.unsqueeze(-1).expand(-1, pnum).eq(gall_label.unsqueeze(-2).expand(pnum, -1)) & ~(torch.eye(pnum).to(torch.bool))

bin = distmat_high_cross[mask].numpy()
n = bin.shape[0]
rd = lambda x: max(round(x, 5), 0)
bin = list(map(rd, bin))
plt.hist(bin, color="darkorange", histtype='stepfilled', alpha=0.3, bins=100, label="cross")

bin = np.hstack((distmat_high_intra_q[maskq].numpy(), distmat_high_intra_p[maskp].numpy()))
idx = random.sample(range(0, bin.shape[0]), n)
bin = bin[idx]
rd = lambda x: max(round(x, 5), 0)
bin = list(map(rd, bin))
plt.hist(bin, color="deepskyblue", histtype='stepfilled', alpha=0.3, bins=100, label="intra")

plt.yticks([])
plt.xlim(left=0)
plt.legend()
plt.grid()
plt.savefig("figs/histogram_method_high.png")
plt.cla()


bin = distmat_low_cross[mask].numpy()
n = bin.shape[0]
rd = lambda x: max(round(x, 5), 0)
bin = list(map(rd, bin))
plt.hist(bin, color="darkorange", histtype='stepfilled', alpha=0.3, bins=100, label="cross")


bin = np.hstack((distmat_low_intra_q[maskq].numpy(), distmat_low_intra_p[maskp].numpy()))
idx = random.sample(range(0, bin.shape[0]), n)
bin = bin[idx]
rd = lambda x: max(round(x, 5), 0)
bin = list(map(rd, bin))
plt.hist(bin, color="deepskyblue", histtype='stepfilled', alpha=0.3, bins=100, label="intra")

plt.yticks([])
plt.xlim(left=0)
plt.legend()
plt.grid()
plt.savefig("figs/histogram_method_low.png")
plt.cla()
