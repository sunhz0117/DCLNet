import torch
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import random

from IPython import embed

tsne=TSNE()
c, h, w = 256, 72, 36
part = 6

f = open('figs/tsne_method.pkl','rb')
dic = pickle.load(f)
f.close()

# f = open('figs/tsne_baseline.pkl','rb')
# dic2 = pickle.load(f)
# f.close()

# pid = [4]

# feat_rgb = dic["map_rgb"][pid].permute((0, 2, 3, 1)).reshape(-1, c)
# feat_ir = dic["map_ir"][pid].permute((0, 2, 3, 1)).reshape(-1, c)
# mask_rgb = dic["mask_rgb"][pid].reshape(-1)
# mask_ir = dic["mask_ir"][pid].reshape(-1)
# id = dic["id"][pid]
# len, c = feat_rgb.shape

# feat = torch.cat((feat_rgb, feat_ir), 0)
# res = tsne.fit_transform(feat) # to 2 dims
# res_rgb = res[:len]
# res_ir = res[len:]

# colors = ['gray', 'orangered', 'gold', 'darkorange', 'limegreen', 'deepskyblue', 'darkviolet']

# # print("RGB")
# for i in range(part + 1):
#     # print(torch.sum(mask_rgb == i).item())
#     d = res_rgb[mask_rgb == i]
#     idx = random.sample(range(0, d.shape[0]), 20)
#     d = d[idx]
#     plt.scatter(d[:,0],d[:,1], color=colors[i], marker='.', label="part-{}".format(i))

# # print("IR")
# for i in range(part + 1):
#     # print(torch.sum(mask_ir == i).item())
#     d = res_ir[mask_ir == i]
#     idx = random.sample(range(0, d.shape[0]), 20)
#     d = d[idx]
#     plt.scatter(d[:,0],d[:,1], color=colors[i], marker='^')

# plt.xticks([])
# plt.yticks([])
# plt.savefig("figs/tsne_baseline.png")
# plt.cla()

for lb in range(32):
    pid = [lb]
    print(lb)

    feat_rgb = dic["map_rgb"][pid].permute((0, 2, 3, 1)).reshape(-1, c)
    feat_ir = dic["map_ir"][pid].permute((0, 2, 3, 1)).reshape(-1, c)
    mask_rgb = dic["mask_rgb"][pid].reshape(-1)
    mask_ir = dic["mask_ir"][pid].reshape(-1)
    id = dic["id"][pid]
    len, c = feat_rgb.shape

    feat = torch.cat((feat_rgb, feat_ir), 0)
    res = tsne.fit_transform(feat) # to 2 dims
    res_rgb = res[:len]
    res_ir = res[len:]

    colors = ['gray', 'orangered', 'gold', 'darkorange', 'limegreen', 'deepskyblue', 'darkviolet']

    # print("RGB")
    for i in range(part + 1):
        # print(torch.sum(mask_rgb == i).item())
        d = res_rgb[mask_rgb == i]
        if d.shape[0] < 20:
            print(lb)
            continue
        idx = random.sample(range(0, d.shape[0]), 20)
        d = d[idx]
        plt.scatter(d[:,0],d[:,1], color=colors[i], marker='.', label="part-{}".format(i))

    # print("IR")
    for i in range(part + 1):
        # print(torch.sum(mask_ir == i).item())
        d = res_ir[mask_ir == i]
        if d.shape[0] < 20:
            print(lb)
            continue
        idx = random.sample(range(0, d.shape[0]), 20)
        d = d[idx]
        plt.scatter(d[:,0],d[:,1], color=colors[i], marker='^')

    plt.savefig("figs/tsne/tsne_method_{}.png".format(lb))
    plt.axis('off')
    plt.cla()