from re import M
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

import pickle
from IPython import embed

def normalize_rank(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def rank_loss(dist_mat, labels, margin,alpha,tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])
        
        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]

        ap_is_pos = torch.clamp(torch.add(dist_ap,margin-alpha),min=0.0)
        ap_pos_num = ap_is_pos.size(0) +1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum,float(ap_pos_num))

        an_is_pos = torch.lt(dist_an,alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval*(-1*an_less_alpha+alpha))
        an_weight_sum = torch.sum(an_weight) +1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm,an_weight))
        loss_an = torch.div(an_ln_sum,an_weight_sum)
        #print(dist_ap.max())
        #print(loss_an)

        total_loss = total_loss+loss_ap+loss_an
    total_loss = total_loss*1.0/N
    return total_loss


class RankedLoss(nn.Module):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    
    def __init__(self, margin=1.5, alpha=2, tval=1):
        super(RankedLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        
    def __call__(self, global_feat, labels, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize_rank(global_feat, axis=-1)
        dist_mat = euclidean_dist_rank(global_feat, global_feat)
        total_loss = rank_loss(dist_mat,labels,self.margin,self.alpha,self.tval)
         
        return total_loss


class Rank_loss(nn.Module):
 
    ## Basic idea for cross_modality rank_loss 8

    def __init__(self, margin_1=2.0, margin_2=2.0, alpha_1=2.4, alpha_2=2.2, tval=1.0):
        super(Rank_loss, self).__init__()
        self.margin_1 = margin_1 # for same modality0
        self.margin_2 = margin_2 # for different modalities
        self.alpha_1 = alpha_1 # for same modality
        self.alpha_2 = alpha_2 # for different modalities
        self.tval = tval

    def forward(self, x, targets, sub, norm = True):
        if norm:
            #x = self.normalize(x)
            x = torch.nn.functional.normalize(x, dim=1, p=2)

        dist_mat = self.euclidean_dist(x, x) # compute the distance


        loss = self.rank_loss(dist_mat, targets, sub)

        return loss,dist_mat

    def rank_loss(self, dist, targets, sub):
        loss = 0.0
        for i in range(dist.size(0)):
            is_pos = targets.eq(targets[i])
            is_pos[i] = 0
            is_neg = targets.ne(targets[i])

            intra_modality = sub.eq(sub[i])
            cross_modality = ~intra_modality

            mask_pos_intra = is_pos* intra_modality
            mask_pos_cross = is_pos* cross_modality
            mask_neg_intra = is_neg* intra_modality
            mask_neg_cross = is_neg* cross_modality

            ap_pos_intra = torch.clamp(torch.add(dist[i][mask_pos_intra], self.margin_1-self.alpha_1),0)
            ap_pos_cross = torch.clamp(torch.add(dist[i][mask_pos_cross], self.margin_2-self.alpha_2),0)

            loss_ap = torch.div(torch.sum(ap_pos_intra), ap_pos_intra.size(0)+1e-5)
            loss_ap += torch.div(torch.sum(ap_pos_cross), ap_pos_cross.size(0)+1e-5)

            dist_an_intra = dist[i][mask_neg_intra]
            dist_an_cross = dist[i][mask_neg_cross]

            an_less_intra = dist_an_intra[torch.lt(dist[i][mask_neg_intra], self.alpha_1)]
            an_less_cross = dist_an_cross[torch.lt(dist[i][mask_neg_cross], self.alpha_2)]

            an_weight_intra = torch.exp(self.tval*(-1* an_less_intra +self.alpha_1))
            an_weight_intra_sum = torch.sum(an_weight_intra)+1e-5
            an_weight_cross = torch.exp(self.tval*(-1* an_less_cross +self.alpha_2))
            an_weight_cross_sum = torch.sum(an_weight_cross)+1e-5
            an_sum_intra = torch.sum(torch.mul(self.alpha_1-an_less_intra,an_weight_intra))
            an_sum_cross = torch.sum(torch.mul(self.alpha_2-an_less_cross,an_weight_cross))

            loss_an =torch.div(an_sum_intra,an_weight_intra_sum ) +torch.div(an_sum_cross,an_weight_cross_sum )
            #loss_an = torch.div(an_sum_cross,an_weight_cross_sum )
            loss += loss_ap + loss_an
            #loss += loss_an

        return loss * 1.0/ dist.size(0)
         
    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x

    def euclidean_dist(self, x, y):
        m, n =x.size(0), y.size(0)

        xx = torch.pow(x,2).sum(1, keepdim= True).expand(m,n)
        yy = torch.pow(y,2).sum(1, keepdim= True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min =1e-12).sqrt()

        return dist

    def Circle_dist(self, dist_mat, i, js, is_pos):
        circle_dist = -dist_mat[i][j] # original distance

        # for postive pair
        if is_pos:
            circle_dist += torch.sum(dist[i][mask_pos_cross]+torch.sum(dist[j][mask_pos_cross]))- dist[j][j]## since ij is iq

        # for negative pair
        else:
            neg1 = targets.ne(targets[i])
            neg2 = targets.ne(targets[j])
            neg = neg1 * neg2

            ## find hard-samples

            if dist[i][dist_mat[i][neg].argmin()]+dist[j][dist_mat[i][neg].argmin()] < dist[i][dist_mat[j][neg].argmin()]+dist[j][dist_mat[j][neg].argmin()] :
                circle_dist += dist[i][dist_mat[i][neg].argmin()]+dist[j][dist_mat[i][neg].argmin()]

            else:
                circle_dist += dist[i][dist_mat[j][neg].argmin()]+dist[j][dist_mat[j][neg].argmin()]


        return circle_dist


class Rank_loss_F(nn.Module):
    def __init__(self, margin_1=0.5, margin_2=1.5, alpha_1=2.2, alpha_2=2.2, tval=1.0):
        super(Rank_loss_F, self).__init__()
        self.margin_1 = margin_1 # for same modality
        self.margin_2 = margin_2 # for different modalities
        self.alpha_1 = alpha_1 # for same modality
        self.alpha_2 = alpha_2 # for different modalities
        self.tval = tval

    def forward(self, x, targets, sub, norm = True):
        
        if norm:
            #x = self.normalize(x)
            x = torch.nn.functional.normalize(x, dim=1, p=2)
        
        n = x.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        modality_mask = sub.expand(n, n).eq(sub.expand(n, n).t())

        dist_mat = self.euclidean_dist(x, x) # compute the distance

        loss = self.rank_loss(dist_mat, mask, modality_mask, sub)

        return loss, dist_mat

    def rank_loss(self, dist, mask, modality_mask, sub):
        loss = 0.0
        for i in range(dist.size(0)):
            is_pos = mask[i, :]
            is_pos[i] = 0
            is_neg = ~is_pos
            intra_modality = modality_mask[i,:]
            cross_modality = ~intra_modality
            

            
            mask_pos_intra = is_pos* intra_modality
            mask_pos_cross = is_pos* cross_modality
            mask_neg_intra = is_neg* intra_modality
            mask_neg_cross = is_neg* cross_modality

            ap_pos_intra = torch.clamp(torch.add(dist[i][mask_pos_intra], self.margin_1-self.alpha_1),0)
            ap_pos_cross = torch.clamp(torch.add(dist[i][mask_pos_cross], self.margin_2-self.alpha_2),0)

            loss_ap = torch.div(torch.sum(ap_pos_intra), ap_pos_intra.size(0)+1e-5)
            loss_ap += torch.div(torch.sum(ap_pos_cross), ap_pos_cross.size(0)+1e-5)
          

            dist_an_intra = dist[i][mask_neg_intra]
            dist_an_cross = dist[i][mask_neg_cross]

            an_less_intra = dist_an_intra[torch.lt(dist[i][mask_neg_intra], self.alpha_1)]
            an_less_cross = dist_an_cross[torch.lt(dist[i][mask_neg_cross], self.alpha_2)]

            an_weight_intra = torch.exp(self.tval*(-1* an_less_intra +self.alpha_1))
            an_weight_intra_sum = torch.sum(an_weight_intra)+1e-5
            an_weight_cross = torch.exp(self.tval*(-1* an_less_cross +self.alpha_2))
            an_weight_cross_sum = torch.sum(an_weight_cross)+1e-5
            an_sum_intra = torch.sum(torch.mul(self.alpha_1-an_less_intra,an_weight_intra))
            an_sum_cross = torch.sum(torch.mul(self.alpha_2-an_less_cross,an_weight_cross))

            loss_an =torch.div(an_sum_intra,an_weight_intra_sum ) +torch.div(an_sum_cross,an_weight_cross_sum )
            #loss_an = torch.div(an_sum_cross,an_weight_cross_sum )

            loss += loss_ap + loss_an
            #loss += loss_an
        return loss * 1.0/ dist.size(0)


    def euclidean_dist(self, x, y):
        m, n =x.size(0), y.size(0)

        xx = torch.pow(x,2).sum(1, keepdim= True).expand(m,n)
        yy = torch.pow(y,2).sum(1, keepdim= True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min =1e-12).sqrt()

        return dist


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.1):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class CenterTripletLoss(nn.Module):
    def __init__(self, k_size, margin=0):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        centers = torch.stack(centers)
                
        dist_pc = (inputs - centers)**2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, centers, centers.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n, self.k_size):
            dist_an.append( (self.margin - dist[i][mask[i] == 0]).clamp(min=0.0).mean() )
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = dist_pc.mean() + dist_an.mean()
        return loss, dist_pc.mean(), dist_an.mean()


class InfoNCE(nn.Module):
    def __init__(self):
        super(InfoNCE, self).__init__()
        self.BCE_Loss = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        n = inputs.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t()).to(torch.float32)

        # Compute pairwise distance
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # Compute cosine distance
        dist = pdist_cos(inputs, inputs)
        dist = dist.clamp(min=1e-12)  # for numerical stability

        # F.cosine_similarity(inputs[0].repeat(inputs.size(0), 1), inputs)

        loss = self.BCE_Loss(dist, mask)
        return loss


def remask(feat_map, feat_mask, part, mask=None):
    N, C, H, W = feat_map.size()

    feat_map = F.normalize(feat_map, dim=1)
    
    feat_logits = feat_map.view(N, C, -1) #(N, C, H*W)
    feat_mask = feat_mask.view(N, -1) #(N, H*W)
    if mask is not None:
        mask = mask.view(N, -1).detach()
    
    # global logit
    logit = torch.bmm(feat_logits.transpose(1, 2), feat_logits).clamp(min=-1, max=1)

    refine_mask = torch.zeros_like(feat_mask)

    pnum = feat_mask.shape[1] // part # number of pixels in each part

    for id in range(N):
        feat_mean = torch.cat([torch.mean(feat_logits[id].transpose(0, 1)[feat_mask[id] == p+1], 0).unsqueeze(0) for p in range(part)], 0)
        feat_mean = F.normalize(feat_mean, dim=1)

        # part logit
        plogit = torch.mm(feat_logits[id].transpose(0, 1), feat_mean.transpose(0, 1)).clamp(min=-1, max=1)

        for i in range(part):
            st, fi = max(0, i-1), min(part, i+2)
            refine_mask[id][i*pnum: (i+1)*pnum] = torch.max(plogit[i*pnum: (i+1)*pnum, st: fi], 1)[1] + max(1, i)
            if mask is None:
                conf_mean = torch.mean(torch.max(plogit[i*pnum: (i+1)*pnum, st: fi], 1)[0]).detach()
                fg_mask = (torch.max(plogit[i*pnum: (i+1)*pnum, st: fi], 1)[0] > conf_mean).to(torch.int32)
                refine_mask[id][i*pnum: (i+1)*pnum] *= fg_mask
            else:
                fg_mask = (mask[id][i*pnum: (i+1)*pnum] > 0).to(torch.int32)
                refine_mask[id][i*pnum: (i+1)*pnum] *= fg_mask
        
        # a = feat_mask[id].reshape(H, W)
        # for i in range(H):
        #     for j in range(W):
        #         print(a[i][j].item(), end="")
        #     print()

        # a = refine_mask[id].reshape(H, W)
        # for i in range(H):
        #     for j in range(W):
        #         print(a[i][j].item(), end="")
        #     print()

        # get heat map
        # heat_map = torch.mm(feat_mean, feat_mean.permute(1, 0)).detach().cpu().numpy()
        # f = open('figs/heatmap_method.pkl', 'wb')
        # pickle.dump(heat_map, f)
        # f.close()
        # embed()

    refine_mask = refine_mask.reshape(N, H, W)

    return refine_mask


class SelfPixContrastive(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.

    Code imported from https://github.com/WXinlong/DenseCL/blob/main/openselfsup/models/heads/contrastive_head.py
    """

    def __init__(self, temperature=0.1):
        super(SelfPixContrastive, self).__init__()
        self.criterion = nn.BCELoss(reduce=False)
        self.temperature = temperature

    def forward(self, rgb_map, ir_map, rgb_mask, ir_mask):

        assert rgb_map.size() == ir_map.size()
        N, C, H, W = rgb_map.size()

        rgb_map = F.normalize(rgb_map, dim=1)
        ir_map = F.normalize(ir_map, dim=1)

        rgb_logits = rgb_map.view(N, C, -1) #(N, C, H*W)
        ir_logits = ir_map.view(N, C, -1) #(N, C, H*W)
        rgb_mask = rgb_mask.view(N, -1) #(N, H*W)
        ir_mask = ir_mask.view(N, -1) #(N, H*W)

        fg_mask = (rgb_mask > 0).to(torch.int32)

        mask = rgb_mask.unsqueeze(-1).expand(-1, -1, H*W).eq(ir_mask.unsqueeze(-2).expand(-1, H*W, -1)).to(torch.float32)
        logit = torch.bmm(rgb_logits.transpose(1, 2), ir_logits).clamp(min=-1, max=1)

        logit = torch.exp(logit / self.temperature)
        loss = (logit * mask).sum(-1) / (logit.sum(-1) + 1e-6)
        loss = loss * fg_mask  # filter ignored pixel
        loss = - torch.log(loss[loss != 0]).mean()
        
        return loss


class CrossPixContrastive(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.01.

    Code rewriten from https://github.com/WXinlong/DenseCL/blob/main/openselfsup/models/heads/contrastive_head.py
    """

    def __init__(self, temperature=0.05):
        super(CrossPixContrastive, self).__init__()
        self.criterion = nn.BCELoss(reduce=False)
        self.temperature = temperature

    def forward(self, rgb_map, ir_map, rgb_mask, ir_mask):

        assert rgb_map.size() == ir_map.size()
        N, C, H, W = rgb_map.size()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        rgb_map = F.normalize(rgb_map, dim=1)
        ir_map = F.normalize(ir_map, dim=1)

        rgb_logits = rgb_map.view(N, C, -1) #(N, C, H*W)
        ir_logits = ir_map.view(N, C, -1) #(N, C, H*W)
        rgb_mask = rgb_mask.view(N, -1) #(N, H*W)
        ir_mask = ir_mask.view(N, -1) #(N, H*W)

        mask = rgb_mask.unsqueeze(-1).expand(-1, -1, H*W).eq(ir_mask.unsqueeze(-2).expand(-1, H*W, -1)).to(torch.float32)
        fg_mask = (torch.cat((rgb_mask, ir_mask), 0) > 0)
        logit = torch.bmm(rgb_logits.transpose(1, 2), ir_logits).clamp(min=-1, max=1)

        logit = torch.exp(logit / self.temperature)
        loss = torch.cat(((logit * mask).sum(-1) / (logit.sum(-1) + 1e-6), (logit * mask).sum(-2) / (logit.sum(-2) + 1e-6)), 0)
        loss = loss * fg_mask  # filter ignored pixel
        loss = - torch.log(loss[loss != 0]).mean()

        return loss

class CrossPixContrastiveL2(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.

    Code imported from https://github.com/WXinlong/DenseCL/blob/main/openselfsup/models/heads/contrastive_head.py
    """

    def __init__(self, temperature=10):
        super(CrossPixContrastiveL2, self).__init__()
        self.criterion = nn.BCELoss(reduce=False)
        self.temperature = temperature

    def forward(self, rgb_map, ir_map, rgb_mask, ir_mask):

        assert rgb_map.size() == ir_map.size()
        N, C, H, W = rgb_map.size()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        rgb_logits = rgb_map.view(N, C, -1).transpose(1, 2) #(N, H*W, C)
        ir_logits = ir_map.view(N, C, -1).transpose(1, 2) #(N, H*W, C)
        rgb_mask = rgb_mask.view(N, -1) #(N, H*W)
        ir_mask = ir_mask.view(N, -1) #(N, H*W)

        # L2 Distance
        dist1 = torch.pow(rgb_logits, 2).sum(dim=2, keepdim=True).expand(N, H*W, H*W)
        dist2 = torch.pow(ir_logits, 2).sum(dim=2, keepdim=True).expand(N, H*W, H*W)
        dist = dist1 + dist2.transpose(1, 2)
        logit = dist - 2*torch.bmm(rgb_logits, ir_logits.transpose(1, 2))

        logit = torch.exp(-logit)

        mask = rgb_mask.unsqueeze(-1).expand(-1, -1, H*W).eq(ir_mask.unsqueeze(-2).expand(-1, H*W, -1)).to(torch.float32)
        fg_mask = (torch.cat((rgb_mask, ir_mask), 0) > 0)

        logit = torch.exp(logit / self.temperature)
        loss = torch.cat(((logit * mask).sum(-1) / (logit.sum(-1) + 1e-6), (logit * mask).sum(-2) / (logit.sum(-2) + 1e-6)), 0)
        loss = loss * fg_mask  # filter ignored pixel
        loss = - torch.log(loss[loss != 0]).mean()

        return loss


class PixContrastive(nn.Module):
    """Head for contrastive learning.
    Situation: Aligned Images

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.

    Code imported from https://github.com/WXinlong/DenseCL/blob/main/openselfsup/models/heads/contrastive_head.py
    """

    def __init__(self, temperature=0.1):
        super(PixContrastive, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, rgb_map, ir_map, targets):

        assert rgb_map.size() == ir_map.size()
        N, C, H, W = rgb_map.size()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        rgb_map = F.normalize(rgb_map, dim=1)
        ir_map = F.normalize(ir_map, dim=1)

        rgb_logits = rgb_map.view(N, C, -1) #(N, C, H*W)
        ir_logits = ir_map.view(N, C, -1) #(N, C, H*W)
        pos_mask = torch.eye(H*W).to(device).float().detach()
        logit = torch.bmm(rgb_logits.transpose(1, 2), ir_logits)

        # PixContrast
        logit = torch.exp(logit / self.temperature)
        loss = (logit * pos_mask).sum(-1).sum(-1) / (logit.sum(-1).sum(-1) + 1e-6)
        loss = - torch.log(loss).mean()

        return loss


class PixPro(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.

    Code imported from https://github.com/WXinlong/DenseCL/blob/main/openselfsup/models/heads/contrastive_head.py
    """

    def __init__(self, temperature=0.1):
        super(PixPro, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, rgb_map, ir_map, rgb_mask, ir_mask):

        assert rgb_map.size() == ir_map.size()
        N, C, H, W = rgb_map.size()

        rgb_map = F.normalize(rgb_map, dim=1)
        ir_map = F.normalize(ir_map, dim=1)

        rgb_logits = rgb_map.view(N, C, -1) #(N, C, H*W)
        ir_logits = ir_map.view(N, C, -1) #(N, C, H*W)
        rgb_mask = rgb_mask.view(N, -1) #(N, H*W)
        ir_mask = ir_mask.view(N, -1) #(N, H*W)

        mask = rgb_mask.unsqueeze(-1).expand(-1, -1, H*W).eq(ir_mask.unsqueeze(-2).expand(-1, H*W, -1)).to(torch.float32)
        fg_mask = ((rgb_mask.unsqueeze(-1).expand(-1, -1, H*W) > 0) & (ir_mask.unsqueeze(-2).expand(-1, H*W, -1) > 0)).to(torch.float32)
        pos_mask = mask * fg_mask
        logit = torch.bmm(rgb_logits.transpose(1, 2), ir_logits).clamp(min=-1, max=1)

        # PixPro
        loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)
        loss = - torch.mean(loss)

        return loss


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class PixCycleContrastive(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.

    """

    def __init__(self, temperature=0.1):
        super(PixCycleContrastive, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, inputs, targets):

        rgb_map, ir_map = inputs
        N, C, H, W = rgb_map.size()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # rgb_logits = rgb_map.view(N, C, H*W).permute(0, 2, 1), #(N, H*W, C)
        # ir_logits = ir_map.view(N, C, H*W).permute(0, 2, 1), #(N, H*W, C)

        # dist = pdist_cos(rgb_logits, ir_logits)

        # # logits = torch.cat((pos, neg), dim=1)
        # # logits /= self.temperature
        # # labels = torch.zeros((N, ), dtype=torch.long).cuda()

        # From PixPro
        rgb_map = F.normalize(rgb_map, dim=1)
        ir_map = F.normalize(ir_map, dim=1)

        rgb_logits = rgb_map.view(N, C, -1) #(N, C, H*W)
        ir_logits = ir_map.view(N, C, -1) #(N, C, H*W)
        pos_mask = torch.eye(H*W).to(device).float().detach()
        logit = torch.bmm(rgb_logits.transpose(1, 2), ir_logits)

        # PixCycleContrast
        logit = torch.exp(logit / self.temperature)
        rgb2ir2rgb =  torch.max(logit, dim=2)[0] / (logit.sum(-1) + 1e-6) * batched_index_select(torch.max(logit, dim=1)[0], 1, torch.max(logit, dim=2)[1]) / (logit.sum(-2) + 1e-6)
        ir2rgb2ir = torch.max(logit, dim=1)[0] / (logit.sum(-2) + 1e-6) * batched_index_select(torch.max(logit, dim=2)[0], 1, torch.max(logit, dim=1)[1]) / (logit.sum(-1) + 1e-6)
        loss = torch.cat((rgb2ir2rgb, ir2rgb2ir))
        loss = - torch.log(loss).mean()

        return loss


class ChannelContrastive(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.

    """

    def __init__(self, temperature=1):
        super(ChannelContrastive, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, inputs, targets):

        rgb_map, ir_map = inputs
        N, C, H, W = rgb_map.size()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        rgb_map = F.normalize(rgb_map, dim=1)
        ir_map = F.normalize(ir_map, dim=1)

        rgb_logits = rgb_map.view(N, C, -1) #(N, C, H*W)
        ir_logits = ir_map.view(N, C, -1) #(N, C, H*W)
        pos_mask = torch.eye(C).to(device).float().detach()
        logit = torch.bmm(rgb_logits, ir_logits.transpose(1, 2))

        # ChannelPro
        # loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)
        # loss = - loss.mean()

        # ChannelContrast
        logit = torch.exp(logit / self.temperature)
        loss = (logit * pos_mask).sum(-1).sum(-1) / (logit.sum(-1).sum(-1) + 1e-6)
        loss = - torch.log(loss).mean()

        return loss

################################################
# Metric Loss
################################################

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, batch_size, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    
    
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)
        
        # Compute pairwise distance, replace by the official when merged
        dist = pdist_torch(input1, input2)
        
        # For each anchor, find the hardest positive and negative
        # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i,i].unsqueeze(0))
            dist_an.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct*2
        

class BiTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.suffix
    """
    def __init__(self, batch_size, margin=0.5):
        super(BiTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)
        
        # Compute pairwise distance, replace by the official when merged
        dist = pdist_torch(input1, input2)
        
        # For each anchor, find the hardest positive and negative
        # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i,i].unsqueeze(0))
            dist_an.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss1 = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct1  =  torch.ge(dist_an, dist_ap).sum().item() 
        
        # Compute pairwise distance, replace by the official when merged
        dist2 = pdist_torch(input2, input1)
        
        # For each anchor, find the hardest positive and negative
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist2[i,i].unsqueeze(0))
            dist_an2.append(dist2[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        
        # Compute ranking hinge loss
        y2 = torch.ones_like(dist_an2)
        # loss2 = self.ranking_loss(dist_an2, dist_ap2, y2)
        
        loss2 = torch.sum(torch.nn.functional.relu(dist_ap2 + self.margin - dist_an2))
        
        # compute accuracy
        correct2  =  torch.ge(dist_an2, dist_ap2).sum().item()
        
        loss = torch.add(loss1, loss2)
        return loss, correct1 + correct2
        
        
class BDTRLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.suffix
    """
    def __init__(self, batch_size, margin=0.5):
        super(BDTRLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, inputs, targets):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        correct  =  torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct
        

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx


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


class Circle_Rank_loss(nn.Module):

    ## Basic idea for cross_modality rank_loss 8

    def __init__(self, margin_1=2.0, margin_2=2.0, alpha_1=2.4, alpha_2=2.2, tval=1.0):
        super(Circle_Rank_loss, self).__init__()
        self.margin_1 = margin_1 # for same modality
        self.margin_2 = margin_2 # for different modalities
        self.alpha_1 = alpha_1 # for same modality
        self.alpha_2 = alpha_2 # for different modalities
        self.tval = tval

    def forward(self, x, targets, sub, norm = True):
        if norm:
            #x = self.normalize(x)
            x = torch.nn.functional.normalize(x, dim=1, p=2)

        dist_mat = self.euclidean_dist(x, x) # compute the distance


        loss = self.rank_loss(dist_mat, targets, sub)

        return loss, dist_mat

    def rank_loss(self, dist, targets, sub):
        loss = 0.0
        for i in range(dist.size(0)):
            is_pos = targets.eq(targets[i])
            is_pos[i] = 0
            is_neg = targets.ne(targets[i])


            intra_modality = sub.eq(sub[i])
            cross_modality = ~intra_modality

            mask_pos_intra = is_pos * intra_modality
            mask_pos_cross = is_pos * cross_modality
            mask_neg_intra = is_neg * intra_modality
            mask_neg_cross = is_neg * cross_modality

            circ_pos = self.Circle_dist_mat(dist, i, mask_pos_intra, mask_pos_cross, targets) 
            circ_neg = self.Circle_dist_mat(dist, i, mask_neg_intra, mask_neg_cross, targets, False)
            # print(circ_pos.max())
            # print(circ_neg.max())
            # print(dist[i][mask_pos_cross].max())
            # print(dist[i][mask_neg_cross].min())
            # time.sleep(10)

            ap_pos_intra = torch.clamp(torch.add(dist[i][mask_pos_intra], self.margin_1-self.alpha_1), 0)
            ap_pos_cross = torch.clamp(torch.add(dist[i][mask_pos_cross], self.margin_2-self.alpha_2), 0)

            loss_ap = torch.div(torch.sum(ap_pos_intra), ap_pos_intra.size(0) + 1e-5)
            loss_ap += torch.div(torch.sum(ap_pos_cross), ap_pos_cross.size(0) + 1e-5)
 
            dist_an_intra = dist[i][mask_neg_intra]
            dist_an_cross = dist[i][mask_neg_cross]

            an_less_intra = dist_an_intra[torch.lt(dist[i][mask_neg_intra], self.alpha_1)]
            an_less_cross = dist_an_cross[torch.lt(dist[i][mask_neg_cross], self.alpha_2)]

            an_weight_intra = torch.exp(self.tval*(-1* an_less_intra + self.alpha_1))
            an_weight_intra_sum = torch.sum(an_weight_intra) + 1e-5

            # an_weight_intra = torch.exp(self.tval *(-1*circ_neg+ self.alpha_1))
            # an_weight_intra_sum = torch.sum(an_weight_intra)+1e-5

            an_weight_cross = torch.exp(self.tval*(-1* an_less_cross +self.alpha_2))
            an_weight_cross_sum = torch.sum(an_weight_cross) + 1e-5
            an_sum_intra = torch.sum(torch.clamp(torch.mul(self.alpha_1-an_less_intra, an_weight_intra), 0))
            an_sum_cross = torch.sum(torch.mul(self.alpha_2-an_less_cross, an_weight_cross))
            # an_sum_intra = torch.clamp(self.alpha_1-circ_neg, 0).sum()

            loss_an = torch.div(an_sum_intra, an_weight_intra_sum) + torch.div(an_sum_cross, an_weight_cross_sum)
            #loss_an = torch.div(an_sum_intra, circ_neg.size(0)+1e-5) +torch.div(an_sum_cross,an_weight_cross_sum )
            #loss_an = torch.div(an_sum_cross,an_weight_cross_sum )

            loss += loss_ap + loss_an
            #loss += loss_an

        return loss * 1.0 / dist.size(0)

    def normalize(self, x, axis=-1):
        x = 1. * x / (torch.norm(x, 2, axis, keepdim = True).expand_as(x) + 1e-12)
        return x

    def euclidean_dist(self, x, y):
        m, n =x.size(0), y.size(0)

        xx = torch.pow(x,2).sum(1, keepdim= True).expand(m,n)
        yy = torch.pow(y,2).sum(1, keepdim= True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min =1e-12).sqrt()

        return dist

    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x

    def euclidean_dist(self, x, y):
        m, n =x.size(0), y.size(0)

        xx = torch.pow(x,2).sum(1, keepdim= True).expand(m,n)
        yy = torch.pow(y,2).sum(1, keepdim= True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min =1e-12).sqrt()

        return dist

    def Circle_dist_mat(self, dist_mat, i, js, mask_cross, targets, is_pos = True):
        pos_dist_i = torch.sum(dist_mat[i][mask_cross]) # original distance 
        neg1 = targets.ne(targets[i])
        neg1 = neg1 * mask_cross
        tmp = torch.index_select(dist_mat, 0, torch.nonzero(js).squeeze())
        #circle_dist=[]
        if is_pos: 

            tmp2 = tmp[:, mask_cross]
            circle_dist = (torch.sum(tmp2, dim =1) + pos_dist_i)/8
        else: 
        ## find hard sample 
            a = targets[js]
            a = a.repeat(targets.size(0),1)
            #a = targets.ne(a)
            a = a.t().ne(targets)
            neg_1 = neg1.repeat(a.size(0),1) * a ## (i,j) denotes 
            
            tmp_dist = dist_mat[i,:].repeat(a.size(0),1)
            hard_index_mat = torch.argmin(tmp_dist * neg_1, dim=1).detach()

            tmp_dist2 = dist_mat[js,:]
                    
            j_hard_index_mat = torch.argmin(tmp_dist2 * neg_1, dim =1).detach()

            tmp = dist_mat[i,hard_index_mat]+ torch.diag(tmp[:, j_hard_index_mat]) + dist_mat[i,j_hard_index_mat] + torch.diag(tmp[:, hard_index_mat])

            circle_dist = tmp/4.0

        circle_dist += dist_mat[i][js] 
        return circle_dist/2.0

    def Circle_dist(self, dist_mat, i, js, mask_cross,  targets, is_pos=True):

        #circle_dist = -dist_mat[i][js] # original distance

        pos_dist_i = torch.sum(dist_mat[i][mask_cross])

        ## we will explore this after

        # print( dist_mat[i][mask_pos_cross].shape)
        # similar = dist_mat[js][mask_pos_cross].view(-1,1).mm(dist_mat[js][mask_pos_cross].view(-1,1).t())
        # print(similar.shape)

        neg1 = targets.ne(targets[i])

        circle_dist=[]

        # for postive pair
        if is_pos:
            tmp = 0
            for j in torch.nonzero(js):

                #print(torch.sum(dist_mat[j][mask_pos_cross], dim=1))
                tmp = pos_dist_i+torch.sum(dist_mat[j.item()][mask_cross])

                circle_dist.append(tmp/8.0) # we think

        # for negative pair
        else:
            neg1 = neg1*mask_cross
            for jt in torch.nonzero(js):
                j = jt.item()
                neg2 = targets.ne(targets[j])

                neg = neg2*neg1

            ## find hard-samples
                tmp =  dist_mat[i][dist_mat[i][neg].argmin()]+dist_mat[j][dist_mat[i][neg].argmin()]+ \
                        dist_mat[i][dist_mat[j][neg].argmin()]+dist_mat[j][dist_mat[j][neg].argmin()]
                # if dist[i][dist_mat[i][neg].argmin()]+dist[j][dist_mat[i][neg].argmin()] < dist[i][dist_mat[j][neg].argmin()]+dist[j][dist_mat[j][neg].argmin()] :
                #     tmp= dist[i][dist_mat[i][neg].argmin()]+dist[j][dist_mat[i][neg].argmin()]
                # else:
                #     circle_dist += dist[i][dist_mat[j][neg].argmin()]+dist[j][dist_mat[j][neg].argmin()]
                circle_dist.append(tmp/4.0)

        circle_dist = torch.stack(circle_dist, dim=0)
        circle_dist += dist_mat[i][js]
        # print(circle_dist)

        return circle_dist/2.0


class ASS_loss(nn.Module):
    def __init__(self, walker_loss=1.0, visit_loss=1.0):
        super(ASS_loss, self).__init__()
        self.walker_loss = walker_loss
        self.visit_loss = visit_loss
        self.ce = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, feature, sub, targets):
        ## normalize
        feature = torch.nn.functional.normalize(feature, dim=1, p=2)
        loss = 0.0
        for i in range(feature.size(0)):
            cross_modality = sub.ne(sub[i])

            # is_pos = targets.eq(targets[i])
            # is_neg = targets.ne(targets[i])
            p_logit_ab, v_loss_ab = self.probablity(feature, cross_modality,  targets)
            p_logit_ba, v_loss_ba = self.probablity(feature, ~cross_modality, targets)
            n1 = targets[cross_modality].size(0)
            n2 = targets[~cross_modality].size(0)

            is_pos_ab = targets[cross_modality].expand(n1,n1).eq(targets[cross_modality].expand(n1,n1).t())

            p_target_ab = is_pos_ab.float()/torch.sum(is_pos_ab, dim=1).float().expand_as(is_pos_ab)

            is_pos_ba = targets[~cross_modality].expand(n2,n2).eq(targets[~cross_modality].expand(n2,n2).t())
            
            #a = targets[cross_modality].repeat(1,n2).reshape_as(p_logit_ba)

            #b = targets[~cross_modality].repeat(1,n1).reshape_as(p_logit_ba)

            #p_target_ba = b.eq(a)
            
            p_target_ba = is_pos_ba.float()/torch.sum(is_pos_ba, dim=1).float().expand_as(is_pos_ba)


            p_logit_ab = self.logsoftmax(p_logit_ab)
            p_logit_ba = self.logsoftmax(p_logit_ba)

            #loss += self.ce(p_logit_ab, p_target_ab) + self.ce(p_logit_ab,p_target_ba )
            loss += (- p_target_ab * p_logit_ab).mean(0).sum()+ (- p_target_ba * p_logit_ba).mean(0).sum()

            loss += 1.0*(v_loss_ab+v_loss_ba)

        return loss/feature.size(0)/4

    def probablity(self, feature, cross_modality, target):
        a = feature[cross_modality]
        b = feature[~cross_modality]

        match_ab = a.mm(b.t())

        p_ab = F.softmax(match_ab, dim=-1)
        p_ba = F.softmax(match_ab.t(), dim=-1)
        p_aba = torch.log(1e-8+p_ab.mm(p_ba))

        #visit_loss = self.visit(p_ab)
        visit_loss = self.new_visit(p_ab, target, cross_modality)

        return p_aba, visit_loss

    def visit(self, p_ab):
        p_ab = torch.log(1e-8 +p_ab)
        visit_probability = p_ab.mean(dim=0).expand_as(p_ab)

        p_target_ab = torch.zeros_like(p_ab).fill_(1).div(p_ab.size(0))

        loss = (- p_target_ab * visit_probability).mean(0).sum()

        return loss

    def new_visit(self, p_ab, target, cross_modality):
        p_ab = torch.log(1e-8 +p_ab)
        visit_probability = p_ab.mean(dim=0).expand_as(p_ab)
        
        n1 = target[cross_modality].size(0)
        n2 = target[~cross_modality].size(0)
        #p_target_ab = target[cross_modality].expand(n1,n1).eq(target[~cross_modality].expand(n2,n2))
     
        a = target[cross_modality].repeat(1,n2).reshape_as(p_ab)

        b = target[~cross_modality].repeat(1,n1).reshape_as(p_ab)

        p_target_ab = a.eq(b)

        p_target_ab = p_target_ab.float()/(torch.sum(p_target_ab, dim=1).float().unsqueeze(0).expand(n2,n1).t())
        loss = (- p_target_ab * visit_probability).mean(0).sum()
        return loss

    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x

################################################
# Orthogonal Loss
################################################

class Zero_OrthogonalLoss(nn.Module):
    def __init__(self):
        super(Zero_OrthogonalLoss, self).__init__()

    def forward(self, x, y, labels, sub):
        assert x.size() == y.size()
        # change to > 0
        x = torch.mul(x, x).pow(0.5)
        y = torch.mul(y, y).pow(0.5)
        loss = torch.bmm(x.unsqueeze(1), y.unsqueeze(-1))
        loss = loss.view(loss.size(0))
        loss = torch.mean(loss) / x.size(1)

        return loss

class Instance_OrthogonalLoss(nn.Module):
    def __init__(self):
        super(Instance_OrthogonalLoss, self).__init__()

    def forward(self, x, y, labels, sub):
        assert x.size() == y.size()
        loss = torch.bmm(x.unsqueeze(1), y.unsqueeze(-1))
        loss = loss.view(loss.size(0))
        loss = torch.mean(loss) / x.size(1)

        return loss

class Batch_OrthogonalLoss(nn.Module):
    def __init__(self):
        super(Batch_OrthogonalLoss, self).__init__()

    def forward(self, x, y, labels, sub):
        assert x.size() == y.size()
        loss = torch.mm(x, y.permute(1,0))
        targets = torch.zeros_like(loss)
        loss = F.smooth_l1_loss(loss, targets)
        loss = loss / x.size(1)

        return loss

class Contrastive_OrthogonalLoss(nn.Module):
    def __init__(self):
        super(Contrastive_OrthogonalLoss, self).__init__()

    def forward(self, x, y, labels, sub):
        assert x.size() == y.size()
        n = labels.size(0)

        task_loss = torch.abs(torch.mm(x, y.permute(1,0)))
        task_loss = torch.mean(task_loss) / x.size(1)

        # 3*mask + ~mask : [pos: 1, neg: -1]
        # 2*mask + ~mask : [pos: 0, neg: -1]
        id_mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(torch.int64)
        id_mask = 2*id_mask + ~id_mask # - torch.eye(n).to(torch.int64).cuda()
        mod_mask = sub.expand(n, n).eq(sub.expand(n, n).t()).to(torch.int64)
        mod_mask = 2*mod_mask + ~mod_mask # - torch.eye(n).to(torch.int64).cuda()

        id_loss = -id_mask * torch.abs(torch.mm(x, x.permute(1,0)))
        id_loss = torch.mean(id_loss) / x.size(1)
        mod_loss = -mod_mask * torch.abs(torch.mm(y, y.permute(1,0)))
        mod_loss = torch.mean(mod_loss) / x.size(1)

        loss = task_loss + id_loss + mod_loss

        return loss

################################################
# Similarity Loss
################################################

class SimiLoss(nn.Module):
    def __init__(self, batch_size):
        super(SimiLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mask = torch.eye(batch_size).to(device)
        self.CE_loss = nn.CrossEntropyLoss(reduce=False).to(device)
        self.bs = batch_size

    def forward(self, x):
        loss_d = self.CE_loss(x, self.mask.view(-1, 1).squeeze(1).to(torch.int64)).view(self.bs, self.bs)

        dist_ap, dist_an = [], []
        for i in range(self.bs):
            dist_ap.append(loss_d[i,i].unsqueeze(0))
            dist_an.append(loss_d[i][self.mask[i] == 0].max().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        loss = torch.mean(dist_ap) + torch.mean(dist_an)

        return loss