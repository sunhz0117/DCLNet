from math import e
import torch
from torch.nn import functional as F

from IPython import embed

def get_CAM(net, feat_map, fc, sz, top_k=2):
    feat_map = feat_map.detach()
    fc = fc.detach()
    logits = F.softmax(fc, dim=1)
    scores, pred_labels = torch.topk(logits, k=top_k, dim=1)
    pred_labels = pred_labels[:, 0]
    bs, c, h, w = feat_map.size()

    # normalize
    we = net.classifier.weight[pred_labels,:].detach().unsqueeze(1)
    fe = feat_map.view(bs,c,h*w)
    part = 6

    cam = torch.bmm(we, fe).squeeze(1)
    cam = (cam - torch.min(cam, axis=1)[0].unsqueeze(1).repeat((1, h*w))) / (torch.max(cam, axis=1)[0] - torch.min(cam, axis=1)[0]).unsqueeze(1).repeat((1, h*w))
    cam = cam.view(bs,1,h,w)
    cam = F.interpolate(cam, size=(sz[0], sz[1]), mode='bilinear', align_corners=False).squeeze(1)

    weight = torch.zeros_like(cam).to(torch.int32)
    conf_mean = torch.mean(cam.view(bs, -1), axis=1).unsqueeze(1).unsqueeze(2).repeat((1, sz[0], sz[1]))
    weight = (cam > conf_mean).to(torch.int32)
    for i in range(part):
        weight[:,int(i * (sz[0] // part)):int((i+1) * (sz[0] // part)),:] *= (i+1)
    
    return weight

def get_CAM_part(net, feat_map, fc, sz, top_k=2):
    feat_map = feat_map.detach()
    fc = fc.detach()
    logits = F.softmax(fc, dim=1)
    scores, pred_labels = torch.topk(logits, k=top_k, dim=1)
    pred_labels = pred_labels[:, 0]
    bs, c, h, w = feat_map.size()

    # normalize
    we = net.part_classifier.weight[pred_labels, :].detach().unsqueeze(1)
    fe = feat_map.view(bs,c,h*w)
    part = 6
        
    camp = []
    for i in range(part):
        cam = torch.bmm(we[:, :, int(i * c): int((i+1) * c)], fe[:, :, int(i * h * w // part): int((i+1) * h * w // part)]).squeeze(1)
        cam = (cam - torch.min(cam, axis=1)[0].unsqueeze(1).repeat((1, h * w // part))) / (torch.max(cam, axis=1)[0] - torch.min(cam, axis=1)[0]).unsqueeze(1).repeat((1, h * w // part))
        cam = cam.view(bs,1,h//part,w)
        cam = F.interpolate(cam, size=(sz[0] // part, sz[1]), mode='bilinear', align_corners=False).squeeze(1)
        camp.append(cam)
    camp = torch.cat(camp, 1)

    weight = torch.zeros_like(camp).to(torch.int32)
    for i in range(part):
        conf_mean = torch.mean(camp[:,int(i * (sz[0] // part)):int((i+1) * (sz[0] // part)),:].view(bs, -1), axis=1).reshape(bs, 1, 1).repeat(1, sz[0] // part, sz[1])
        fg_mask = (camp[:,int(i * (sz[0] // part)):int((i+1) * (sz[0] // part)),:] > conf_mean).to(torch.int32)
        weight[:,int(i * (sz[0] // part)):int((i+1) * (sz[0] // part)),:] = fg_mask * (i+1)

    return weight
