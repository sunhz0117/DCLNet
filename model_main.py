from unicodedata import mirrored
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from resnet import resnet50, resnet18
import torch.nn.functional as F
import math
from attention import GraphAttentionLayer, IWPA

from IPython import embed

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class MLP2d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()

        self.linear1 = nn.Conv2d(in_dim, inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Conv2d(inner_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

class res50(nn.Module):
    def __init__(self, arch='resnet50', layer=[0,1,2,3,4]):
        super(res50, self).__init__()

        self.base = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        self.layer = layer

    def forward(self, x):
        x_map_bin = []
        if 0 in self.layer:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)
            x_map_bin.append(x.detach())
        if 1 in self.layer:
            x = self.base.layer1(x)
            x_map_bin.append(x.detach())
        if 2 in self.layer:
            x = self.base.layer2(x)
            x_map_bin.append(x.detach())
        if 3 in self.layer:
            x = self.base.layer3(x)
            x_map_bin.append(x.detach())
        if 4 in self.layer:
            x = self.base.layer4(x)
            x_map_bin.append(x.detach())

        return x, x_map_bin


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class gray_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(gray_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class base_resnet_share(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet_share, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        # x = self.base.layer4(x)
        return x


class id_resnet_specific(nn.Module):
    def __init__(self, arch='resnet50'):
        super(id_resnet_specific, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        # x = self.base.layer1(x)
        # x = self.base.layer2(x)
        # x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class mod_resnet_specific(nn.Module):
    def __init__(self, arch='resnet50'):
        super(mod_resnet_specific, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        # x = self.base.layer1(x)
        # x = self.base.layer2(x)
        # x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class cross_discriminator(nn.Module):
    def __init__(self, num_features, num_classes=2, norm=False):
        super(cross_discriminator, self).__init__()
        nc = num_features * 2
        self.norm = norm
        self.bn_mod = nn.BatchNorm1d(num_features)
        self.bn_id = nn.BatchNorm1d(num_features)
        self.classifier = nn.Sequential(
            nn.Linear(nc, num_features),
            nn.BatchNorm1d(num_features),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Linear(num_features, num_features // 2),
            nn.BatchNorm1d(num_features // 2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Linear(num_features // 2, num_classes),
        )

    def forward(self, id, mod):
        if self.norm:
            id = self.bn_id(id)
            mod = self.bn_mod(mod)
        input = torch.cat((id, mod), dim=1)
        output = self.classifier(input)

        return output


class adversarial_discriminator(nn.Module):
    def __init__(self, num_features, num_classes=2, norm=False):
        super(adversarial_discriminator, self).__init__()
        nc = num_features
        self.norm = norm
        self.bn = nn.BatchNorm1d(num_features)
        self.classifier = nn.Sequential(
            nn.Linear(nc, num_features),
            nn.BatchNorm1d(num_features),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Linear(num_features, num_features // 2),
            nn.BatchNorm1d(num_features // 2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Linear(num_features // 2, num_classes),
        )

    def forward(self, id):
        if self.norm:
            id = self.bn(id)
        output = self.classifier(id)

        return output


class embed_net(nn.Module):
    def __init__(self, args, class_num, part=0, arch='resnet50'):
        super(embed_net, self).__init__()

        self.args = args
        self.thermal_module = res50(arch=arch, layer=[*range(0, self.args.specific_layer)])
        self.visible_module = res50(arch=arch, layer=[*range(0, self.args.specific_layer)])
        self.shared_module = res50(arch=arch, layer=[*range(self.args.specific_layer, 5)])
        channel = [3, 64, 256, 512, 1024, 2048]
        self.low_dim = channel[self.args.specific_layer]
        self.pool_dim = channel[-1]
        self.pixel_dim = 64 + 256 + 512 + 1024 + 2048
        self.part = part
        self.l2norm = Normalize(2)

        self.gm_pool = True
        
        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)
        self.part_bottleneck = nn.BatchNorm1d(self.pool_dim * self.part)
        self.part_classifier = nn.Linear(self.pool_dim * self.part, class_num, bias=False)

        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.part_bottleneck.bias.requires_grad_(False)  # no shift
        self.part_bottleneck.apply(weights_init_kaiming)
        self.part_classifier.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # pixel classifier
        self.pixel_classifier_rgb = nn.Linear(self.pixel_dim, 8, bias=False) # 2cls 8cls
        self.pixel_classifier_rgb.apply(weights_init_classifier)
        self.pixel_classifier_ir = nn.Linear(self.pixel_dim, 8, bias=False) # 2cls 8cls
        self.pixel_classifier_ir.apply(weights_init_classifier)

    def xsample(self, x, sz):
        # upsample or downsample
        return F.interpolate(x, size=sz, mode='bilinear')

    def forward(self, img_rgb, img_ir, mode=0):
        # RGB/IR Specific Branch
        if mode == 0:
            x_rgb, x_rgb_bin = self.visible_module(img_rgb)
            x_ir, x_ir_bin = self.thermal_module(img_ir)
            x_mid = torch.cat((x_rgb, x_ir), 0)
            x_specific_bin = [torch.cat((rgbmap, irmap), 0) for rgbmap, irmap in zip(x_rgb_bin, x_ir_bin)]
        elif mode == 1:
            x_mid, x_specific_bin = self.visible_module(img_rgb)
        elif mode == 2:
            x_mid, x_specific_bin= self.thermal_module(img_ir)

        # Modality Shared Branch
        x_map, x_shared_bin = self.shared_module(x_mid)

        # pixel feature
        pixel_featmap_bin = x_specific_bin + x_shared_bin

        bs = x_map.size(0)
        sz = tuple(x_mid.shape[-2:])

        if mode == 0:
            pixel_feat_rgb = [self.xsample(m[:int(bs // 2)], sz)  for m in pixel_featmap_bin]
            pixel_feat_rgb = torch.cat(pixel_feat_rgb, 1)
            pixel_feat_ir = [self.xsample(m[int(bs // 2):], sz)  for m in pixel_featmap_bin]
            pixel_feat_ir = torch.cat(pixel_feat_ir, 1)
        elif mode == 1:
            pixel_feat_rgb = [self.xsample(m, sz)  for m in pixel_featmap_bin]
            pixel_feat_rgb = torch.cat(pixel_feat_rgb, 1)
        elif mode == 2:
            pixel_feat_ir = [self.xsample(m, sz)  for m in pixel_featmap_bin]
            pixel_feat_ir = torch.cat(pixel_feat_ir, 1)
        
        # Global Pooling
        if self.gm_pool:
            b, c, h, w = x_map.shape
            x_pool = x_map.view(b, c, -1)
            p = 3
            x_pool = (torch.mean(x_pool**p, dim=-1) + 1e-12) ** (1/p)
        else:
            x_pool = self.avgpool(x_map)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        feat = self.bottleneck(x_pool)

        # Part Pooling
        p_pool = []
        for i in range(self.part):
            b, c, h, w = x_map.shape
            pf = x_map[:, :, int(i * (h//self.part)): int((i+1) * (h//self.part)), :]
            if self.gm_pool:
                pf = pf.view(b, c, -1)
                p = 3
                pf = (torch.mean(pf**p, dim=-1) + 1e-12) ** (1/p)
            else:
                pf = self.avgpool(pf)
                pf = pf.view(pf.size(0), -1)
            p_pool.append(pf)
        p_pool = torch.cat(p_pool, 1)
        p_pool = p_pool.detach()  # considered
        pfeat = self.part_bottleneck(p_pool)

        if mode == 0:
            map_rgb = x_rgb
            map_ir = x_ir
            pixel_feat_rgb = pixel_feat_rgb.permute(0, 2, 3, 1).reshape(-1, self.pixel_dim).detach()
            pixel_feat_ir = pixel_feat_ir.permute(0, 2, 3, 1).reshape(-1, self.pixel_dim).detach()
            pixel_logit = torch.cat((self.pixel_classifier_rgb(pixel_feat_rgb), self.pixel_classifier_ir(pixel_feat_ir)), 0)
            low_level_map = (map_rgb, map_ir)
            high_level_map = x_map
        elif mode == 1:
            pixel_feat_rgb = pixel_feat_rgb.permute(0, 2, 3, 1).reshape(-1, self.pixel_dim)
            pixel_logit = self.pixel_classifier_rgb(pixel_feat_rgb)
            pixel_mask = torch.max(pixel_logit, axis=1)[1].reshape(bs, *sz)
        elif mode == 2:
            pixel_feat_ir = pixel_feat_ir.permute(0, 2, 3, 1).reshape(-1, self.pixel_dim)
            pixel_logit = self.pixel_classifier_ir(pixel_feat_ir)
            pixel_mask = torch.max(pixel_logit, axis=1)[1].reshape(bs, *sz)

        if self.training:
            return x_pool, self.classifier(feat), p_pool, self.part_classifier(pfeat), low_level_map, high_level_map, pixel_logit# , y_pool, self.mod_classifier(y_pool), self.cam_classifier(y_pool), cross_score #, adversarial_discriminator
        else:
            return self.l2norm(feat), pixel_mask, self.l2norm(self.avgpool(x_mid).view(x_mid.size(0), x_mid.size(1))), x_pool
