from __future__ import print_function
import argparse
from re import S
import sys
import time
from numpy.core.fromnumeric import mean, shape
import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import erase
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, accuracy, eval_sysu2
from model_main import embed_net
from utils import *
from loss import ChannelContrastive, OriTripletLoss, PixCycleContrastive, Rank_loss, RankedLoss, Circle_Rank_loss, ASS_loss, CenterTripletLoss, \
    Contrastive_OrthogonalLoss, Batch_OrthogonalLoss, SimiLoss, InfoNCE, PixContrastive, PixPro, CrossPixContrastive, SelfPixContrastive, pdist_cos_np, remask, CrossPixContrastiveL2
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import math
from PIL import Image
import pickle
from visualize import UnNormalize
from CAM import get_CAM, get_CAM_part
import pickle
import scipy.io as sio

from IPython import embed

parser = argparse.ArgumentParser(description='DenseReID')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=8, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log_ddag/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--grayscale', default=0.5, type=float,
                    metavar='margin', help='grayscale possibility only for training')
parser.add_argument('--random_erasing', default=0.5, type=float,
                    metavar='margin', help='random_erasing possibility only for training')
parser.add_argument('--part', default=6, type=int,
                    metavar='tb', help='part number')
parser.add_argument('--method', default='id+tri', type=str,
                    metavar='m', help='method type')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--ckpt', default='xxx', type=str, help='ckpt str')
parser.add_argument('--specific_layer', default=2, type=int, metavar='sl', help='number of specific layers (default: 2[res0-res1])')
parser.add_argument('--vis', default=False, action='store_true', help='stop and vis')
parser.add_argument('--tsne', default=False, action='store_true', help='stop and tsne')
parser.add_argument('--histogram', default=False, action='store_true', help='stop and histogram')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.autograd.set_detect_anomaly(True)

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    # TODO: define your data path for SYSU-MM01 dataset
    data_path = './data/sysu/'
    log_path = args.log_path + 'sysu_log_ddag/'
    test_mode = [1, 2] # infrared to visible
elif dataset =='regdb':
    # TODO: define your data path for RegDB dataset
    data_path = './data/regdb/'
    log_path = args.log_path + 'regdb_log_ddag/'
    # test_mode = [2, 1] # visible to infrared
    test_mode = [1, 2] # infrared to visible

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

# log file name
suffix = args.ckpt + '_' + dataset + '_2*{}*{}_lr_{}_{}'.format(args.num_pos, args.batch_size, args.lr, args.optim)
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim
if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

test_log_file = open(log_path + suffix + '.txt', "w")
# sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
print("==========\nArgs:{}\n==========".format(args), file=test_log_file)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
end = time.time()

print('==> Loading data..')
# Data loading code
transform_train_rgb = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Pad(10),
    # transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomGrayscale(p=args.grayscale),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=args.random_erasing),
])
transform_train_ir = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Pad(10),
    # transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=args.random_erasing),
])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Pad(10),
    # transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform_rgb=transform_train_rgb, transform_ir=transform_train_ir)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam, gall_path = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform_rgb=transform_train_rgb, transform_ir=transform_train_ir)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    if test_mode == [2, 1]:
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
    elif test_mode == [1, 2]:
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)) , ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


print('==> Building model..')
net = embed_net(args, n_class, part=args.part, arch=args.arch)

net.to(device)
cudnn.benchmark = True

loader_batch = args.batch_size * args.num_pos

# define loss function
criterion_CE = nn.CrossEntropyLoss(ignore_index=-1).to(device)
criterion_Triplet = OriTripletLoss(batch_size=loader_batch, margin=args.margin).to(device)
criterion_CenterTriplet = CenterTripletLoss(k_size=args.batch_size, margin=0.7).to(device)
criterion_CRank = Circle_Rank_loss().to(device) #Rank_loss()
criterion_ASS = ASS_loss().to(device)
criterion_Rank = Rank_loss().to(device)
criterion_Contrastive = InfoNCE().to(device)
criterion_PixContrastive = PixContrastive().to(device)
criterion_PixPro = PixPro().to(device)
criterion_PixCycleContrastive = PixCycleContrastive().to(device)
criterion_ChannelContrastive = ChannelContrastive().to(device)
criterion_SelfPixContrastive = SelfPixContrastive().to(device)
criterion_CrossPixContrastive = CrossPixContrastive().to(device)
criterion_CrossPixContrastiveL2 = CrossPixContrastiveL2().to(device)
criterion_L1 = nn.L1Loss(size_average=True).to(device)
criterion_L2 = nn.MSELoss().to(device)
criterion_BCE = nn.BCELoss().to(device)

# load checkpoint
if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'], strict=False)
        print('==> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# optimizer
if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                   + list(map(id, net.classifier.parameters())) \
                   + list(map(id, net.part_bottleneck.parameters())) \
                   + list(map(id, net.part_classifier.parameters())) \

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_P = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        {'params': net.part_bottleneck.parameters(), 'lr': args.lr},
        {'params': net.part_classifier.parameters(), 'lr': args.lr},
        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    
elif args.optim == 'adam':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                   + list(map(id, net.classifier.parameters())) \
                   + list(map(id, net.part_bottleneck.parameters())) \
                   + list(map(id, net.part_classifier.parameters())) \

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_P = optim.Adam([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        {'params': net.part_bottleneck.parameters(), 'lr': args.lr},
        {'params': net.part_classifier.parameters(), 'lr': args.lr},
        ],
        weight_decay=5e-4)


def adjust_learning_rate(optimizer_P, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if epoch < 5:
        lr = args.lr * (epoch + 1) / 5
    elif 5 <= epoch < 15:
        lr = args.lr
    elif 15 <= epoch < 27:
        lr = args.lr * 0.1
    elif epoch >= 27:
        lr = args.lr * 0.01

    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    return lr


def train(epoch):

    # adjust learning rate
    current_lr = adjust_learning_rate(optimizer_P, epoch)

    total_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    cc_loss = AverageMeter()
    rank_loss = AverageMeter()
    crank_loss = AverageMeter()
    ass_loss = AverageMeter()
    self_pcon_loss = AverageMeter()
    cross_pcon_loss = AverageMeter()
    part_loss = AverageMeter()
    pixel_loss = AverageMeter()
    continuous_loss = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    loss_total = torch.Tensor([0])
    loss_id = torch.Tensor([0])
    loss_tri = torch.Tensor([0])
    loss_cc = torch.Tensor([0])
    loss_rank = torch.Tensor([0])
    loss_crank = torch.Tensor([0])
    loss_ass = torch.Tensor([0])
    loss_self_pcon = torch.Tensor([0])
    loss_cross_pcon = torch.Tensor([0])
    loss_part = torch.Tensor([0])
    loss_pixel = torch.Tensor([0])
    loss_continuous = torch.Tensor([0])

    # switch to train mode
    net.train()
    end = time.time()
    a = trainset[0]

    for batch_idx, (img_rgb, img_ir, target_rgb, target_ir) in enumerate(trainloader):

        imgs = torch.cat((img_rgb, img_ir), 0)
        labels = torch.cat((target_rgb, target_ir), 0)

        img_rgb = Variable(img_rgb.cuda())
        img_ir = Variable(img_ir.cuda())
        labels = Variable(labels.cuda())

        data_time.update(time.time() - end)

        # modality label
        sub1 = torch.zeros(img_rgb.size(0))
        sub2 = torch.ones(img_ir.size(0))
        sub =  Variable(torch.cat((sub1, sub2), 0).cuda())

        # feat, out0, mod_feat, mod, cam, d_score = net(img_rgb, img_ir, adj_norm)
        feat, feat_logit, pfeat, pfeat_logit, low_level_map, high_level_map, pixel_logit = net(img_rgb, img_ir, mode=0)
        map_rgb, map_ir = low_level_map
        low_map = torch.cat((map_rgb, map_ir), 0)

        bs, c, h, w = low_map.size()
        sbs = bs // 2

        # part mask inference & ignore
        # part_mask = torch.max(pixel_logit[:, :2], axis=1)[1] # 2cls
        part_mask = torch.max(pixel_logit[:, :2], axis=1)[1] * (torch.max(pixel_logit[:, 2:], axis=1)[1] + 1) # 8cls

        part_mask = part_mask.reshape(bs, h, w)

        not_erase_mask = ((imgs[:, 0, :, :] != 0) | (imgs[:, 1, :, :] != 0) | (imgs[:, 2, :, :] != 0)).to(torch.int64).to(device)
        not_erase_mask = F.interpolate(not_erase_mask.unsqueeze(1).to(torch.float32), size=(h, w), mode='nearest').squeeze(1).to(torch.int64)

        cam_mask = get_CAM(net, high_level_map, feat_logit, (h, w))
        cam_part_mask = get_CAM_part(net, high_level_map, feat_logit, (h, w))
        
        cam_mask *= not_erase_mask
        part_mask *= not_erase_mask
        cam_part_mask *= not_erase_mask

        ########################### vis
        if args.vis:
            un = UnNormalize()
            ori = torch.cat((img_rgb, img_ir), 0)
            for i in range(bs):
                img=un(ori[i]) * 256
                img=img.clamp(min=0, max=255)
                img=img.permute((1, 2, 0)).to(torch.int8).cpu().numpy().astype(np.uint8)
                img=Image.fromarray(img)
                img.save("vis/ori/{}.png".format(i))
                img.close()

            for i in range(bs):
                img=Image.fromarray(cam_mask[i].cpu().numpy().astype(np.int32))
                img.save("vis/mask/{}.png".format(i))
                img.close()

            for i in range(bs):
                img=Image.fromarray(part_mask[i].cpu().numpy().astype(np.int32))
                img.save("vis/part/{}.png".format(i))
                img.close()

            for i in range(bs):
                img=Image.fromarray(cam_part_mask[i].cpu().numpy().astype(np.int32))
                img.save("vis/cam/{}.png".format(i))
                img.close()

            for i in range(bs):
                img=Image.fromarray(not_erase_mask[i].cpu().numpy().astype(np.int32))
                img.save("vis/erase/{}.png".format(i))
                img.close()

        ########################### vis

        # baseline loss: identity loss + triplet loss Eq. (1)　
        loss_id = criterion_CE(feat_logit, labels)
        loss_tri, batch_acc = criterion_Triplet(feat, labels)
        loss_cc, _, _ = criterion_CenterTriplet(feat, labels)
        loss_id_part = criterion_CE(pfeat_logit, labels)

        loss_part = loss_id_part

        correct += (batch_acc / 2)
        _, predicted = feat_logit.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        # Semantic Rectification Module
        # pixel_label = torch.zeros_like(part_mask).to(torch.int64)
        # for i in range(args.part):
        #     st, fi = i * h // args.part, (i+1) * h // args.part
        #     pixel_label[:, st:fi, :] = i+1
        # pixel_label = Variable(pixel_label.cuda())  # Idol Mask

        # 2cls, baseline_part_refine
        # mask_rgb, mask_ir = remask(map_rgb, cam_part_mask[:sbs], args.part, mask=part_mask[:sbs]), remask(map_ir, cam_part_mask[sbs:], args.part, mask=part_mask[sbs:])

        # 8cls remask
        # mask_rgb, mask_ir = remask(map_rgb, part_mask[:sbs], args.part, mask=part_mask[:sbs]), remask(map_ir, part_mask[sbs:], args.part, mask=part_mask[sbs:])

        # 8cls not remask best
        mask_rgb, mask_ir = part_mask[:loader_batch], part_mask[loader_batch:]

        # baseline_global
        # mask_rgb, mask_ir = cam_mask[:loader_batch], cam_mask[loader_batch:]

        # baseline_global_refine
        # mask_rgb, mask_ir = remask(map_rgb, cam_mask[:sbs], args.part, mask=part_mask[:sbs]), remask(map_ir, cam_mask[sbs:], args.part, mask=part_mask[sbs:])

        # baseline_part
        # mask_rgb, mask_ir = cam_part_mask[:loader_batch], cam_part_mask[loader_batch:]

        print("Used pixel: {}".format((torch.sum(mask_rgb != 0) + torch.sum(mask_ir != 0)) / (bs * h * w)), end="\r")
        

        # a = pixel_label[0].reshape(h, w)
        # for i in range(h):
        #     for j in range(w):
        #         print(a[i][j].item(), end="")
        #     print()

        ########################### vis
        mask_ot = torch.cat((mask_rgb, mask_ir), 0)
        if args.vis:
            for i in range(bs):
                img=Image.fromarray(mask_ot[i].cpu().numpy().astype(np.int32))
                img.save("vis/ot/{}.png".format(i))
                img.close()
            embed()
            os.system("python vis.py")
        ########################### vis

        ########################### T-SNE
        if args.tsne:
            dic = {
                "map_rgb": map_rgb.detach().cpu(),
                "map_ir": map_ir.detach().cpu(),
                "id": target_rgb.detach().cpu(),
                "mask_rgb": mask_rgb.detach().cpu(),
                "mask_ir": mask_ir.detach().cpu()
            }
            f = open('figs/tsne_method.pkl','wb')
            pickle.dump(dic, f)
            f.close()
            embed()
        ########################### 

        loss_cross_pcon = criterion_CrossPixContrastive(map_rgb, map_ir, mask_rgb, mask_ir)
        # loss_cross_pcon = criterion_CrossPixContrastiveL2(map_rgb, map_ir, mask_rgb, mask_ir)
        # loss_cross_pcon = criterion_PixPro(map_rgb, map_ir, mask_rgb, mask_ir)

        # foreground and background classification using res2
        # bg = ((mask_ot == 0) & (cam_part_mask == 0)).view(-1)
        # fg = ((mask_ot != 0) & (cam_part_mask != 0)).view(-1)
        # pixel_label = torch.zeros_like(cam_part_mask.reshape(-1).to(torch.int64)) - 1
        # pixel_label[bg] = 0
        # pixel_label[fg] = 1

        # foreground and background classification using res2 class-activate-map
        # mask_rgb, mask_ir = remask(map_rgb, cam_part_mask[:loader_batch], args.part), remask(map_ir, cam_part_mask[loader_batch:], args.part)
        # mask_ot = torch.cat((mask_rgb, mask_ir), 0)
        # mask_ot[erase_mask == 1] = -1
        # pixel_label_2 = (cam_mask != 0).to(torch.int64).view(-1) # baseline_global_refine
        pixel_label_2 = (cam_part_mask != 0).to(torch.int64).view(-1) # baseline_part_refine
        pixel_label_7 = cam_part_mask.to(torch.int64).view(-1)
        pixel_label_6 = pixel_label_7 - 1

        # loss_pixel = criterion_CE(pixel_logit[:, :], pixel_label_2) # 2cls
        loss_pixel = criterion_CE(pixel_logit[:, :2], pixel_label_2) + criterion_CE(pixel_logit[:, 2:], pixel_label_6) # 8cls

        # continuous loss
        pixel_map = pixel_logit.reshape(bs, h, w, 8) # 2cls 8cls
        h_continuous = pixel_map[:, 1:, :, :] - pixel_map[:, :-1, :, :]
        w_continuous = pixel_map[:, :, 1:, :] - pixel_map[:, :, :-1, :]
        loss_continuous = (criterion_L1(h_continuous, torch.zeros_like(h_continuous)) + criterion_L1(w_continuous, torch.zeros_like(w_continuous))) * 0.5

        
        
        # loss = loss_id + loss_rank + 0.2*loss_ass
        # loss = loss_id + loss_tri
        loss = loss_id + loss_cc
        
        
        # loss += loss_self_pcon

        # loss += loss_cross_pcon
        # loss += loss_part
        # loss += loss_pixel
        # loss += loss_continuous

        # Overall loss
        loss_total = loss

        # optimization of encoder
        optimizer_P.zero_grad()
        loss_total.backward()
        optimizer_P.step()

        # log different loss components
        total_loss.update(loss_total.item(), 2 * img_rgb.size(0))
        id_loss.update(loss_id.item(), 2 * img_rgb.size(0))
        tri_loss.update(loss_tri.item(), 2 * img_rgb.size(0))
        rank_loss.update(loss_rank.item(), 2 * img_rgb.size(0))
        crank_loss.update(loss_crank.item(), 2 * img_rgb.size(0))
        ass_loss.update(loss_ass.item(), 2 * img_rgb.size(0))
        cc_loss.update(loss_cc.item(), 2 * img_rgb.size(0))
        self_pcon_loss.update(loss_self_pcon.item(), 2 * img_rgb.size(0))
        cross_pcon_loss.update(loss_cross_pcon.item(), 2 * img_rgb.size(0))
        part_loss.update(loss_part.item(), 2 * img_rgb.size(0))
        pixel_loss.update(loss_pixel.item(), 2 * img_rgb.size(0))
        continuous_loss.update(loss_continuous.item(), 2 * img_rgb.size(0))
        total += labels.size(0)

        id_acc = accuracy(feat_logit, labels)[0][0]
        tri_acc = torch.Tensor([100. * correct / total])[0].to(device)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} '# ({batch_time.avg:.3f}) '
                  'lr: {:.6f} ' 
                  'TotalLoss: {total_loss.val:.4f} '# ({total_loss.avg:.4f}) '
                  'IDLoss: {id_loss.val:.4f} '# ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} '# ({tri_loss.avg:.4f}) '
                #   'RankLoss: {rank_loss.val:.4f} '# ({rank_loss.avg:.4f}) '
                #   'AssLoss: {ass_loss.val:.4f} '# ({ass_loss.avg:.4f}) '
                #   'CRankLoss: {crank_loss.val:.4f} '# ({crank_loss.avg:.4f}) '
                #   'CCLoss: {cc_loss.val:.4f} '# ({cc_loss.avg:.4f}) '
                #   'CPConLoss: {cross_pcon_loss.val:.4f} '# ({cross_pcon_loss.avg:.4f}) '
                #   'SPConLoss: {self_pcon_loss.val:.4f} '# ({self_pcon_loss.avg:.4f}) '
                #   'PartLoss: {part_loss.val:.4f} '# ({part_loss.avg:.4f}) '
                #   'PixelLoss: {pixel_loss.val:.4f} '# ({pixel_loss.avg:.4f}) '
                #   'ContinuousLoss: {continuous_loss.val:.4f} '# ({continuous_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                   epoch, batch_idx, len(trainloader), current_lr,
                   tri_acc, batch_time=batch_time,
                   total_loss=total_loss, 
                   id_loss=id_loss, 
                   tri_loss=tri_loss,
                   rank_loss=rank_loss, 
                   ass_loss=ass_loss, 
                   crank_loss = crank_loss, 
                   cc_loss = cc_loss, 
                   cross_pcon_loss=cross_pcon_loss,
                   self_pcon_loss=self_pcon_loss,
                   part_loss=part_loss,
                   pixel_loss=pixel_loss,
                   continuous_loss=continuous_loss,
                   ))

    writer.add_scalar('total_loss', total_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('rank_loss', rank_loss.avg, epoch)
    writer.add_scalar('ass_loss', ass_loss.avg, epoch)
    writer.add_scalar('crank_loss', crank_loss.avg, epoch)
    writer.add_scalar('cc_loss', cc_loss.avg, epoch)
    writer.add_scalar('cpcon_loss', cross_pcon_loss.avg, epoch)
    writer.add_scalar('spcon_loss', self_pcon_loss.avg, epoch)
    writer.add_scalar('part_loss', part_loss.avg, epoch)
    writer.add_scalar('pixel_loss', pixel_loss.avg, epoch)
    writer.add_scalar('continuous_loss', continuous_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)

def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_high = np.zeros((ngall, 2048))
    gall_feat_low = np.zeros((ngall, 256))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, pixel_mask, feat_low, feat_high = net(input, input, mode=test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_low[ptr:ptr + batch_num, :] = feat_low.detach().cpu().numpy()
            gall_feat_high[ptr:ptr + batch_num, :] = feat_high.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_high = np.zeros((nquery, 2048))
    query_feat_low = np.zeros((nquery, 256))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, pixel_mask, feat_low, feat_high = net(input, input, mode=test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_low[ptr:ptr + batch_num, :] = feat_low.detach().cpu().numpy()
            query_feat_high[ptr:ptr + batch_num, :] = feat_high.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the pairwise similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    # distmat_low = np.matmul(query_feat_low, np.transpose(gall_feat_low))
    # compute the cosine similarity
    # distmat = pdist_cos_np(query_feat, gall_feat)

    if args.histogram:
        dic = {
            "query_feat_high": torch.Tensor(query_feat),
            "gall_feat_high": torch.Tensor(gall_feat),
            "query_feat_low": torch.Tensor(query_feat_low),
            "gall_feat_low": torch.Tensor(gall_feat_low),
            "query_label": torch.Tensor(query_label),
            "gall_label": torch.Tensor(gall_label),
        }
        f = open('figs/histogram_method.pkl','wb')
        pickle.dump(dic, f)
        f.close()
        embed()

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        query_feat, gall_feat = torch.Tensor(query_feat), torch.Tensor(gall_feat)
        # perm = sio.loadmat(os.path.join(data_path, 'exp', 'rand_perm_cam.mat'))['rand_perm_cam']
        # eval_sysu2(query_feat, query_label, query_cam, gall_feat, gall_label, gall_cam, gall_path, perm, mode=args.mode, num_shots=1, num_trials=1, rerank=False)
        # eval_sysu2(query_feat, query_label, query_cam, gall_feat, gall_label, gall_cam, gall_path, perm, mode=args.mode, num_shots=10, num_trials=1, rerank=False)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    return cmc, mAP, mINP


# training
print('==> Start Training...')
for epoch in range(start_epoch, 150):
    print('==> Preparing Data Loader...')
    # identity sampler: 
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # infrared index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    if not args.test_only:
        train(epoch)

    if epoch >= 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)

        # testing
        cmc, mAP, mINP = test(epoch)
        # log output
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP), file=test_log_file)

        test_log_file.flush()

        # save model
        if cmc[0] > best_acc and not args.test_only:  # not the real best for sysu-mm01
            best_acc = cmc[0]
            best_epoch = epoch
            best_mAP = mAP
            best_mINP = mINP
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'mINP': mINP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        if not args.test_only:
            print('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, best_acc, best_mAP, best_mINP))
            print('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, best_acc, best_mAP, best_mINP), file=test_log_file)

    if args.test_only:
        break
