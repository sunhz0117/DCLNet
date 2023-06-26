from __future__ import print_function, absolute_import
import os
import sys
import numpy as np
import random

from IPython import embed

def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

# class SYSUDataset(Dataset):
#     def __init__(self, root, mode='train', transform=None):
#         assert os.path.isdir(root)
#         assert mode in ['train', 'gallery', 'query']

#         if mode == 'train':
#             train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
#             val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()

#             train_ids = train_ids.strip('\n').split(',')
#             val_ids = val_ids.strip('\n').split(',')
#             selected_ids = train_ids + val_ids
#         else:
#             test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
#             selected_ids = test_ids.strip('\n').split(',')

#         selected_ids = [int(i) for i in selected_ids]
#         num_ids = len(selected_ids)

#         img_paths = glob(os.path.join(root, 'cam*/*/*.jpg'), recursive=True)
#         img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

#         if mode == 'gallery':
#             img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
#         elif mode == 'query':
#             img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]

#         img_paths = sorted(img_paths)
#         self.img_paths = img_paths
#         self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
#         self.num_ids = num_ids
#         self.transform = transform

#         if mode == 'train':
#             id_map = dict(zip(selected_ids, range(num_ids)))
#             self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
#         else:
#             self.ids = [int(path.split('/')[-2]) for path in img_paths]

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, item):
#         path = self.img_paths[item]
#         img = Image.open(path)
#         if self.transform is not None:
#             img = self.transform(img)

#         label = torch.tensor(self.ids[item], dtype=torch.long)
#         cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
#         item = torch.tensor(item, dtype=torch.long)

#         return img, label, cam, path, item


def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
                # files_rgb += new_files

    files_rgb = sorted(files_rgb)

    gall_img = []
    gall_id = []
    gall_cam = []
    gall_path = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
        gall_path.append(img_path)
    return gall_img, np.array(gall_id), np.array(gall_cam), gall_path
    
    
def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)