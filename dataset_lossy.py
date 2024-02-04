import numpy as np
import open3d
import torch
import torch.utils.data as data
from os.path import join
import os
import MinkowskiEngine as ME
import random


class Dataset(data.Dataset):
    def __init__(self, root_dir, split, bit=10, maximum=20475, type='train', scaling_factor=1, time_step=1, format='npy'):
        self.maximum = maximum
        self.type = type
        self.scaling_factor = scaling_factor
        self.format = format
        sequence_list = ['soldier', 'redandblack', 'loot', 'longdress', 'andrew', 'basketballplayer', 'dancer', 'david', 'exercise', 'phil', 'queen', 'ricardo', 'sarah', 'model']
        self.sequence_list = sequence_list
        start = [536, 1450, 1000, 1051, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1]
        end = [835, 1749, 1299, 1350, 317, 600, 600, 215, 600, 244, 249, 215, 206, 600]
        num = [end[i] - start[i] for i in range(len(start))]
        self.lookup = []
        for i in split:
            sequence_dir = join(root_dir, sequence_list[i]+'_ori')
            # sequence_dir = join(root_dir, sequence_list[i])
            file_prefix = sequence_list[i]+'_vox'+str(bit)+'_'
            file_subfix = '.'+self.format
            if type == 'train':
                s = start[i]
                e = int((end[i]-start[i])*0.95+start[i])
            elif type == 'val':
                s = int((end[i]-start[i])*0.95 +start[i])
                e = end[i]-time_step+1
            else:
                s = start[i]
                e = end[i]
            for s in range(s, e):
                s1 = str(s+time_step).zfill(4)
                s0 = str(s).zfill(4)
                file_name0 = file_prefix + s0 + file_subfix
                file_name1 = file_prefix + s1 + file_subfix
                file_dir = join(sequence_dir, file_name0)
                file_dir1 = join(sequence_dir, file_name1)
                self.lookup.append([file_dir, file_dir1])

    def __getitem__(self, item):
        file_dir, file_dir1 = self.lookup[item]
        if self.format == 'npy':
            p, p1 = np.load(file_dir), np.load(file_dir1)
        elif self.format == 'ply':
            p = np.asarray(open3d.io.read_point_cloud(file_dir).points)
            p1 = np.asarray(open3d.io.read_point_cloud(file_dir1).points)
        pc = torch.tensor(p[:, :3]).cuda()
        pc1 = torch.tensor(p1[:, :3]).cuda()

        if self.scaling_factor != 1:
            pc = torch.unique(torch.floor(pc / self.scaling_factor), dim=0)
            pc1 = torch.unique(torch.floor(pc1 / self.scaling_factor), dim=0)
        xyz, point = pc, torch.ones_like(pc[:, :1])
        xyz1, point1 = pc1, torch.ones_like(pc1[:, :1])

        return xyz, point, xyz1, point1

    def __len__(self):
        return len(self.lookup)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    # coords, feats, labels = list(zip(*list_data))
    xyz, point, xyz1, point1 = list(zip(*list_data))

    xyz_batch = ME.utils.batched_coordinates(xyz)
    point_batch = torch.vstack(point).float()
    xyz1_batch = ME.utils.batched_coordinates(xyz1)
    point1_batch = torch.vstack(point1).float()
    return xyz_batch, point_batch, xyz1_batch, point1_batch


