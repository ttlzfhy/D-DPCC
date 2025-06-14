import numpy as np
import open3d
import torch
import torch.utils.data as data
from os.path import join
import os
import MinkowskiEngine as ME
import random
import glob


class StaticDataset(data.Dataset):
    def __init__(self, filedirs, type='train', scaling_factor=1, augment=False, scaling_type='floor'):
        filedirs = sorted(filedirs)
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type
        self.augment = augment
        self.format = os.path.basename(filedirs[0]).split('.')[-1]
        self.lookup = filedirs

    def __getitem__(self, item):
        file_dir = self.lookup[item]
        if self.format == 'npy':
            p = np.load(file_dir)
        elif self.format == 'ply':
            p = np.asarray(open3d.io.read_point_cloud(file_dir).points)
        else:
            raise ValueError(f"Unsupported file format: {self.format}")
        pc = torch.tensor(p[:, :3]).cuda()
        if self.scaling_factor != 1:
            if self.scaling_type == 'floor':
                pc = torch.unique(torch.floor(pc / self.scaling_factor), dim=0)
            elif self.scaling_type == 'round':
                pc = torch.unique(torch.round(pc / self.scaling_factor), dim=0)
            else:
                print('Please choose the scaling_type between floor, round!!!!!')
                assert False

        if self.augment:
            random_factor = random.uniform(0.5, 1)
            # print(random_factor)
            pc = torch.unique(torch.round(pc * random_factor), dim=0)

        xyz, point = pc, torch.ones_like(pc[:, :1])

        return xyz, point

    def __len__(self):
        return len(self.lookup)

def static_collate_pointcloud_fn(list_data):
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
    xyz, point = list(zip(*list_data))

    xyz_batch = ME.utils.batched_coordinates(xyz)
    point_batch = torch.vstack(point).float()
    return xyz_batch, point_batch


def get_seqs_list(filedirs):
    filedirs = sorted(filedirs)
    seqs_list = []

    for idx in range(len(filedirs)):
        if idx == 0: continue
        curr_frame = filedirs[idx]
        ref_frame = filedirs[idx - 1]

        Idx0 = os.path.split(curr_frame)[-1].split('.')[0].split('_')[-1]
        Idx1 = os.path.split(ref_frame)[-1].split('.')[0].split('_')[-1]
        try:
            Idx0 = int(Idx0)
            Idx1 = int(Idx1)
        except (ValueError) as e:
            Idx0 = int(str(Idx0).split('-')[-1])
            Idx1 = int(str(Idx1).split('-')[-1])

        if (Idx0 - Idx1 != 1 or
                os.path.basename(curr_frame).split('_')[0] != os.path.basename(ref_frame).split('_')[0]): continue
        seqs_list.append([ref_frame, curr_frame])

    return seqs_list


class DynamicDataset(data.Dataset):
    def __init__(self, seqs_list, train=True, scaling_factor=1, augment=False, scaling_type='floor'):
        self.train = train
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type
        self.augment = augment
        self.format = os.path.basename(seqs_list[0][0]).split('.')[-1]
        self.lookup = seqs_list

    def __getitem__(self, item):
        file_dir, file_dir1 = self.lookup[item]
        if self.format == 'npy':
            p, p1 = np.load(file_dir), np.load(file_dir1)
        elif self.format == 'ply':
            p = np.asarray(open3d.io.read_point_cloud(file_dir).points)
            p1 = np.asarray(open3d.io.read_point_cloud(file_dir1).points)
            # p = np.concatenate([np.asarray(open3d.io.read_point_cloud(file_dir).points), np.asarray(open3d.io.read_point_cloud(file_dir).colors)], axis=1)
            # p1 = np.concatenate([np.asarray(open3d.io.read_point_cloud(file_dir1).points), np.asarray(open3d.io.read_point_cloud(file_dir1).colors)], axis=1)
        else:
            raise ValueError(f"Unsupported file format: {self.format}")
        
        pc = torch.tensor(p[:, :3]).cuda()
        pc1 = torch.tensor(p1[:, :3]).cuda()            
        
        # print(pc.size(), pc1.size())
        if self.scaling_factor != 1:
            if self.scaling_type == 'floor':
                pc = torch.unique(torch.floor(pc / self.scaling_factor), dim=0)
                pc1 = torch.unique(torch.floor(pc1 / self.scaling_factor), dim=0)
            elif self.scaling_type == 'round':
                pc = torch.unique(torch.round(pc / self.scaling_factor), dim=0)
                pc1 = torch.unique(torch.round(pc1 / self.scaling_factor), dim=0)
            else:
                print('Please choose the scaling_type between floor, round!!!!!')
                assert False

        if self.augment:
            random_factor = random.uniform(0.5, 1)
            # print(random_factor)
            pc = torch.unique(torch.round(pc * random_factor), dim=0)
            pc1 = torch.unique(torch.round(pc1 * random_factor), dim=0)

        xyz, point = pc, torch.ones_like(pc[:, :1])
        xyz1, point1 = pc1, torch.ones_like(pc1[:, :1])

        return xyz, point, xyz1, point1

    def __len__(self):
        return len(self.lookup)


def dynamic_collate_pointcloud_fn(list_data):
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


def load_sparse_tensor(filedir, device, dtype='int32'):
    if filedir.endswith('.ply'):
        p = np.asarray(open3d.io.read_point_cloud(filedir).points).astype(dtype)
    elif filedir.endswith('.npy'):
        p = np.load(filedir).astype(dtype)
    else:
        raise ValueError(f"Unsupported file format: {filedir}")
    coords = torch.tensor(p[:, :3]).to(device)
    feats = torch.ones_like(coords[:, :1]).float()
    coords_, feats_ = ME.utils.sparse_collate([coords], [feats])
    sparse_tensor = ME.SparseTensor(feats_, coords_, device=device)
    return sparse_tensor





