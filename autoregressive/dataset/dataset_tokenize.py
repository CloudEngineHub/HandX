import os
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


class VQMotionDataset(data.Dataset):  #dataset_name, feat_bias = 5, window_size = 64, unit_length = 8, motion_type=None, text_type=None, version=None
    def __init__(self, dataset_name, feat_bias = 5, window_size = 64, unit_length = 8, motion_type=None, text_type=None, version=None):
        # if motion_dim == 258:
        #     self.mean = np.load(f'{_DATA_DIR}/mean_duet_worot.npy')
        #     self.std = np.load(f'{_DATA_DIR}/std_duet_worot.npy')+1e-6
        #     self.std_tensor = torch.from_numpy(self.std)
        #     self.mean_tensor = torch.from_numpy(self.mean)
        #     self.data = dict(np.load(f'{_DATA_DIR}/{split}_full_duet_repre_worot.npz',allow_pickle=True))
        # elif motion_dim == 438:
        #     self.mean = np.load(f'{_DATA_DIR}/mean_duet.npy')
        #     self.std = np.load(f'{_DATA_DIR}/std_duet.npy')+1e-6
        #     self.std_tensor = torch.from_numpy(self.std)
        #     self.mean_tensor = torch.from_numpy(self.mean)
        #     self.data = dict(np.load(f'{_DATA_DIR}/{split}_full_duet_repre.npz',allow_pickle=True))
        # elif motion_dim == 288:
        self.mean = np.load(f'{_DATA_DIR}/mean_correct_duet_scalar_rot.npy')
        self.std = np.load(f'{_DATA_DIR}/std_correct_duet_scalar_rot.npy')+1e-6
        self.std_tensor = torch.from_numpy(self.std)
        self.mean_tensor = torch.from_numpy(self.mean)
        self.data_train = dict(np.load(f'{_DATA_DIR}/train_full_correct_duet_scalar_rot.npz',allow_pickle=True))
        
        self.data_test = dict(np.load(f'{_DATA_DIR}/test_full_correct_duet_scalar_rot.npz',allow_pickle=True))
        # self.data = dict(np.load(f'{_DATA_DIR}/{split}_full_duet_repre_worot.npz',allow_pickle=True))
        self.id_list_train = list(self.data_train.keys())
        self.id_list_test = list(self.data_test.keys())
        self.id_list = self.id_list_test+ self.id_list_train
        self.user = [0] * len(self.id_list_test) + [1] * len(self.id_list_train) 
        # if debug:
        #     self.id_list = id_list[:400]
        self.window_size = window_size
        self.unit_length = unit_length
        
        # self.indices = self._create_indices()
    def _create_indices(self):
        # Create a list of tuples (sample_index, time_index[, hand_idx])
        indices = []
        # if self.mode in ['left', 'right', 'split']:
        #     for i, _ in enumerate(self.keys):
        #         for start_idx in range(0, 60 - self.opt.window_size + 1, self.opt.window_stride):
        #             indices.append((i, start_idx))
        # elif self.mode == 'joint':
        
        for i, _ in enumerate(self.id_list):
            for start_idx in range(0, 60 - self.window_size + 1, 3):
                indices.append((i, start_idx))
                # indices.append((i, 1, start_idx))
        return indices
        # self.dataset_name = dataset_name
        # self.motion_type = motion_type
        # self.text_type = text_type
        # self.version = version

        # if dataset_name == 't2m':
        #     self.data_root = './dataset/HumanML3D'
        #     self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        #     self.text_dir = pjoin(self.data_root, 'texts')
        #     self.joints_num = 22
        #     self.max_motion_length = 196
        #     self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        #     mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        #     std = np.load(pjoin(self.meta_dir, 'std.npy'))
        #     split_file = pjoin(self.data_root, 'train.txt')

        # elif dataset_name == 'kit':
        #     self.data_root = './dataset/KIT-ML'
        #     self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        #     self.text_dir = pjoin(self.data_root, 'texts')
        #     self.joints_num = 21

        #     self.max_motion_length = 196
        #     self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        #     mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        #     std = np.load(pjoin(self.meta_dir, 'std.npy'))
        #     split_file = pjoin(self.data_root, 'train.txt')

        # elif dataset_name == 'motionmillion':
        #     self.data_root = './dataset/MotionMillion'
        #     self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
        #     self.text_dir = pjoin(self.data_root, self.text_type)
        #     self.joints_num = 22
        #     self.max_motion_length = 300
        #     mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
        #     std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
        #     split_file = pjoin(self.data_root, 'split', self.version, split + '.txt')
            
        # else:
        #     raise KeyError('Dataset Does not Exists')
        
        # joints_num = self.joints_num
        # id_list = []
        
        # self.data = []
        # self.lengths = []
        # self.id_list = []
        
        # with cs.open(split_file, 'r') as f:
        #     for line in f.readlines():
        #         id_list.append(line.strip())

        # if debug:
        #     id_list = id_list[:1000]
            
        # for name in tqdm(id_list):
        #     motion = np.load(pjoin(self.motion_dir, name + '.npy'))
        #     if motion.shape[0] < self.window_size:
        #         continue
        #     self.id_list.append(name)
        #     self.lengths.append(motion.shape[0] - self.window_size)
        #     self.data.append(motion)
        # self.mean = mean
        # self.std = std
        # print("Total number of motions {}".format(len(self.id_list)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def transform(self, data):
        return (data - self.mean) / self.std
    def inv_transform_torch(self, data):
        device = data.device
        return data * self.std_tensor.to(device) + self.mean_tensor.to(device)
    
    
    def transform_torch(self, data):
        device = data.device
        return (data - self.mean_tensor.to(device)) / self.std_tensor.to(device)
    
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        # sample_idx, time_idx = self.indices[idx]
        name = self.id_list[item]
        use = self.user[item]
        if use==0:
            
        
            motion = self.data_test[name].item()['motion'][:]
        else:
            motion = self.data_train[name].item()['motion'][:]
            
        
        # m_length = len(motion)
        # if self.unit_length < 10:
        #     coin2 = np.random.choice(['single', 'single', 'double'])
        # else:
        #     coin2 = 'single'

        # if coin2 == 'double':
        #     m_length = (m_length // self.unit_length - 1) * self.unit_length
        # elif coin2 == 'single':
        #     m_length = (m_length // self.unit_length) * self.unit_length
        # idx = random.randint(0, len(motion) - m_length)
        # motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # if m_length < self.max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        #                              ], axis=0)
        
        return motion, name

class VQMotionDataset_old(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, window_size = 64, unit_length = 8, motion_type=None, text_type=None, version=None):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version
        self.dataset_name = dataset_name
        
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
            split_file = pjoin(self.data_root, 'all.txt')
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
            split_file = pjoin(self.data_root, 'all.txt')
        elif dataset_name == 'handx':
            self.data_root = './dataset/HandX'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
            self.text_dir = pjoin(self.data_root, self.text_type)
            self.joints_num = 22
            radius = 4
            fps = 30
            self.max_motion_length = 300
            dim_pose = 272
            mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
            std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
            split_file = pjoin(self.data_root, 'split', self.version, 'all.txt')
        else:
            raise KeyError('Dataset Does not Exists')
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        for name in tqdm(id_list):
            new_name_list.append(name)

        self.mean = mean
        self.std = std
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        motion = np.load(pjoin(self.motion_dir, name + '.npy'))
        m_length = len(motion)

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name

def DATALoader(dataset_name,
                batch_size = 1,
                num_workers = 8, unit_length = 4, motion_type=None, text_type=None, version=None) : 
    
    dataset = VQMotionDataset(dataset_name, unit_length=unit_length, motion_type=motion_type, text_type=text_type, version=version)
    train_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader, dataset.mean, dataset.std

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
