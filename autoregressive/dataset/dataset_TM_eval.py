import os
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def collate_fn(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)



class MotionMillionFSQDataset(data.Dataset):
    # dataset_name, is_test, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4, version = "version1/tokenizer_no_mirror"
    def __init__(self, motion_dim, is_test, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4, version = "version1/tokenizer_no_mirror"):
        if motion_dim == 258:
            self.mean = np.load(f'{_DATA_DIR}/mean_duet_worot.npy')
            self.std = np.load(f'{_DATA_DIR}/std_duet_worot.npy')+1e-6
            self.std_tensor = torch.from_numpy(self.std)
            self.mean_tensor = torch.from_numpy(self.mean)
            self.data = dict(np.load(f'{_DATA_DIR}/test_full_duet_repre_worot.npz',allow_pickle=True))
        elif motion_dim == 438:
            self.mean = np.load(f'{_DATA_DIR}/mean_duet.npy')
            self.std = np.load(f'{_DATA_DIR}/std_duet.npy')+1e-6
            self.std_tensor = torch.from_numpy(self.std)
            self.mean_tensor = torch.from_numpy(self.mean)
            self.data = dict(np.load(f'{_DATA_DIR}/test_full_duet_repre.npz',allow_pickle=True))
        elif motion_dim in [168,288]:
            self.mean = np.load(f'{_DATA_DIR}/mean_correct_duet_scalar_rot.npy')
            self.std = np.load(f'{_DATA_DIR}/std_correct_duet_scalar_rot.npy')+1e-6
            self.mean = self.mean[:motion_dim]
            self.std = self.std[:motion_dim]
            # self.std[6:18] = 1.0
            # self.std[:3] /= 4
            # self.std[3:6] /= 6.0
            # self.std[6:18] /= 8.0
            # self.std[48:168] /= 1.5
            
            # self.std[48:168] /= 4.0
            # self.mean[6:18] = 0.0
            self.std_tensor = torch.from_numpy(self.std)
            self.mean_tensor = torch.from_numpy(self.mean)
            self.data = dict(np.load(f'{_DATA_DIR}/test_full_correct_duet_scalar_rot.npz',allow_pickle=True))
        elif motion_dim in [291]:
            self.mean = np.load(f'{_DATA_DIR}/mean_rvel_duet_scalar_rot.npy')
            self.std = np.load(f'{_DATA_DIR}/std_rvel_duet_scalar_rot.npy')+1e-6
            self.mean = self.mean[:motion_dim]
            self.std = self.std[:motion_dim]
            # self.std[6:18] = 1.0
            # self.std[3:6] /= 6.0
            # self.std[6:18] /= 8.0
            # self.std[48:168] /= 1.5
            # self.std[48:168] /= 4.0
            # self.mean[6:18] = 0.0
            self.std_tensor = torch.from_numpy(self.std)
            self.mean_tensor = torch.from_numpy(self.mean)
            self.data = dict(np.load(f'{_DATA_DIR}/test_full_rvel_duet_scalar_rot.npz',allow_pickle=True))
        elif motion_dim in [164,134,284]:
            self.mean = np.load(f'{_DATA_DIR}/mean_quat_duet_scalar_rot.npy')
            self.std = np.load(f'{_DATA_DIR}/std_quat_duet_scalar_rot.npy')+1e-6
            self.mean = self.mean[:motion_dim]
            self.std = self.std[:motion_dim]
            # self.std[6:14] = 1.0
            # self.std[:3] /= 4
            # self.std[3:6] /= 6.0
            # self.std[6:14] /= 6
            # self.std[14:14+120] /= 1.5d
            
            # self.std[6:14] /= 4
            # self.mean[6:14] = 0.0
            self.std_tensor = torch.from_numpy(self.std)
            self.mean_tensor = torch.from_numpy(self.mean)
            self.data = dict(np.load(f'{_DATA_DIR}/test_full_quat_duet_scalar_rot.npz',allow_pickle=True))
        self.motion_dim = motion_dim
        self.id_list = list(self.data.keys())
        L = len(self.id_list)//10+1
        self.id_list = self.id_list[:L]
        # if debug:
        #     self.id_list = id_list[:400]
        # self.window_size = window_size
        self.unit_length = unit_length
        # self.indices = self._create_indices()
    # def _create_indices(self):
    #     # Create a list of tuples (sample_index, time_index[, hand_idx])
    #     indices = []
    #     # if self.mode in ['left', 'right', 'split']:
    #     #     for i, _ in enumerate(self.keys):
    #     #         for start_idx in range(0, 60 - self.opt.window_size + 1, self.opt.window_stride):
    #     #             indices.append((i, start_idx))
    #     # elif self.mode == 'joint':
        
    #     for i, _ in enumerate(self.id_list):
    #         for start_idx in range(0, 60 - self.window_size + 1, 3):
    #             indices.append((i, start_idx))
    #             # indices.append((i, 1, start_idx))
    #     return indices
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
        motion = self.data[name].item()['motion'][:,:self.motion_dim]
        
        m_length = len(motion)
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
        
        return motion, m_length, name
# '''For use of training text-2-motion generative model'''
# class MotionMillionFSQDataset(data.Dataset):
#     def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4, version = "version1/tokenizer_no_mirror"):
        
#         self.max_length = 20
#         self.pointer = 0
#         self.dataset_name = dataset_name
#         self.is_test = is_test
#         self.max_text_len = max_text_len
#         self.unit_length = unit_length
#         self.w_vectorizer = w_vectorizer
#         self.version = version
        
#         if dataset_name == 't2m':
#             self.data_root = './dataset/HumanML3D'
#             self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
#             self.text_dir = pjoin(self.data_root, 'texts')
#             self.joints_num = 22
#             radius = 4
#             fps = 20
#             self.max_motion_length = 196
#             dim_pose = 263
#             kinematic_chain = paramUtil.t2m_kinematic_chain
#             self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
#             if is_test:
#                 split_file = pjoin(self.data_root, 'test.txt')
#             else:
#                 split_file = pjoin(self.data_root, 'val.txt')
#         elif dataset_name == 'kit':
#             self.data_root = './dataset/KIT-ML'
#             self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
#             self.text_dir = pjoin(self.data_root, 'texts')
#             self.joints_num = 21
#             radius = 240 * 8
#             fps = 12.5
#             dim_pose = 251
#             self.max_motion_length = 196
#             kinematic_chain = paramUtil.kit_kinematic_chain
#             self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
#             if is_test:
#                 split_file = pjoin(self.data_root, 'test.txt')
#             else:
#                 split_file = pjoin(self.data_root, 'val.txt')
#         elif dataset_name == 'motionmillion':
#             # self.data_root = './dataset/HumanML3D'
#             # self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
#             # self.text_dir = pjoin(self.data_root, 'texts')
#             self.data_root = './dataset/MotionMillion'
#             self.motion_dir = pjoin(self.data_root, 'motion_data', "vector_272")
#             self.text_dir = pjoin(self.data_root, "texts")
#             self.joints_num = 22
#             radius = 4
#             fps = 60
#             self.max_motion_length = 600
#             dim_pose = 263
#             kinematic_chain = paramUtil.t2m_kinematic_chain
#             self.meta_dir = pjoin(self.data_root, 'mean_std', "vector_272")
#             if is_test:
#                 split_file = pjoin(self.data_root, 'split', self.version, 'test.txt')
#             else:
#                 split_file = pjoin(self.data_root, 'split', self.version, 'val.txt')
            
#         mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
#         std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
#         if self.dataset_name == 'motionmillion':
#             min_motion_len = 120 # 192
#         elif self.dataset_name == 't2m':
#             min_motion_len = 40 # 192
#         else:
#             min_motion_len = 24

#         id_list = []
#         self.id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())

#         length_list = []
#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(self.motion_dir, name + '.npy'))
#                 if (len(motion)) < min_motion_len or (len(motion) > self.max_motion_length):
#                     continue
#                 self.id_list.append(name)
#                 length_list.append(len(motion))
                
#             except Exception as e:
#                 print(e)
        
#         self.id_list.sort()
#         self.mean = mean
#         self.std = std
#         self.length_arr = np.array(length_list)
#         self.reset_max_len(self.max_length)
#         print(len(self.id_list)) # 
#     def reset_max_len(self, length):
#         assert length <= self.max_motion_length
#         self.pointer = np.searchsorted(self.length_arr, length)
#         print("Pointer Pointing at %d"%self.pointer)
#         self.max_length = length

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def transform(self, data):
#         return (data - self.mean) / self.std

#     def __len__(self):
#         return len(self.id_list) - self.pointer

#     def __getitem__(self, item):
#         idx = self.pointer + item
#         name = self.id_list[idx]
#         motion = np.load(pjoin(self.motion_dir, name + '.npy'))

#         m_length = len(motion)
#         if self.unit_length < 10:
#             coin2 = np.random.choice(['single', 'single', 'double'])
#         else:
#             coin2 = 'single'

#         if coin2 == 'double':
#             m_length = (m_length // self.unit_length - 1) * self.unit_length
#         elif coin2 == 'single':
#             m_length = (m_length // self.unit_length) * self.unit_length
#         idx = random.randint(0, len(motion) - m_length)
#         motion = motion[idx:idx+m_length]

#         "Z Normalization"
#         motion = (motion - self.mean) / self.std

#         if m_length < self.max_motion_length:
#             motion = np.concatenate([motion,
#                                      np.zeros((self.max_motion_length - m_length, motion.shape[1]))
#                                      ], axis=0)
#         motion = motion.astype(np.float32)
#         return motion, m_length, name



def MotionMillionFSQDATALoader(motion_dim, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4, version = "version1/tokenizer_no_mirror") : 
    
    val_dataset = MotionMillionFSQDataset(motion_dim, is_test, w_vectorizer, unit_length=unit_length, version=version)
    val_loader = torch.utils.data.DataLoader( val_dataset, 
                                              batch_size,
                                              shuffle = True,
                                              num_workers=40, # num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True,
                                              prefetch_factor=2)
    return val_loader, val_dataset.mean, val_dataset.std

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
