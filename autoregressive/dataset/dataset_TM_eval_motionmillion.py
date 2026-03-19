import os
import torch
from torch.utils import data
import numpy as np
import random
from tqdm import tqdm

from torch.utils.data._utils.collate import default_collate

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def collate_fn(batch):
    return default_collate(batch)


'''For use of evaluating text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias=5, max_text_len=20, unit_length=4, split="val"):

        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        self.joints_num = 22
        self.max_motion_length = 300
        self.motion_dim = 288

        # Load mean/std from data/ directory
        mean = np.load(os.path.join(_DATA_DIR, 'mean_correct_duet_scalar_rot.npy'))
        std = np.load(os.path.join(_DATA_DIR, 'std_correct_duet_scalar_rot.npy')) + 1e-6

        # Load motion data from npz
        npz_split = 'test' if split in ('val', 'val_debug', 'test') else split
        npz_path = os.path.join(_DATA_DIR, f'{npz_split}_full_correct_duet_scalar_rot.npz')
        raw_data = dict(np.load(npz_path, allow_pickle=True))

        min_motion_len = 60
        data_dict = {}
        new_name_list = []
        length_list = []
        self.id_list = []

        for name in tqdm(raw_data.keys(), desc=f"Loading eval data ({npz_split})"):
            try:
                entry = raw_data[name].item()
                motion = entry['motion']  # (T, 288)

                if len(motion) < min_motion_len or len(motion) > 200:
                    continue

                # Build text from annotations
                text_data = []
                l_texts = entry.get('left_annotation', [])
                r_texts = entry.get('right_annotation', [])
                i_texts = entry.get('interaction_annotation', [])

                if l_texts and r_texts and i_texts:
                    # Create captions matching training format
                    for li in range(len(l_texts)):
                        lt = l_texts[min(li, len(l_texts) - 1)]
                        rt = r_texts[min(li, len(r_texts) - 1)]
                        it = i_texts[min(li, len(i_texts) - 1)]
                        caption = f"<extra_id_0> {lt} <extra_id_1> {rt} <extra_id_2> {it}"
                        text_data.append({'caption': caption})

                if not text_data:
                    continue

                self.id_list.append(name)
                length_list.append(len(motion))
                data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'text': text_data
                }
                new_name_list.append(name)

            except Exception as e:
                print(f"Skipping {name}: {e}")

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        print(f"Loaded {len(self.id_list)} eval samples")

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.id_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return "_", "_", caption, 0, motion, m_length, '_', name



def DATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers=8, unit_length=4, split="val"):

    val_dataset = Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length, split=split)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last=True)
    return val_loader, val_dataset.mean, val_dataset.std


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
