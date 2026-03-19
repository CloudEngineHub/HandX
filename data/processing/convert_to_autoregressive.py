"""
Convert HandX data (data/handx/) to autoregressive representation (autoregressive/data/).

Input:
    data/handx/train_can_pos_all_wotextfeat.npz   # motion: (60, 2, 21, 3) + annotations
    data/handx/test_can_pos_all_wotextfeat.npz
    data/handx/train_mano.npz                      # left/right_pose: (60, 48), etc.
    data/handx/test_mano.npz

Output (288-dim "correct_duet_scalar_rot" representation):
    autoregressive/data/train_full_correct_duet_scalar_rot.npz
    autoregressive/data/test_full_correct_duet_scalar_rot.npz
    autoregressive/data/texts_all.pkl
    autoregressive/data/mean_correct_duet_scalar_rot.npy
    autoregressive/data/std_correct_duet_scalar_rot.npy

288-dim layout per frame:
    [0:3]     vel_A          — right hand root velocity
    [3:6]     relative_B_pos — left root position relative to right root
    [6:18]    rot6d          — root rotation in 6D (2 hands * 6)
    [18:48]   scalar_rot     — scalar rotation for joints 1-15 (2 hands * 15)
    [48:168]  local_pos      — local joint positions (2 hands * 20 joints * 3)
    [168:288] local_vel      — local joint position velocity (2 hands * 20 joints * 3)

Usage:
    cd data/processing
    python convert_to_autoregressive.py [--output_dir OUTPUT_DIR]
"""

import argparse
import os
import pickle
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation'))
from constant import SKELETON_CHAIN
from single_code import MotionCoder


def _get_scalar_rotation(single_motion_seq: np.ndarray, side: str) -> np.ndarray:
    """
    Compute scalar rotation angles at each joint from joint positions.

    Args:
        single_motion_seq: (T, 21, 3) joint positions for one hand
        side: 'left' or 'right'

    Returns:
        (T, 21) scalar rotation per joint
    """
    coder = MotionCoder(single_motion_seq, isright=(side == 'right'))
    coder.get_local_coordinate()
    local_motion = coder.local_motion  # (T, 21, 3)

    T, J = single_motion_seq.shape[0], single_motion_seq.shape[1]
    scalar_rot = np.zeros((T, J))

    for chain in SKELETON_CHAIN:
        scalar_rot[:, chain[0]] = 0
        scalar_rot[:, chain[-1]] = 0
        for s in range(1, len(chain) - 1):
            j, pre_j, nxt_j = chain[s], chain[s - 1], chain[s + 1]
            v1 = (local_motion[:, j] - local_motion[:, pre_j])[:, [0, 2]]
            v2 = (local_motion[:, nxt_j] - local_motion[:, j])[:, [0, 2]]
            angle_diff = np.arctan2(v2[:, 1], v2[:, 0]) - np.arctan2(v1[:, 1], v1[:, 0])
            if side == 'right':
                angle_diff = -angle_diff
            scalar_rot[:, j] = angle_diff

    return np.nan_to_num(scalar_rot, nan=0.0, posinf=0.0, neginf=0.0)


def convert_sample(position: np.ndarray, left_pose: np.ndarray, right_pose: np.ndarray,
                   device: torch.device) -> np.ndarray:
    """
    Convert a single sample from (can_pos + mano) to 288-dim representation.

    Args:
        position: (60, 2, 21, 3) canonical joint positions
        left_pose: (60, 48) MANO left hand pose parameters (axis-angle)
        right_pose: (60, 48) MANO right hand pose parameters (axis-angle)
        device: torch device

    Returns:
        (60, 288) autoregressive representation
    """
    T = position.shape[0]
    pos = torch.from_numpy(position).float().to(device)

    # MANO root rotation → rotation matrix → 6D
    rotation = torch.from_numpy(np.concatenate([
        left_pose.reshape(T, 1, -1, 3),
        right_pose.reshape(T, 1, -1, 3)
    ], axis=1)).float().to(device)                         # (T, 2, 16, 3)
    rotation_m = axis_angle_to_matrix(rotation)            # (T, 2, 16, 3, 3)
    R = rotation_m[:, :, 0]                                # (T, 2, 3, 3) root rotation
    rot6d = matrix_to_rotation_6d(R.float())               # (T, 2, 6)

    # Local joint positions: rotate relative positions by R
    rel = pos[:, :, 1:] - pos[:, :, :1]                   # (T, 2, 20, 3)
    local = torch.einsum('thjc,thck->thjk', rel, R)       # (T, 2, 20, 3)

    # Local velocity
    velocity = torch.cat([local[1:] - local[:-1],
                          torch.zeros_like(local[:1])], dim=0)  # (T, 2, 20, 3)

    # Root velocity (right hand) and relative position (left - right)
    vel_A = torch.cat([pos[1:, 1, 0] - pos[:-1, 1, 0],
                       torch.zeros(1, 3, device=device)], dim=0)  # (T, 3)
    relative_B_pos = pos[:, 0, 0] - pos[:, 1, 0]                 # (T, 3)

    # Scalar rotation from joint positions (CPU, numpy)
    pos_np = position
    scalar_l = _get_scalar_rotation(pos_np[:, 0], side='left')[:, 1:16]   # (T, 15)
    scalar_r = _get_scalar_rotation(pos_np[:, 1], side='right')[:, 1:16]  # (T, 15)
    scalar = np.concatenate([scalar_l, scalar_r], axis=-1)                 # (T, 30)
    scalar_t = torch.from_numpy(scalar).float().to(device)

    # Concatenate: [vel_A(3), rel_B(3), rot6d(12), scalar(30), local(120), vel(120)] = 288
    result = torch.cat([
        vel_A.reshape(T, 3),
        relative_B_pos.reshape(T, 3),
        rot6d.reshape(T, 12),
        scalar_t.reshape(T, 30),
        local.reshape(T, 120),
        velocity.reshape(T, 120),
    ], dim=-1)

    return result.detach().cpu().numpy()


def process_split(split: str, input_dir: str, output_dir: str, device: torch.device,
                   running_stats: dict = None) -> dict:
    """
    Process one split (train/test) and return texts dict for merging.
    If running_stats is provided (dict with 'sum', 'sum_sq', 'count'), accumulate
    online statistics for mean/std computation (train split only).
    """
    pos_path = os.path.join(input_dir, f'{split}_can_pos_all_wotextfeat.npz')
    mano_path = os.path.join(input_dir, f'{split}_mano.npz')

    print(f"Loading {split} position data...")
    pos_data = np.load(pos_path, allow_pickle=True)
    keys = list(pos_data.keys())
    print(f"  Loaded position data: {len(keys)} samples")

    print(f"Loading {split} mano data...")
    mano_data = np.load(mano_path, allow_pickle=True)
    print(f"  Loaded mano data: {len(mano_data.keys())} samples")
    dct_out = {}
    texts_out = {}

    for name in tqdm(keys, desc=f"Processing {split}"):
        pos_sample = pos_data[name].item()
        mano_sample = mano_data[name].item()

        position = pos_sample['motion']          # (60, 2, 21, 3)
        left_pose = mano_sample['left_pose']     # (60, 48)
        right_pose = mano_sample['right_pose']   # (60, 48)

        motion_288 = convert_sample(position, left_pose, right_pose, device)

        if running_stats is not None:
            running_stats['sum'] += motion_288.sum(axis=0)
            running_stats['sum_sq'] += (motion_288 ** 2).sum(axis=0)
            running_stats['count'] += motion_288.shape[0]

        sample_out = {'motion': motion_288}
        text_out = {}
        for ann_key in ['left_annotation', 'right_annotation', 'interaction_annotation']:
            sample_out[ann_key] = pos_sample[ann_key]
            text_out[ann_key] = pos_sample[ann_key]

        key_with_prefix = f"{split}_{name}"
        dct_out[key_with_prefix] = sample_out
        texts_out[key_with_prefix] = text_out

    out_path = os.path.join(output_dir, f'{split}_full_correct_duet_scalar_rot.npz')
    print(f"Saving {out_path} ({len(dct_out)} samples)...")
    np.savez(out_path, **dct_out)

    return texts_out


def main():
    parser = argparse.ArgumentParser(description="Convert HandX data to autoregressive 288-dim representation")
    parser.add_argument('--input_dir', type=str, default='../handx',
                        help='Path to data/handx/ directory')
    parser.add_argument('--output_dir', type=str, default='../../autoregressive/data',
                        help='Output directory (default: autoregressive/data/)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Online mean/std accumulator for train split
    stats = {'sum': np.zeros(288, dtype=np.float64),
             'sum_sq': np.zeros(288, dtype=np.float64),
             'count': 0}

    all_texts = {}

    # Train split — accumulate stats
    texts = process_split('train', args.input_dir, args.output_dir, device, running_stats=stats)
    all_texts.update(texts)

    # Test split — no stats needed
    texts = process_split('test', args.input_dir, args.output_dir, device)
    all_texts.update(texts)

    # Save texts_all.pkl
    texts_path = os.path.join(args.output_dir, 'texts_all.pkl')
    with open(texts_path, 'wb') as f:
        pickle.dump(all_texts, f)
    print(f"Saved {texts_path} ({len(all_texts)} entries)")

    # Compute and save mean/std from accumulated stats
    mean = stats['sum'] / stats['count']
    std = np.sqrt(stats['sum_sq'] / stats['count'] - mean ** 2)
    std[std < 1e-6] = 1.0
    np.save(os.path.join(args.output_dir, 'mean_correct_duet_scalar_rot.npy'), mean)
    np.save(os.path.join(args.output_dir, 'std_correct_duet_scalar_rot.npy'), std)
    print(f"Saved mean/std with shape {mean.shape}")

    print("Done.")


if __name__ == '__main__':
    main()
