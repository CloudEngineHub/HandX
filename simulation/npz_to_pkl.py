"""
Extract MANO sequences from HandX npz files and save as pkl files
compatible with mano_to_pt.py (and the simulation pipeline).

Usage:
  # Extract specific source keys
  python npz_to_pkl.py --npz ../data/handx/test_mano.npz --keys hot3d_0001 handx_0002 --output_dir ./pkl_out
"""

import argparse
import os
import pickle
import numpy as np


def sanitize_key(key):
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in key)


def npz_entry_to_pkl_dict(entry):
    """Convert an npz entry to the pkl format expected by mano_to_pt.py."""
    return {
        'left': {
            'shape': entry['left_shape'],
            'pose': entry['left_pose'],
            'trans': entry['left_trans'],
        },
        'right': {
            'shape': entry['right_shape'],
            'pose': entry['right_pose'],
            'trans': entry['right_trans'],
        },
    }


def save_pkl(data_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)


def main():
    parser = argparse.ArgumentParser(
        description='Extract MANO sequences from HandX npz and save as pkl for simulation')
    parser.add_argument('--npz', required=True,
                        help='Path to train_mano.npz or test_mano.npz')
    parser.add_argument('--keys', type=str, nargs='+', required=True,
                        help='Source keys to extract (e.g. hot3d_0001 handx_0002)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for pkl files')
    args = parser.parse_args()

    print(f'Loading {args.npz} ...')
    npz = np.load(args.npz, allow_pickle=True)
    all_keys = list(npz.keys())
    print(f'  Total sequences: {len(all_keys)}')

    basename = os.path.splitext(os.path.basename(args.npz))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    keys_to_save = []
    for key in args.keys:
        if key not in all_keys:
            print(f'  Warning: key {key} not found in npz, skipping')
        else:
            keys_to_save.append(key)

    print(f'  Extracting {len(keys_to_save)} sequences ...')

    for key in keys_to_save:
        entry = npz[key].item()
        pkl_dict = npz_entry_to_pkl_dict(entry)
        out_path = os.path.join(args.output_dir, f'{basename}_{sanitize_key(key)}.pkl')
        save_pkl(pkl_dict, out_path)

    print(f'Done. {len(keys_to_save)} pkl files saved to {args.output_dir}')


if __name__ == '__main__':
    main()
