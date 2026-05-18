"""
Quick verification script to test that data preprocessing and loading work.

Usage:
    python tools/verify_data.py --test_read   # test reading a single skeleton file
    python tools/verify_data.py --test_load   # test loading preprocessed data
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_read_skeleton():
    """Read a single .skeleton file and print stats."""
    from tools.preprocess_ntu import read_skeleton, _get_bodies, _select_bodies, normalize_skeleton

    skeleton_dir = '/data/lxc/datasets/ntu/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons'
    files = sorted([f for f in os.listdir(skeleton_dir) if f.endswith('.skeleton')])[:5]

    for fname in files:
        fpath = os.path.join(skeleton_dir, fname)
        print(f'\n--- {fname} ---')
        seq = read_skeleton(fpath)
        print(f'  Frames: {seq["numFrame"]}')
        print(f'  Bodies per frame: {[f["numBody"] for f in seq["frameInfo"][:5]]}...')

        bodies = _get_bodies(seq)
        print(f'  Detected bodies: {len(bodies)}')

        body_arr = _select_bodies(bodies)
        print(f'  Selected body shape: {body_arr.shape}')  # [M, T, V, C]

        norm = normalize_skeleton(body_arr)
        print(f'  Normalized shape: {norm.shape}')  # [C, T, V, M]

        # stats
        valid = np.abs(norm).sum(axis=(0, 2, 3)) > 0
        print(f'  Valid frames: {valid.sum()}')
        print(f'  Value range: [{norm.min():.4f}, {norm.max():.4f}]')

    print('\n[OK] Skeleton reading test passed!')


def test_load_preprocessed():
    """Load preprocessed .npy data and check shapes."""
    import pickle

    base = 'data/ntu/processed'
    found = False
    for split_dir in sorted(os.listdir(base)):
        split_path = os.path.join(base, split_dir)
        if not os.path.isdir(split_path):
            continue

        for part in ['train', 'val']:
            data_file = os.path.join(split_path, f'{part}_data.npy')
            label_file = os.path.join(split_path, f'{part}_label.pkl')

            if not os.path.exists(data_file):
                continue

            found = True
            data = np.load(data_file, mmap_mode='r')
            with open(label_file, 'rb') as f:
                names, labels = pickle.load(f)

            print(f'\n--- {split_dir}/{part} ---')
            print(f'  Data shape: {data.shape}')  # [N, C, T, V, M]
            print(f'  Num labels: {len(labels)}')
            print(f'  Label range: [{min(labels)}, {max(labels)}]')
            print(f'  Num classes: {len(set(labels))}')

            # check a sample
            sample = data[0]
            valid = np.abs(sample).sum(axis=(0, 2, 3)) > 0
            print(f'  Sample[0] valid frames: {valid.sum()}')
            print(f'  Sample[0] value range: [{sample.min():.4f}, {sample.max():.4f}]')

            # zero-sample check
            all_zero = np.sum(np.abs(data).sum(axis=(1, 2, 3, 4)) == 0)
            print(f'  All-zero samples: {all_zero}')

    if not found:
        print('[WARN] No preprocessed data found. Run preprocess_ntu.py first.')
        return

    print('\n[OK] Data loading test passed!')


def test_dataset_loader():
    """Test NTUDataset and DataLoader."""
    import pickle
    from datasets.ntu_dataset import NTUDataset

    base = 'data/ntu/processed'
    for split_dir in sorted(os.listdir(base)):
        split_path = os.path.join(base, split_dir)
        if not os.path.isdir(split_path):
            continue

        data_file = os.path.join(split_path, 'train_data.npy')
        label_file = os.path.join(split_path, 'train_label.pkl')

        if not os.path.exists(data_file):
            continue

        print(f'\n--- Testing DataLoader for {split_dir} ---')
        dataset = NTUDataset(
            data_path=data_file,
            label_path=label_file,
            num_frames=64,
            split='train',
        )
        print(f'  Dataset size: {len(dataset)}')

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch_data, batch_label, batch_idx = next(iter(loader))
        print(f'  Batch data shape: {batch_data.shape}')  # [B, C, T, V, M]
        print(f'  Batch label: {batch_label.tolist()}')
        print(f'  Data dtype: {batch_data.dtype}')
        print(f'  Value range: [{batch_data.min():.4f}, {batch_data.max():.4f}]')

        print(f'\n[OK] DataLoader test passed for {split_dir}!')
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_read', action='store_true', help='Test reading raw .skeleton files')
    parser.add_argument('--test_load', action='store_true', help='Test loading preprocessed data')
    parser.add_argument('--test_loader', action='store_true', help='Test NTUDataset + DataLoader')
    args = parser.parse_args()

    if not any([args.test_read, args.test_load, args.test_loader]):
        args.test_read = True

    if args.test_read:
        test_read_skeleton()
    if args.test_load:
        test_load_preprocessed()
    if args.test_loader:
        test_dataset_loader()
