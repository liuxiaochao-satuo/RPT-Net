"""
NTU RGB+D 60/120 skeleton data preprocessing.

Reads raw .skeleton files, filters bad samples, normalizes coordinates,
and saves [N, C, T, V, M] arrays for each benchmark split.

Usage:
    python tools/preprocess_ntu.py --dataset ntu60
    python tools/preprocess_ntu.py --dataset ntu120
    python tools/preprocess_ntu.py --dataset ntu60 --benchmark xsub
"""

import argparse
import os
import sys
import math
import pickle
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────
# NTU skeleton constants
# ──────────────────────────────────────────────
NUM_JOINT = 25
MAX_BODY_TRUE = 2
MAX_BODY_KINECT = 4
MAX_FRAME = 300
NUM_CHANNEL = 3  # x, y, z

NTU60_TRAINING_SUBJECTS = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
]
NTU60_TRAINING_CAMERAS = [2, 3]

NTU120_TRAINING_SUBJECTS = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56,
    57, 58, 59, 70, 74, 78, 80, 81, 82, 83,
    84, 85, 86, 89, 91, 92, 93, 94, 95, 97,
    98, 100, 103,
]


# ──────────────────────────────────────────────
# Rotation utilities
# ──────────────────────────────────────────────
def _rotation_matrix(axis, theta):
    """Rodrigues rotation matrix for *axis* by *theta* radians."""
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
    ])


def _angle_between(v1, v2):
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# ──────────────────────────────────────────────
# Skeleton reading
# ──────────────────────────────────────────────
def read_skeleton(file_path):
    """Parse a single .skeleton file into a dict structure."""
    with open(file_path, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for _ in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': int(f.readline()), 'bodyInfo': []}
            for _ in range(frame_info['numBody']):
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState',
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['bodyID'] = int(body_info['bodyID'])
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for _ in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState',
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


# ──────────────────────────────────────────────
# Body extraction & filtering
# ──────────────────────────────────────────────
def _get_bodies(skeleton_seq):
    """Extract per-body coordinate arrays from parsed skeleton dict.

    Returns dict {body_id: ndarray[C, T, V]} where C=3 (xyz).
    """
    bodies = {}
    for t, frame in enumerate(skeleton_seq['frameInfo']):
        seen_ids = []
        for body in frame['bodyInfo']:
            bid = body['bodyID']
            while bid in seen_ids:
                bid += 1
            seen_ids.append(bid)
            if bid not in bodies:
                bodies[bid] = np.zeros((NUM_CHANNEL, MAX_FRAME, NUM_JOINT), dtype=np.float32)
            for j in range(NUM_JOINT):
                bodies[bid][0, t, j] = body['jointInfo'][j]['x']
                bodies[bid][1, t, j] = body['jointInfo'][j]['y']
                bodies[bid][2, t, j] = body['jointInfo'][j]['z']
    return bodies


def _body_energy(body_ctv):
    """Motion energy: sum of per-channel std on valid (non-zero) frames."""
    s = body_ctv - body_ctv[:, :, 0:1]
    valid = s.sum(0).sum(-1) != 0
    s = s[:, valid]
    if s.size == 0:
        return 0.0
    return float(s[0].std() + s[1].std() + s[2].std())


def _select_bodies(bodies, max_bodies=MAX_BODY_TRUE):
    """Keep at most *max_bodies* by motion energy, returns [M, T, V, C]."""
    arr = np.array(list(bodies.values()))  # [M_raw, C, T, V]
    energies = np.array([_body_energy(b) for b in arr])
    idx = energies.argsort()[::-1][:max_bodies]
    arr = arr[idx]
    arr = arr.transpose(0, 2, 3, 1)  # [M, T, V, C]
    return arr


# ──────────────────────────────────────────────
# Skeleton normalisation
# ──────────────────────────────────────────────
def normalize_skeleton(skeleton):
    """Normalize skeleton [M, T, V, C(xyz)]:
      1. Shift top-padded zeros to front
      2. Translate so main body center joint (#1 spine-base → #0 in 0-idx) at origin
      3. Scale by spine bone length (joint 0 → joint 20)
      4. Rotate z-axis and x-axis alignment
    Returns [C, T, V, M].
    """
    M, T, V, C = skeleton.shape
    if skeleton.sum() == 0:
        return np.zeros((C, T, V, M), dtype=np.float32)

    # shift non-zero frames to the front
    if skeleton[:, 0].sum() == 0:
        valid = skeleton.sum(-1).sum(-1).sum(0) != 0
        tmp = skeleton[:, valid].copy()
        skeleton[:] = 0
        skeleton[:, :tmp.shape[1]] = tmp

    # translate: subtract center joint (joint 0) of main body at frame 0
    center = skeleton[0, 0, 0].copy()  # [C]
    for m in range(M):
        if skeleton[m].sum() == 0:
            continue
        mask = (skeleton[m].sum(-1) != 0).reshape(T, V, 1)
        skeleton[m] = (skeleton[m] - center) * mask

    # scale by spine bone length (joint 0 → joint 20)
    t = 0
    spine_len = 0.0
    while t < T and spine_len == 0:
        spine_len = np.linalg.norm(skeleton[0, t, 20] - skeleton[0, t, 0])
        t += 1
    if spine_len > 0:
        skeleton /= spine_len

    # rotate: align spine (joint 0→20) to z-axis
    joint_bottom = skeleton[0, 0, 0]
    joint_top = skeleton[0, 0, 20]
    axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
    angle = _angle_between(joint_top - joint_bottom, [0, 0, 1])
    rot_z = _rotation_matrix(axis, angle)
    for m in range(M):
        if skeleton[m].sum() == 0:
            continue
        for t_i in range(T):
            if skeleton[m, t_i].sum() == 0:
                continue
            for j in range(V):
                skeleton[m, t_i, j] = rot_z @ skeleton[m, t_i, j]

    # rotate: align shoulder line (joint 20→5) to x-axis (projected on xy-plane)
    jl = skeleton[0, 0, 20].copy()
    jr = skeleton[0, 0, 5].copy()
    jl[2] = 0
    jr[2] = 0
    axis = np.cross(jr - jl, [1, 0, 0])
    angle = _angle_between(jr - jl, [1, 0, 0])
    rot_x = _rotation_matrix(axis, angle)
    for m in range(M):
        if skeleton[m].sum() == 0:
            continue
        for t_i in range(T):
            if skeleton[m, t_i].sum() == 0:
                continue
            for j in range(V):
                skeleton[m, t_i, j] = rot_x @ skeleton[m, t_i, j]

    # [M, T, V, C] → [C, T, V, M]
    return skeleton.transpose(3, 1, 2, 0)


# ──────────────────────────────────────────────
# Filename parsing helpers
# ──────────────────────────────────────────────
def parse_filename(filename):
    """Parse SsssCcccPpppRrrrAaaa into (setup, camera, subject, replication, action)."""
    setup = int(filename[filename.find('S') + 1:filename.find('S') + 4])
    camera = int(filename[filename.find('C') + 1:filename.find('C') + 4])
    subject = int(filename[filename.find('P') + 1:filename.find('P') + 4])
    replication = int(filename[filename.find('R') + 1:filename.find('R') + 4])
    action = int(filename[filename.find('A') + 1:filename.find('A') + 4])
    return setup, camera, subject, replication, action


# ──────────────────────────────────────────────
# Main generation routine
# ──────────────────────────────────────────────
def generate_dataset(data_paths, out_path, ignored_samples, benchmark, part,
                     training_subjects, training_cameras):
    """Process raw skeletons and save as .npy + label .pkl."""

    sample_names = []
    sample_labels = []

    for data_path in data_paths:
        for filename in sorted(os.listdir(data_path)):
            if not filename.endswith('.skeleton'):
                continue
            if filename in ignored_samples:
                continue
            setup, camera, subject, _, action = parse_filename(filename)

            if benchmark == 'xsub':
                is_training = subject in training_subjects
            elif benchmark == 'xview':
                is_training = camera in training_cameras
            elif benchmark == 'xset':
                is_training = (setup % 2 == 1)
            else:
                raise ValueError(f'Unknown benchmark: {benchmark}')

            if (part == 'train') == is_training:
                sample_names.append((data_path, filename))
                sample_labels.append(action - 1)

    print(f'  [{benchmark}/{part}] {len(sample_names)} samples')

    # save labels
    os.makedirs(out_path, exist_ok=True)
    label_path = os.path.join(out_path, f'{part}_label.pkl')
    with open(label_path, 'wb') as f:
        pickle.dump(([name for _, name in sample_names], list(sample_labels)), f)

    # process skeletons
    data = np.zeros(
        (len(sample_labels), NUM_CHANNEL, MAX_FRAME, NUM_JOINT, MAX_BODY_TRUE),
        dtype=np.float32,
    )

    bad_count = 0
    for i, (data_path, fname) in enumerate(tqdm(sample_names, desc=f'{benchmark}/{part}')):
        try:
            seq_info = read_skeleton(os.path.join(data_path, fname))
            bodies = _get_bodies(seq_info)
            if len(bodies) == 0:
                bad_count += 1
                continue
            body_arr = _select_bodies(bodies)  # [M, T, V, C]
            norm = normalize_skeleton(body_arr)  # [C, T, V, M]
            num_body = min(norm.shape[-1], MAX_BODY_TRUE)
            data[i, :, :, :, :num_body] = norm[:, :, :, :num_body]
        except Exception as e:
            bad_count += 1
            tqdm.write(f'  [WARN] Failed to process {fname}: {e}')

    if bad_count > 0:
        print(f'  [INFO] {bad_count} samples failed during processing')

    data_path_out = os.path.join(out_path, f'{part}_data.npy')
    np.save(data_path_out, data)
    print(f'  Saved {data_path_out}  shape={data.shape}')


def load_ignored_samples(path):
    """Load the missing-skeleton list, return set of filenames."""
    ignored = set()
    if path is None or not os.path.exists(path):
        return ignored
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line[0].isdigit() and line[0] != 'S':
                continue
            if line.startswith('S'):
                ignored.add(line + '.skeleton')
    return ignored


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='NTU RGB+D Skeleton Preprocessor')
    parser.add_argument('--dataset', type=str, default='ntu60',
                        choices=['ntu60', 'ntu120'],
                        help='Which dataset to preprocess')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Specific benchmark (xsub/xview/xset). '
                             'If not set, generates all applicable benchmarks.')
    parser.add_argument('--ntu60_path', type=str,
                        default='/data/lxc/datasets/ntu/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons')
    parser.add_argument('--ntu120_path', type=str,
                        default='/data/lxc/datasets/ntu/nturgbd_skeletons_s018_to_s032')
    parser.add_argument('--missing_60', type=str,
                        default='dataset/error_sample_index/ntu60_missing.txt')
    parser.add_argument('--missing_120', type=str,
                        default='dataset/error_sample_index/ntu120_missing.txt')
    parser.add_argument('--out_folder', type=str,
                        default='data/ntu/processed')
    args = parser.parse_args()

    if args.dataset == 'ntu60':
        data_paths = [args.ntu60_path]
        ignored = load_ignored_samples(args.missing_60)
        training_subjects = NTU60_TRAINING_SUBJECTS
        benchmarks = ['xsub', 'xview']
    else:
        data_paths = [args.ntu60_path, args.ntu120_path]
        ignored = load_ignored_samples(args.missing_120)
        training_subjects = NTU120_TRAINING_SUBJECTS
        benchmarks = ['xsub', 'xset']

    if args.benchmark:
        benchmarks = [args.benchmark]

    print(f'Dataset: {args.dataset}')
    print(f'Data paths: {data_paths}')
    print(f'Ignored samples: {len(ignored)}')
    print(f'Benchmarks: {benchmarks}')
    print()

    for benchmark in benchmarks:
        for part in ['train', 'val']:
            out_path = os.path.join(args.out_folder, f'{args.dataset}_{benchmark}')
            generate_dataset(
                data_paths=data_paths,
                out_path=out_path,
                ignored_samples=ignored,
                benchmark=benchmark,
                part=part,
                training_subjects=training_subjects,
                training_cameras=NTU60_TRAINING_CAMERAS,
            )
        print()

    print('Done!')


if __name__ == '__main__':
    main()
