#!/usr/bin/env python3
"""
Quick offline pipeline verification for OpenBCI logs.

Performs the following checks (no training):
  1) Loads base Offline_EEGNet config and overrides for OpenBCI 16ch
  2) Selects the latest exp_* session under data/raspy/ (or --session)
  3) Builds the h5 dataset (preprocess, window, resample)
  4) Prints h5 shapes and inferred (electrodes, time)
  5) Instantiates EEGNet with matching dims and runs a forward pass on a batch

Usage:
  python scripts/tests/test_offline_verify.py [--session exp_YYYY-mm-dd_HH-MM-SS] [--h5 quick_verify.h5]
"""
import os
import sys
import glob
import argparse
import h5py
try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# Repo root
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from Offline_EEGNet.shared_utils import utils, dataset
if HAS_TORCH:
    from Offline_EEGNet.EEGNet import EEGNet


def pick_session(exp_name: str | None) -> str:
    data_dir = os.path.join(REPO, 'data', 'raspy')
    if exp_name:
        sess_path = os.path.join(data_dir, exp_name)
        if not os.path.isdir(sess_path):
            raise FileNotFoundError(f"Session not found: {sess_path}")
        if not (os.path.isfile(os.path.join(sess_path, 'eeg.bin')) and os.path.isfile(os.path.join(sess_path, 'task.bin'))):
            raise FileNotFoundError(f"Session missing eeg.bin/task.bin: {sess_path}")
        return exp_name
    sessions = sorted([d for d in os.listdir(data_dir) if d.startswith('exp_')])
    for sess in reversed(sessions):
        sess_path = os.path.join(data_dir, sess)
        if os.path.isfile(os.path.join(sess_path, 'eeg.bin')) and os.path.isfile(os.path.join(sess_path, 'task.bin')):
            return sess
    raise FileNotFoundError(f"No exp_* folder with eeg.bin/task.bin found in {data_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', help='exp_* folder name under data/raspy to use')
    ap.add_argument('--h5', default='quick_verify.h5', help='Output h5 name (under preprocessed_data)')
    args = ap.parse_args()

    # 1) Load base config and override for OpenBCI + chosen session
    cfg_path = os.path.join(REPO, 'Offline_EEGNet', 'config.yaml')
    cfg = utils.read_config(cfg_path)
    # Use absolute path so create_dataset works regardless of CWD
    data_dir = os.path.join(REPO, 'data', 'raspy')
    sess = pick_session(args.session)
    print('Using session:', sess)

    # Debug: print unique states in task data
    task_path = os.path.join(REPO, 'data', 'raspy', sess, 'task.bin')
    try:
        task_dict = utils.read_data_file_to_dict(task_path)
        import numpy as np
        states = np.asarray(task_dict['state_task']).astype(int)
        uniq, counts = np.unique(states, return_counts=True)
        print('State distribution in task.bin:', dict(zip(uniq.tolist(), counts.tolist())))
        print('Total ticks:', len(states))
        print('First 20 states:', states[:20].tolist())
        print('Last 20 states:', states[-20:].tolist())
        state_changes = np.flatnonzero(np.diff(states.flatten())) + 1
        print('Number of transitions:', len(state_changes))
        print('First 10 transition indices:', state_changes[:10].tolist())
        if len(state_changes) > 10:
            print('Last 10 transition indices:', state_changes[-10:].tolist())
    except Exception as e:
        print('Warning: could not inspect task.bin states:', e)

    cfg['data_dir'] = data_dir + '/'  # Offline_EEGNet expects trailing slash
    cfg['data_names'] = [sess]
    cfg['data_kinds'] = ['OL']

    # OpenBCI 16 ch preproc
    dp = cfg['data_preprocessor']
    dp['eeg_cap_type'] = 'openbci16'
    dp['sampling_frequency'] = 125
    dp['apply_laplacian'] = False
    # normalization skip is Fs-aware via code; can optionally set:
    # dp['skip_seconds'] = 2.0

    # Dataset generation (samples at input Fs)
    dg = cfg['dataset_generator']
    dg['window_length'] = 125       # 1 s at 125 Hz
    dg['first_ms_to_drop'] = 125    # ~1 s initial drop (units are input samples)

    # Augmentation (resample window to model Fs)
    aug = cfg['augmentation']
    aug['new_sampling_frequency'] = 100

    # 2) Create h5 dataset
    h5_dir_abs = os.path.join(REPO, 'data', 'raspy', 'preprocessed_data')
    os.makedirs(h5_dir_abs, exist_ok=True)
    h5_path = os.path.join(h5_dir_abs, args.h5)
    print('Creating h5 at:', h5_path)
    dataset.create_dataset(cfg, h5_path=h5_path)
    print('Created:', h5_path)

    # 3) Inspect shapes from h5
    with h5py.File(h5_path, 'r') as f:
        # Take fold 0 as example
        X = f['0_trials']
        y = f['0_labels']
        print('h5 shapes -> X:', X.shape, 'y:', y.shape)
        n_elec, T = X.shape[1], X.shape[2]
        print('n_electrodes:', n_elec, 'T (time samples):', T)

    # 4) Build model and run a single forward
    if HAS_TORCH:
        from Offline_EEGNet.EEGData import CachedEEGData
        output_dim = len(cfg['dataset_generator']['dataset_operation']['selected_labels'])
        model = EEGNet(cfg['model'], output_dim, n_elec)
        model.eval()

        ds = CachedEEGData(h5_path, [0], train=True)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        xb, yb = next(iter(dl))
        with torch.no_grad():
            probs, logits, hidden = model(xb, return_logits=False, return_dataclass=True)
        print('Forward OK:', xb.shape, '->', probs.shape, logits.shape, hidden.shape)
    else:
        print('Torch not installed; skipped forward pass. Dataset creation verified.')
    print('Verification complete. Offline pipeline is ready for training.')


if __name__ == '__main__':
    main()
