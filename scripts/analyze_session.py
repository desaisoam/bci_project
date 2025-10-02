#!/usr/bin/env python3
"""
Analyze a recorded RASPy session (task.bin + eeg.bin) thoroughly.

Usage examples:
  python scripts/analyze_session.py --latest
  python scripts/analyze_session.py --session data/raspy/exp_2025-10-02_14-14-07

Outputs a report covering:
  - File presence and sizes
  - Task stream rate, label distribution, transition counts
  - EEG stream rate, per‑channel basic stats, counter step summary
  - Cross‑stream alignment sanity (eeg_step monotonicity and endpoints)

This does not modify data. It reads binary logs using Offline_EEGNet/shared_utils.
"""

import argparse
import os
import sys
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_UTILS = os.path.join(REPO_ROOT, 'Offline_EEGNet', 'shared_utils')
sys.path.insert(0, SHARED_UTILS)
try:
    # lazy import utils from shared_utils
    import utils as su
except Exception as e:
    print('[FAIL] Could not import Offline_EEGNet/shared_utils/utils.py:', e)
    sys.exit(2)


def human_bytes(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def list_sessions(data_dir):
    base = os.path.join(data_dir, 'raspy')
    if not os.path.isdir(base):
        return []
    return sorted(
        [os.path.join(base, d) for d in os.listdir(base) if d.startswith('exp_')],
        key=lambda p: os.path.getmtime(p)
    )


def rate_from_time_ns(ts):
    if len(ts) < 3:
        return 0.0
    diffs = np.diff(ts.astype(np.int64)) / 1e9  # seconds
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    med = np.median(diffs)
    mean = np.mean(diffs)
    return 1.0/med if med > 0 else (1.0/mean if mean > 0 else 0.0)


def print_header(title):
    print('\n' + '='*len(title))
    print(title)
    print('='*len(title))


def analyze(session_dir):
    # Validate presence of files
    task_path = os.path.join(session_dir, 'task.bin')
    eeg_path = os.path.join(session_dir, 'eeg.bin')
    if not os.path.isfile(task_path) or not os.path.isfile(eeg_path):
        print('[FAIL] Missing task.bin or eeg.bin in', session_dir)
        sys.exit(2)
    print_header('Files')
    print('task.bin:', human_bytes(os.path.getsize(task_path)))
    print('eeg.bin :', human_bytes(os.path.getsize(eeg_path)))

    # Load streams
    print_header('Load Streams')
    task = su.read_data_file_to_dict(task_path)
    eeg = su.read_data_file_to_dict(eeg_path)
    print('Loaded task keys:', list(task.keys()))
    print('Loaded eeg keys :', list(eeg.keys()))

    # Task stats
    print_header('Task Stream')
    tn = task['time_ns']
    t_rate = rate_from_time_ns(tn)
    print(f'Task rows: {len(tn)}, approx rate: {t_rate:.2f} Hz')
    if 'state_task' in task:
        st = task['state_task'].astype(int)
        vals, cnts = np.unique(st, return_counts=True)
        print('state_task distribution:', dict(zip(vals.tolist(), cnts.tolist())))
        # transitions
        if len(st) > 1:
            n_trans = int(np.sum(np.diff(st) != 0))
            print('state_task transitions:', n_trans)
    if 'eeg_step' in task:
        es = task['eeg_step'].astype(int)
        mono = np.all(np.diff(es) >= 0)
        print('eeg_step monotonic:', bool(mono), 'min(max):', int(es.min()) if len(es) else 0, int(es.max()) if len(es) else 0)

    # EEG stats
    print_header('EEG Stream')
    en = eeg['time_ns']
    e_rate = rate_from_time_ns(en)
    print(f'EEG rows: {len(en)}, approx rate: {e_rate:.2f} Hz')
    # Convert structured (N, 17) fields to ndarray
    try:
        sig = np.vstack(eeg['eegbuffersignal'])  # (N, 17)
        dbf = np.vstack(eeg['databuffer'])       # (N, 17)
        print('eegbuffersignal shape:', sig.shape, 'databuffer shape:', dbf.shape)
        # channel stats (first 16 columns)
        ch_mean = sig[:, :16].mean(axis=0)
        ch_std = sig[:, :16].std(axis=0)
        print('EEG ch mean (first 8):', np.round(ch_mean[:8], 6).tolist())
        print('EEG ch std  (first 8):', np.round(ch_std[:8], 6).tolist())
        # Counter step summary
        if sig.shape[1] >= 17:
            diffs = np.diff(sig[:, -1].astype(np.int64))
            if diffs.size:
                unique, counts = np.unique(diffs, return_counts=True)
                step_map = {int(u): int(c) for u, c in zip(unique, counts)}
                print('Last-column counter diff histogram (top):', step_map)
    except Exception as e:
        print('[WARN] Could not stack eegbuffersignal/databuffer:', e)

    # Cross-stream endpoint sanity
    print_header('Cross-Stream Alignment')
    if 'eeg_step' in task:
        es = task['eeg_step'].astype(int)
        last_es = int(es.max()) if len(es) else 0
        n_eeg = len(eeg['time_ns'])
        print('task.eeg_step last:', last_es, '| EEG rows:', n_eeg, '| diff:', n_eeg - last_es)
        if abs(n_eeg - last_es) > int(0.25 * e_rate):
            print('[WARN] Last eeg_step differs notably from EEG rows; minor drift is normal, large gaps warrant review')

    print_header('Summary')
    print('- Task and EEG streams loaded and parsed')
    print('- Task rate ~{:.2f} Hz, EEG rate ~{:.2f} Hz'.format(t_rate, e_rate))
    print('- state_task present with {} unique values'.format(len(np.unique(task['state_task'].astype(int))) if 'state_task' in task else 0))
    print('- Counter column is diagnostic only; offline pipeline strips it for openbci16')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', help='Path to exp_* folder (containing task.bin and eeg.bin)')
    ap.add_argument('--latest', action='store_true', help='Analyze most recent folder under data/raspy')
    args = ap.parse_args()

    if not args.session and not args.latest:
        print('Provide --session <path> or --latest')
        sys.exit(2)

    sess = args.session
    if args.latest:
        root = os.path.join(REPO_ROOT, 'data')
        sessions = list_sessions(root)
        if not sessions:
            print('[FAIL] No sessions found under', os.path.join(root, 'raspy'))
            sys.exit(2)
        sess = sessions[-1]

    print('[INFO] Analyzing session:', sess)
    analyze(sess)


if __name__ == '__main__':
    main()

