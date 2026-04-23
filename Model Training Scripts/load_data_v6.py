"""
load_data_v6.py  —  Shared data loader for evaluation.

Keeps the run filtering in lockstep with the training scripts so the
evaluation sees the same distribution the model was trained on.

Set TASK when calling load_all_subjects(..., task="MR"|"LR"):
  - MR: includes baseline runs R01/R02 as pure rest, plus hand-vs-hand
        runs R03/R04/R07/R08/R11/R12 for both rest (T0) and movement (T1/T2).
  - LR: only hand-vs-hand runs, and only T1 (LEFT=0) / T2 (RIGHT=1).
"""

import os
import re
import mne
import numpy as np

SELECTED_CHANNELS = ["Fc3.", "Fcz.", "Fc4.", "C3..", "Cz..", "C4.."]

# Runs where T1 = left hand, T2 = right hand
HAND_RUNS     = {3, 4, 7, 8, 11, 12}
# Baseline resting runs (R01 = eyes open, R02 = eyes closed)
BASELINE_RUNS = {1, 2}


def _run_number(filename: str) -> int:
    m = re.search(r'R(\d+)\.edf$', filename)
    return int(m.group(1)) if m else -1


def _extract_windows(data, segment_len, stride, label, X, y, subjects, subj):
    """Slide windows across the given data array, z-scoring each window."""
    n = 0
    for s in range(0, data.shape[1] - segment_len + 1, stride):
        seg = data[:, s:s + segment_len]
        seg = (seg - seg.mean(axis=1, keepdims=True)) / (
                seg.std(axis=1, keepdims=True) + 1e-6)
        X.append(seg)
        y.append(label)
        subjects.append(subj)
        n += 1
    return n


def load_all_subjects(root, task="MR", segment_len=640, stride=None):
    """
    Load windows for the given task, run-filtered to match training.

    Parameters
    ----------
    root : path to the PhysioNet EEGMMI root directory
    task : "MR" or "LR"
    segment_len : samples per window (640 = 4.0 s @ 160 Hz)
    stride : window stride in samples. Defaults to segment_len (non-overlap)
             so the eval set matches training's evaluation behaviour.
    """
    assert task in ("MR", "LR"), f"Unknown task {task!r}"
    if stride is None:
        stride = segment_len   # non-overlapping windows for evaluation

    X, y, subjects = [], [], []

    # For MR we label T0/T1/T2; for LR we keep only T1/T2.
    if task == "MR":
        ann_label_map = {'T0': 0, 'T1': 1, 'T2': 1}   # rest vs any movement
    else:   # LR
        ann_label_map = {'T1': 0, 'T2': 1}            # LEFT=0, RIGHT=1

    n_windows_total = 0
    n_runs_kept, n_runs_skipped = 0, 0

    for subj in sorted(s for s in os.listdir(root) if s.startswith('S')):
        subj_dir = os.path.join(root, subj)
        print(subj)
        for run in sorted(f for f in os.listdir(subj_dir) if f.endswith('.edf')):
            run_num = _run_number(run)

            # ── Baseline runs: only used for MR, as pure rest ──
            if run_num in BASELINE_RUNS:
                if task != "MR":
                    n_runs_skipped += 1
                    continue
                raw = mne.io.read_raw_edf(os.path.join(subj_dir, run),
                                          preload=True, verbose=False)
                if not all(c in raw.ch_names for c in SELECTED_CHANNELS):
                    missing = set(SELECTED_CHANNELS) - set(raw.ch_names)
                    print(f"  [warn] {run}: missing {missing}, skipping")
                    n_runs_skipped += 1
                    continue
                raw.pick(SELECTED_CHANNELS)
                raw.filter(8., 30., verbose=False)
                data = raw.get_data()
                n_windows_total += _extract_windows(
                    data, segment_len, stride, 0,
                    X, y, subjects, subj
                )
                n_runs_kept += 1
                continue

            # ── Hand-vs-hand runs: used for both MR and LR ──
            if run_num not in HAND_RUNS:
                n_runs_skipped += 1
                continue

            raw = mne.io.read_raw_edf(os.path.join(subj_dir, run),
                                      preload=True, verbose=False)
            if not all(c in raw.ch_names for c in SELECTED_CHANNELS):
                missing = set(SELECTED_CHANNELS) - set(raw.ch_names)
                print(f"  [warn] {run}: missing {missing}, skipping")
                n_runs_skipped += 1
                continue
            raw.pick(SELECTED_CHANNELS)
            raw.filter(8., 30., verbose=False)
            data  = raw.get_data()
            sfreq = raw.info['sfreq']
            n_runs_kept += 1

            for ann in raw.annotations:
                if ann['description'] not in ann_label_map:
                    continue
                start = int(ann['onset'] * sfreq)
                end   = start + segment_len
                if end > data.shape[1]:
                    continue
                seg = data[:, start:end]
                seg = (seg - seg.mean(axis=1, keepdims=True)) / (
                        seg.std(axis=1, keepdims=True) + 1e-6)
                X.append(seg)
                y.append(ann_label_map[ann['description']])
                subjects.append(subj)
                n_windows_total += 1

    print(f"[Data] Runs kept: {n_runs_kept}, skipped: {n_runs_skipped}")
    print(f"[Data] {n_windows_total} windows total  (task={task})")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]   # (N, C, T, 1)
    y = np.array(y, dtype=np.int32)
    subjects = np.array(subjects)
    return X, y, subjects