import mne
import os
import numpy as np
SELECTED_CHANNELS = ['C3..', 'C4..', 'Cz..', 'Fcz.', 'Fc3.', 'Fc4.', 'Fp1.', 'Fp2.']

def load_all_subjects(root, segment_len=640):
    X, y, subjects = [], [], []
    label_map = {'T0':0, 'T1':1, 'T2':2}

    for subj in sorted(s for s in os.listdir(root) if s.startswith('S')):
        subj_dir = os.path.join(root, subj)
        print(subj)
        for run in sorted(f for f in os.listdir(subj_dir) if f.endswith('.edf')):
            if run.endswith(('R01.edf', 'R02.edf')):
                continue

            raw = mne.io.read_raw_edf(os.path.join(subj_dir, run),
                                      preload=True, verbose=False)
            # available_channels = [ch for ch in SELECTED_CHANNELS if ch in raw.ch_names]
            # raw.pick_channels(available_channels, ordered=True)

            raw.filter(8., 30., verbose=False)

            data = raw.get_data()
            sfreq = raw.info['sfreq']

            for ann in raw.annotations:
                if ann['description'] not in label_map:
                    continue

                start = int(ann['onset'] * sfreq)
                end = start + segment_len
                if end > data.shape[1]:
                    continue

                seg = data[:, start:end]
                seg = (seg - seg.mean(axis=1, keepdims=True)) / (
                        seg.std(axis=1, keepdims=True) + 1e-6)

                X.append(seg)
                y.append(label_map[ann['description']])
                subjects.append(subj)

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int32)
    subjects = np.array(subjects)

    return X, y, subjects