import os
import numpy as np
import mne

sfreq = 256
segment_length = 1.5
segment_samples = int(sfreq * segment_length)

label_map = {'T0':0, 'T1':1, 'T2':2}  # rest, left, right

def load_segments(preprocessed_root):
    X, y, subjects = [], [], []

    for subj in sorted(os.listdir(preprocessed_root)):
        if not subj.startswith("S"):
            continue

        subj_path = os.path.join(preprocessed_root, subj)

        for file in sorted(os.listdir(subj_path)):
            if not file.endswith("_raw.fif"):
                continue

            raw = mne.io.read_raw_fif(os.path.join(subj_path, file),
                                      preload=True, verbose=False)
            data = raw.get_data()

            for ann in raw.annotations:
                if ann["description"] not in label_map:
                    continue

                # 🔥 critical improvement: delay window
                start = int((ann["onset"] + 0.5) * sfreq)
                end = start + segment_samples

                if end > data.shape[1]:
                    continue

                seg = data[:, start:end]   # NO normalization
                X.append(seg)
                y.append(label_map[ann["description"]])
                subjects.append(subj)

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int32)
    subjects = np.array(subjects)

    print("Loaded:", X.shape, y.shape)
    return X, y, subjects
