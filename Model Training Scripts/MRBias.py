import os
import numpy as np
import tensorflow as tf
import mne
from keras.models import load_model
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
mne.set_log_level('ERROR')
tf.config.set_visible_devices([], 'GPU')
from sklearn.metrics import confusion_matrix

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

DATA_ROOT = "/Users/carterlawrence/Downloads/files"

X_all, y_all, subject_ids = load_all_subjects(DATA_ROOT)
print("test 123")
# ----------------------------
# LOAD MODEL
# ----------------------------
model = load_model("eegnet_M_New.h5")
print("load model")
# ============================
# MOVEMENT vs REST EVALUATION
# ============================

# movement=1, rest=0
y_mr = (y_all != 0).astype(int)

print("predicting movement vs rest")
probs_mr = model.predict(X_all)
preds_mr = (probs_mr > 0.5).astype(int).squeeze()

print("Class distribution (true):", np.bincount(y_mr))
print("Prediction distribution:", np.bincount(preds_mr))

cm_mr = confusion_matrix(y_mr, preds_mr)
print("\nMovement vs Rest Confusion Matrix:")
print(cm_mr)

for i in range(cm_mr.shape[0]):
    acc = cm_mr[i, i] / cm_mr[i].sum()
    label = "REST" if i == 0 else "MOVE"
    print(f"{label} accuracy: {acc:.3f}")
