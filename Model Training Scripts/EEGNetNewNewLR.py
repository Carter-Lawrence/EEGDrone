# ============================
# REAL-TIME EEGNet BINARY BCI
# ============================

import os
import numpy as np
import tensorflow as tf
import mne

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D,
                                     SeparableConv2D, AveragePooling2D,
                                     Dropout, Dense, Flatten,
                                     BatchNormalization, Activation, Add)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import SpatialDropout2D
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

mne.set_log_level('ERROR')
tf.config.set_visible_devices([], 'GPU')

# ----------------------------
# DATA LOADING
# ----------------------------
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


# ----------------------------
# EEGNET MODEL (REAL-TIME SAFE)
# ----------------------------
def EEGNet_V6(chans, samples):

    inp = Input(shape=(chans, samples, 1))

    # -------- Temporal Convolution (frequency learning)
    x = Conv2D(32, (1, 64), padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # -------- Spatial Convolution (channel interactions)
    x = DepthwiseConv2D((chans, 1), depth_multiplier=2,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(0.4)(x)

    # -------- Spectral abstraction
    x = SeparableConv2D(64, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(0.4)(x)

    # -------- Residual-like block (extra depth)
    res = SeparableConv2D(64, (1, 8), padding='same')(x)
    res = BatchNormalization()(res)
    res = Activation('elu')(res)
    x = Add()([x, res])

    # -------- Classifier
    x = Flatten()(x)
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inp, out)

# ----------------------------
# LOAD DATA
# ----------------------------
DATA_ROOT = "/Users/carterlawrence/Downloads/files"

X_all, y_all, subject_ids = load_all_subjects(DATA_ROOT)

# Binary labels: rest=0, movement=1
# ----------------------------
# LEFT vs RIGHT LABELS
# ----------------------------

mask = (y_all == 1) | (y_all == 2)   # keep only left/right trials
X_all = X_all[mask]
y_lr = y_all[mask]
subject_ids = subject_ids[mask]

# Left = 1, Right = 0
y_binary = (y_lr == 1).astype(np.int32)[:, np.newaxis]


# ----------------------------
# SUBJECT-WISE TRAIN / VAL SPLIT
# ----------------------------
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, val_idx = next(gss.split(X_all, y_binary, groups=subject_ids))

X_train, X_val = X_all[train_idx], X_all[val_idx]
y_train, y_val = y_binary[train_idx], y_binary[val_idx]

# ----------------------------
# CLASS WEIGHTS
# ----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train.squeeze()
)
class_weight = {0: class_weights[0], 1: class_weights[1]}

# ----------------------------
# DATASETS
# ----------------------------
BATCH_SIZE = 64

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = EEGNet_V6(
    chans=X_all.shape[1],
    samples=X_all.shape[2]
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
]

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_binary.squeeze()
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ----------------------------x
# THRESHOLD TUNING (IMPORTANT)
# ----------------------------
probs = model.predict(X_val)

for t in [0.2, 0.3, 0.4, 0.5]:
    preds = (probs > t).astype(int)
    acc = (preds == y_val).mean()
    print(f"Threshold {t:.2f} → accuracy {acc:.3f}")

model.save("eegnet_LR_4.h5")

