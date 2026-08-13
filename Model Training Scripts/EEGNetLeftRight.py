# REAL-TIME EEGNet BINARY BCI  —  LEFT vs RIGHT HAND  (6-channel)
# CRITICAL FIX: PhysioNet EEGMMI has 4 MI task types that share T1/T2 labels:
#   runs R03, R07, R11:  executed  left fist (T1) vs right fist (T2)   ← keep
#   runs R04, R08, R12:  imagined  left fist (T1) vs right fist (T2)   ← keep
#   runs R05, R09, R13:  executed  both fists (T1) vs both feet (T2)   ← SKIP
#   runs R06, R10, R14:  imagined  both fists (T1) vs both feet (T2)   ← SKIP
# Previous training mixed all of them, so "LEFT" was polluted with
# "both fists" trials and "RIGHT" with "both feet" trials — that's what
# caused the inference-time LEFT-bias.
# Also includes:
#   - Overlapping sliding windows  (≈4× more training data per trial)
#   - On-the-fly augmentation       (time-shift + gaussian noise)
#   - Mixup                         (soft labels, smoother decision boundary)

import os
import re
import numpy as np
import tensorflow as tf
import mne

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D,
                                     SeparableConv2D, AveragePooling2D,
                                     Dropout, Dense, Flatten,
                                     BatchNormalization, Activation, Add)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

mne.set_log_level('ERROR')
tf.config.set_visible_devices([], 'GPU')

# Configuration
PICK_CHANNELS = ["Fc3.", "Fcz.", "Fc4.", "C3..", "Cz..", "C4.."]
SFREQ         = 160
SEGMENT_LEN   = 640
TRIAL_LEN     = int(4.0 * SFREQ)
STRIDE        = int(0.25 * SFREQ)
BATCH_SIZE    = 64
EPOCHS        = 150

# Only runs where T1 = left hand, T2 = right hand
LR_RUNS = {3, 4, 7, 8, 11, 12}

AUG_TIME_SHIFT    = 50
AUG_NOISE_STDDEV  = 0.02
MIXUP_ALPHA       = 0.0


# Data loading  (LR-only runs, overlapping windows)
def _run_number(filename: str) -> int:
    m = re.search(r'R(\d+)\.edf$', filename)
    return int(m.group(1)) if m else -1


def load_lr_subjects(root, segment_len=SEGMENT_LEN,
                     trial_len=TRIAL_LEN, stride=STRIDE):
    """Load only hand-vs-hand runs.  T1 → LEFT (0), T2 → RIGHT (1)."""
    X, y, subjects = [], [], []
    label_map = {'T1': 0, 'T2': 1}   # LEFT=0, RIGHT=1

    n_windows_total = 0
    n_runs_kept, n_runs_skipped = 0, 0

    for subj in sorted(s for s in os.listdir(root) if s.startswith('S')):
        subj_dir = os.path.join(root, subj)
        print(subj)
        for run in sorted(f for f in os.listdir(subj_dir) if f.endswith('.edf')):
            if _run_number(run) not in LR_RUNS:
                n_runs_skipped += 1
                continue
            n_runs_kept += 1

            raw = mne.io.read_raw_edf(os.path.join(subj_dir, run),
                                      preload=True, verbose=False)

            if not all(c in raw.ch_names for c in PICK_CHANNELS):
                missing = set(PICK_CHANNELS) - set(raw.ch_names)
                print(f"  [warn] {run}: missing {missing}, skipping")
                continue
            raw.pick(PICK_CHANNELS)

            raw.filter(8., 30., verbose=False)
            data  = raw.get_data()
            sfreq = raw.info['sfreq']

            for ann in raw.annotations:
                if ann['description'] not in label_map:
                    continue  # skip T0 (rest) — only want L vs R

                trial_start = int(ann['onset'] * sfreq)
                trial_end   = trial_start + trial_len
                if trial_end > data.shape[1]:
                    continue

                last_start = trial_end - segment_len
                for s in range(trial_start, last_start + 1, stride):
                    seg = data[:, s:s + segment_len]
                    seg = (seg - seg.mean(axis=1, keepdims=True)) / (
                            seg.std(axis=1, keepdims=True) + 1e-6)
                    X.append(seg)
                    y.append(label_map[ann['description']])
                    subjects.append(subj)
                    n_windows_total += 1

    print(f"[Data] Runs kept: {n_runs_kept}, skipped (fists/feet): {n_runs_skipped}")
    print(f"[Data] {n_windows_total} windows total "
          f"(stride={stride}, window={segment_len})")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int32)
    subjects = np.array(subjects)
    return X, y, subjects


# EEGNet model architecture
def EEGNet_V6(chans, samples):
    inp = Input(shape=(chans, samples, 1))

    x = Conv2D(32, (1, 64), padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = DepthwiseConv2D((chans, 1), depth_multiplier=2,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(0.4)(x)

    x = SeparableConv2D(64, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(0.4)(x)

    res = SeparableConv2D(64, (1, 8), padding='same')(x)
    res = BatchNormalization()(res)
    res = Activation('elu')(res)
    x = Add()([x, res])

    x = Flatten()(x)
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inp, out)


# Data augmentation functions
def augment(x, y):
    shift = tf.random.uniform([], -AUG_TIME_SHIFT, AUG_TIME_SHIFT + 1,
                              dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=1)
    noise = tf.random.normal(tf.shape(x), stddev=AUG_NOISE_STDDEV,
                             dtype=x.dtype)
    x = x + noise
    return x, y


def mixup_batch(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y

    batch_size = tf.shape(x)[0]
    gamma1 = tf.random.gamma([batch_size], alpha)
    gamma2 = tf.random.gamma([batch_size], alpha)
    lam    = gamma1 / (gamma1 + gamma2 + 1e-8)
    lam    = tf.maximum(lam, 1.0 - lam)

    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])

    idx = tf.random.shuffle(tf.range(batch_size))
    x2  = tf.gather(x, idx)
    y2  = tf.gather(y, idx)

    x_mix = lam_x * x + (1.0 - lam_x) * x2
    y_mix = lam_y * tf.cast(y, tf.float32) + \
            (1.0 - lam_y) * tf.cast(y2, tf.float32)
    return x_mix, y_mix



# Load data
DATA_ROOT = "/Users/carterlawrence/Downloads/files"

X_all, y_lr, subject_ids = load_lr_subjects(DATA_ROOT)
print(f"[Data] X shape = {X_all.shape}  (expect C={len(PICK_CHANNELS)})")

# Labels: LEFT=0, RIGHT=1  (matches T1→0, T2→1)
y_binary = y_lr.astype(np.int32)[:, np.newaxis]
print(f"[Labels] LEFT(0): {(y_lr == 0).sum()}   RIGHT(1): {(y_lr == 1).sum()}")

# Subject-wise split
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, val_idx = next(gss.split(X_all, y_binary, groups=subject_ids))

X_train, X_val = X_all[train_idx], X_all[val_idx]
y_train, y_val = y_binary[train_idx], y_binary[val_idx]
print(f"[Split] train={len(X_train)}  val={len(X_val)}")
print(f"[Split] train-LEFT={ (y_train==0).sum() }  train-RIGHT={ (y_train==1).sum() }")
print(f"[Split] val-LEFT={   (y_val==0).sum()   }  val-RIGHT={   (y_val==1).sum()   }")

# Datasets
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (tf.data.Dataset
            .from_tensor_slices((X_train, y_train))
            .shuffle(2048)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .map(lambda x, y: mixup_batch(x, y, MIXUP_ALPHA),
                 num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset
          .from_tensor_slices((X_val, y_val))
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

# Train model
model = EEGNet_V6(chans=X_all.shape[1], samples=X_all.shape[2])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.BinaryAccuracy(name='acc'),
    ]
)

checkpoint = ModelCheckpoint(
    "eegnet_LR_best.h5",
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

callbacks = [
    EarlyStopping(monitor="val_auc", mode="max",
                  patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max",
                      patience=6, factor=0.5, min_lr=1e-5),
    checkpoint,
]

try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
except KeyboardInterrupt:
    print("\n[Train] Interrupted — saving current weights.")
finally:
    model.save("eegnet_LR_4.h5")
    print("Saved: eegnet_LR_4.h5")

# Validation diagnostics
probs = model.predict(X_val, batch_size=128, verbose=0)
print("\n===== VAL THRESHOLD SWEEP =====")
print(f"Prob distribution on val: mean={probs.mean():.3f}  "
      f"median={np.median(probs):.3f}  "
      f"pct>0.5={(probs>0.5).mean():.3f}")
for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]:
    preds = (probs > t).astype(int)
    acc = (preds == y_val).mean()
    print(f"Threshold {t:.2f} → accuracy {acc:.3f}")
