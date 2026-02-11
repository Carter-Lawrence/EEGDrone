import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     AveragePooling2D, Dropout, Dense, Flatten,
                                     BatchNormalization, Activation)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import mne


# ----------------------------
# PARAMETERS
# ----------------------------
sfreq = 256
tmin, tmax = -0.25, 1
baseline = None
reject_criteria = dict(eeg=200e-6)
event_id = {'T1': 1, 'T2': 1, 'T0': 0}

base_path = "/Users/carterlawrence/Downloads/preprocessed_eeg"
wanted_runs = ["R04", "R08","R12"]
mne.set_log_level('ERROR')

# for subj in range(1, 109):  # exclude subj 109 for testing
#     subj_str = f"S{str(subj).zfill(3)}"
#     subj_folder = f"{base_path}/{subj_str}"
#     for run in wanted_runs:
#         file = f"{subj_folder}/{subj_str}{run}_clean_raw.fif"
#         raw_clean = mne.io.read_raw_fif(file, preload=True)
#         try:
#             events, event_dict = mne.events_from_annotations(raw_clean, event_id=event_id)
#             epochs = mne.Epochs(raw_clean, events, event_id=event_id,
#                                 tmin=tmin, tmax=tmax, baseline=baseline,
#                                 preload=True, reject=None)

#             if len(epochs) == 0:
#                 print(f"No epochs for {subj_str}{run}")
#                 continue

#             X = epochs.get_data()
#             y = epochs.events[:, -1]

#             all_X.append(X)
#             all_y.append(y)
#             all_subjects.append(np.full(len(y), subj))
#         except Exception as e:
#             print(f"Error epoching {subj_str}{run}: {e}")
#             continue

def load_all_subjects(root, segment_len=640):
    X, y = [], []
    label_map = {'T0':0, 'T1':1, 'T2':2}

    subjects = sorted([s for s in os.listdir(root) if s.startswith('S')])
    for subj in subjects:
        print(subj)
        subj_dir = os.path.join(root, subj)
        runs = sorted([f for f in os.listdir(subj_dir) if f.endswith('.edf')])

        for run in runs:
            # Skip baseline-only runs
            if run.endswith(('R01.edf', 'R02.edf')):
                continue

            fpath = os.path.join(subj_dir, run)
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
            raw.filter(8., 30., verbose=False)
            data = raw.get_data()  # (64, time)
            sfreq = raw.info['sfreq']

            for ann in raw.annotations:
                label_str = str(ann['description'])
                if label_str not in label_map:
                    continue
                label = label_map[label_str]
                start = int(ann['onset'] * sfreq)
                end = start + segment_len
                if end > data.shape[1]:
                    continue
                segment = data[:, start:end]

                # Normalize per-channel
                segment = (segment - segment.mean(axis=1, keepdims=True)) / (
                          segment.std(axis=1, keepdims=True) + 1e-6)
                X.append(segment)
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X = X[..., np.newaxis]  # shape: (n, 64, segment_len, 1)
    return X, y


def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5):
    # ---- Modifying the structure a bit_length
    
    input_main = Input(shape=(Chans, Samples, 1))
    #breakpoint()
    
    # First part - temporal convolution
    block1 = Conv2D(8, (1, 64), padding='same', use_bias=False)(input_main)
    block1 = BatchNormalization()(block1)
    
    # Second part - Depthwise spatial convolution
    block1 = DepthwiseConv2D((Chans, 1), depth_multiplier=2, use_bias=False, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    # Third part - Separable convolution block
    block1 = SeparableConv2D(16, (1, 16), padding='same', use_bias=False)(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 8))(block1)
    block1 = Dropout(dropoutRate)(block1)
    
    # Last part - Classifier
    flatten = Flatten()(block1)
    # ------- Change from softmax to sigmoid
    outputs = Dense(nb_classes, activation='sigmoid', dtype='float32')(flatten)
    

    return Model(inputs=input_main, outputs=outputs)


X_all, y_all = load_all_subjects("/Users/carterlawrence/Downloads/files")

# --------------
# Modify X_mean and X_std to use X_all instead of X (what it was using before)
X_mean = X_all.mean(axis=(0, 2), keepdims=True)
X_std = X_all.std(axis=(0, 2), keepdims=True) + 1e-6
X_all = (X_all - X_all.mean(axis=2, keepdims=True)) / (X_all.std(axis=2, keepdims=True) + 1e-6)
# -------------- 
# Changing from using categories to using binary 0/1
y_binary = np.where(y_all == 0, 0, 1)  # rest=0, movement=1
y_binary = y_binary[:, np.newaxis]  # shape (n_samples,1)

print(np.unique(y_binary, return_counts=True)
)
model = EEGNet(1, Chans=X_all.shape[1], Samples=X_all.shape[2])

optimizer = Adam(learning_rate=1e-3)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0,1]),
    y=y_binary.squeeze()
)

class_weight = {0: class_weights[0], 1: class_weights[1]}

metrics=[
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall()
]
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=metrics
)

# --- Adding test/train split

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_binary, test_size=0.2, random_state=42, shuffle=True
)

BATCH_SIZE = 64

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 1024, seed = 42).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]

EPOCHS = 100



history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose = 1,
    class_weight = class_weights

)
   