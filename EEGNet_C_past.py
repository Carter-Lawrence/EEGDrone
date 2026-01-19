import numpy as np
import tensorflow as tf
import mne
from keras.models import Model
from keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                          AveragePooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation)
from keras.constraints import max_norm
from keras.utils import to_categorical
from keras.layers import (Input, Conv2D, BatchNormalization, DepthwiseConv2D,
                                     Activation, AveragePooling2D, Dropout, SeparableConv2D,
                                     Flatten, Dense, Permute, Reshape, Lambda,
                                     Bidirectional, LSTM, Layer, Multiply, GlobalAveragePooling2D,
                                     GlobalAveragePooling1D)
from keras.constraints import max_norm
from keras.models import Model
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Lambda, Dense, Reshape, Multiply
from keras import backend as K
from keras.layers import LSTM, Bidirectional
import keras.backend as K
# ----------------------------
# PARAMETERS
# ----------------------------
sfreq = 256
tmin, tmax = -0.25, 1
baseline = (0, 0)
reject_criteria = dict(eeg=200e-6)
event_id = {'T1': 0, 'T2': 1}

base_path = "/Users/carterlawrence/Downloads/preprocessed_eeg"
wanted_runs = ["R04", "R08","R12"]
mne.set_log_level('ERROR')

all_X, all_y, all_subjects = [], [], []

for subj in range(1, 109):  # exclude subj 109 for testing
    subj_str = f"S{str(subj).zfill(3)}"
    subj_folder = f"{base_path}/{subj_str}"
    for run in wanted_runs:
        file = f"{subj_folder}/{subj_str}{run}_clean_raw.fif"
        raw_clean = mne.io.read_raw_fif(file, preload=True)
        try:
            events, event_dict = mne.events_from_annotations(raw_clean, event_id=event_id)
            epochs = mne.Epochs(raw_clean, events, event_id=event_id,
                                tmin=tmin, tmax=tmax, baseline=baseline,
                                preload=True, reject=reject_criteria)

            if len(epochs) == 0:
                print(f"No epochs for {subj_str}{run}")
                continue

            X = epochs.get_data()
            y = epochs.events[:, -1]

            # z-score normalization per channel
            X_mean = X.mean(axis=(0, 2), keepdims=True)
            X_std = X.std(axis=(0, 2), keepdims=True) + 1e-6
            X = (X - X_mean) / X_std


            all_X.append(X)
            all_y.append(y)
            all_subjects.append(np.full(len(y), subj))
        except Exception as e:
            print(f"Error epoching {subj_str}{run}: {e}")
            continue

X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
subjects = np.concatenate(all_subjects)
X_all = X_all[..., np.newaxis]
y_all_cat = to_categorical(y_all, num_classes=2)  # adjust num_classes if needed

print("Preprocessing Done")

def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5):
    input_main = Input(shape=(Chans, Samples, 1))
    
    block1 = Conv2D(16, (1, 64), padding='same', use_bias=False, kernel_constraint=max_norm(2.))(input_main)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), depth_multiplier=4, use_bias=False, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(32, (1, 16), padding='same', use_bias=False,
                             depthwise_constraint=max_norm(1.),
                             pointwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2) 
    dense1 = Dense(64, activation='elu', kernel_constraint=max_norm(0.25))(flatten) 
    dense1 = Dropout(0.5)(dense1) 
    
    dense2 = Dense(nb_classes, kernel_constraint=max_norm(0.25))(dense1) 
    softmax = Activation('softmax')(dense2)

    return Model(inputs=input_main, outputs=softmax)

model = EEGNet(nb_classes=y_all_cat.shape[1], Chans=X_all.shape[1], Samples=X_all.shape[2])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_all, y_all_cat, batch_size=32, epochs=100, validation_split=0.2, verbose=1)
test_loss, test_acc = model.evaluate(X_all, y_all_cat)
model.save("eegnet_C_1.h5")
