import numpy as np
import tensorflow as tf
import mne
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     AveragePooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
import asrpy
# ----------------------------
# PARAMETERS
# ----------------------------
sfreq = 256
tmin, tmax = 0.0, 3.0
baseline = (0, 0)
reject_criteria = dict(eeg=200e-6)
event_id = {'T1': 0, 'T2': 1}

base_path = "/Users/carterlawrence/Downloads/files"
wanted_runs = ["R03", "R04", "R07", "R08", "R11", "R12"]
mne.set_log_level('ERROR')

all_X, all_y, all_subjects = [], [], []


for subj in range(1, 109):#esclude subj 109 for testing
    subj_folder = f"{base_path}/S{str(subj).zfill(3)}"
    file_baseline = f"{subj_folder}/S{str(subj).zfill(3)}R02.edf"
    print(file_baseline)
    raw_baseline = mne.io.read_raw_edf(file_baseline, preload=True, verbose=False)
    raw_baseline.resample(256, verbose=False)            # make sure sfreq matches your task
    raw_baseline.set_eeg_reference('average', verbose=False)
    raw_baseline.filter(1., 40., fir_design='firwin', verbose=False)  # optional, wide-band
    asr = asrpy.ASR(sfreq = sfreq,cutoff = 20)
    asr.fit(raw_baseline)                     # fit ASR on baseline
    print(f"{subj}...")
    for run in wanted_runs:
        file = f"{subj_folder}/S{str(subj).zfill(3)}{run}.edf"
        try:
            raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
            raw.resample(sfreq, verbose=False)                   # 1. resample first
            raw.set_eeg_reference('average', verbose=False)     # 2. reference
            raw.filter(8., 30., fir_design='firwin', verbose=False)  # 3. task filter
            raw_clean = asr.transform(raw) #artifact subspacial reconstruction

            events,event_dict = mne.events_from_annotations(raw_clean, event_id=event_id)
            epochs = mne.Epochs(raw_clean, events, event_id=event_id,
                                tmin=tmin, tmax=tmax, baseline=baseline,
                                preload=True, reject=reject_criteria)
            if len(epochs) == 0:
                print("epoch didnt wokr")
                continue
            # get data and labels
            X = epochs.get_data()
            y = epochs.events[:, -1]

            # z-score normalization per channel
            X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-6)
            # append to master lists
            all_X.append(X)
            all_y.append(y)
            all_subjects.append(np.full(len(y), subj))
        except Exception as e:
            print("error")
            continue

# ----------------------------
# STEP 3: Concatenate and Train EEGNet
# ----------------------------
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
subjects = np.concatenate(all_subjects)
X_all = X_all[..., np.newaxis]
y_all_cat = to_categorical(y_all, num_classes=3)  # adjust num_classes if needed

print("Preprocessing Done")

def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5):
    input_main = Input(shape=(Chans, Samples, 1))
    
    block1 = Conv2D(8, (1, 64), padding='same', use_bias=False, kernel_constraint=max_norm(2.))(input_main)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), depth_multiplier=2, use_bias=False, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(16, (1, 16), padding='same', use_bias=False,
                             depthwise_constraint=max_norm(1.),
                             pointwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.25))(flatten)
    softmax = Activation('softmax')(dense)
    return Model(inputs=input_main, outputs=softmax)

model = EEGNet(nb_classes=y_all_cat.shape[1], Chans=X_all.shape[1], Samples=X_all.shape[2])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_all, y_all_cat, batch_size=32, epochs=100, validation_split=0.2, verbose=1)
test_loss, test_acc = model.evaluate(X_all, y_all_cat)
model.save("eegnet_C.h5")
print("Test accuracy:", test_acc)
