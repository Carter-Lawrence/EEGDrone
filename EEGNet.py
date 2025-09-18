import numpy as np
import tensorflow as tf
import mne
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     AveragePooling2D, Dropout, Dense, Flatten, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

sfreq = 256
tmin, tmax = -0.5, 3.0
baseline = (None, 0.5)
reject_criteria = dict(eeg=200e-6)
event_id = {'T1': 1, 'T2': 2}

all_X, all_y, all_subjects = [], [], []

base_path = "/Users/carterlawrence/Downloads/files"
wanted_runs = ["R04", "R08", "R12"]

for subj in range(1, 110):
    subj_folder = f"{base_path}/S{str(subj).zfill(3)}"
    
    for run in wanted_runs:
        file = f"{subj_folder}/S{str(subj).zfill(3)}{run}.edf"
        raw = mne.io.read_raw_edf(file, preload=True)

        # --- Reference, resample, filter ---
        raw.set_eeg_reference('average')
        raw.resample(256)
        raw.filter(8., 30., fir_design='firwin')

        # --- ICA artifact removal without EOG channels ---
        ica = mne.preprocessing.ICA(n_components=20, random_state=97, method='fastica')
        ica.fit(raw.copy().filter(1., None))  # wide-band for ICA

        # automatically find bad components by variance threshold
        # components with unusually high variance are likely artifacts
        component_variances = np.var(ica.get_sources(raw).get_data(), axis=1)
        threshold = np.percentile(component_variances, 95)  # mark top 5% variance as bad
        ica.exclude = list(np.where(component_variances > threshold)[0])

        raw_clean = ica.apply(raw.copy())

        # --- Extract events and epoch ---
        events, _ = mne.events_from_annotations(raw_clean, event_id=event_id)
        epochs = mne.Epochs(raw_clean, events, event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=baseline,
                            preload=True, reject=reject_criteria)
        
        if len(epochs) == 0:
            continue

        # --- Get data & labels ---
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        y = epochs.events[:, -1]

        # Z-score normalization per channel
        X = (X - X.mean(axis=(0,2), keepdims=True)) / X.std(axis=(0,2), keepdims=True)

        # Append to master lists
        all_X.append(X)
        all_y.append(y)
        all_subjects.append(np.full(len(y), subj))  # for LOSO

# Concatenate all subjects
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
subjects = np.concatenate(all_subjects)

# --- Prepare for EEGNet ---
X_all = X_all[..., np.newaxis]               # add channel-last dimension
y_all_cat = to_categorical(y_all, num_classes=3)  # adjust num_classes if needed

def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5):
    input_main = Input(shape=(Chans, Samples, 1))
    
    # Block 1: Temporal convolution
    block1 = Conv2D(8, (1, 64), padding='same', use_bias=False)(input_main)
    block1 = BatchNormalization()(block1)
    
    # Depthwise spatial convolution
    block1 = DepthwiseConv2D((Chans, 1), depth_multiplier=2, use_bias=False)(block1)
    block1 = BatchNormalization()(block1)
    block1 = tf.keras.layers.Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)
    
    # Block 2: Separable convolution
    block2 = SeparableConv2D(16, (1, 16), padding='same', use_bias=False)(block1)
    block2 = BatchNormalization()(block2)
    block2 = tf.keras.layers.Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)
    
    # Classification
    flatten = Flatten()(block2)
    dense = Dense(nb_classes, kernel_constraint=tf.keras.constraints.max_norm(0.25))(flatten)
    softmax = tf.keras.layers.Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

model = EEGNet(nb_classes=y_all_cat.shape[1], Chans=X_all.shape[1], Samples=X_all.shape[2])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

trains = model.fit(X_all, y_all_cat, 
                    batch_size=16, 
                    epochs=100, 
                    validation_split=0.2,
                    verbose=1)

test_loss, test_acc = model.evaluate(X_all, y_all_cat)
print("Test accuracy:", test_acc)
