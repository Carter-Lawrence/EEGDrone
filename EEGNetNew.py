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
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Lambda, Dense, Reshape, Multiply
from keras import backend as K
from keras.layers import LSTM, Bidirectional
import numpy as np
import tensorflow as tf
import mne
from keras.models import Model
from keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D, AveragePooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation)
from keras.constraints import max_norm
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sfreq = 256
target_sfreq = 256
tmin, tmax = -0.25, 1
baseline = (0, 0)
reject_criteria = dict(eeg=10)
event_id = {'T1': 1, 'T2': 1, 'T0': 0}

base_path = "/Users/carterlawrence/Downloads/files"
wanted_runs = ["R04", "R08","R12"]
mne.set_log_level('ERROR')

all_X, all_y, all_subjects = [], [], []
save_base = f"{base_path}preprocessed_eeg_V2"


for subj in range(1, 109):  # exclude subj 109 for testing
    subj_str = f"S{str(subj).zfill(3)}"
    subj_folder = f"{base_path}/{subj_str}"
    for run in wanted_runs:
        file = f"{subj_folder}/{subj_str}{run}.edf"
        raw_clean = mne.io.read_raw_edf(file, preload=True)
        raw_clean.resample(target_sfreq)
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

            all_X.append(X)
            all_y.append(y)
            all_subjects.append(np.full(len(y), subj))
        except Exception as e:
            print(f"Error epoching {subj_str}{run}: {e}")
            continue


            # z-score normalization per channel

X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
subjects = np.concatenate(all_subjects)#does nothing lol
X_mean = X.mean(axis=(0, 2), keepdims=True)
X_std = X.std(axis=(0, 2), keepdims=True) + 1e-6
X = (X - X_mean) / X_std
X_all = X_all[..., np.newaxis]
y_all_cat = to_categorical(y_all, num_classes=2)  # adjust num_classes if needed
#breakpoint()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_flat = X_all.reshape(len(X_all), -1)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_flat, y_all)

pred = clf.predict(X_flat)
print("Baseline accuracy:", accuracy_score(y_all, pred))



# for i in range(10):
#     plt.figure(figsize=(10, 6))
    
#     offset = 0  # vertical spacing between channels
#     for ch in range(64):
#         plt.plot(
#             X_all[i, ch, :, 0] + ch * offset
#         )
    
#     plt.title(f'Epoch {i} | Label = {y_all[i]}')
#     plt.xlabel('Time (samples)')
#     plt.tight_layout()
#     plt.show()





# def SimpleEEGNet(
#     num_channels=64,
#     time_samples=256,
#     dropout_rate=0.5
# ):
#     input_layer = Input(shape=(num_channels, time_samples, 1))

#     # ----- Block 1: Temporal Convolution -----
#     x = Conv2D(
#         filters=8,
#         kernel_size=(1, 64),
#         padding='same',
#         use_bias=False
#     )(input_layer)
#     x = BatchNormalization()(x)
#     x = Activation('elu')(x)

#     # ----- Block 2: Spatial (Depthwise) Convolution -----
#     x = DepthwiseConv2D(
#         kernel_size=(num_channels, 1),
#         depth_multiplier=2,
#         use_bias=False,
#         depthwise_constraint=tf.keras.constraints.max_norm(1.0)
#     )(x)
#     x = BatchNormalization()(x)
#     x = Activation('elu')(x)
#     x = AveragePooling2D(pool_size=(1, 4))(x)
#     x = Dropout(dropout_rate)(x)

#     # ----- Block 3: Separable Convolution -----
#     x = SeparableConv2D(
#         filters=16,
#         kernel_size=(1, 16),
#         padding='same',
#         use_bias=False
#     )(x)
#     x = BatchNormalization()(x)
#     x = Activation('elu')(x)

#     # ----- Classifier -----
#     x = Flatten()(x)
#     output = Dense(1, activation='sigmoid')(x)

#     model = Model(inputs=input_layer, outputs=output)
#     return model

# # Ensure channel-last + singleton dimension
# if X_all.ndim == 3:
#     X_all = X_all[..., np.newaxis]  # (N, 64, T, 1)

# # Convert labels to float
# y_all = y_all.astype(np.float32)

# # Train / validation split
# X_train, X_val, y_train, y_val = train_test_split(
#     X_all,
#     y_all,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_all
# )


# # model = SimpleEEGNet(
# #     num_channels=64,
#     time_samples=X_train.shape[2],
#     dropout_rate=0.5
# )


# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )


# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=100,
#     batch_size=32,
#     verbose=1
# )

# model.summary()
