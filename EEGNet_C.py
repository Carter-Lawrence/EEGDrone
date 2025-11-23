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


class GradientNormEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, threshold=1e-5, patience=3, batch_size=64):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.wait = 0
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        # Take just a small batch from the training data
        x, y = self.model._training_data
        self.x_batch = x[:self.batch_size]
        self.y_batch = y[:self.batch_size]

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            preds = self.model(self.x_batch, training=True)
            loss = self.model.compiled_loss(self.y_batch, preds)

        grads = tape.gradient(loss, self.model.trainable_weights)
        grads = [g for g in grads if g is not None]

        grad_norm = tf.sqrt(sum(tf.reduce_sum(g ** 2) for g in grads))

        print(f"Epoch {epoch+1}: Gradient norm = {grad_norm.numpy():.6e}")

        if grad_norm < self.threshold:
            self.wait += 1
            if self.wait >= self.patience:
                print("GradientNormEarlyStopping: stopping training.")
                self.model.stop_training = True
        else:
            self.wait = 0


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

unique_subjects = np.unique(subjects)
train_subs = unique_subjects[:-5]   # all except last 5 subjects
val_subs = unique_subjects[-5:]
X_train = X_all[np.isin(subjects, train_subs)]
y_train = y_all_cat[np.isin(subjects, train_subs)]
X_val = X_all[np.isin(subjects, val_subs)]
y_val = y_all_cat[np.isin(subjects, val_subs)]


print("Preprocessing Done")

def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5):
    input_main = Input(shape=(Chans, Samples, 1))
    
    block1 = Conv2D(16, (1, 16), 
                    padding='same', 
                    use_bias=False, 
                    kernel_constraint=max_norm(2.))(input_main)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), 
                             depth_multiplier=4,
                             use_bias=False, 
                             depthwise_constraint=max_norm(1.))(block1)
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

    #flatten = Flatten()(block2) 
    flatten = GlobalAveragePooling2D()(block2)
    dense1 = Dense(64, activation='elu', 
                   kernel_constraint=max_norm(0.25))(flatten) 
    dense1 = Dropout(0.5)(dense1) 
    dense2 = Dense(nb_classes, 
                   kernel_constraint=max_norm(0.25))(dense1) 
    softmax = Activation('softmax')(dense2)

    return Model(inputs=input_main, outputs=softmax)

model = EEGNet(nb_classes=y_all_cat.shape[1], Chans=X_all.shape[1], Samples=X_all.shape[2])
optimizer = AdamW(
    learning_rate=3e-4,
    weight_decay=1e-6
)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/gradients")

grad_logger = GradientNormEarlyStopping()
model._training_data = (X_train, y_train)

model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=1000,
    validation_split=0.2,
    callbacks=[grad_logger],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_val, y_val)
model.save("eegnet_C_AdamW.h5")

