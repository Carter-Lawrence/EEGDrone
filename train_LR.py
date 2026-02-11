import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from load_segments import load_segments

DATA_ROOT = "/Users/carterlawrence/Downloads/preprocessed_eeg_V4"

X, y, subjects = load_segments(DATA_ROOT)

# keep only left/right
mask = (y == 1) | (y == 2)
X_lr = X[mask]
y_lr = y[mask]
subjects_lr = subjects[mask]

# LEFT=1, RIGHT=0
y_lr_bin = (y_lr == 1).astype(int)

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, val_idx = next(gss.split(X_lr, y_lr_bin, groups=subjects_lr))

X_train, X_val = X_lr[train_idx], X_lr[val_idx]
y_train, y_val = y_lr_bin[train_idx], y_lr_bin[val_idx]

# 🔥 EEGNet-style model (good for spatial patterns)
inputs = tf.keras.Input(shape=X_train.shape[1:])

x = tf.keras.layers.Conv2D(16, (1, 64), padding="same", use_bias=False)(inputs)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.DepthwiseConv2D((X_train.shape[1], 1),
                                    depth_multiplier=2,
                                    use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("elu")(x)
x = tf.keras.layers.AveragePooling2D((1, 4))(x)

x = tf.keras.layers.SeparableConv2D(32, (1, 16), padding="same")(x)
x = tf.keras.layers.Activation("elu")(x)
x = tf.keras.layers.AveragePooling2D((1, 8))(x)

x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC()]
)

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=64,
          verbose=1)

probs = model.predict(X_val)
preds = (probs > 0.5).astype(int).squeeze()

cm = confusion_matrix(y_val, preds)
print("LR Confusion Matrix:\n", cm)

model.save("model_LR.h5")
