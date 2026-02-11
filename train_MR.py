import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from load_segments import load_segments

DATA_ROOT = "/Users/carterlawrence/Downloads/preprocessed_eeg_V4"

X, y, subjects = load_segments(DATA_ROOT)

# MR labels: rest=0, movement=1
y_mr = (y != 0).astype(int)

# subject-wise split (CRITICAL)
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, val_idx = next(gss.split(X, y_mr, groups=subjects))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y_mr[train_idx], y_mr[val_idx]

print("Train shape:", X_train.shape)

# 🔥 MR-friendly model (simpler than EEGNet)
inputs = tf.keras.Input(shape=X_train.shape[1:])

x = tf.keras.layers.Conv2D(16, (1, 64), padding="same", activation="elu")(inputs)
x = tf.keras.layers.AveragePooling2D((1, 4))(x)

x = tf.keras.layers.Conv2D(32, (1, 32), padding="same", activation="elu")(x)
x = tf.keras.layers.AveragePooling2D((1, 4))(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
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
print("MR Confusion Matrix:\n", cm)

model.save("model_MR.h5")
