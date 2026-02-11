import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from eegnet_v6 import EEGNet_V6
from load_data_v6 import load_all_subjects

tf.config.set_visible_devices([], 'GPU')

X, y, subjects = load_all_subjects("/Users/carterlawrence/Downloads/files")

# ------------------------
# CHOOSE TASK
# ------------------------

TASK = "LR"   # "MR" or "LR"

if TASK == "MR":
    y_bin = (y != 0).astype(int)
elif TASK == "LR":
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y_bin = (y[mask] == 1).astype(int)
    subjects = subjects[mask]

# reshape for CNN
X = X[..., np.newaxis]

# ------------------------
# SUBJECT-WISE SPLIT
# ------------------------
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y_bin, groups=subjects))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y_bin[train_idx], y_bin[val_idx]

# ------------------------
# CLASS WEIGHTS
# ------------------------
cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
cw = {0:cw[0], 1:cw[1]}

# ------------------------
# MODEL
# ------------------------
model = EEGNet_V6(X.shape[1], X.shape[2])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC()]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    class_weight=cw,
    callbacks=callbacks,
    verbose=1
)

model.save(f"EEGNet_{TASK}_V6.h5")
