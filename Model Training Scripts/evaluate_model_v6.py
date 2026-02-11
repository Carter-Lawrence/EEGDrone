import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from load_data_v6 import load_all_subjects
import mne

TASK = "LR"   # "MR" or "LR"
mne.set_log_level('ERROR')
tf.config.set_visible_devices([], 'GPU')
# ------------------------
# LOAD DATA
# ------------------------
X, y, subjects = load_all_subjects("/Users/carterlawrence/Downloads/files")

if TASK == "MR":
    y_bin = (y != 0).astype(int)
elif TASK == "LR":
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y_bin = (y[mask] == 1).astype(int)  # LEFT=1, RIGHT=0
    subjects = subjects[mask]

X = X[..., np.newaxis]

# ------------------------
# LOAD MODEL
# ------------------------
model = tf.keras.models.load_model("eegnet_LR_4.h5")

# ------------------------
# PREDICT
# ------------------------
probs = model.predict(X, batch_size=128)
preds = (probs > 0.5).astype(int).squeeze()

# ------------------------
# DISTRIBUTIONS (VERY IMPORTANT)
# ------------------------
print("\n===== CLASS DISTRIBUTIONS =====")
print("True labels:", np.bincount(y_bin))
print("Predictions:", np.bincount(preds))

# ------------------------
# CONFUSION MATRIX
# ------------------------
cm = confusion_matrix(y_bin, preds)

print("\n===== CONFUSION MATRIX =====")
print(cm)

# ------------------------
# PER-CLASS ACCURACY
# ------------------------
print("\n===== PER-CLASS ACCURACY =====")
for i in range(cm.shape[0]):
    acc = cm[i, i] / cm[i].sum()
    label = "Class " + str(i)
    if TASK == "MR":
        label = "REST" if i == 0 else "MOVE"
    if TASK == "LR":
        label = "RIGHT" if i == 0 else "LEFT"
    print(f"{label} accuracy: {acc:.3f}")

# ------------------------
# FULL REPORT
# ------------------------
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_bin, preds, digits=4))
