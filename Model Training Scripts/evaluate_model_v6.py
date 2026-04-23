"""
evaluate_model_v6.py  —  Offline evaluation for MR or LR models.

Uses the same run filter as training so the eval distribution matches.
Prints per-class accuracy, full classification report, and a balanced-
accuracy threshold sweep to pick a good MOVE_THRESHOLD for BciReplay.
"""

import numpy as np
import tensorflow as tf
import mne
from sklearn.metrics import confusion_matrix, classification_report

from load_data_v6 import load_all_subjects

# ------------------------
# CONFIG
# ------------------------
TASK        = "LR"                 # "MR" or "LR"
MODEL_PATH  = "eegnet_LR_4.h5"     # or eegnet_LR_4.h5
DATA_ROOT   = "/Users/carterlawrence/Downloads/files"

mne.set_log_level('ERROR')
tf.config.set_visible_devices([], 'GPU')

# ------------------------
# LOAD DATA  (run-filtered to match training)
# ------------------------
X, y, subjects = load_all_subjects(DATA_ROOT, task=TASK)
print(f"[Eval] X shape from loader = {X.shape}")

# Binary label is now already set by the loader (0/1)
y_bin = y.astype(int)

# Guard: only add trailing axis if the loader didn't already
if X.ndim == 3:
    X = X[..., np.newaxis]
print(f"[Eval] X shape to model = {X.shape}   (expect (N, 6, 640, 1))")

# ------------------------
# LOAD MODEL
# ------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[Eval] Loaded {MODEL_PATH}  "
      f"(input_shape={model.input_shape})")

# ------------------------
# PREDICT
# ------------------------
probs = model.predict(X, batch_size=128, verbose=0).squeeze()
preds = (probs > 0.5).astype(int)

# ------------------------
# DISTRIBUTIONS
# ------------------------
print("\n===== CLASS DISTRIBUTIONS =====")
print("True labels:", np.bincount(y_bin, minlength=2))
print("Predictions:", np.bincount(preds, minlength=2))

# ------------------------
# CONFUSION MATRIX
# ------------------------
cm = confusion_matrix(y_bin, preds, labels=[0, 1])
print("\n===== CONFUSION MATRIX =====")
print(cm)

# ------------------------
# PER-CLASS ACCURACY
# ------------------------
print("\n===== PER-CLASS ACCURACY (threshold=0.50) =====")
class_names = ("REST", "MOVE") if TASK == "MR" else ("LEFT", "RIGHT")
for i in (0, 1):
    row_sum = cm[i].sum()
    acc = cm[i, i] / row_sum if row_sum else 0.0
    print(f"{class_names[i]} accuracy: {acc:.3f}")

# ------------------------
# FULL REPORT
# ------------------------
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_bin, preds, digits=4,
                            target_names=list(class_names)))

# ------------------------
# BALANCED-ACCURACY THRESHOLD SWEEP
# ------------------------
# On a class-imbalanced eval set, overall accuracy is misleading —
# a model that always predicts the majority class looks "good".
# Balanced accuracy = mean of per-class accuracies, which is what
# you actually care about for a BCI.
print("\n===== BALANCED THRESHOLD SWEEP =====")
print(f"Prob distribution: mean={probs.mean():.3f}  "
      f"median={np.median(probs):.3f}  "
      f"pct>0.5={(probs>0.5).mean():.3f}")
print(f"{'thresh':>6} {'c0_acc':>8} {'c1_acc':>8} "
      f"{'balanced':>10} {'overall':>9}  "
      f"({class_names[0]}/{class_names[1]})")
best_t, best_balanced = 0.5, -1.0
for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    preds_t = (probs > t).astype(int)
    cm_t = confusion_matrix(y_bin, preds_t, labels=[0, 1])
    c0 = cm_t[0, 0] / cm_t[0].sum() if cm_t[0].sum() else 0.0
    c1 = cm_t[1, 1] / cm_t[1].sum() if cm_t[1].sum() else 0.0
    balanced = (c0 + c1) / 2
    overall  = (preds_t == y_bin).mean()
    marker = "  ←" if balanced > best_balanced else ""
    if balanced > best_balanced:
        best_balanced, best_t = balanced, t
    print(f"{t:>6.2f} {c0:>8.3f} {c1:>8.3f} "
          f"{balanced:>10.3f} {overall:>9.3f}{marker}")

print(f"\n[Best] threshold={best_t:.2f}  "
      f"balanced_accuracy={best_balanced:.3f}")
print(f"[Suggest] Use MOVE_THRESHOLD_ON ≈ {best_t + 0.05:.2f}  "
      f"and MOVE_THRESHOLD_OFF ≈ {max(0.0, best_t - 0.05):.2f}  "
      f"(hysteresis gap)")