import numpy as np
import tensorflow as tf
import mne
mne.set_log_level('ERROR')

from load_data_v6 import load_all_subjects

X, y, _ = load_all_subjects("/Users/carterlawrence/Downloads/files", task="LR")
print(f"X shape: {X.shape}  y distribution: {np.bincount(y)}")

model = tf.keras.models.load_model("eegnet_LR_4.h5")
probs = model.predict(X, batch_size=128, verbose=0).squeeze()

print(f"\nProb stats on eval:")
print(f"  min = {probs.min():.4f}")
print(f"  max = {probs.max():.4f}")
print(f"  mean = {probs.mean():.4f}")
print(f"  median = {np.median(probs):.4f}")
print(f"  pct < 0.1 = {(probs < 0.1).mean():.3f}")
print(f"  pct in [0.1, 0.9] = {((probs >= 0.1) & (probs <= 0.9)).mean():.3f}")
print(f"  pct > 0.9 = {(probs > 0.9).mean():.3f}")

# Per-class prob distribution
print(f"\nLEFT (true label=0) prob: mean={probs[y==0].mean():.3f}  median={np.median(probs[y==0]):.3f}")
print(f"RIGHT (true label=1) prob: mean={probs[y==1].mean():.3f}  median={np.median(probs[y==1]):.3f}")