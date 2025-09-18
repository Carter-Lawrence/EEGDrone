import glob
import mne
import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from joblib import parallel_backend
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


# -------------------------------
# 1. Settings
# -------------------------------
sfreq = 256
tmin, tmax = 0.5, 1
baseline = None
reject_criteria = dict(eeg=200e-6)
event_id = {'T1': 1, 'T2': 2}  # only T1/T2

# -------------------------------
# 2. Load all subjects and runs
# -------------------------------
all_X, all_y, all_subjects = [], [], []

base_path = "/Users/carterlawrence/Downloads/files"
wanted_runs = ["R04", "R08", "R12"]

for subj in range(1, 110):
    subj_folder = f"{base_path}/S{str(subj).zfill(3)}"
    subj_epochs = []

    for run in wanted_runs:
        file = f"{subj_folder}/S{str(subj).zfill(3)}{run}.edf"
        raw = mne.io.read_raw_edf(file, preload=True)

        raw.set_eeg_reference('average')
        raw.resample(sfreq)
        raw.filter(8., 30., fir_design='firwin')

        # Extract events and epoch
        events, _ = mne.events_from_annotations(raw, event_id=event_id)
        epochs = mne.Epochs(raw, events, event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=baseline,
                            preload=True, reject=reject_criteria)
        
        # Skip if no epochs survived
        if len(epochs) == 0:
            continue

        
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        y = epochs.events[:, -1]

        # Z-score normalization per channel for this subject
        X = (X - X.mean(axis=(0,2), keepdims=True)) / X.std(axis=(0,2), keepdims=True)

        all_X.append(X)
        all_y.append(y)
        all_subjects.append(np.full(len(y), subj))  # track subject for LOOCV

# Concatenate
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
subjects = np.concatenate(all_subjects)

print(f"Total epochs: {X_all.shape[0]}, Channels: {X_all.shape[1]}, Times: {X_all.shape[2]}")
print(f"Class distribution: {np.unique(y_all, return_counts=True)}")

# -------------------------------
# 3. Covariance + Tangent Space + SVM
# -------------------------------
cov = Covariances(estimator='lwf', reg=1e-3)  # regularization avoids non-positive definite
print("cov done")
ts = TangentSpace()
print("ts done")
svm = SVC(kernel='linear', class_weight='balanced')
print("svm done")

# Pipeline
clf = Pipeline([
    ('Cov', cov),
    ('TS', ts),
    ('Scaler', StandardScaler()),
    ('SVM', svm)
])

# Leave-One-Subject-Out CV
logo = LeaveOneGroupOut()
print("logo done")
with tqdm_joblib(tqdm(desc="LOSO CV", total=logo.get_n_splits())):
    cv_scores = cross_val_score(clf, X_all, y_all, groups=subjects, cv=logo, n_jobs=-1)
print(f"LOSO CV accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
