import mne
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load raw EEG
# -------------------------------
raw = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/S001R04.edf", preload=True)

# -------------------------------
# 2. Preprocessing
# -------------------------------
# Set reference
raw.set_eeg_reference('average')

# Resample to speed things up
sfreq = 256
raw.resample(sfreq)

# Notch filter to remove line noise
raw.notch_filter(freqs=[60, 120])

# Bandpass filter for mu/beta rhythm
raw.filter(8., 30., fir_design='firwin')

# -------------------------------
# 3. ICA artifact removal
# -------------------------------
ica = mne.preprocessing.ICA(n_components=20, random_state=97, method='fastica')
ica.fit(raw.copy().filter(1., None))  # wide-band for ICA

# Inspect ICA sources and manually mark bad components
ica.plot_sources(raw)
ica.exclude = []  # manually mark components to exclude, e.g. [0,3]

# Apply ICA to clean data
raw_clean = ica.apply(raw.copy())

# Select only relevant channels
plot_channels = ['C3..', 'C4..', 'Cz..', 'Fcz.', 'Fc3.', 'Fc4.', 'F3..', 'F4..', 'Fz..', 'Iz..']
raw_clean.pick_channels(plot_channels)

# -------------------------------
# 4. Extract events and epoch
# -------------------------------
event_id = {'T0': 0, 'T1': 1, 'T2': 2}
events, _ = mne.events_from_annotations(raw_clean, event_id=event_id)

tmin, tmax = 0., 1.0  # epoch window
epochs = mne.Epochs(raw_clean, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    baseline=None, preload=True)

# Reject noisy epochs
reject_criteria = dict(eeg=200e-6)  # 200 µV
epochs.drop_bad(reject=reject_criteria)

# Get data and labels
X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, -1]  # labels

print("Epochs shape:", X.shape)
print("Labels shape:", y.shape)

# -------------------------------
# 5. CSP + SVM classifier
# -------------------------------
n_csp_components = 4
csp = CSP(n_components=n_csp_components, reg=None, log=True, norm_trace=False)
svm = SVC(kernel='linear', class_weight='balanced')

clf = Pipeline([
    ('CSP', csp),
    ('Scaler', StandardScaler()),
    ('SVM', svm)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

# Fit classifier
clf.fit(X_train, y_train)

# Evaluate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
cv_scores = cross_val_score(clf, X, y, cv=5)

print(f"Train accuracy: {train_acc:.2f}")
print(f"Test accuracy: {test_acc:.2f}")
print(f"5-fold CV accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")



# -------------------------------
# 6. Real-time usage (pseudo)
# -------------------------------
# new_epoch_preprocessed -> shape (n_channels, n_times)
# predicted_label = clf.predict(new_epoch_preprocessed[np.newaxis, :, :])
