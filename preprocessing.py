import mne
import matplotlib.pyplot as plt
import numpy as np
# --- 1. Load raw EEG ---
raw = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/S001R04.edf", preload=True)

# map annotation codes to numeric event IDs
event_id = {'T0': 0, 'T1': 1, 'T2': 2}

# convert annotations to events
events, event_dict = mne.events_from_annotations(raw, event_id=event_id)
print("Event dictionary:", event_dict)

# --- 2. Resample & filter ---
raw.resample(256)                # optional, speeds things up
raw.set_eeg_reference('average') # common average reference
raw.notch_filter(freqs=[60, 120])# remove line noise
raw.filter(8., 30., fir_design='firwin')  # keep mu/beta bands

# --- 3. ICA artifact removal ---
ica = mne.preprocessing.ICA(n_components=20, random_state=97, method='fastica')
ica.fit(raw.copy().filter(1., None))  # wide-band for ICA

# Instead of topomap plots (which require digitization),
# just inspect ICA sources as time series:
ica.plot_sources(raw)  

# manually mark bad components (after inspecting plots)
ica.exclude = []  # e.g. [0, 3]

# apply ICA to clean data
raw_clean = ica.apply(raw.copy())

# --- 4. Epoch data ---
tmin, tmax = -0.5, 3.0  # 0.5s before cue to 3s after
epochs = mne.Epochs(raw_clean, events, event_id=event_id,
                    tmin=tmin, tmax=tmax,
                    baseline=None, preload=True)

# reject noisy epochs
reject_criteria = dict(eeg=200e-6)  # 200 µV threshold
epochs.drop_bad(reject=reject_criteria)
epochs.plot(n_epochs=10, n_channels=20, scalings='auto')

# --- 5. Ready to use ---
#print(epochs)
#epochs.get_data() → (n_trials, n_channels, n_times)
#epochs.events[:, 2] → labels

# Get the data: (n_epochs, n_channels, n_times)

# get indices for those channels

# Plot the first epoch across channels
X = epochs.get_data()
y = epochs.events[:, 2]  # labels

# choose channels of interest
picks = ['C3..', 'C4..', 'Cz..', 'Fcz.', 'Fc3.', 'Fc4.', 'F3..', 'F4..', 'Fz.. ', 'Iz..']

# get indices for those channels
pick_idx = [epochs.ch_names.index(ch) for ch in picks if ch in epochs.ch_names]

# loop through epochsx
for epoch_idx in range(len(X)):
    data = X[epoch_idx] * 1e6  # scale to µV

    fig, axes = plt.subplots(4, 4, figsize=(8, 12))  # 5x2 grid for 10 channels
    fig.subplots_adjust(hspace=0.5, wspace=0.3, left=0.05, right=0.95, bottom=0.05, top=0.95)
    axes = axes.flatten()

    for i, ax in enumerate(axes[:len(pick_idx)]):
        ch = pick_idx[i]
        ax.plot(data[ch])
        ax.set_title(f"{epochs.ch_names[ch]} (label={y[epoch_idx]})")
        ax.set_xlim(0, data.shape[1])

    plt.show()
