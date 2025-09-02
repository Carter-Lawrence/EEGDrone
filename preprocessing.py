import mne

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
# run ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, method='fastica')
ica.fit(raw.copy().filter(1., None))  # wide-band for ICA

# plot components to inspect
ica.plot_components()  # interactive GUI, pick which components to exclude

# manually mark bad components
ica.exclude = [0, 3]  # example: replace with indices of artifacts from plot

# apply ICA
raw_clean = ica.apply(raw.copy())

# --- 4. Epoch data ---
tmin, tmax = -0.5, 3.0  # 0.5s before cue to 3s after
epochs = mne.Epochs(raw_clean, events, event_id=event_id,
                    tmin=tmin, tmax=tmax,
                    baseline=None, preload=True)

# reject noisy epochs
reject_criteria = dict(eeg=200e-6)  # 200 µV threshold
epochs.drop_bad(reject=reject_criteria)

# --- 5. Ready to use ---
print(epochs)
# epochs.get_data() → (n_trials, n_channels, n_times)
# epochs.events[:,2] → labels
