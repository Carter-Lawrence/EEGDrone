import mne

# --- 1. Load raw EEG ---
raw = mne.io.read_raw_fif("/Users/carterlawrence/Downloads/S001R02.edf", preload=True)  # or .edf/.bdf/.gdf

# --- 2. Resample ---
if raw.info['sfreq'] > 500:
    raw.resample(256)

# --- 3. Re-reference ---
raw.set_eeg_reference('average')

# --- 4. Notch filter (remove line noise) ---
raw.notch_filter(freqs=[60, 120])

# --- 5. Band-pass filter (keep motor imagery bands) ---
raw.filter(8., 30., fir_design='firwin')

# --- 6. ICA for artifacts ---
ica = mne.preprocessing.ICA(n_components=20, random_state=97, method='fastica')
ica.fit(raw.copy().filter(1., None))  # fit on wide band
eog_inds, _ = ica.find_bads_eog(raw)  # detect eye blinks
ica.exclude = eog_inds
raw = ica.apply(raw)

# --- 7. Extract events from stim channel ---
events = mne.find_events(raw, stim_channel='STI 014')
event_id = {'left_hand': 1, 'right_hand': 2}  # adjust to your markers

# --- 8. Epoch the data ---
epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=-0.5, tmax=3.0,
                    baseline=None, preload=True)

# --- 9. Drop noisy epochs ---
reject_criteria = dict(eeg=200e-6)  # 200 µV threshold
epochs.drop_bad(reject=reject_criteria)

# ✅ Now you have clean, labeled epochs ready for CSP
print(epochs)
