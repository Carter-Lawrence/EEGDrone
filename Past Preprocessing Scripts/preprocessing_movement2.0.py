import numpy as np
import mne
import os

# --------------------
# PARAMETERS
# --------------------
SFREQ = 256
WINDOW = 2.0      # seconds used for training
DELAY = 0.5       # ignore cue onset transient
SAMPLES = int(SFREQ * WINDOW)

CHANNELS = [
    'C3..','C4..','Cz..',
    'Cp3.','Cp4.',
    'Fc3.','Fc4.',
    'Fcz.'
]

LABEL_MAP = {'T0':0, 'T1':1, 'T2':2}  # rest, left, right

BASE = "/Users/carterlawrence/Downloads/"
RAW_PATH = BASE + "files"
SAVE_PATH = BASE + "eeg_segments_v6"
os.makedirs(SAVE_PATH, exist_ok=True)

mne.set_log_level('ERROR')

# --------------------
# LOOP SUBJECTS
# --------------------
for subj in range(1, 109):
    sid = f"S{str(subj).zfill(3)}"
    subj_dir = f"{RAW_PATH}/{sid}"
    
    X, y, subjects = [], [], []
    print("processing", sid)

    for file in sorted(os.listdir(subj_dir)):
        if not file.endswith(".edf"): 
            continue
        
        raw = mne.io.read_raw_edf(f"{subj_dir}/{file}", preload=True, verbose=False)

        # pick channels
        available = [ch for ch in CHANNELS if ch in raw.ch_names]
        raw.pick_channels(available)

        # resample + reference + filter
        if raw.info['sfreq'] != SFREQ:
            raw.resample(SFREQ)
        raw.set_eeg_reference('average')
        raw.filter(8., 30.)

        data = raw.get_data()
        sfreq = raw.info['sfreq']

        # segment around cues
        for ann in raw.annotations:
            if ann['description'] not in LABEL_MAP:
                continue
            
            start = int((ann['onset'] + DELAY) * sfreq)
            end = start + SAMPLES
            if end > data.shape[1]:
                continue
            
            seg = data[:, start:end]
            X.append(seg)
            y.append(LABEL_MAP[ann['description']])
            subjects.append(sid)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    subjects = np.array(subjects)

    np.savez_compressed(f"{SAVE_PATH}/{sid}.npz", X=X, y=y, subjects=subjects)

print("DONE preprocessing")
