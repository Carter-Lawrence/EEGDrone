import numpy as np
import tensorflow as tf
import mne
from keras.models import Model
from keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D, AveragePooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation)
from keras.constraints import max_norm
from keras.utils import to_categorical
import asrpy
import os
# ----------------------------
# PARAMETERS
# ----------------------------
sfreq = 256

base_path = "/Users/carterlawrence/Downloads/"
save_base = f"{base_path}preprocessed_eeg_V3"
wanted_runs = ["R04","R08", "R12"]
mne.set_log_level('ERROR')

all_X, all_y, all_subjects = [], [], []


for subj in range(1,2):  # exclude subj 109 for testing
    subj_str = f"S{str(subj).zfill(3)}" 
    subj_folder = f"{base_path}files/{subj_str}"
    save_folder = f"{save_base}/{subj_str}"
    os.makedirs(save_folder, exist_ok=True)

    # baseline
    file_baseline = f"{subj_folder}/{subj_str}R02.edf"
    raw_baseline = mne.io.read_raw_edf(file_baseline, preload=True, verbose=False)
    raw_baseline.resample(sfreq, verbose=False)
    raw_baseline.set_eeg_reference('average', verbose=False)
    raw_baseline.filter(8., 30., fir_design='firwin', verbose=False)
# ensure block-aligned data
    raw_baseline.crop(tmax=(raw_baseline.n_times // 4096 * 4096 - 1) / raw_baseline.info['sfreq'])
    asr = asrpy.ASR(sfreq=sfreq, cutoff=10)
    asr.fit(raw_baseline)
    print(f"{subj_str}...")
    for run in wanted_runs:
        file = f"{subj_folder}/{subj_str}{run}.edf"
        save_file = f"{save_folder}/{subj_str}{run}_clean_raw.fif"
        # --- Check if preprocessed file already exists ---
        if os.path.exists(save_file):
            print("already preprocessed")
        else:
            try:
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                raw.resample(sfreq, verbose=False)
                raw.set_eeg_reference('average', verbose=False)
                raw.filter(8., 30., fir_design='firwin', verbose=False)
                raw_clean = asr.transform(raw)

                # Save preprocessed file
                raw_clean.save(save_file, overwrite=True)

            except Exception as e:
                print(f"Error in {subj_str}{run}: {e}")
                continue
            '''
        try:
            events, event_dict = mne.events_from_annotations(raw_clean, event_id=event_id)
            epochs = mne.Epochs(raw_clean, events, event_id=event_id,
                                tmin=tmin, tmax=tmax, baseline=baseline,
                                preload=True, reject=reject_criteria)

            if len(epochs) == 0:
                print(f"No epochs for {subj_str}{run}")
                continue

            X = epochs.get_data()
            y = epochs.events[:, -1]

            # z-score normalization per channel
            X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-6)

            all_X.append(X)
            all_y.append(y)
            all_subjects.append(np.full(len(y), subj))
        except Exception as e:
            print(f"Error epoching {subj_str}{run}: {e}")
            continue
            '''
    