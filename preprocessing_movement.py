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
wanted_runs = ["R03", "R04","R05", "R06", "R07","R08", "R09", "R10", "R11","R12"]
mne.set_log_level('ERROR')

all_X, all_y, all_subjects = [], [], []


for subj in range(1,109):  # exclude subj 109 for testing
    subj_str = f"S{str(subj).zfill(3)}" 
    subj_folder = f"{base_path}files/{subj_str}"
    save_folder = f"{save_base}/{subj_str}"
    os.makedirs(save_folder, exist_ok=True)
# ensure block-aligned data
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
                raw.crop(tmin=1, tmax=raw.times[-1] - 1)

                # Save preprocessed file
                raw.save(save_file, overwrite=True)

            except Exception as e:
                print(f"Error in {subj_str}{run}: {e}")
                continue