import numpy as np
import mne
import os

# ----------------------------
# PARAMETERS
# ----------------------------
sfreq = 256
segment_length = 1.5
segment_samples = int(sfreq * segment_length)

SELECTED_CHANNELS = ['C3..', 'C4..', 'Cz..', 'Fcz.', 'Fc3.', 'Fc4.', 
                     'F3..', 'F4..', 'Fz..', 'Iz..']

base_path = "/Users/carterlawrence/Downloads/"
save_base = f"{base_path}preprocessed_eeg_V4"
wanted_runs = ["R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12"]
mne.set_log_level('ERROR')

# ----------------------------
# STAGE 1: PREPROCESSING ONLY
# Save filtered, segmented data WITHOUT normalization
# ----------------------------

for subj in range(1, 109):
    subj_str = f"S{str(subj).zfill(3)}" 
    subj_folder = f"{base_path}files/{subj_str}"
    save_folder = f"{save_base}/{subj_str}"
    os.makedirs(save_folder, exist_ok=True)
    
    print(f"Processing {subj_str}...")
    
    for run in wanted_runs:
        file = f"{subj_folder}/{subj_str}{run}.edf"
        save_file = f"{save_folder}/{subj_str}{run}_raw.fif"
        
        # Skip if already exists
        if os.path.exists(save_file):
            print(f"  {run} already preprocessed, skipping...")
            continue
            
        try:
            raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
            
            # Channel selection
            available_channels = [ch for ch in SELECTED_CHANNELS if ch in raw.ch_names]

            if len(available_channels) < len(SELECTED_CHANNELS):
                print(f"⚠️ Missing channels in {subj_str}{run}: {set(SELECTED_CHANNELS) - set(available_channels)}")

            raw.pick_channels(available_channels, ordered=True)

            
            # Signal processing
            if raw.info['sfreq'] != sfreq:
                raw.resample(sfreq, verbose=False)
            raw.set_eeg_reference('average', verbose=False)
            raw.filter(8., 30., fir_design='firwin', verbose=False)
            raw.crop(tmin=2, tmax=raw.times[-1] - 2)
            raw.save(save_file, overwrite=True)
            data = raw.get_data()
            print("EEG stats:", data.min(), data.max(), data.std())

            # # Extract segments
            # data = raw.get_data()
            # run_X, run_y, run_subjects = [], [], []
            
            # for ann in raw.annotations:
            #     if ann['description'] not in label_map:
            #         continue
                
            #     start_sample = int(ann['onset'] * sfreq)
            #     end_sample = start_sample + segment_samples
                
            #     if end_sample > data.shape[1]:
            #         continue
                
            #     # **KEY**: Store segment WITHOUT any normalization
            #     segment = data[:, start_sample:end_sample]
                
            #     run_X.append(segment)
            #     run_y.append(label_map[ann['description']])
            #     run_subjects.append(subj_str)
            
            # Save UNNORMALIZED preprocessed data
            # if run_X:
            #     np.savez_compressed(
            #         save_file,
            #         X=np.array(run_X),
            #         y=np.array(run_y),
            #         subjects=np.array(run_subjects)
            #     )
            #     print(f"  {run}: {len(run_X)} segments saved (unnormalized)")
            
        except Exception as e:
            print(f"  Error in {subj_str}{run}: {e}")
            continue

print("preprocessing done")