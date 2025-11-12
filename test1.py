import mne
import numpy as np
import time
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
from mne.filter import notch_filter, filter_data
from keras.models import load_model
from datetime import datetime
import asrpy

# --- Load trained models ---
movement_model = load_model("eegnet_M_1.h5", compile=False)  # rest vs movement
type_model = load_model("eegnet_C_1.h5", compile=False)      # left vs right

# --- Load EDF and start streaming ---
raw = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/files/S109/S109R04.edf", preload=True)
player = PlayerLSL(raw, chunk_size=32)
player.start()
time.sleep(1)
print("Streaming started...")

# --- Connect to LSL stream ---
stream = StreamLSL(stype='eeg', bufsize=1.25).connect()
print(f"Connected to LSL stream: {stream.name}")

# --- Parameters ---
sfreq = 256
n_channels = len(raw.ch_names)
window_samples = 321    # 3-second window → 768 samples
stride_samples = int(0.5 * sfreq)       # slide forward 0.5 s → 128 samples
buffer_proc = np.zeros((n_channels, window_samples))

# --- Fit ASR baseline ---
baseline_raw = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/files/S109/S109R02.edf", preload=True)
baseline_raw.resample(sfreq, verbose=False)
baseline_raw.set_eeg_reference('average', verbose=False)
baseline_raw.filter(1., 40., fir_design='firwin', verbose=False)

asr = asrpy.ASR(sfreq=sfreq, cutoff=20)
asr.fit(baseline_raw)
print("ASR fitted on baseline.")

# --- Preprocessing for EEGNet ---
def preprocess_for_model(data):
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)
    return data[np.newaxis, ..., np.newaxis]  # (1, channels, samples, 1)

print("Starting live prediction loop...\n")

try:
    while True:
        # --- get latest 0.5 s of new data ---
        latest_data, timestamps = stream.get_data(winsize=stride_samples)

        if latest_data is None or latest_data.size == 0:
            print("No more EEG data available — stream ended.")
            break

        n_new = latest_data.shape[1]

        # --- slide window: drop oldest 0.5 s, add newest 0.5 s ---
        buffer_proc = np.hstack([buffer_proc[:, n_new:], latest_data])

        # --- process current 3 s window ---
        proc_window = buffer_proc.copy()
        proc_window -= proc_window.mean(axis=0, keepdims=True)

        # Apply ASR and filters
        info = mne.create_info(ch_names=stream.info['ch_names'], sfreq=sfreq, ch_types='eeg')
        raw_tmp = mne.io.RawArray(proc_window, info, verbose=False)
        raw_tmp.set_eeg_reference('average', verbose=False)
        raw_tmp = asr.transform(raw_tmp)
        proc_window = raw_tmp.get_data()

        #proc_window = notch_filter(proc_window, sfreq, freqs=[60], method='iir', verbose=False)
        proc_window = filter_data(proc_window, sfreq, l_freq=8., h_freq=30., method='iir', verbose=False)

        # pad to match model input (769)
        #if proc_window.shape[1] == 321:
        #    proc_window = np.pad(proc_window, ((0, 0), (0, 1)), mode='edge')

        # --- model inference ---
        model_input = preprocess_for_model(proc_window)
        eeg_time = timestamps[-1]
        time_str = datetime.fromtimestamp(eeg_time).strftime("%H:%M:%S.%f")[:-3]

        move_pred = movement_model.predict(model_input, verbose=0)[0]
        move_class = np.argmax(move_pred)
        move_conf = move_pred[move_class]

        if move_class == 0 or move_conf < 0.75:
            # Treat as REST if below confidence threshold
            print(f"[{time_str}] REST  (conf: {move_conf:.2f})")
        else:
            # Only classify movement if confidence >= 75%
            type_pred = type_model.predict(model_input, verbose=0)[0]
            type_class = np.argmax(type_pred)
            type_conf = type_pred[type_class]
            if type_conf > 0.75:
                direction = "LEFT" if type_class == 0 else "RIGHT"
                print(f"[{time_str}] MOVEMENT → {direction}  (conf: {type_conf:.2f})")

        # --- wait until next 0.5 s stride ---
        time.sleep(0.25)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    player.stop()
    print("PlayerLSL stopped.")
