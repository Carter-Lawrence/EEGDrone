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
movement_model = load_model("eegnet_M.h5", compile=False)  # rest vs movement
type_model = load_model("eegnet_C.h5", compile=False)   # left vs right

# --- Load EDF and start streaming ---
raw = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/files/S109/S109R04.edf", preload=True)
player = PlayerLSL(raw, chunk_size=32)
player.start()
time.sleep(1)
print("Streaming started...")

# --- Connect to LSL stream ---
stream = StreamLSL(stype='eeg', bufsize=0.5).connect()
print(f"Connected to LSL stream: {stream.name}")

# --- Parameters ---
sfreq = 256
n_channels = len(raw.ch_names)
window_samples = int(3.0 * sfreq)  # = 768
buffer_proc = np.zeros((n_channels, window_samples))
buffer_timestamps = np.zeros(window_samples)
update_interval = 0.5  # seconds

baseline_raw = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/files/S109/S109R02.edf", preload=True)
baseline_raw.resample(sfreq, verbose=False)
baseline_raw.set_eeg_reference('average', verbose=False)
baseline_raw.filter(1., 40., fir_design='firwin', verbose=False)

# Fit ASR on baseline
asr = asrpy.ASR(sfreq=sfreq, cutoff=20)
asr.fit(baseline_raw)
print("ASR fitted on baseline.")

# --- Preprocess for EEGNet ---
def preprocess_for_model(data):
    # per-channel z-score normalization (match training)
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)
    return data[np.newaxis, ..., np.newaxis]  # shape (1, channels, samples, 1)

print("Starting live prediction loop...\n")
try:
    while True:
        latest_data, timestamps = stream.get_data()

        # --- end when raw/stream is empty ---
        if latest_data is None or latest_data.size == 0:
            print("No more EEG data available — stream ended.")
            break

        n_new = latest_data.shape[1]

        # --- update rolling buffer ---
        if n_new >= window_samples:
            buffer_proc = latest_data[:, -window_samples:]
            buffer_timestamps = timestamps[-window_samples:]
        else:
            buffer_proc = np.hstack([buffer_proc[:, n_new:], latest_data])
            buffer_timestamps = np.hstack([buffer_timestamps[n_new:], timestamps[-n_new:]])

        # --- process when buffer is full ---
        if buffer_proc.shape[1] == window_samples:
            proc_window = buffer_proc.copy()
            proc_window -= proc_window.mean(axis=0, keepdims=True)

            info = mne.create_info(ch_names=stream.info['ch_names'], sfreq=sfreq, ch_types='eeg')
            raw_tmp = mne.io.RawArray(proc_window, info, verbose=False)
            raw_tmp = asr.transform(raw_tmp)
            proc_window = raw_tmp.get_data()

            proc_window = notch_filter(proc_window, sfreq, freqs=[60], method='iir', verbose=False)
            proc_window = filter_data(proc_window, sfreq, l_freq=8., h_freq=30., method='iir', verbose=False)

            if proc_window.shape[1] == 768:
                proc_window = np.pad(proc_window, ((0,0),(0,1)), mode='constant')

            model_input = preprocess_for_model(proc_window)
            eeg_time = buffer_timestamps[-1]
            time_str = datetime.fromtimestamp(eeg_time).strftime("%H:%M:%S.%f")[:-3]

            move_pred = movement_model.predict(model_input, verbose=0)
            move_class = np.argmax(move_pred)

            if move_class == 0:
                eeg_time = buffer_timestamps[-1]  # seconds
                print(f"[{eeg_time:.3f}s] REST")
            else:
                type_pred = type_model.predict(model_input, verbose=0)
                type_class = np.argmax(type_pred)
                direction = "LEFT" if type_class == 0 else "RIGHT"
                eeg_time = buffer_timestamps[-1]  # seconds
                print(f"[{eeg_time}] MOVEMENT → {direction}")

        time.sleep(update_interval)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    player.stop()
    print("PlayerLSL stopped.")

