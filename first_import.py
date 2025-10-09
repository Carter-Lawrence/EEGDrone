import mne
import numpy as np
import time
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
from mne.filter import filter_data, notch_filter
from tensorflow.keras.models import load_model
from datetime import datetime

# --- Load trained EEGNet models ---
movement_model = load_model("eegnet_movement_model.h5", compile=False)  # rest vs movement
type_model = load_model("eegnet_model_workingMVP.h5", compile=False)          # left vs right

# --- Load EDF and start streaming ---
raw = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/S001R04.edf", preload=True)
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
window_samples = 897  # [-0.5, 3] s → 3.5 s window
buffer_proc = np.zeros((n_channels, window_samples))
buffer_timestamps = np.zeros(window_samples)
update_interval = 0.1  # seconds

def preprocess_for_model(data):
    """Normalize and reshape EEG window for model input."""
    data = (data - np.mean(data)) / np.std(data)
    return data[np.newaxis, ..., np.newaxis]  # (1, 64, 897, 1)

print("Starting live prediction loop...\n")

try:
    while True:
        latest_data, timestamps = stream.get_data()
        if latest_data is not None and latest_data.size > 0:
            n_new = latest_data.shape[1]

            # --- update rolling buffer ---
            if n_new >= window_samples:
                # chunk longer than window → keep last 897 samples
                buffer_proc = latest_data[:, -window_samples:]
                buffer_timestamps = timestamps[-window_samples:]
            else:
                buffer_proc = np.hstack([buffer_proc[:, n_new:], latest_data])
                buffer_timestamps = np.hstack([buffer_timestamps[n_new:], timestamps[-n_new:]])

            # --- filter & predict only if buffer is full ---
            if buffer_proc.shape[1] == window_samples:
                proc_window = buffer_proc.copy()
                proc_window -= proc_window.mean(axis=0, keepdims=True)  # CAR
                #proc_window = notch_filter(proc_window, sfreq, freqs=[60], verbose=False, filter_length=window_samples)
                proc_window = filter_data(proc_window, sfreq, l_freq=8., h_freq=30.,
                                          fir_design='firwin', verbose=False, filter_length='auto')

                model_input = preprocess_for_model(proc_window)
                eeg_time = buffer_timestamps[-1]  # timestamp of last sample
                time_str = datetime.fromtimestamp(eeg_time).strftime("%H:%M:%S.%f")[:-3]

                # Stage 1: rest vs movement
                move_pred = movement_model.predict(model_input, verbose=0)
                move_class = np.argmax(move_pred)

                if move_class == 0:
                    print(f"[{time_str}] REST")
                else:
                    # Stage 2: left vs right
                    type_pred = type_model.predict(model_input, verbose=0)
                    type_class = np.argmax(type_pred)
                    direction = "LEFT" if type_class == 0 else "RIGHT"
                    print(f"[{time_str}] MOVEMENT → {direction}")

        time.sleep(update_interval)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    player.stop()
    print("PlayerLSL stopped.")
