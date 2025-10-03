import mne
import time
import numpy as np
import matplotlib.pyplot as plt
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
from mne.filter import filter_data, notch_filter

# --- 1. Load EDF and start streaming raw data ---
raw = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/S001R04.edf", preload=True)

player = PlayerLSL(raw, chunk_size=32)
player.start()
print("PlayerLSL started, streaming raw data...")

time.sleep(1)  # let stream announce itself

# --- 2. Connect to stream ---
stream = StreamLSL(stype='eeg', bufsize=0.5).connect()
print(f"Connected to LSL stream: {stream.name}")

# --- 3. Plot setup (2 panels: raw vs processed) ---
sfreq = int(raw.info["sfreq"])
n_channels = len(raw.ch_names)

plt.ion()
fig, (ax_raw, ax_proc) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

lines_raw = [ax_raw.plot([], [])[0] for _ in range(n_channels)]
lines_proc = [ax_proc.plot([], [])[0] for _ in range(n_channels)]

ax_raw.set_xlim(0, 500)
ax_raw.set_ylim(-200e-6, 200e-6)
ax_raw.set_title("Unprocessed EEG")

ax_proc.set_xlim(0, 500)
ax_proc.set_ylim(-200e-6, 200e-6)
ax_proc.set_title("Processed EEG (CAR + Notch + Band-pass)")

buffer_raw = np.zeros((n_channels, 500))
buffer_proc = np.zeros((n_channels, 500))

# --- 4. Real-time loop ---
duration = 60  # seconds
start_time = time.time()

try:
    while time.time() - start_time < duration:
        latest_data, timestamps = stream.get_data()

        if latest_data is not None and latest_data.size > 0:
            # --- copy raw ---
            raw_chunk = latest_data.copy()

            # --- preprocessing copy ---
            proc_chunk = latest_data.copy()
            proc_chunk -= proc_chunk.mean(axis=0, keepdims=True)  # CAR
            proc_chunk = notch_filter(proc_chunk, sfreq, freqs=[60], verbose=False)
            proc_chunk = filter_data(proc_chunk, sfreq, l_freq=8., h_freq=30.,
                                     fir_design='firwin', verbose=False)

            # update rolling buffers
            n_new = latest_data.shape[1]
            buffer_raw = np.hstack([buffer_raw[:, n_new:], raw_chunk])
            buffer_proc = np.hstack([buffer_proc[:, n_new:], proc_chunk])

            # update plots
            for i, line in enumerate(lines_raw):
                line.set_ydata(buffer_raw[i])
                line.set_xdata(np.arange(buffer_raw.shape[1]))
            for i, line in enumerate(lines_proc):
                line.set_ydata(buffer_proc[i])
                line.set_xdata(np.arange(buffer_proc.shape[1]))

            plt.pause(0.01)

except Exception as e:
    print(f"Error: {e}")

finally:
    player.stop()
    print("PlayerLSL stopped.")
