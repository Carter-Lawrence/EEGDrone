import mne
import time
import numpy as np
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
import matplotlib.pyplot as plt

# --- Step 1: Load EDF file ---
raw_edf = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/S001R02.edf", preload=True)
sfreq = 100
n_channels = 64
ch_names = [f"Ch{i}" for i in range(1, n_channels + 1)]
info = mne.create_info(ch_names, sfreq, ch_types='eeg')

# --- Step 2: Set up plotting ---
plt.ion()
fig, axes = plt.subplots(8, 8, figsize=(16, 16))
fig.subplots_adjust(hspace=0.6, wspace=0.3)  # more space between plots
axes = axes.flatten()

lines = []
for i, ax in enumerate(axes):
    line, = ax.plot([], [])
    ax.set_xlim(0, 1000)
    ax.set_ylim(-3.5, 3.5)  # adjust depending on your signal
    ax.set_title(raw_edf.ch_names[i])
    lines.append(line)

axes[-1].set_xlabel("Samples")

buffer = np.zeros((n_channels, 1000))  # rolling window

# --- Step 3: Create and start the PlayerLSL ---
player = PlayerLSL(raw_edf, chunk_size=32)
player.start()
print("PlayerLSL started, streaming data...")

# --- Step 4: Receive the stream ---
time.sleep(1)  # allow discovery

try:
    stream = StreamLSL(stype='eeg', bufsize=0.5).connect()
    print(f"Connected to LSL stream with name: {stream.name}")
    print("Acquiring data for 120 seconds...")

    duration = 120
    start_time = time.time()

    while time.time() - start_time < duration:
        latest_data, timestamps = stream.get_data()
        if latest_data is not None and latest_data.size > 0:
            n_new = latest_data.shape[1]
            buffer = np.hstack([buffer[:, n_new:], latest_data])

            for i, line in enumerate(lines):
                line.set_ydata(buffer[i] * 10000)
                line.set_xdata(np.arange(buffer.shape[1]))

            plt.pause(0.01)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    player.stop()
    print("PlayerLSL stopped.")
