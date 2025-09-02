import mne
import time
import numpy as np
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
import matplotlib.pyplot as plt

sfreq = 100
n_channels = 64
ch_names = [f"Ch{i}" for i in range(1, n_channels + 1)]
info = mne.create_info(ch_names, sfreq, ch_types='eeg')

raw_edf = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/S001R02.edf", preload=True)
print(raw_edf)

event_id = {'T0': 0, 'T1': 1, 'T2': 2}

# convert annotations into events
events, event_dict = mne.events_from_annotations(raw_edf, event_id=event_id)

print(event_dict)
#Plotting Section
plt.ion()
fig, ax = plt.subplots()
lines = [ax.plot([], [])[0] for _ in range(n_channels)]
ax.set_xlim(0, 500)   # show last 500 samples
ax.set_ylim(-0.00035, 0.00035)    # adjust to your signal scale
buffer = np.zeros((n_channels, 500))  # rolling window

# --- Step 2: Create and start the PlayerLSL ---
# Create a PlayerLSL from the MNE Raw object
player = PlayerLSL(raw_edf, chunk_size=32)

# Start the player in the background
player.start()
print("PlayerLSL started, streaming data...")

# --- Step 3: Receive the stream with StreamLSL ---
# Pause briefly to allow the stream to be discoverable on the network
time.sleep(1)

try:
    # Connect to the stream. You can specify by name, source_id, or stype.
    # The stype is 'eeg' because we specified that in our mock `mne.Info` object.
    stream = StreamLSL(stype='eeg', bufsize=0.5).connect()
    print(f"Connected to LSL stream with name: {stream.name}")
    print("Acquiring data for 5 seconds...")

    duration = 120  # seconds
    start_time = time.time()
    while time.time() - start_time < duration:
        latest_data, timestamps = stream.get_data()
        if latest_data is not None and latest_data.size > 0:
            # shift old data out, append new samples
            n_new = latest_data.shape[1]
            buffer = np.hstack([buffer[:, n_new:], latest_data])

            # update plot lines
            for i, line in enumerate(lines):
                line.set_ydata(buffer[i])
                line.set_xdata(np.arange(buffer.shape[1]))
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # --- Clean up by stopping the player ---
    player.stop()
    print("PlayerLSL stopped.")