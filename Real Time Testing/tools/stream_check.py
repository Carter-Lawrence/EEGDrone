# ================= REAL-TIME EEG STREAM PLOT =================

import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

# ---------------- USER SETTINGS ----------------
SERIAL_PORT = "/dev/tty.usbserial-DP05IYGX"   # CHANGE
WINDOW = 640          # samples per channel in the buffer
PLOT_LEN = 500        # number of points to display in plot
BANDPASS_LOW = 8.0    # Hz
BANDPASS_HIGH = 30.0  # Hz
# ----------------------------------------------

# ---------------- CONNECT TO BOARD ----------------
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT
print("test1")
board = BoardShim(BoardIds.CYTON_BOARD, params)
print("test2")
board.prepare_session()
print("test3")
board.start_stream(1024)
print("test4")
time.sleep(5)
print("Streaming started")

# ---------------- SETUP ----------------
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD)
sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD)
n_channels = len(eeg_channels)

# Real-time buffer for each channel
buffer = [deque(maxlen=WINDOW) for _ in range(n_channels)]

# ---------------- PLOT SETUP ----------------
plt.style.use('ggplot')
fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2 * n_channels), sharex=True)
lines = []

for i in range(n_channels):
    axes[i].set_ylim(-50, 50)  # adjust depending on your EEG scale
    axes[i].set_ylabel(f"Ch {eeg_channels[i]}")
    line, = axes[i].plot([], [])
    lines.append(line)

axes[-1].set_xlabel("Samples")
fig.suptitle("Real-Time EEG Stream")

# ---------------- UPDATE FUNCTION ----------------
def update(frame):
    # read latest data
    data = board.get_current_board_data(32)  # get latest 32 samples
    eeg = data[eeg_channels, :]              # shape: (channels, samples)

    # bandpass each channel
    for ch in range(n_channels):
        DataFilter.perform_bandpass(
            eeg[ch],
            sfreq,
            BANDPASS_LOW,
            BANDPASS_HIGH,
            4,
            FilterTypes.BUTTERWORTH,
            0
        )
        # append to buffer
        for val in eeg[ch]:
            buffer[ch].append(val)

    # update plot lines
    for i in range(n_channels):
        lines[i].set_data(np.arange(len(buffer[i])), buffer[i])
        axes[i].set_xlim(0, max(PLOT_LEN, len(buffer[i])))

    return lines

# ---------------- ANIMATION ----------------
ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.show()

# ---------------- CLEANUP ----------------
board.stop_stream()
board.release_session()
print("Streaming stopped")
