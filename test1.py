import mne
import numpy as np
import time
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
from mne.filter import filter_data
from keras.models import load_model
from datetime import datetime
from collections import deque

# --- Load trained models ---
movement_model = load_model("eegnet_M_New.h5", compile=False)
type_model = load_model("eegnet_LR_3.h5", compile=False)

# --- Load EDF and start streaming ---
raw = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/files/S109/S109R04.edf", preload=True)
event_id = {'T1': 1, 'T2': 2, 'T0': 0}
events, event_dict = mne.events_from_annotations(raw, event_id=event_id)

player = PlayerLSL(raw, chunk_size=32)
player.start()
time.sleep(1)
print("Streaming started...")

# --- Connect to LSL stream ---
stream = StreamLSL(stype='eeg', bufsize=2.5).connect()
print(f"Connected to LSL stream: {stream.name}")

# --- Parameters ---
sfreq = 256
n_channels = len(raw.ch_names)
window_samples = 640
stride_samples = int(0.25 * sfreq)
buffer_proc = np.zeros((n_channels, window_samples))

sample_end = 0
sample_beginning = -stride_samples
# ============================
# BCI STABILIZATION PARAMETERS
# ============================

MOVE_THRESHOLD_ON = 0.54
MOVE_THRESHOLD_OFF = 0.54
TYPE_THRESHOLD = 0.58

SMOOTH_WINDOW = 8
MIN_MOVE_FRAMES = 4
MIN_REST_FRAMES = 6

move_buffer = deque(maxlen=SMOOTH_WINDOW)
type_buffer = deque(maxlen=SMOOTH_WINDOW)

movement_state = False
move_counter = 0
rest_counter = 0
current_direction = None


# ----------------------------
# MATCH TRAINING NORMALIZATION
# ----------------------------
def preprocess_for_model(data):
    data = filter_data(data, sfreq, l_freq=8., h_freq=30.,
                       method='fir', phase='zero', verbose=False)
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-6
    data = (data - mean) / std
    return data[np.newaxis, ..., np.newaxis]


print("Starting live prediction loop...\n")

try:
    while True:
        sample_end += stride_samples
        sample_beginning += stride_samples

        latest_data, timestamps = stream.get_data(winsize=stride_samples)
        if latest_data is None or latest_data.size == 0:
            print("Stream ended.")
            break

        n_new = latest_data.shape[1]
        buffer_proc = np.hstack([buffer_proc[:, n_new:], latest_data])

        model_input = preprocess_for_model(buffer_proc.copy())

        eeg_time = timestamps[-1]
        time_str = datetime.fromtimestamp(eeg_time).strftime("%H:%M:%S.%f")[:-3]

        # ----------------------------
        # MODEL PREDICTIONS
        # ----------------------------
        move_prob = float(movement_model.predict(model_input, verbose=0)[0][0])
        type_prob = float(type_model.predict(model_input, verbose=0)[0][0])

        move_buffer.append(move_prob)
        type_buffer.append(type_prob)

        move_smooth = np.mean(move_buffer)
        type_smooth = np.mean(type_buffer)

        # ----------------------------
        # MOVEMENT STATE MACHINE
        # ----------------------------
        if not movement_state:
            if move_smooth > MOVE_THRESHOLD_ON:
                move_counter += 1
                if move_counter >= MIN_MOVE_FRAMES:
                    movement_state = True
                    move_counter = 0
                    rest_counter = 0
                    type_buffer.clear()
                    print(f"[{time_str}] MOVEMENT STARTED (p={move_smooth:.3f})")
            else:
                move_counter = 0
                print(f"[{time_str}] REST (p={move_smooth:.3f})")

        else:
            # movement active
            if move_smooth < MOVE_THRESHOLD_OFF:
                rest_counter += 1
                if rest_counter >= MIN_REST_FRAMES:
                    movement_state = False
                    current_direction = None
                    rest_counter = 0
                    print(f"[{time_str}] MOVEMENT ENDED (p={move_smooth:.3f})")
            else:
                rest_counter = 0

                # ----------------------------
                # LEFT / RIGHT DECISION (only during movement)
                # ----------------------------
                if type_smooth > TYPE_THRESHOLD:
                    direction = "LEFT"   # keep your flipped mapping
                elif type_smooth < (1 - TYPE_THRESHOLD):
                    direction = "RIGHT"
                else:
                    direction = "UNCERTAIN"

                if direction != "UNCERTAIN":
                    current_direction = direction

                if current_direction:
                    print(f"[{time_str}] MOVEMENT → {current_direction} "
                          f"(move={move_smooth:.2f}, type={type_smooth:.2f})")
                else:
                    print(f"[{time_str}] MOVEMENT → STABILIZING "
                          f"(move={move_smooth:.2f}, type={type_smooth:.2f})")

        # ----------------------------
        # GROUND TRUTH DEBUG
        # ----------------------------
        for event_sample, _, event_code in events:
            if sample_beginning < event_sample < sample_end:
                actual = {0: "REST", 1: "LEFT", 2: "RIGHT"}[event_code]
                print(f"    *** ACTUAL: {actual} ***")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    player.stop()
    print("PlayerLSL stopped.")
