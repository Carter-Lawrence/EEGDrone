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

# --- ADJUSTED THRESHOLDS ---
MOVE_THRESHOLD_ON = 0.65    # Threshold to START movement
MOVE_THRESHOLD_OFF = 0.30   # Much lower - allow temporary dips
TYPE_THRESHOLD = 0.60       # Confidence for direction

VOTE_WINDOW = 7             # Longer voting window
MIN_MOVE_VOTES = 4          # Need 4/7 to start
MIN_REST_VOTES = 5          # Need 5/7 LOW probabilities to end movement
MIN_TYPE_VOTES = 4          # Need 4/7 for direction

move_history = deque(maxlen=VOTE_WINDOW)
type_history = deque(maxlen=VOTE_WINDOW)

state = "REST"
current_direction = None
frames_in_movement = 0      # Track how long we've been in movement

# --- Preprocessing ---
def preprocess_for_model(data):
    data = filter_data(data, sfreq, l_freq=8., h_freq=30., 
                      method='fir', phase='zero', verbose=False)
    data = (data - data.mean(axis=1, keepdims=True)) / (
        data.std(axis=1, keepdims=True) + 1e-6
    )
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

        # --- Movement detection ---
        move_prob = float(movement_model.predict(model_input, verbose=0)[0][0])
        type_prob = float(type_model.predict(model_input, verbose=0)[0][0])

        if frames_in_movement % 4 == 0:  # Print every 4th frame to reduce spam
            print(f"    DEBUG: type_prob={type_prob:.3f} (>0.5=RIGHT, <0.5=LEFT)")
        # Vote on movement
        move_history.append(1 if move_prob > MOVE_THRESHOLD_ON else 0)
        move_votes = sum(move_history)
        
        # State machine with persistence
        if state == "REST":
            if move_votes >= MIN_MOVE_VOTES:
                state = "MOVEMENT"
                frames_in_movement = 0
                print(f"[{time_str}] 🔵 MOVEMENT STARTED (move_prob={move_prob:.3f})")
            else:
                print(f"[{time_str}] REST (move_prob={move_prob:.3f}, votes={move_votes}/{VOTE_WINDOW})")
        
        else:  # state == "MOVEMENT"
            frames_in_movement += 1
            
            # Count how many recent frames are BELOW threshold
            rest_votes = sum(1 for p in move_history if p == 0)
            
            # Only exit if we've been in movement for at least 8 frames (2 seconds)
            # AND we have strong evidence of rest
            if frames_in_movement > 8 and rest_votes >= MIN_REST_VOTES:
                state = "REST"
                current_direction = None
                type_history.clear()
                frames_in_movement = 0
                print(f"[{time_str}] ⚪ MOVEMENT ENDED → REST (move_prob={move_prob:.3f})")
            else:
                # Classify direction
                type_prob = float(type_model.predict(model_input, verbose=0)[0][0])
                
                
                # Vote on direction
# To this (FLIPPED):
                if type_prob > TYPE_THRESHOLD:
                    type_history.append("LEFT")   # SWAPPED!
                elif type_prob < (1 - TYPE_THRESHOLD):
                    type_history.append("RIGHT")  # SWAPPED!
                else:
                    type_history.append("UNCERTAIN")
                
                # Determine majority direction
                if len(type_history) > 0:
                    votes = {"LEFT": type_history.count("LEFT"),
                            "RIGHT": type_history.count("RIGHT"),
                            "UNCERTAIN": type_history.count("UNCERTAIN")}
                    
                    direction = max(votes, key=votes.get)
                    direction_votes = votes[direction]
                    
                    if direction != "UNCERTAIN" and direction_votes >= MIN_TYPE_VOTES:
                        current_direction = direction
                    
                    if current_direction:
                        print(f"[{time_str}] ➡️  MOVEMENT → {current_direction} "
                              f"(move={move_prob:.2f}, type={type_prob:.2f}, "
                              f"dur={frames_in_movement}, votes={direction_votes}/{VOTE_WINDOW})")
                    else:
                        print(f"[{time_str}] 🔄 MOVEMENT → STABILIZING "
                              f"(move={move_prob:.2f}, type={type_prob:.2f}, dur={frames_in_movement})")
        
        # --- Ground truth ---
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