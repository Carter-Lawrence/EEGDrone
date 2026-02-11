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

raw_edf = mne.io.read_raw_edf("/Users/carterlawrence/Downloads/files/S109/S109R04.edf", preload=True)
#print(raw_edf.annotations)
#print(raw_edf)

event_id = {'T0': 0, 'T1': 1, 'T2': 2}

# convert annotations into events
events, event_dict = mne.events_from_annotations(raw_edf, event_id=event_id)

#print(event_dict)

sfreq = raw_edf.info['sfreq']  # sampling rate
print("First 10 events with timestamps:")
for ev in events:
    sample_idx, _, code = ev
    time_sec = sample_idx / sfreq
    print(f"Time: {time_sec:.3f}s, Event code: {code}")