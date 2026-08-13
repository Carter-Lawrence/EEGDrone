# EEGDrone

A brain–computer interface that decodes motor imagery from live EEG and turns it
into movement commands. You imagine moving your left or right hand, an EEGNet
model classifies the signal in real time, and the decoded command is sent over
serial to an Arduino that drives the hardware.

<!-- TODO: drop in a short demo GIF or a photo of the rig here — this is the
single most convincing thing you can show. -->
<!-- ![demo](docs/demo.gif) -->

## What it does

The system runs two lightweight EEGNet binary classifiers on a rolling window of
6-channel EEG:

1. **Move vs. Rest** — is the person intentionally moving/imagining movement?
2. **Left vs. Right** — if so, which hand?

A small state machine smooths the predictions and debounces them (so a single
noisy frame doesn't flip the output), then emits one of three commands — `REST`,
`LEFT`, `RIGHT` — to an Arduino over serial.

## How it works

**Data.** Trained on the [PhysioNet EEG Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/).
Only the executed/imagined left-vs-right-hand runs (R03, R04, R07, R08, R11, R12)
are used, plus the baseline runs (R01, R02) as pure rest. Isolating just these
runs was important — the dataset mixes several motor-imagery task types under the
same T1/T2 labels, and training on all of them polluted the "left/right" classes
and caused a strong directional bias at inference time.

**Signal processing.** Six motor-cortex channels (Fc3, Fcz, Fc4, C3, Cz, C4),
sampled at 160 Hz, are cut into 4-second windows (640 samples) with a 0.25 s
stride. Each window is band-pass filtered 8–30 Hz (mu/beta bands) and z-scored
per channel — the same normalization is applied identically at training and
inference time so the live signal matches the training distribution.

**Model.** A compact EEGNet-style CNN: a temporal convolution to learn frequency
filters, a depthwise spatial convolution across channels, a separable
convolution for spectral abstraction, and a sigmoid output for binary
classification. Training uses overlapping windows and feeds the model slightly varied copies of
each example — nudged a little in time and with a bit of random noise added — so
it learns the general brain pattern instead of memorizing exact recordings.
Train/validation splits are grouped by subject, so all of one person's data
stays on one side of the split and accuracy reflects how well the model works on
a brand-new person it has never seen.

**Real-time loop.** `BciLive.py` connects to an OpenBCI Cyton board via BrainFlow,
runs both models in a background thread, visualizes the live EEG with pyqtgraph,
and streams decoded commands to the Arduino.

## Results

| Config | Channels | Left vs. Right | Move vs. Rest |
|---|---|---|---|
| Deployed (motor cortex subset) | 6 | 67.75% | 75.2% |
| Full montage | 64 | 73.7% | 84.7% |

Chance is 50% on both tasks. Evaluated on held-out subjects (grouped split so no
subject appears in both train and test).

## Repo layout

```
Model Training Scripts/
  EEGNetLeftRight.py      # train the Left/Right classifier
  EEGNetMovementRest.py   # train the Move/Rest classifier
  load_data_v6.py         # shared data loader (run-filtered to match training)
  evaluate_model_v6.py    # offline evaluation + threshold sweep
  LRBias.py / MRBias.py   # bias/sanity checks on trained models

Real Time Testing/
  BciLive.py              # live board -> models -> Arduino
  BciReplay.py            # replay a recorded file through the live pipeline
  tools/
    stream_check.py       # verify the EEG board is streaming
    board_monitor.py      # live multi-channel plot
    servo_test.py         # manually test the Arduino / servo output
    view_edf.py           # inspect a recorded EDF file
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Train a model:

```bash
python "Model Training Scripts/EEGNetLeftRight.py"
```

Evaluate:

```bash
python "Model Training Scripts/evaluate_model_v6.py"
```

Run live (with hardware connected):

```bash
python "Real Time Testing/BciLive.py" \
  --serial-port  /dev/cu.usbserial-XXXX \
  --arduino-port /dev/cu.usbmodemXXXX \
  --arduino-baud 9600
```

Trained `.h5` model files are not committed to keep the repo light — train your
own with the scripts above.

## Hardware

- OpenBCI Cyton board (8-channel), 6 channels used over motor cortex, 2 channels as baseline on earlobes
- Arduino (serial, 9600 baud) driving the movement output

## Notes & next steps

Next steps for this project include testing the model in real time on a full 64 electrode setup, as the current model architecture after being trained on 64 channles from the training dataset indicate the full setup would boost accuracy by 8.95% and 9.5% for left vs. right and movement vs. rest respectively. A more interesting and economical test would be to use the 64 channel model to pick the best 16 channels to Daisy-chain to the live board, and test the accuracy live for those best 16 channels.  
