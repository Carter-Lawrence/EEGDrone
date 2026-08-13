"""
bci_live.py  —  Real-time EEG visualisation + EEGNet prediction + Arduino output
Connects to an OpenBCI Cyton board via BrainFlow, runs two EEGNet binary
classifiers in a background thread, and sends decoded commands to an Arduino
over a serial port:
    0  →  REST
    1  →  LEFT
    2  →  RIGHT
Usage:
    python bci_live.py \\
        --serial-port  /dev/cu.usbserial-DP05IYGX \\
        --arduino-port /dev/cu.usbmodem14201 \\
        --arduino-baud 9600
"""
import argparse
import logging
import os
import threading
import time 

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from keras.models import load_model
    from mne.filter import filter_data
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("[WARN] keras/mne not installed – running in visualisation-only mode.")

try:
    import serial as pyserial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed – Arduino output disabled. pip install pyserial")

# Configuration, must be the same as used during training for correct operation
MOVEMENT_MODEL_PATH    = "YOUR MODEL HERE"
TYPE_MODEL_PATH        = "YOUR MODEL HERE"

MOVE_THRESHOLD_ON      = 0.5
MOVE_THRESHOLD_OFF     = 0.5
MIN_MOVE_FRAMES        = 2
MIN_REST_FRAMES        = 2
SMOOTH_WINDOW          = 8
TYPE_THRESHOLD         = 0.5     # dead zone set in state machine (0.45 / 0.55)
PREDICTION_WINDOW_SAMP = 640     # 2.5 s @ 256 Hz
PREDICTION_STRIDE_SAMP = 64      # 0.25 s stride

# Channel mapping
# Must match the electrode order used in mne.pick() during training AND
# the physical wiring of your Cyton headset.
#
# Cyton pin index (0-based, pin N1P → 0) → EEG 10-10 label
CYTON_CHANNEL_MAP = {
    0: "FC3",   # pin N1P
    1: "FCz",   # pin N2P
    2: "FC4",   # pin N3P
    3: "C3",    # pin N4P
    4: "Cz",    # pin N5P
    5: "C4",    # pin N6P
}
# Order in which electrodes were fed to the model during training.
# This is the order rows must appear in when passed to .predict().
TRAINING_CHANNEL_ORDER = ["FC3", "FCz", "FC4", "C3", "Cz", "C4"]

# Arduino command bytes
CMD_REST  = b'0'
CMD_LEFT  = b'1'
CMD_RIGHT = b'2'

def preprocess_for_model(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Band-pass 8–30 Hz + per-channel z-score → (1, C, T, 1)."""
    data = filter_data(data.astype(np.float64), sfreq,
                       l_freq=8., h_freq=30.,
                       method='fir', phase='zero', verbose=False)
    mean = data.mean(axis=1, keepdims=True)
    std  = data.std(axis=1,  keepdims=True) + 1e-6
    return ((data - mean) / std)[np.newaxis, ..., np.newaxis]   # (1,C,T,1)

class ArduinoLink:
    """Thread-safe Arduino serial writer.  Sends only when command changes."""

    def __init__(self, port: str, baud: int = 9600):
        self._ser       = None
        self._last_cmd  = None
        self._lock      = threading.Lock()
        self.connected  = False

        if not SERIAL_AVAILABLE:
            print("[Arduino] pyserial not available.")
            return
        if not port:
            print("[Arduino] No port specified – output disabled.")
            return

        try:
            self._ser      = pyserial.Serial(port, baud, timeout=1)
            self.connected = True
            print(f"[Arduino] Connected on {port} @ {baud} baud ✓")
        except Exception as e:
            print(f"[Arduino] Could not open {port}: {e}")

    def send(self, command: bytes):
            """Send a byte only when the command has changed."""
            with self._lock:
                if self._ser and self.connected and command != self._last_cmd:
                    try:
                        self._ser.write(command)
                        self._last_cmd = command
                        print(f"[Arduino] → {command.decode()}")
                    except Exception as e:
                        print(f"[Arduino] Write error: {e}")
                        self.connected = False   # ← stop retrying

    def close(self):
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()


class PredictionEngine:
    """Runs EEGNet inference in a background thread and drives Arduino output."""

    def __init__(self, board_shim, exg_channels, sfreq: float,
                 arduino: ArduinoLink):
        self.board_shim   = board_shim
        self.exg_channels = list(exg_channels)   # BrainFlow row indices for each pin
        self.sfreq        = sfreq
        self.arduino      = arduino
        self.running      = False

        # Build the channel-reorder index map 
        # Goal: produce an array of BrainFlow row indices such that
        # self._buf[i] corresponds to TRAINING_CHANNEL_ORDER[i].
        label_to_pin = {v: k for k, v in CYTON_CHANNEL_MAP.items()}
        self._training_pins = []      # Cyton pin indices in training order
        self._training_rows = []      # BrainFlow row indices in training order
        for label in TRAINING_CHANNEL_ORDER:
            if label not in label_to_pin:
                raise ValueError(
                    f"[ChannelMap] '{label}' in TRAINING_CHANNEL_ORDER "
                    f"has no entry in CYTON_CHANNEL_MAP"
                )
            pin = label_to_pin[label]
            if pin >= len(self.exg_channels):
                raise ValueError(
                    f"[ChannelMap] Cyton pin {pin} (for '{label}') is out of "
                    f"range — board reports only {len(self.exg_channels)} EXG channels"
                )
            self._training_pins.append(pin)
            self._training_rows.append(self.exg_channels[pin])

        self.n_channels = len(TRAINING_CHANNEL_ORDER)
        self._print_channel_map()

        self._buf       = np.zeros((self.n_channels, PREDICTION_WINDOW_SAMP))
        self._move_buf  = collections.deque(maxlen=SMOOTH_WINDOW)
        self._type_buf  = collections.deque(maxlen=SMOOTH_WINDOW)
        self._lock      = threading.Lock()

        # public state (read by GUI)
        self.movement_state    = False
        self.current_direction = "INITIALISING…"
        self.move_prob_smooth  = 0.0
        self.type_prob_smooth  = 0.0
        self._move_counter     = 0
        self._rest_counter     = 0

        self._models_ok = False
        if MODELS_AVAILABLE:
            try:
                print("[Model] Loading movement model …")
                self._mv_model = load_model(MOVEMENT_MODEL_PATH, compile=False)
                print("[Model] Loading type model …")
                self._ty_model = load_model(TYPE_MODEL_PATH, compile=False)

                # sanity-check model channel
                # EEGNet input is (None, C, T, 1); channel dim is index 1.
                for name, m in [("movement", self._mv_model),
                                ("type",     self._ty_model)]:
                    expected = m.input_shape[1]
                    if expected != self.n_channels:
                        raise ValueError(
                            f"[Model] {name} model expects {expected} channels "
                            f"but TRAINING_CHANNEL_ORDER has {self.n_channels}. "
                            f"Retrain or fix the channel map."
                        )
                self._models_ok = True
                print("[Model] Both models loaded ✓  "
                      f"(input channels = {self.n_channels} ✓)")
            except Exception as e:
                print(f"[Model] Failed to load: {e}")

    def _print_channel_map(self):
        print("─" * 60)
        print("  CHANNEL MAP  (training order → Cyton pin → BrainFlow row)")
        print("─" * 60)
        for i, label in enumerate(TRAINING_CHANNEL_ORDER):
            pin  = self._training_pins[i]
            row  = self._training_rows[i]
            print(f"  model row {i}: {label:<5s}  "
                  f"pin N{pin+1}P  →  brainflow row {row}")
        print("─" * 60)

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    # Internal methods
    def _loop(self):
        stride = PREDICTION_STRIDE_SAMP
        while self.running:
            raw = self.board_shim.get_current_board_data(stride)
            if raw.shape[1] < stride:
                time.sleep(0.05)
                continue

            # Pick rows in TRAINING_CHANNEL_ORDER so the model's spatial
            # filters see each electrode in the row it was trained on.
            eeg = raw[self._training_rows, -stride:]
            with self._lock:
                self._buf = np.hstack([self._buf[:, stride:], eeg])

            if self._models_ok:
                inp  = preprocess_for_model(self._buf.copy(), self.sfreq)
                mv_p = float(self._mv_model.predict(inp, verbose=0)[0][0])
                ty_p = float(self._ty_model.predict(inp, verbose=0)[0][0])

                self._move_buf.append(mv_p)
                self._type_buf.append(ty_p)

                with self._lock:
                    self.move_prob_smooth = float(np.mean(self._move_buf))
                    self.type_prob_smooth = float(np.mean(self._type_buf))
                    self._update_state_machine()

            time.sleep(0.05)

    def _update_state_machine(self):
        ms = self.move_prob_smooth
        ts = self.type_prob_smooth

        if not self.movement_state:
            # waiting for movement onset
            if ms > MOVE_THRESHOLD_ON:
                self._move_counter += 1
                if self._move_counter >= MIN_MOVE_FRAMES:
                    self.movement_state    = True
                    self._move_counter     = 0
                    self._rest_counter     = 0
                    self._type_buf.clear()
                    self.current_direction = "MOVEMENT STARTED"
            else:
                self._move_counter     = 0
                self.current_direction = "REST"
                self.arduino.send(CMD_REST)        # ← Arduino: 0

        else:
            # movement active
            if ms < MOVE_THRESHOLD_OFF:
                self._rest_counter += 1
                if self._rest_counter >= MIN_REST_FRAMES:
                    self.movement_state    = False
                    self.current_direction = "REST"
                    self._rest_counter     = 0
                    self.arduino.send(CMD_REST)    # ← Arduino: 0
            else:
                self._rest_counter = 0

                # LEFT / RIGHT decision
                # Current LR model convention: LEFT=0 (low prob), RIGHT=1 (high prob).
                # Dead zone [0.45, 0.55] where direction is held.
                if ts > 0.55:
                    direction = "RIGHT"
                elif ts < 0.45:
                    direction = "LEFT"
                else:
                    direction = "UNCERTAIN"

                if direction == "LEFT":
                    self.current_direction = "LEFT"
                    self.arduino.send(CMD_LEFT)    # ← Arduino: 1
                elif direction == "RIGHT":
                    self.current_direction = "RIGHT"
                    self.arduino.send(CMD_RIGHT)   # ← Arduino: 2
                else:
                    # hold last known direction, don't spam Arduino
                    pass

    def get_state(self) -> dict:
        with self._lock:
            return {
                "direction":  self.current_direction,
                "moving":     self.movement_state,
                "move_prob":  self.move_prob_smooth,
                "type_prob":  self.type_prob_smooth,
                "models_ok":  self._models_ok,
                "arduino_ok": self.arduino.connected,
            }

# Graphical interface for real-time EEG visualization and control
#Used Claude for this FYI, have not checked or read over code
class Graph:
    BG      = "#0d1117"
    PANEL   = "#161b22"
    ACCENT  = "#58a6ff"
    GREEN   = "#3fb950"
    RED     = "#f85149"
    AMBER   = "#d29922"
    TEXT    = "#e6edf3"
    DIM     = "#8b949e"

    DIR_COLORS = {
        "LEFT":             "#58a6ff",
        "RIGHT":            "#f78166",
        "REST":             "#3fb950",
        "MOVEMENT STARTED": "#d29922",
        "STABILISING…":     "#d29922",
        "UNCERTAIN":        "#8b949e",
        "INITIALISING…":    "#8b949e",
        "NO MODELS":        "#8b949e",
    }
    CURVE_COLORS = ["#58a6ff","#3fb950","#f78166","#d29922",
                    "#bc8cff","#79c0ff","#56d364","#ffa657"]

    def __init__(self, board_shim, arduino: ArduinoLink):
        self.board_shim        = board_shim
        self.arduino           = arduino
        bid                    = board_shim.get_board_id()
        self.exg_channels      = BoardShim.get_exg_channels(bid)
        self.sfreq             = BoardShim.get_sampling_rate(bid)
        self.num_points        = 4 * self.sfreq

        # Use the training channel order for the plot rows so the on-screen
        # traces visually match what the model actually sees.
        label_to_pin = {v: k for k, v in CYTON_CHANNEL_MAP.items()}
        self.display_rows      = [self.exg_channels[label_to_pin[lbl]]
                                  for lbl in TRAINING_CHANNEL_ORDER]
        self.display_labels    = list(TRAINING_CHANNEL_ORDER)
        self.n_channels        = len(self.display_rows)

        self.engine = PredictionEngine(board_shim, self.exg_channels,
                                       self.sfreq, arduino)
        self.engine.start()

        self.app = QtWidgets.QApplication([])
        self._build_ui()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(50)
        QtWidgets.QApplication.instance().exec()

    def _build_ui(self):
        pg.setConfigOption('background', self.BG)
        pg.setConfigOption('foreground', self.TEXT)

        self.win = QtWidgets.QWidget()
        self.win.setStyleSheet(f"background:{self.BG};")
        self.win.setWindowTitle("BCI Live  ·  EEG + EEGNet + Arduino")
        self.win.resize(1340, 880)

        root = QtWidgets.QVBoxLayout(self.win)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # header row
        hrow = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("⬤  BCI LIVE")
        title.setStyleSheet(f"color:{self.ACCENT};font-size:13px;"
                            f"font-family:'SF Mono','Fira Code',monospace;"
                            f"letter-spacing:4px;")
        hrow.addWidget(title)
        hrow.addStretch()
        self.arduino_badge = QtWidgets.QLabel("● Arduino")
        self.arduino_badge.setStyleSheet(
            f"color:{self.GREEN if self.arduino.connected else self.RED};"
            f"font-size:11px;font-family:'SF Mono','Fira Code',monospace;")
        hrow.addWidget(self.arduino_badge)
        root.addLayout(hrow)

        # body
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body)

        # plot area
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground(self.BG)
        body.addWidget(self.plot_widget, stretch=4)

        # sidebar
        side = QtWidgets.QWidget()
        side.setFixedWidth(250)
        side.setStyleSheet(f"background:{self.PANEL};border:1px solid #30363d;"
                           f"border-radius:8px;")
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(16, 20, 16, 20)
        sl.setSpacing(12)
        body.addWidget(side)

        def lbl(text, size=11, color=None, bold=False):
            w = QtWidgets.QLabel(text)
            w.setStyleSheet(
                f"color:{color or self.DIM};font-size:{size}px;"
                f"font-family:'SF Mono','Fira Code',monospace;"
                f"{'font-weight:bold;' if bold else ''}")
            w.setWordWrap(True)
            return w

        sl.addWidget(lbl("PREDICTION", 9))

        self.dir_label = QtWidgets.QLabel("—")
        self.dir_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.dir_label.setStyleSheet(
            f"color:{self.ACCENT};font-size:36px;font-weight:bold;"
            f"font-family:'SF Mono','Fira Code',monospace;")
        sl.addWidget(self.dir_label)

        self.state_label = lbl("initialising", 10, self.DIM)
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(self.state_label)

        # arduino cmd display
        sl.addSpacing(4)
        sl.addWidget(lbl("ARDUINO CMD", 9))
        self.cmd_label = QtWidgets.QLabel("—")
        self.cmd_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.cmd_label.setStyleSheet(
            f"color:{self.AMBER};font-size:28px;font-weight:bold;"
            f"font-family:'SF Mono','Fira Code',monospace;"
            f"background:#1c2128;border-radius:6px;padding:6px;")
        sl.addWidget(self.cmd_label)

        # prob bars
        sl.addSpacing(8)
        sl.addWidget(lbl("MOVE PROB", 9))
        self.move_bar = self._make_bar(sl, self.GREEN)
        self.move_val = lbl("0.00", 9, self.GREEN)
        sl.addWidget(self.move_val)

        sl.addSpacing(4)
        sl.addWidget(lbl("TYPE PROB  (L < 0.5 < R)", 9))
        self.type_bar = self._make_bar(sl, self.ACCENT)
        self.type_val = lbl("0.00", 9, self.ACCENT)
        sl.addWidget(self.type_val)

        sl.addStretch()
        model_ok = MODELS_AVAILABLE
        sl.addWidget(lbl(
            "● models loaded" if model_ok else "○ vis-only mode",
            9, self.GREEN if model_ok else self.DIM))
        sl.addWidget(lbl(
            f"● arduino {'connected' if self.arduino.connected else '○ not connected'}",
            9, self.GREEN if self.arduino.connected else self.DIM))

        # EEG curves
        self.plots, self.curves = [], []
        for i in range(self.n_channels):
            p = self.plot_widget.addPlot(row=i, col=0)
            p.showAxis('left', True)
            p.showAxis('bottom', i == self.n_channels - 1)
            p.getAxis('left').setLabel(self.display_labels[i],
                                       color=self.DIM, size='8pt')
            p.getAxis('left').setWidth(52)
            for ax in ('left', 'bottom'):
                p.getAxis(ax).setPen(pg.mkPen('#30363d'))
                p.getAxis(ax).setTextPen(pg.mkPen(self.DIM))
            p.setMenuEnabled(False)
            p.hideButtons()
            color = self.CURVE_COLORS[i % len(self.CURVE_COLORS)]
            self.curves.append(p.plot(pen=pg.mkPen(color, width=1.2)))
            self.plots.append(p)

        self.win.show()

    def _make_bar(self, layout, color):
        c = QtWidgets.QWidget()
        c.setFixedHeight(14)
        c.setStyleSheet("background:#21262d;border-radius:7px;")
        b = QtWidgets.QWidget(c)
        b.setGeometry(0, 0, 0, 14)
        b.setStyleSheet(f"background:{color};border-radius:7px;")
        layout.addWidget(c)
        return (c, b)

    def _set_bar(self, bar_tuple, value):
        c, b = bar_tuple
        b.setFixedWidth(int(np.clip(value, 0, 1) * c.width()))

    # Update loop
    def update(self):
        # EEG traces — display in TRAINING_CHANNEL_ORDER so the on-screen
        # trace layout mirrors what the model consumes row-by-row.
        data = self.board_shim.get_current_board_data(self.num_points)
        for idx, ch in enumerate(self.display_rows):
            if data.shape[1] == 0:
                continue
            sig = data[ch].copy()
            DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(sig, self.sfreq, 3., 45., 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(sig, self.sfreq, 48., 52., 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(sig, self.sfreq, 58., 62., 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            self.curves[idx].setData(sig.tolist())

        # prediction panel
        s = self.engine.get_state()
        direction = s["direction"] if s["models_ok"] else "NO MODELS"
        color = self.DIR_COLORS.get(direction, self.DIM)

        self.dir_label.setStyleSheet(
            f"color:{color};font-size:36px;font-weight:bold;"
            f"font-family:'SF Mono','Fira Code',monospace;")
        self.dir_label.setText(direction)

        # Arduino cmd badge
        cmd_map = {"LEFT": "1  (LEFT)", "RIGHT": "2  (RIGHT)", "REST": "0  (REST)"}
        cmd_color_map = {"LEFT": self.ACCENT, "RIGHT": self.RED, "REST": self.GREEN}
        cmd_txt   = cmd_map.get(direction, "—")
        cmd_color = cmd_color_map.get(direction, self.AMBER)
        self.cmd_label.setStyleSheet(
            f"color:{cmd_color};font-size:28px;font-weight:bold;"
            f"font-family:'SF Mono','Fira Code',monospace;"
            f"background:#1c2128;border-radius:6px;padding:6px;")
        self.cmd_label.setText(cmd_txt)

        status = "● MOVING" if s["moving"] else "○ at rest"
        sc = self.GREEN if s["moving"] else self.DIM
        self.state_label.setText(status)
        self.state_label.setStyleSheet(
            f"color:{sc};font-size:10px;"
            f"font-family:'SF Mono','Fira Code',monospace;")

        self._set_bar(self.move_bar, s["move_prob"])
        self._set_bar(self.type_bar, s["type_prob"])
        self.move_val.setText(f"{s['move_prob']:.3f}")
        self.type_val.setText(f"{s['type_prob']:.3f}")

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    p = argparse.ArgumentParser(
        description="BCI Live – EEG visualisation + EEGNet prediction + Arduino output")
    p.add_argument('--timeout',       type=int, default=0)
    p.add_argument('--ip-port',       type=int, default=0)
    p.add_argument('--ip-protocol',   type=int, default=0)
    p.add_argument('--ip-address',    type=str, default='')
    p.add_argument('--serial-port',   type=str, default="/dev/cu.usbserial-DP05IYGX",
                   help="OpenBCI serial port")
    p.add_argument('--mac-address',   type=str, default='')
    p.add_argument('--other-info',    type=str, default='')
    p.add_argument('--serial-number', type=str, default='')
    p.add_argument('--board-id',      type=int, default=BoardIds.CYTON_BOARD.value)
    p.add_argument('--file',          type=str, default='')
    p.add_argument('--master-board',  type=int, default=BoardIds.NO_BOARD)
    # Arduino
    p.add_argument('--arduino-port',  type=str, default="/dev/cu.usbmodem2101",
                   help="Arduino serial port, e.g. /dev/cu.usbmodem14201 or COM3")
    p.add_argument('--arduino-baud',  type=int, default=9600)
    args = p.parse_args()

    params = BrainFlowInputParams()
    params.ip_port       = args.ip_port
    params.serial_port   = args.serial_port
    params.mac_address   = args.mac_address
    params.other_info    = args.other_info
    params.serial_number = args.serial_number
    params.ip_address    = args.ip_address
    params.ip_protocol   = args.ip_protocol
    params.timeout       = args.timeout
    params.file          = args.file
    params.master_board  = args.master_board

    print(f"EEG  port : {args.serial_port} (exists={os.path.exists(args.serial_port)})")
    print(f"Arduino   : {args.arduino_port or '(none)'}")

    arduino   = ArduinoLink(args.arduino_port, args.arduino_baud)
    board_shim = BoardShim(int(args.board_id), params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream()
        Graph(board_shim, arduino)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('Shutting down …')
        arduino.close()
        if board_shim.is_prepared():
            board_shim.release_session()
            logging.info('Session released.')


if __name__ == '__main__':
    main()