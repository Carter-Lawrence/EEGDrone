"""
bci_replay_test.py  —  Offline EDF replay with live EEGNet prediction + Arduino output
"""

import collections
import threading
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

import mne
mne.set_log_level('ERROR')

try:
    from keras.models import load_model
    from mne.filter import filter_data
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("[WARN] keras/mne not installed – vis-only mode.")

try:
    import serial as pyserial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed.")

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
MOVEMENT_MODEL_PATH    = "eegnet_MR_4.h5"
TYPE_MODEL_PATH        = "eegnet_LR_4.h5"

MOVE_THRESHOLD_ON      = 0.58   # nudge down from 0.62
MOVE_THRESHOLD_OFF     = 0.52   # asymmetric: easier to stay in movement than enter
MIN_MOVE_FRAMES        = 5      # keep — this is what's preventing FPs
MIN_REST_FRAMES        = 6      # keep
SMOOTH_WINDOW          = 12     # keep
TYPE_THRESHOLD         = 0.50   # keep — this is now correct
PREDICTION_WINDOW_SAMP = 640    # try 2.0s instead of 2.5s for faster response
PREDICTION_STRIDE_SAMP = 64   # smaller = more predictions per second

CMD_REST  = b'0'
CMD_LEFT  = b'1'
CMD_RIGHT = b'2'

LABEL_MAP = {0: "REST", 1: "LEFT", 2: "RIGHT"}
EVENT_MAP = {"T0": 0, "T1": 1, "T2": 2}

PATH         = "/Users/carterlawrence/Downloads/files/S001/S001R04.edf"
SPEED        = 1.0
NO_ARDUINO   = False
ARDUINO_PORT = "/dev/cu.usbmodem1101"   # e.g. "/dev/cu.usbmodem14201"
ARDUINO_BAUD = 9600
WINDOW       = 6
SHOW_GROUND  = True
# ═════════════════════════════════════════════════════════════════════════════

#Imma boop yo schmoopty
class ArduinoLink:
    def __init__(self, port: str, baud: int = 9600, enabled: bool = True):
        self._ser      = None
        self._last_cmd = None
        self._lock     = threading.Lock()
        self.connected = False          # ← this line must be here
        if not enabled or not SERIAL_AVAILABLE or not port:
            return
        try:
            self._ser      = pyserial.Serial(port, baud, timeout=1)
            self._ser.dtr  = False
            time.sleep(2)
            self._ser.reset_input_buffer()
            self.connected = True
            print(f"[Arduino] Connected on {port} @ {baud} baud ✓")
        except Exception as e:
            print(f"[Arduino] Could not open {port}: {e}")

    def send(self, command: bytes):
        with self._lock:
            if self._ser and self.connected and command != self._last_cmd:
                try:
                    self._ser.write(command)
                    self._last_cmd = command
                except Exception as e:
                    print(f"[Arduino] Write error: {e}")
                    self.connected = False   # ← stop retrying

    def close(self):
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()


class EdfReplayer:
    def __init__(self, edf_path: str, speed: float = 1.0):
        print(f"[EDF] Loading {edf_path} …")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        self.sfreq       = int(raw.info['sfreq'])
        self.ch_names    = raw.ch_names
        self.n_channels  = len(self.ch_names)
        self._data       = raw.get_data()
        self._speed      = speed
        self._total_samp = self._data.shape[1]

        self.events = []
        for ann in raw.annotations:
            if ann['description'] in EVENT_MAP:
                samp  = int(ann['onset'] * self.sfreq)
                label = LABEL_MAP[EVENT_MAP[ann['description']]]
                self.events.append((samp, label))
        print(f"[EDF] {self.n_channels} ch, {self.sfreq} Hz, "
              f"{self._total_samp} samples, {len(self.events)} events")

        buf_samp       = max(PREDICTION_WINDOW_SAMP * 2, WINDOW * self.sfreq + 256)
        self._ring     = np.zeros((self.n_channels, buf_samp))
        self._buf_size = buf_samp
        self._head     = 0
        self._lock     = threading.Lock()
        self._finished = False

    def start(self):
        threading.Thread(target=self._stream_loop, daemon=True).start()

    @property
    def finished(self):
        return self._finished

    def _stream_loop(self):
        chunk = PREDICTION_STRIDE_SAMP
        delay = chunk / (self.sfreq * self._speed)
        ptr   = 0
        while ptr < self._total_samp:
            end   = min(ptr + chunk, self._total_samp)
            block = self._data[:, ptr:end]
            n     = block.shape[1]
            with self._lock:
                self._ring = np.hstack([self._ring[:, n:], block])
                self._head += n
            ptr += chunk
            time.sleep(delay)
        self._finished = True
        print("[EDF] Replay finished.")

    def get_data(self, n_samples: int) -> np.ndarray:
        with self._lock:
            available = min(n_samples, self._buf_size)
            return self._ring[:, -available:].copy()

    @property
    def head(self) -> int:
        with self._lock:
            return self._head


def preprocess_for_model(data: np.ndarray, sfreq: float) -> np.ndarray:
    data = filter_data(data.astype(np.float64), sfreq,
                       l_freq=8., h_freq=30.,
                       method='fir', phase='zero', verbose=False)
    mean = data.mean(axis=1, keepdims=True)
    std  = data.std(axis=1,  keepdims=True) + 1e-6
    return ((data - mean) / std)[np.newaxis, ..., np.newaxis]


class PredictionEngine:
    def __init__(self, replayer: EdfReplayer, arduino: ArduinoLink,
                 show_ground: bool = True):
        self.replayer    = replayer
        self.arduino     = arduino
        self.show_ground = show_ground
        self.sfreq       = replayer.sfreq
        self.n_channels  = replayer.n_channels
        self.running     = False

        self._buf      = np.zeros((self.n_channels, PREDICTION_WINDOW_SAMP))
        self._move_buf = collections.deque(maxlen=SMOOTH_WINDOW)
        self._type_buf = collections.deque(maxlen=SMOOTH_WINDOW)
        self._lock     = threading.Lock()

        # public state
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
                self._models_ok = True
                print("[Model] Both models loaded ✓")
            except Exception as e:
                print(f"[Model] Load failed: {e}")

        self._events       = sorted(replayer.events, key=lambda x: x[0])
        self._event_idx    = 0
        self._stride_start = 0

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _loop(self):
        stride = PREDICTION_STRIDE_SAMP

        while self.running and not self.replayer.finished:
            # ── wait for new data ──────────────────────────────────────────
            eeg = self.replayer.get_data(stride)
            if eeg.shape[1] < stride:
                time.sleep(0.005)
                continue

            # ── advance model input buffer ─────────────────────────────────
            with self._lock:
                self._buf = np.hstack([self._buf[:, stride:], eeg[:, -stride:]])

            current_head        = self.replayer.head
            self._stride_start += stride

            # ── ground truth check ─────────────────────────────────────────
            if self.show_ground:
                while (self._event_idx < len(self._events) and
                       self._events[self._event_idx][0] <= current_head):
                    _, label = self._events[self._event_idx]
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{ts}]  ★  GROUND TRUTH: {label}")
                    self._event_idx += 1

            if not self._models_ok:
                continue

            # ── preprocess once, reuse for both models ─────────────────────
            inp = preprocess_for_model(self._buf.copy(), self.sfreq)

            # ── MOVEMENT MODEL: always run ─────────────────────────────────
            mv_p = float(self._mv_model.predict(inp, verbose=0)[0][0])
            self._move_buf.append(mv_p)

            with self._lock:
                self.move_prob_smooth = float(np.mean(self._move_buf))
                currently_moving      = self.movement_state
                near_threshold        = self.move_prob_smooth > (MOVE_THRESHOLD_ON * 0.85)

            # ── TYPE MODEL: only run when moving or approaching threshold ──
            # This is the key optimisation — skip the second model during
            # clear rest periods, roughly halving inference time.
            if currently_moving or near_threshold:
                ty_p = float(self._ty_model.predict(inp, verbose=0)[0][0])
                self._type_buf.append(ty_p)
                with self._lock:
                    self.type_prob_smooth = float(np.mean(self._type_buf))

            # ── state machine ──────────────────────────────────────────────
            with self._lock:
                self._state_machine()

            # NO sleep here — run as fast as inference allows

        self._print_accuracy()

    def _state_machine(self):
        ms = self.move_prob_smooth
        ts = self.type_prob_smooth
        t  = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if not self.movement_state:
            if ms > MOVE_THRESHOLD_ON:
                self._move_counter += 1
                if self._move_counter >= MIN_MOVE_FRAMES:
                    self.movement_state    = True
                    self._move_counter     = 0
                    self._rest_counter     = 0
                    self._type_buf.clear()
                    self.current_direction = "MOVEMENT STARTED"
                    print(f"[{t}]  MOVEMENT STARTED (p={ms:.3f})")
            else:
                self._move_counter     = 0
                self.current_direction = "REST"
                self.arduino.send(CMD_REST)
        else:
            if ms < MOVE_THRESHOLD_OFF:
                self._rest_counter += 1
                if self._rest_counter >= MIN_REST_FRAMES:
                    self.movement_state    = False
                    self.current_direction = "REST"
                    self._rest_counter     = 0
                    self.arduino.send(CMD_REST)
                    print(f"[{t}]  MOVEMENT ENDED   (p={ms:.3f})")
            else:
                self._rest_counter = 0
                # flipped mapping (corrected for label inversion)
                if ts > TYPE_THRESHOLD:
                    direction = "RIGHT"
                elif ts < (1 - TYPE_THRESHOLD):
                    direction = "LEFT"
                else:
                    direction = "UNCERTAIN"

                if direction != "UNCERTAIN":
                    self.current_direction = direction
                    cmd = CMD_LEFT if direction == "LEFT" else CMD_RIGHT
                    self.arduino.send(cmd)
                    print(f"[{t}]  → {direction:5s}  "
                          f"(move={ms:.3f}, type={ts:.3f})")

    def _print_accuracy(self):
        print("\n" + "═" * 50)
        print("  REPLAY COMPLETE")
        print("═" * 50)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "direction":  self.current_direction,
                "moving":     self.movement_state,
                "move_prob":  self.move_prob_smooth,
                "type_prob":  self.type_prob_smooth,
                "models_ok":  self._models_ok,
                "arduino_ok": self.arduino.connected,
                "finished":   self.replayer.finished,
                "head":       self.replayer.head,
                "total":      self.replayer._total_samp,
            }


class ReplayGraph:
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
        "RIGHT":            "#f85149",
        "REST":             "#3fb950",
        "MOVEMENT STARTED": "#d29922",
        "UNCERTAIN":        "#8b949e",
        "INITIALISING…":    "#8b949e",
        "NO MODELS":        "#8b949e",
    }
    CURVE_COLORS = ["#58a6ff","#3fb950","#f78166","#d29922",
                    "#bc8cff","#79c0ff","#56d364","#ffa657"]

    def __init__(self, replayer: EdfReplayer, engine: PredictionEngine):
        self.replayer   = replayer
        self.engine     = engine
        self.sfreq      = replayer.sfreq
        self.n_channels = replayer.n_channels
        self.ch_names   = replayer.ch_names
        self.num_points = WINDOW * self.sfreq

        self.app = QtWidgets.QApplication([])
        self._build_ui()

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.start(50)
        QtWidgets.QApplication.instance().exec()

    def _build_ui(self):
        pg.setConfigOption('background', self.BG)
        pg.setConfigOption('foreground', self.TEXT)

        self.win = QtWidgets.QWidget()
        self.win.setStyleSheet(f"background:{self.BG};")
        self.win.setWindowTitle("BCI Replay Test  ·  Offline EDF → EEGNet → Arduino")
        self.win.resize(1340, 900)

        root = QtWidgets.QVBoxLayout(self.win)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # header + progress
        hrow = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("⬤  BCI REPLAY TEST")
        title.setStyleSheet(f"color:{self.AMBER};font-size:13px;"
                            f"font-family:'SF Mono','Fira Code',monospace;"
                            f"letter-spacing:4px;")
        hrow.addWidget(title)
        hrow.addSpacing(20)

        self.progress_bar_outer = QtWidgets.QWidget()
        self.progress_bar_outer.setFixedHeight(10)
        self.progress_bar_outer.setStyleSheet("background:#21262d;border-radius:5px;")
        self.progress_bar_inner = QtWidgets.QWidget(self.progress_bar_outer)
        self.progress_bar_inner.setGeometry(0, 0, 0, 10)
        self.progress_bar_inner.setStyleSheet(f"background:{self.AMBER};border-radius:5px;")
        hrow.addWidget(self.progress_bar_outer, stretch=1)

        self.progress_label = QtWidgets.QLabel("0%")
        self.progress_label.setStyleSheet(
            f"color:{self.DIM};font-size:10px;"
            f"font-family:'SF Mono','Fira Code',monospace;")
        hrow.addSpacing(8)
        hrow.addWidget(self.progress_label)
        root.addLayout(hrow)

        # body
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body)

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

        def lbl(text, size=11, color=None, bold=False, align_center=False):
            w = QtWidgets.QLabel(text)
            w.setStyleSheet(
                f"color:{color or self.DIM};font-size:{size}px;"
                f"font-family:'SF Mono','Fira Code',monospace;"
                f"{'font-weight:bold;' if bold else ''}")
            if align_center:
                w.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            w.setWordWrap(True)
            return w

        sl.addWidget(lbl("PREDICTION", 9))
        self.dir_label = QtWidgets.QLabel("—")
        self.dir_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.dir_label.setStyleSheet(
            f"color:{self.ACCENT};font-size:36px;font-weight:bold;"
            f"font-family:'SF Mono','Fira Code',monospace;")
        sl.addWidget(self.dir_label)

        self.state_label = lbl("initialising", 10, self.DIM, align_center=True)
        sl.addWidget(self.state_label)

        sl.addSpacing(4)
        sl.addWidget(lbl("ARDUINO CMD", 9))
        self.cmd_label = QtWidgets.QLabel("—")
        self.cmd_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.cmd_label.setStyleSheet(
            f"color:{self.AMBER};font-size:28px;font-weight:bold;"
            f"font-family:'SF Mono','Fira Code',monospace;"
            f"background:#1c2128;border-radius:6px;padding:6px;")
        sl.addWidget(self.cmd_label)

        sl.addSpacing(8)
        sl.addWidget(lbl("MOVE PROB", 9))
        self.move_bar = self._make_bar(sl, self.GREEN)
        self.move_val = lbl("0.00", 9, self.GREEN)
        sl.addWidget(self.move_val)

        sl.addSpacing(4)
        sl.addWidget(lbl("TYPE PROB", 9))
        self.type_bar = self._make_bar(sl, self.ACCENT)
        self.type_val = lbl("0.00", 9, self.ACCENT)
        sl.addWidget(self.type_val)

        sl.addSpacing(8)
        sl.addWidget(lbl("★ = ground truth in terminal", 9, self.AMBER))
        sl.addStretch()
        sl.addWidget(lbl(
            "● models loaded" if MODELS_AVAILABLE else "○ vis-only mode",
            9, self.GREEN if MODELS_AVAILABLE else self.DIM))
        self.finished_label = lbl("", 10, self.AMBER)
        sl.addWidget(self.finished_label)

        # EEG curves
        self.plots, self.curves = [], []
        for i in range(self.n_channels):
            p = self.plot_widget.addPlot(row=i, col=0)
            p.setLabel('left', self.ch_names[i], color=self.DIM, size='8pt')
            if i < self.n_channels - 1:
                p.hideAxis('bottom')
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

    def update(self):
        s = self.engine.get_state()

        # EEG traces
        data = self.replayer.get_data(self.num_points)
        if data.shape[1] > 0:
            for i in range(self.n_channels):
                sig = data[i].copy()
                sig = sig - sig.mean()
                self.curves[i].setData(sig.tolist())

        # progress bar
        total = s["total"]
        head  = s["head"]
        pct   = int(100 * head / total) if total > 0 else 0
        self.progress_bar_inner.setFixedWidth(
            int(pct / 100 * self.progress_bar_outer.width()))
        self.progress_label.setText(f"{pct}%")

        if s["finished"]:
            self.finished_label.setText("✓ REPLAY DONE")

        # direction label
        direction = s["direction"] if s["models_ok"] else "NO MODELS"
        color = self.DIR_COLORS.get(direction, self.DIM)
        self.dir_label.setStyleSheet(
            f"color:{color};font-size:36px;font-weight:bold;"
            f"font-family:'SF Mono','Fira Code',monospace;")
        self.dir_label.setText(direction)

        cmd_map   = {"LEFT": "1  (LEFT)", "RIGHT": "2  (RIGHT)", "REST": "0  (REST)"}
        cmd_color = {"LEFT": self.ACCENT, "RIGHT": self.RED, "REST": self.GREEN}
        self.cmd_label.setStyleSheet(
            f"color:{cmd_color.get(direction, self.AMBER)};font-size:28px;"
            f"font-weight:bold;font-family:'SF Mono','Fira Code',monospace;"
            f"background:#1c2128;border-radius:6px;padding:6px;")
        self.cmd_label.setText(cmd_map.get(direction, "—"))

        sc = self.GREEN if s["moving"] else self.DIM
        self.state_label.setText("● MOVING" if s["moving"] else "○ at rest")
        self.state_label.setStyleSheet(
            f"color:{sc};font-size:10px;"
            f"font-family:'SF Mono','Fira Code',monospace;")

        self._set_bar(self.move_bar, s["move_prob"])
        self._set_bar(self.type_bar, s["type_prob"])
        self.move_val.setText(f"{s['move_prob']:.3f}")
        self.type_val.setText(f"{s['type_prob']:.3f}")

        self.app.processEvents()


def main():
    arduino  = ArduinoLink(ARDUINO_PORT, ARDUINO_BAUD, enabled=not NO_ARDUINO)
    replayer = EdfReplayer(PATH, speed=SPEED)
    engine   = PredictionEngine(replayer, arduino, show_ground=SHOW_GROUND)

    replayer.start()
    engine.start()

    try:
        ReplayGraph(replayer, engine)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        engine.stop()
        arduino.close()
        print("Done.")


if __name__ == '__main__':
    main()