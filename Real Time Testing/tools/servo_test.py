"""
servo_test.py — manual Arduino servo tester.

Type 'l' or 'left' to send LEFT, 'r' or 'right' for RIGHT,
'0' or 'rest' for REST, 'q' to quit.
"""

import sys
import time
import glob
import serial


# ── CONFIG ─────────────────────────────────────────────────────────────────
# Auto-detect the Arduino port. Override this string manually if needed.
def find_arduino_port():
    candidates = glob.glob("/dev/cu.usbmodem*")
    if not candidates:
        return None
    return candidates[0]


ARDUINO_PORT = find_arduino_port() or "/dev/cu.usbmodem2101"
BAUD         = 9600
# ───────────────────────────────────────────────────────────────────────────


def main():
    print(f"Opening {ARDUINO_PORT} @ {BAUD} …")
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"Failed to open port: {e}")
        sys.exit(1)

    # Mirror the BciReplay DTR trick so the Arduino doesn't auto-reset
    # during operation.
    ser.dtr = False
    time.sleep(2)                  # let the Arduino finish any boot sequence
    ser.reset_input_buffer()
    print("Connected. Type commands below (q to quit).")
    print("Commands: 0/rest, 1/l/left, 2/r/right")
    print("-" * 50)

    cmd_map = {
        "0": b"0", "rest": b"0",
        "1": b"1", "l":    b"1", "left":  b"1",
        "2": b"2", "r":    b"2", "right": b"2",
    }

    try:
        while True:
            try:
                raw = input("> ").strip().lower()
            except EOFError:
                break
            if raw in ("q", "quit", "exit"):
                break
            if not raw:
                continue
            if raw not in cmd_map:
                print(f"  Unknown: {raw!r}. Try: 0/1/2 or rest/left/right")
                continue

            byte = cmd_map[raw]
            try:
                ser.write(byte)
                print(f"  sent: {byte.decode()}")
            except serial.SerialException as e:
                print(f"  WRITE FAILED: {e}")
                break

            # Read back any response from the Arduino (non-blocking)
            time.sleep(0.1)
            while ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    print(f"  arduino: {line}")
    finally:
        try:
            ser.write(b"0")   # park at rest on exit
            time.sleep(0.1)
        except Exception:
            pass
        ser.close()
        print("Closed.")


if __name__ == "__main__":
    main()