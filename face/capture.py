"""Photo capture module. Stub — implementation TBD.

Expected to return raw JPEG bytes that downstream `register` / `recognize` can
consume directly. Possible future backends:

    - cv2.VideoCapture(0)            local webcam
    - read from a watched folder     simulated robot
    - HTTP / WebSocket from robot    real hardware

The single public function is `capture()` so callers don't need to know
which backend is wired in.
"""


def capture() -> bytes:
    """Return one captured frame as JPEG bytes."""
    raise NotImplementedError("capture: backend not configured yet")
