# input/camera_source.py

import cv2
import time
from typing import Union


class CameraSource:
    def __init__(self,
                 source: Union[str, int],
                 width:  int  = 1280,
                 height: int  = 720,
                 flip:   bool = True):
        self._source = source
        self._width  = width
        self._height = height
        self._flip   = flip
        self._cap: cv2.VideoCapture = None

    # ── Lifecycle ─────────────────────────────────────────────

    def open(self, retry: int = 3, delay: float = 2.0) -> bool:
        for attempt in range(1, retry + 1):
            print(f"[Camera] Connecting to {self._source}  "
                  f"(attempt {attempt}/{retry}) ...")

            self._cap = cv2.VideoCapture(self._source)

            if self._cap.isOpened():

                # ✅ Only set resolution for webcam (int source)
                if isinstance(self._source, int):
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

                # ✅ Get actual resolution (important for video)
                actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                print(f"[Camera] Connected ✓  ({actual_w}×{actual_h})")

                # Stabilize stream
                for _ in range(3):
                    self._cap.read()

                return True

            print(f"[Camera] Failed — retrying in {delay}s ...")
            time.sleep(delay)

        print("[Camera] Could not connect.")
        return False

    # ── Frame read ────────────────────────────────────────────

    def read(self):
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ok, frame = self._cap.read()

        # 🔁 If video file → loop instead of failing
        if not ok or frame is None:
            if isinstance(self._source, str):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    return False, None
            else:
                print("[Camera] Frame dropped")
                return False, None

        if self._flip:
            frame = cv2.flip(frame, 1)

        return True, frame

    # ── Cleanup ───────────────────────────────────────────────

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()