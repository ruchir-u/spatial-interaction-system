# perception/hand_tracker.py
# ─────────────────────────────────────────────────────────────
# Wraps MediaPipe Hands and returns structured landmark data.
# Only the first detected hand is used (single-hand system).
# ─────────────────────────────────────────────────────────────
import math
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# MediaPipe landmark indices
LM_WRIST        = 0
LM_THUMB_TIP    = 4
LM_INDEX_MCP    = 5    # knuckle — used for hand-size normalisation
LM_INDEX_TIP    = 8
LM_MIDDLE_MCP   = 9
LM_MIDDLE_TIP   = 12
LM_RING_MCP     = 13
LM_RING_TIP     = 16
LM_PINKY_MCP    = 17
LM_PINKY_TIP    = 20


@dataclass
class HandData:
    """
    Processed hand information extracted from one MediaPipe result.

    All x / y values are *normalised* (0-1).
    All z values are MediaPipe's relative depth (smaller = closer).
    pixel_* fields are pre-multiplied to frame dimensions for convenience.
    """
    # Key landmark positions (normalised)
    wrist:       Tuple[float, float, float]
    index_tip:   Tuple[float, float, float]
    thumb_tip:   Tuple[float, float, float]
    middle_tip:  Tuple[float, float, float]
    ring_tip:    Tuple[float, float, float]
    pinky_tip:   Tuple[float, float, float]

    # MCP (knuckle) positions (normalised) — base of each finger
    index_mcp:   Tuple[float, float, float]   # landmark 5
    middle_mcp:  Tuple[float, float, float]   # landmark 9
    ring_mcp:    Tuple[float, float, float]   # landmark 13
    pinky_mcp:   Tuple[float, float, float]   # landmark 17

    # Pixel-space positions (int)
    wrist_px:      Tuple[int, int]
    index_tip_px:  Tuple[int, int]
    thumb_tip_px:  Tuple[int, int]

    # Derived metrics
    pinch_distance_norm: float   # normalised by hand size (0-1)
    hand_size_px: float          # wrist-to-index-MCP distance in pixels

    # Raw landmarks list (all 21) for any future use
    raw_landmarks: object = None


class HandTracker:
    """
    Wraps MediaPipe Hands for single-hand tracking.

    Usage
    -----
    tracker  = HandTracker()
    hand     = tracker.process(frame)   # returns HandData or None
    tracker.close()
    """

    def __init__(self, max_hands: int = 1,
                 model_complexity: int = 0,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5):

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ── Public API ───────────────────────────────────────────

    def process(self, frame: np.ndarray) -> Optional[HandData]:
        """
        Process a BGR frame and return HandData for the first hand found.
        Returns None if no hand is detected.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        lms = result.multi_hand_landmarks[0].landmark

        def lm(idx) -> Tuple[float, float, float]:
            p = lms[idx]
            return (p.x, p.y, p.z)

        def px(idx) -> Tuple[int, int]:
            p = lms[idx]
            return (int(p.x * w), int(p.y * h))

        wrist      = lm(LM_WRIST)
        index_tip  = lm(LM_INDEX_TIP)
        thumb_tip  = lm(LM_THUMB_TIP)
        middle_tip = lm(LM_MIDDLE_TIP)
        ring_tip   = lm(LM_RING_TIP)
        pinky_tip  = lm(LM_PINKY_TIP)
        index_mcp  = lm(LM_INDEX_MCP)
        middle_mcp = lm(LM_MIDDLE_MCP)
        ring_mcp   = lm(LM_RING_MCP)
        pinky_mcp  = lm(LM_PINKY_MCP)

        # Hand size: wrist → index MCP in pixel space (stable reference)
        hand_size_px = math.dist(
            (wrist[0] * w, wrist[1] * h),
            (index_mcp[0] * w, index_mcp[1] * h)
        )
        hand_size_px = max(hand_size_px, 1.0)   # avoid division by zero

        # Pinch = thumb_tip ↔ index_tip, normalised by hand size
        pinch_px = math.dist(
            (thumb_tip[0] * w, thumb_tip[1] * h),
            (index_tip[0] * w, index_tip[1] * h)
        )
        pinch_norm = pinch_px / hand_size_px

        return HandData(
            wrist=wrist,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            middle_tip=middle_tip,
            ring_tip=ring_tip,
            pinky_tip=pinky_tip,
            index_mcp=index_mcp,
            middle_mcp=middle_mcp,
            ring_mcp=ring_mcp,
            pinky_mcp=pinky_mcp,
            wrist_px=px(LM_WRIST),
            index_tip_px=px(LM_INDEX_TIP),
            thumb_tip_px=px(LM_THUMB_TIP),
            pinch_distance_norm=pinch_norm,
            hand_size_px=hand_size_px,
            raw_landmarks=lms,
        )

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()