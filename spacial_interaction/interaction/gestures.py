# interaction/gestures.py
# ─────────────────────────────────────────────────────────────
# Gesture recognisers: Knob (pinch), Swipe up/down,
# Pointing gate, and Open-palm release.
# ─────────────────────────────────────────────────────────────

from dataclasses import dataclass
from collections import deque
from enum import Enum, auto
from typing import Optional

import config
from perception.hand_tracker import HandData


# ── Shared event type ─────────────────────────────────────────

class GestureType(Enum):
    KNOB_CHANGE  = auto()
    SWIPE_UP     = auto()
    SWIPE_DOWN   = auto()
    OPEN_PALM    = auto()


@dataclass
class GestureEvent:
    gesture_type: GestureType
    delta:  float = 0.0
    value:  float = 0.0


# ── Pointing gesture detector ─────────────────────────────────

def is_pointing_gesture(hand: HandData) -> bool:
    """
    Raw single-frame pointing pose check.

    True when:
      - Index tip is clearly ABOVE (lower Y) its MCP  → extended
      - Middle, ring, pinky tips are BELOW their MCPs → curled
      - Thumb-index pinch is open (not pinching)

    Uses MCP joints for per-finger curl detection, making it robust
    regardless of wrist rotation or hand distance from camera.
    """
    # Index extended: tip well above its knuckle
    index_up = (
        hand.index_tip[1] < hand.index_mcp[1] - config.POINT_INDEX_EXTEND_THRESH
    )

    # Middle, ring, pinky curled: tip below (or at) their knuckle
    curl_thresh = config.POINT_CURL_THRESH
    middle_curled = hand.middle_tip[1] > hand.middle_mcp[1] + curl_thresh
    ring_curled   = hand.ring_tip[1]   > hand.ring_mcp[1]   + curl_thresh
    pinky_curled  = hand.pinky_tip[1]  > hand.pinky_mcp[1]  + curl_thresh

    # Not in a pinch (thumb not squeezing index)
    not_pinching = hand.pinch_distance_norm > config.PINCH_MAX_NORM * 0.5

    return index_up and middle_curled and ring_curled and pinky_curled and not_pinching


# ── Open-palm detector ────────────────────────────────────────

def is_open_palm(hand: HandData) -> bool:
    """
    True when all four fingers are clearly extended above their MCPs
    AND the hand is not pinching.

    Uses per-finger MCP references so it works at any distance
    and doesn't depend on wrist position.
    """
    thresh = config.PALM_EXTEND_THRESH

    index_up  = hand.index_tip[1]  < hand.index_mcp[1]  - thresh
    middle_up = hand.middle_tip[1] < hand.middle_mcp[1] - thresh
    ring_up   = hand.ring_tip[1]   < hand.ring_mcp[1]   - thresh
    pinky_up  = hand.pinky_tip[1]  < hand.pinky_mcp[1]  - thresh

    palm_open = hand.pinch_distance_norm > config.PINCH_MAX_NORM * config.PALM_MIN_PINCH_NORM

    return index_up and middle_up and ring_up and pinky_up and palm_open


# ── Pointing gate (debounced) ─────────────────────────────────

class PointingGate:
    """
    Debounced wrapper around is_pointing_gesture().

    Activation:   pose must be held for POINTING_GATE_ENTER_FRAMES
                  consecutive frames before .active becomes True.

    Deactivation: pose must be ABSENT for POINTING_GATE_EXIT_FRAMES
                  consecutive frames before .active becomes False.
                  This prevents brief hand wobble from killing the ray
                  while the user is trying to lock onto a target.

    The gate is bypassed (always reports active) once a marker is
    locked — the state machine controls that via bypass().
    """

    def __init__(self):
        self._enter_count:  int  = 0
        self._exit_count:   int  = 0
        self._bypassed:     bool = False
        self.active:        bool = False

        # Readable progress toward activation (0–1), useful for UI hint
        self.enter_progress: float = 0.0

    def bypass(self, on: bool):
        """
        When on=True the gate always reports active regardless of pose.
        Call with on=True upon marker lock, on=False upon release.
        """
        self._bypassed = on
        if on:
            self.active         = True
            self.enter_progress = 1.0

    def update(self, hand: HandData) -> bool:
        """
        Feed in the current HandData every frame.
        Returns self.active.
        """
        if self._bypassed:
            return True

        pose_seen = is_pointing_gesture(hand)

        if not self.active:
            # ── Waiting to activate ───────────────────────────
            if pose_seen:
                self._enter_count += 1
                self._exit_count   = 0
            else:
                self._enter_count  = 0
                self._exit_count   = 0

            self.enter_progress = self._enter_count / config.POINTING_GATE_ENTER_FRAMES

            if self._enter_count >= config.POINTING_GATE_ENTER_FRAMES:
                self.active         = True
                self.enter_progress = 1.0

        else:
            # ── Active: watching for sustained dropout ────────
            if pose_seen:
                self._exit_count = 0
            else:
                self._exit_count += 1

            self._enter_count = 0   # reset enter counter while active

            if self._exit_count >= config.POINTING_GATE_EXIT_FRAMES:
                self._reset_to_inactive()

        return self.active

    def reset(self):
        """Full reset — used when returning to IDLE."""
        self._reset_to_inactive()
        self._bypassed = False

    def _reset_to_inactive(self):
        self._enter_count   = 0
        self._exit_count    = 0
        self.active         = False
        self.enter_progress = 0.0


# ── Knob controller ───────────────────────────────────────────

class KnobGesture:
    """
    Translates pinch-distance changes into a continuous 0–100 value.
    Pinch distance is normalised by hand size for distance-invariance.
    """

    def __init__(self):
        self._value:      float           = 50.0
        self._prev_pinch: Optional[float] = None
        self._smoothed:   Optional[float] = None

    @property
    def value(self) -> float:
        return self._value

    def reset(self):
        self._prev_pinch = None
        self._smoothed   = None

    def update(self, hand: HandData) -> Optional[GestureEvent]:
        raw = hand.pinch_distance_norm

        alpha = config.KNOB_ALPHA
        if self._smoothed is None:
            self._smoothed = raw
        else:
            self._smoothed = alpha * raw + (1 - alpha) * self._smoothed

        pinch = self._smoothed

        if self._prev_pinch is None:
            self._prev_pinch = pinch
            return None

        delta = pinch - self._prev_pinch
        self._prev_pinch = pinch

        if abs(delta) < config.KNOB_DEADZONE:
            return None

        value_delta = delta * config.KNOB_GAIN
        self._value = float(max(0.0, min(100.0, self._value + value_delta)))

        return GestureEvent(
            gesture_type=GestureType.KNOB_CHANGE,
            delta=value_delta,
            value=self._value,
        )


# ── Swipe detector ────────────────────────────────────────────

class SwipeGesture:
    """
    Detects a deliberate vertical swipe of the index finger.
    Fires SWIPE_UP / SWIPE_DOWN with cooldown to prevent repeats.
    """

    def __init__(self):
        self._history:  deque = deque(maxlen=config.SWIPE_HISTORY_LEN)
        self._cooldown: int   = 0

    def reset(self):
        self._history.clear()
        self._cooldown = 0

    def update(self, hand: HandData, knob: KnobGesture) -> Optional[GestureEvent]:
        if self._cooldown > 0:
            self._cooldown -= 1
            self._history.append(hand.index_tip[1])
            return None

        self._history.append(hand.index_tip[1])

        if len(self._history) < config.SWIPE_HISTORY_LEN:
            return None

        net_dy = self._history[-1] - self._history[0]

        if abs(net_dy) < config.SWIPE_THRESHOLD:
            return None

        # Positive dy in MediaPipe = moving DOWN the screen
        if net_dy < 0:
            gesture_type = GestureType.SWIPE_UP
            knob._value  = float(min(100.0, knob._value + config.SWIPE_STEP))
        else:
            gesture_type = GestureType.SWIPE_DOWN
            knob._value  = float(max(0.0, knob._value - config.SWIPE_STEP))

        self._cooldown = config.SWIPE_COOLDOWN_FRAMES
        self._history.clear()

        return GestureEvent(
            gesture_type=gesture_type,
            delta=config.SWIPE_STEP if gesture_type == GestureType.SWIPE_UP
                                    else -config.SWIPE_STEP,
            value=knob._value,
        )