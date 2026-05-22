# interaction/state_machine.py
# ─────────────────────────────────────────────────────────────
# State flow:
#
#   IDLE    ──[PointingGate activates]──────────────► POINTING
#   POINTING ──[ray hits marker N frames]───────────► LOCKED
#   LOCKED   ──[one grace frame, gate bypassed]─────► CONTROL
#   CONTROL  ──[open palm detected]─────────────────► RELEASE
#   RELEASE  ──[next frame]─────────────────────────► IDLE
#
# Key rules:
#   • Pointing gesture (index up, others curled) must be held for
#     POINTING_GATE_ENTER_FRAMES before the ray activates at all.
#   • Once a marker locks, the gate is BYPASSED — user can relax
#     their hand completely and knob/swipe gestures still work.
#   • Open palm from LOCKED or CONTROL fires OPEN_PALM event and
#     returns to IDLE, where the cycle starts fresh.
#   • If the pointing pose drops during POINTING (pre-lock), the
#     gate has a generous EXIT hysteresis to tolerate wobble.
# ─────────────────────────────────────────────────────────────

from enum import Enum, auto
from typing import Optional, List, Tuple

from perception.aruco_detector import MarkerData
from perception.hand_tracker import HandData
from interaction.pointing import PointingRay
from interaction.lock_manager import LockManager
from interaction.gestures import (
    KnobGesture, SwipeGesture, GestureEvent, GestureType,
    PointingGate, is_open_palm,
)


class State(Enum):
    IDLE     = auto()   # hand visible, waiting for pointing gesture
    POINTING = auto()   # pointing confirmed, ray active, seeking target
    LOCKED   = auto()   # marker just locked — one-frame grace period
    CONTROL  = auto()   # locked + gestures active, pointing not required
    RELEASE  = auto()   # open palm fired — one frame then back to IDLE


class InteractionStateMachine:
    """
    Central coordinator. Call update() every frame.

    Public readable attributes (set after each update())
    ─────────────────────────────────────────────────────
    state            : current State
    locked_id        : locked marker ID or None
    lock_progress    : 0.0–1.0 lock-in progress
    ray_origin       : (x,y) pixel tuple or None  (None when gate inactive)
    ray_tip          : (x,y) pixel tuple or None
    knob_value       : current 0–100 control value
    last_event       : most recent GestureEvent or None
    pointing_active  : True while PointingGate is open
    point_progress   : 0.0–1.0 progress toward gate activation (pre-active)
    """

    def __init__(self, frame_w: int, frame_h: int):
        self._w = frame_w
        self._h = frame_h

        self._ray   = PointingRay()
        self._lock  = LockManager()
        self._knob  = KnobGesture()
        self._swipe = SwipeGesture()
        self._gate  = PointingGate()

        self._prev_locked_id: Optional[int] = None

        # ── Public readable state ──────────────────────────────
        self.state:           State           = State.IDLE
        self.locked_id:       Optional[int]   = None
        self.lock_progress:   float           = 0.0
        self.ray_origin:      Optional[Tuple] = None
        self.ray_tip:         Optional[Tuple] = None
        self.knob_value:      float           = 50.0
        self.last_event:      Optional[GestureEvent] = None
        self.pointing_active: bool            = False
        self.point_progress:  float           = 0.0

    # ── Main update ───────────────────────────────────────────

    def update(
        self,
        hand: Optional[HandData],
        markers: List[MarkerData],
    ) -> Optional[GestureEvent]:

        self.last_event = None

        # ── No hand → full idle reset ──────────────────────────
        if hand is None:
            self._go_idle()
            return None

        # ── Open palm check (highest priority, any locked state) ─
        # Must come before gate / ray logic so it fires even when
        # the user has relaxed their pointing finger post-lock.
        if self.state in (State.LOCKED, State.CONTROL):
            if is_open_palm(hand):
                return self._go_release(hand)

        # ── PointingGate ──────────────────────────────────────
        gate_open = self._gate.update(hand)
        self.pointing_active = gate_open
        self.point_progress  = self._gate.enter_progress

        # ── Gate closed and not locked → IDLE, nothing more ───
        if not gate_open:
            # If we were pointing, cleanly collapse back to IDLE
            if self.state == State.POINTING:
                self._go_idle()
            elif self.state == State.IDLE:
                pass   # already idle
            # LOCKED / CONTROL are gate-bypassed so we never land here
            self.ray_origin = None
            self.ray_tip    = None
            return None

        # ── Gate open: compute ray ─────────────────────────────
        origin, tip = self._ray.update(hand, self._w, self._h)
        self.ray_origin = origin
        self.ray_tip    = tip

        # ── Find candidate marker ─────────────────────────────
        candidate    = PointingRay.find_target(origin, tip, markers)
        candidate_id = candidate.marker_id if candidate else None

        locked_id = self._lock.update(candidate_id)

        just_locked = (self._prev_locked_id is None and locked_id is not None)

        self.locked_id       = locked_id
        self._prev_locked_id = locked_id
        self.lock_progress   = self._lock.lock_progress

        # ── No lock yet ───────────────────────────────────────
        if locked_id is None:
            self._set_state(
                State.POINTING if candidate_id is not None else State.IDLE
            )
            return None

        # ── Marker just locked → bypass gate, grace frame ─────
        if just_locked:
            self._gate.bypass(True)
            self._set_state(State.LOCKED)
            return None

        # ── CONTROL: accept gestures ──────────────────────────
        self._set_state(State.CONTROL)

        event = self._swipe.update(hand, self._knob)
        if event is None:
            event = self._knob.update(hand)

        if event:
            self.knob_value = event.value
            self.last_event = event

        return event

    # ── State helpers ─────────────────────────────────────────

    def _set_state(self, new_state: State):
        self.state = new_state

    def _go_idle(self):
        """Full reset — gate, ray, lock, gestures all cleared."""
        self._set_state(State.IDLE)
        self._gate.reset()
        self._ray.reset()
        self._lock.release()
        self._knob.reset()
        self._swipe.reset()
        self.ray_origin      = None
        self.ray_tip         = None
        self.locked_id       = None
        self.lock_progress   = 0.0
        self.pointing_active = False
        self.point_progress  = 0.0
        self._prev_locked_id = None

    def _go_release(self, hand: HandData) -> GestureEvent:
        """Open palm detected — emit event and reset to IDLE next frame."""
        event = GestureEvent(
            gesture_type=GestureType.OPEN_PALM,
            value=self.knob_value,
        )
        self.last_event = event
        self._set_state(State.RELEASE)

        # Release lock and gestures; gate bypass off so IDLE waits
        # for a fresh pointing gesture before anything can start again.
        self._gate.bypass(False)
        self._gate.reset()
        self._lock.release()
        self._knob.reset()
        self._swipe.reset()
        self._ray.reset()
        self.locked_id       = None
        self.lock_progress   = 0.0
        self._prev_locked_id = None
        self.ray_origin      = None
        self.ray_tip         = None

        return event