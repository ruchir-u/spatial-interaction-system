# interaction/lock_manager.py
# ─────────────────────────────────────────────────────────────
# Implements temporal stability and hysteresis for target locking.
# A marker must be consistently hit for LOCK_FRAMES_REQUIRED
# consecutive frames before it becomes the locked target.
# Switching away requires LOCK_HYSTERESIS_FRAMES of disagreement.
# ─────────────────────────────────────────────────────────────

from typing import Optional
import config


class LockManager:
    """
    Tracks candidate marker detections over time and decides
    when to commit to (lock) a target.

    Attributes
    ----------
    locked_id : int | None
        The currently locked marker ID, or None.
    lock_progress : float
        0.0 → 1.0 — how close we are to locking the candidate.
        Useful for drawing a progress indicator.
    """

    def __init__(self):
        self._candidate_id:   Optional[int] = None
        self._candidate_frames: int         = 0
        self._against_frames:   int         = 0   # frames pointing away

        self.locked_id:    Optional[int] = None
        self.lock_progress: float        = 0.0

    # ── Public API ───────────────────────────────────────────

    def update(self, candidate_id: Optional[int]) -> Optional[int]:
        """
        Feed in the marker ID hit this frame (or None for no hit).

        Returns
        -------
        locked_id — the current locked target (may be unchanged).
        """
        if self.locked_id is None:
            self._handle_pre_lock(candidate_id)
        else:
            self._handle_post_lock(candidate_id)

        return self.locked_id

    def release(self):
        """Manually release the current lock (e.g. open palm gesture)."""
        self._reset()

    # ── Internal ─────────────────────────────────────────────

    def _handle_pre_lock(self, candidate_id: Optional[int]):
        if candidate_id is None:
            self._reset()
            return

        if candidate_id != self._candidate_id:
            # New candidate — restart counter
            self._candidate_id     = candidate_id
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1

        self.lock_progress = self._candidate_frames / config.LOCK_FRAMES_REQUIRED

        if self._candidate_frames >= config.LOCK_FRAMES_REQUIRED:
            self.locked_id     = self._candidate_id
            self.lock_progress = 1.0

    def _handle_post_lock(self, candidate_id: Optional[int]):
        if candidate_id == self.locked_id:
            # Still pointing at locked target — reset against counter
            self._against_frames = 0
            self.lock_progress   = 1.0
        else:
            self._against_frames += 1
            if self._against_frames >= config.LOCK_HYSTERESIS_FRAMES:
                self._reset()

    def _reset(self):
        self._candidate_id     = None
        self._candidate_frames = 0
        self._against_frames   = 0
        self.locked_id         = None
        self.lock_progress     = 0.0
