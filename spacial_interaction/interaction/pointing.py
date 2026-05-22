# interaction/pointing.py
# ─────────────────────────────────────────────────────────────
# Computes the pointing ray from wrist → index_tip and finds
# which ArUco marker (if any) the ray is aimed at.
# ─────────────────────────────────────────────────────────────

import math
import numpy as np
from typing import Optional, Tuple, List

import config
from perception.aruco_detector import MarkerData
from perception.hand_tracker import HandData


class PointingRay:
    """
    Holds the current smoothed pointing ray and exposes target-finding.

    The ray is defined by two points in pixel space:
        origin  — index fingertip pixel position
        tip     — projected endpoint (origin + direction * RAY_SCALE)

    Smoothing is applied via EMA so jitter doesn't flip targets.
    """

    def __init__(self):
        self._smooth_tip_x: Optional[float] = None
        self._smooth_tip_y: Optional[float] = None

    # ── Public API ───────────────────────────────────────────

    def update(self, hand: HandData,
               frame_w: int, frame_h: int) -> Tuple[Tuple[int, int],
                                                     Tuple[int, int]]:
        """
        Compute (and smooth) the ray from the given HandData.

        Returns
        -------
        (origin_px, tip_px)  — both in pixel space (int tuples)
        """
        ix, iy = hand.index_tip_px
        wx, wy = hand.wrist_px

        # Direction vector (normalised coords → scale to pixel deltas)
        dx = hand.index_tip[0] - hand.wrist[0]
        dy = hand.index_tip[1] - hand.wrist[1]

        raw_tip_x = ix + dx * config.RAY_SCALE
        raw_tip_y = iy + dy * config.RAY_SCALE

        # EMA smoothing on tip position
        alpha = config.RAY_SMOOTH_ALPHA
        if self._smooth_tip_x is None:
            self._smooth_tip_x = raw_tip_x
            self._smooth_tip_y = raw_tip_y
        else:
            self._smooth_tip_x = alpha * raw_tip_x + (1 - alpha) * self._smooth_tip_x
            self._smooth_tip_y = alpha * raw_tip_y + (1 - alpha) * self._smooth_tip_y

        origin = (ix, iy)
        tip    = (int(self._smooth_tip_x), int(self._smooth_tip_y))
        return origin, tip

    def reset(self):
        self._smooth_tip_x = None
        self._smooth_tip_y = None

    # ── Target finding ────────────────────────────────────────

    @staticmethod
    def find_target(
        origin: Tuple[int, int],
        tip:    Tuple[int, int],
        markers: List[MarkerData],
    ) -> Optional[MarkerData]:
        """
        Return the marker whose centre is closest to the ray,
        within RAY_HIT_THRESHOLD pixels.  Returns None if no hit.

        Method: perpendicular distance from marker centre to the
        infinite line defined by (origin → tip).
        """
        if not markers:
            return None

        best_marker   = None
        best_distance = float("inf")

        ox, oy = origin
        tx, ty = tip

        # Direction vector of ray
        rdx = tx - ox
        rdy = ty - oy
        ray_len = math.hypot(rdx, rdy)

        if ray_len < 1e-6:
            return None

        for marker in markers:
            mx, my = marker.center

            # Perpendicular distance from point (mx,my) to line (origin→tip)
            dist = abs(rdx * (oy - my) - rdy * (ox - mx)) / ray_len

            # Also check that the marker is in FRONT of the origin
            # (dot product of ray dir with origin→marker should be positive)
            dot = rdx * (mx - ox) + rdy * (my - oy)
            if dot < 0:
                continue

            if dist < best_distance:
                best_distance = dist
                best_marker   = marker

        if best_distance <= config.RAY_HIT_THRESHOLD:
            return best_marker

        return None
