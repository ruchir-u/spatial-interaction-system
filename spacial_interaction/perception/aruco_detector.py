import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import config


@dataclass
class MarkerData:
    marker_id: int
    corners: np.ndarray
    center: Tuple[int, int]
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None


class ArucoDetector:

    def __init__(self):
        # ── Dictionary ────────────────────────────────────────
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, config.ARUCO_DICT)
        )

        # ── Detection parameters (tuned for webcam/DroidCam) ──
        self.parameters = cv2.aruco.DetectorParameters()

        # Adaptive threshold window sizes — wider range handles dark feeds
        self.parameters.adaptiveThreshWinSizeMin  = 3
        self.parameters.adaptiveThreshWinSizeMax  = 53
        self.parameters.adaptiveThreshWinSizeStep = 4

        # Accept smaller markers (farther away / lower res)
        self.parameters.minMarkerPerimeterRate    = 0.02
        self.parameters.maxMarkerPerimeterRate    = 4.0

        # Max error correction — more forgiving decoding
        self.parameters.errorCorrectionRate       = 0.5

        # Sub-pixel corner refinement for stable pose
        self.parameters.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX

        # ── New API detector (required in OpenCV 4.8+) ────────
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        # ── Dummy camera matrix (replace with calibrated values) ─
        self._camera_matrix = np.array([
            [800,   0, 320],
            [  0, 800, 240],
            [  0,   0,   1]
        ], dtype=np.float32)
        self._dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # 3D object points for a marker of size marker_length metres
        self._marker_length = 0.05
        hl = self._marker_length / 2
        self._obj_pts = np.array([
            [-hl,  hl, 0],
            [ hl,  hl, 0],
            [ hl, -hl, 0],
            [-hl, -hl, 0],
        ], dtype=np.float32)

    # ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[MarkerData]:

        # ── Preprocessing: fix dark/low-contrast camera feeds ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # ── Detection (new API — works on OpenCV 4.8+) ────────
        corners_list, ids, rejected = self.detector.detectMarkers(gray)

        print(f"[DEBUG] IDs: {ids.flatten().tolist() if ids is not None else None} | "
              f"Rejected candidates: {len(rejected) if rejected else 0}")

        # Draw raw rejected candidates in blue (useful for debugging angle/border issues)
        if rejected:
            for r in rejected:
                pts = r.reshape(4, 2).astype(int)
                cv2.polylines(frame, [pts], True, (255, 100, 0), 1)

        # Draw confirmed markers in green
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

        # Overlay debug text
        id_text = ids.flatten().tolist() if ids is not None else "None"
        cv2.putText(frame, f"IDs: {id_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if ids is None:
            return []

        # ── Process each valid marker ─────────────────────────
        results: List[MarkerData] = []

        for corners, marker_id in zip(corners_list, ids.flatten()):

            pts = corners.reshape(4, 2)

            perimeter = cv2.arcLength(pts, closed=True)
            if perimeter < config.MIN_MARKER_PERIMETER:
                continue

            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())

            rvec, tvec = self._estimate_pose(pts)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            if rvec is not None and tvec is not None:
                try:
                    cv2.drawFrameAxes(
                        frame,
                        self._camera_matrix,
                        self._dist_coeffs,
                        rvec, tvec,
                        self._marker_length * 0.6
                    )
                except Exception:
                    pass

            results.append(MarkerData(
                marker_id=int(marker_id),
                corners=pts,
                center=(cx, cy),
                rvec=rvec,
                tvec=tvec,
            ))

        return results

    # ─────────────────────────────────────────────────────────

    def _estimate_pose(self, pts: np.ndarray):
        """
        Estimate pose using solvePnP — works on all OpenCV versions.
        estimatePoseSingleMarkers was removed in OpenCV 4.8.
        """
        try:
            img_pts = pts.astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                self._obj_pts,
                img_pts,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE   # best method for square markers
            )
            if success:
                return rvec, tvec
        except Exception as e:
            print(f"[Pose] solvePnP failed: {e}")
        return None, None
