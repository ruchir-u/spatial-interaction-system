# config.py
# ─────────────────────────────────────────────────────────────
# All tunable parameters for the Spatial Interaction System.
# Edit this file to adjust behaviour without touching logic code.
# ─────────────────────────────────────────────────────────────

# ── Camera ────────────────────────────────────────────────────
DROIDCAM_IP   = "10.171.199.114"
DROIDCAM_PORT = 4747
CAMERA_URL    = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video"

USE_DROIDCAM  = True
WEBCAM_INDEX  = 0

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ── ArUco ─────────────────────────────────────────────────────
ARUCO_DICT           = "DICT_4X4_50"
MIN_MARKER_PERIMETER = 20

# ── Pointing Ray ──────────────────────────────────────────────
RAY_SCALE         = 300
RAY_HIT_THRESHOLD = 80

# ── Lock Manager ──────────────────────────────────────────────
LOCK_FRAMES_REQUIRED   = 15   # ~0.5 s at 30 fps
LOCK_HYSTERESIS_FRAMES = 20

# ── Pointing Gesture Gate ─────────────────────────────────────
# Frames the pointing pose must be held continuously to activate
# the ray. Prevents accidental triggering from a passing gesture.
POINTING_GATE_ENTER_FRAMES = 6    # ~0.2 s at 30 fps — quick but deliberate

# Frames the pointing pose must be absent before the ray deactivates
# while in POINTING state (pre-lock). High value = forgiving of wobble.
# Has no effect in CONTROL state — once locked the ray stays until palm.
POINTING_GATE_EXIT_FRAMES  = 10   # ~0.33 s — don't kill on momentary drop

# ── Pointing Gesture Thresholds ───────────────────────────────
# Index finger is considered EXTENDED when its tip Y is this much
# above (less than) the index MCP Y. Normalised 0-1.
POINT_INDEX_EXTEND_THRESH  = 0.06

# A finger is considered CURLED when its tip Y is this much
# below (greater than) its MCP Y. Negative = must be below MCP.
POINT_CURL_THRESH          = -0.02   # tip.y > mcp.y + this → curled

# ── Open Palm Thresholds ──────────────────────────────────────
# All four fingers must be extended by at least this much above
# their own MCP to count as an open palm.
PALM_EXTEND_THRESH         = 0.04

# Additionally require pinch to be wide open (not pinching)
PALM_MIN_PINCH_NORM        = 0.8    # fraction of PINCH_MAX_NORM

# ── Gestures ──────────────────────────────────────────────────
PINCH_MIN_NORM   = 0.05
PINCH_MAX_NORM   = 0.35

KNOB_GAIN        = 150.0
KNOB_ALPHA       = 0.25
KNOB_DEADZONE    = 0.005

SWIPE_THRESHOLD       = 0.06
SWIPE_COOLDOWN_FRAMES = 18
SWIPE_HISTORY_LEN     = 10
SWIPE_STEP            = 10.0

# ── Smoothing ─────────────────────────────────────────────────
RAY_SMOOTH_ALPHA = 0.4

# ── Visualiser ────────────────────────────────────────────────
COL_IDLE        = (100, 100, 100)
COL_POINTING    = (0, 200, 255)
COL_LOCKED      = (0, 255, 80)
COL_CONTROL     = (255, 180, 0)
COL_RAY         = (0, 220, 255)
COL_MARKER_BOX  = (200, 200, 200)
COL_MARKER_HIT  = (0, 255, 80)
COL_TEXT        = (255, 255, 255)
COL_BAR_BG      = (40, 40, 40)
COL_BAR_FG      = (0, 200, 255)

FONT            = 0
FONT_SCALE      = 0.55
FONT_THICKNESS  = 1
