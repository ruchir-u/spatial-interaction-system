"""
main.py — Spatial Interaction System Entry Point
=================================================
Run:
    python main.py

Override DroidCam IP at runtime:
    python main.py --ip 192.168.1.10

Use local webcam instead:
    python main.py --no-droidcam --camera 0

Press  Q  or  ESC  to quit.
Recording saves automatically to a timestamped .mp4 in the project folder.
"""

import cv2
import time
import argparse
import datetime

import config
from input.camera_source import CameraSource
from perception.aruco_detector import ArucoDetector
from perception.hand_tracker import HandTracker
from interaction.state_machine import InteractionStateMachine
from visualizer import Visualizer
from output.event_bus import EventBus, console_handler


def parse_args():
    p = argparse.ArgumentParser(description="Spatial Interaction System")
    p.add_argument("--ip",          default=config.DROIDCAM_IP,
                   help="DroidCam IP address")
    p.add_argument("--port",        default=config.DROIDCAM_PORT, type=int,
                   help="DroidCam port (default 4747)")
    p.add_argument("--no-droidcam", action="store_true",
                   help="Use local webcam instead of DroidCam")
    p.add_argument("--camera",      default=config.WEBCAM_INDEX, type=int,
                   help="Local webcam index (used with --no-droidcam)")
    p.add_argument("--width",       default=config.FRAME_WIDTH,  type=int)
    p.add_argument("--height",      default=config.FRAME_HEIGHT, type=int)
    p.add_argument("--no-record",   action="store_true",
                   help="Disable video recording")
    return p.parse_args()


def main():

    args = parse_args()

    # ── Camera source ─────────────────────────────────────────
    if args.no_droidcam:
        source = args.camera
    else:
        source = f"http://{args.ip}:{args.port}/video"

    cam = CameraSource(source, width=args.width, height=args.height, flip=False)
    if not cam.open():
        return

    # ── Subsystems ────────────────────────────────────────────
    aruco   = ArucoDetector()
    tracker = HandTracker(model_complexity=0)
    sm      = InteractionStateMachine(args.width, args.height)
    viz     = Visualizer()

    bus = EventBus()
    bus.subscribe(console_handler)

    # ── Recorder ─────────────────────────────────────────────
    out = None
    if not args.no_record:
        timestamp = datetime.datetime.now().strftime("rec_%Y%m%d_%H%M%S.mp4")
        fourcc    = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = datetime.datetime.now().strftime("rec_%Y%m%d_%H%M%S.avi")
        ok, frame = cam.read()   # read one test frame first
        actual_h, actual_w = frame.shape[:2]
        out = cv2.VideoWriter(timestamp, fourcc, 30.0, (actual_w, actual_h))
        print(f"[Recorder] Saving to {timestamp}")

    print("\n[System] Running — press Q or ESC to quit.\n")

    fps_timer   = time.perf_counter()
    fps         = 0.0
    frame_count = 0

    # ── Main loop ─────────────────────────────────────────────
    while True:
        ok, frame = cam.read()
        if not ok:
            print("[Camera] Frame read failed — stream may have dropped.")
            time.sleep(0.1)
            continue

        # Perception
        markers = aruco.detect(frame)
        hand    = tracker.process(frame)

        # Interaction
        event = sm.update(hand, markers)

        # Output
        if event is not None and sm.locked_id is not None:
            bus.emit(sm.locked_id, event)

        # Render overlays onto frame
        viz.draw(frame, sm, markers, fps, hand)

        # Record AFTER drawing so overlays are captured
        if out is not None:
            out.write(frame)

        # Display
        cv2.namedWindow("Spatial Interaction", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Spatial Interaction", 1280, 720)
        cv2.imshow("Spatial Interaction", frame)

        # FPS
        frame_count += 1
        now = time.perf_counter()
        if now - fps_timer >= 0.5:
            fps         = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer   = now

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    # ── Cleanup ───────────────────────────────────────────────
    cam.release()
    if out is not None:
        out.release()
        print("[Recorder] File saved.")
    tracker.close()
    cv2.destroyAllWindows()
    print("[System] Exited cleanly.")


if __name__ == "__main__":
    main()