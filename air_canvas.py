"""
Air Canvas AI
=============
A real-time, touchless drawing system. A webcam feed is analyzed with
MediaPipe's HandLandmarker to find 21 hand landmarks per frame. The index
fingertip acts as a virtual brush:

    - Point with only your index finger extended  -> draw
    - Make a fist (all fingers curled)             -> cycle brush color
    - Anything else (open palm, etc.)               -> pen lifted, no drawing

Keyboard shortcuts (window must be focused):
    c - clear the canvas
    s - save the current drawing to saved_drawings/
    q / ESC - quit

Pipeline:
    Webcam -> OpenCV -> MediaPipe HandLandmarker -> Hand Landmarks
           -> Gesture Detection -> Virtual Canvas -> Composited Output
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_drawings")

CAM_WIDTH, CAM_HEIGHT = 1280, 720

BRUSH_THICKNESS = 8
ERASER_THICKNESS = 45

# (name, BGR color). "Eraser" paints with the canvas background color, which
# the compositor treats as transparent -- so it visually erases strokes.
COLOR_PALETTE = [
    ("Blue", (255, 0, 0)),
    ("Green", (0, 200, 0)),
    ("Red", (0, 0, 255)),
    ("Yellow", (0, 220, 220)),
    ("Eraser", (0, 0, 0)),
]

FIST_COOLDOWN_SEC = 1.0  # minimum time between fist-triggered color changes

# 21-point hand landmark indices (MediaPipe hand model)
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20


# --------------------------------------------------------------------------- #
# Gesture helpers
# --------------------------------------------------------------------------- #

def landmark_pixels(hand_landmarks, width, height):
    """Convert normalized landmarks to (x, y) pixel coordinates."""
    return [(lm.x * width, lm.y * height) for lm in hand_landmarks]


def fingers_extended(pts):
    """
    Return a dict of booleans for which fingers are extended, using simple
    geometric heuristics on the 21 landmark points (in pixel coordinates).
    """
    def dist(a, b):
        return ((pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2) ** 0.5

    # Non-thumb fingers: extended when the tip is above (smaller y) its PIP joint.
    index = pts[INDEX_TIP][1] < pts[INDEX_PIP][1]
    middle = pts[MIDDLE_TIP][1] < pts[MIDDLE_PIP][1]
    ring = pts[RING_TIP][1] < pts[RING_PIP][1]
    pinky = pts[PINKY_TIP][1] < pts[PINKY_PIP][1]

    # Thumb: extended when it's sticking out away from the palm, independent
    # of handedness. Compare thumb-tip-to-index-MCP distance against the
    # palm's own scale (wrist-to-index-MCP distance).
    palm_scale = dist(WRIST, INDEX_MCP) + 1e-6
    thumb = dist(THUMB_TIP, INDEX_MCP) > 0.8 * palm_scale

    return {"thumb": thumb, "index": index, "middle": middle, "ring": ring, "pinky": pinky}


def classify_gesture(fingers):
    """Map finger states to one of: 'draw', 'fist', 'other'."""
    if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "draw"
    if not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "fist"
    return "other"


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #

def draw_hand_skeleton(frame, pts):
    connections = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
    for conn in connections:
        p1 = (int(pts[conn.start][0]), int(pts[conn.start][1]))
        p2 = (int(pts[conn.end][0]), int(pts[conn.end][1]))
        cv2.line(frame, p1, p2, (255, 255, 255), 1)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)


def draw_color_palette(frame, active_index):
    swatch_w = 90
    for i, (name, color) in enumerate(COLOR_PALETTE):
        x0 = 10 + i * (swatch_w + 8)
        y0 = 10
        x1, y1 = x0 + swatch_w, y0 + 55
        display_color = (60, 60, 60) if name == "Eraser" else color
        cv2.rectangle(frame, (x0, y0), (x1, y1), display_color, -1)
        border_color = (255, 255, 255) if i == active_index else (120, 120, 120)
        thickness = 4 if i == active_index else 1
        cv2.rectangle(frame, (x0, y0), (x1, y1), border_color, thickness)
        cv2.putText(frame, name, (x0 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)


def composite_canvas_on_frame(frame, canvas):
    """Overlay the drawing canvas on top of the webcam frame, treating
    near-black canvas pixels as transparent background."""
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv_mask_bgr = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_mask_bgr)
    frame = cv2.bitwise_or(frame, canvas)
    return frame


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Hand landmarker model not found at {MODEL_PATH}.\n"
            "Download it with:\n"
            "  curl -L -o models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        )
    os.makedirs(SAVE_DIR, exist_ok=True)

    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    canvas = None
    prev_point = None
    color_index = 0
    last_fist_time = 0.0
    prev_frame_time = time.time()
    fps_history = deque(maxlen=30)

    print("Air Canvas AI running.")
    print("  Point with your index finger to draw.")
    print("  Make a fist to cycle to the next color / eraser.")
    print("  Keys: [c] clear canvas   [s] save drawing   [q] quit")

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        height, width = frame.shape[:2]

        if canvas is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        gesture = "none"
        if result.hand_landmarks:
            pts = landmark_pixels(result.hand_landmarks[0], width, height)
            fingers = fingers_extended(pts)
            gesture = classify_gesture(fingers)
            draw_hand_skeleton(frame, pts)

            index_tip = (int(pts[INDEX_TIP][0]), int(pts[INDEX_TIP][1]))

            if gesture == "draw":
                name, color = COLOR_PALETTE[color_index]
                thickness = ERASER_THICKNESS if name == "Eraser" else BRUSH_THICKNESS
                if prev_point is not None:
                    cv2.line(canvas, prev_point, index_tip, color, thickness)
                else:
                    cv2.circle(canvas, index_tip, thickness // 2, color, -1)
                prev_point = index_tip
                cv2.circle(frame, index_tip, 10, (0, 255, 0), 2)

            elif gesture == "fist":
                prev_point = None
                now = time.time()
                if now - last_fist_time > FIST_COOLDOWN_SEC:
                    color_index = (color_index + 1) % len(COLOR_PALETTE)
                    last_fist_time = now

            else:
                prev_point = None
        else:
            prev_point = None

        output = composite_canvas_on_frame(frame, canvas)

        draw_color_palette(output, color_index)

        now = time.time()
        fps_history.append(1.0 / max(now - prev_frame_time, 1e-6))
        prev_frame_time = now
        fps = sum(fps_history) / len(fps_history)
        cv2.putText(output, f"FPS: {fps:.1f}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, f"Gesture: {gesture}", (10, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Air Canvas AI", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('c'):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            prev_point = None
            print("Canvas cleared.")
        elif key == ord('s'):
            filename = os.path.join(SAVE_DIR, f"drawing_{int(time.time())}.png")
            cv2.imwrite(filename, canvas)
            print(f"Saved drawing to {filename}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
