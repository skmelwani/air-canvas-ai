# Air Canvas AI

A real-time, touchless drawing system. A webcam feed is analyzed with MediaPipe's
hand-landmark model to track your hand, and your index fingertip becomes a virtual
brush — no mouse, stylus, or touchscreen required.

```
Webcam -> OpenCV -> MediaPipe HandLandmarker -> Hand Landmarks
       -> Gesture Detection -> Virtual Canvas -> Composited Output
```

## Features

- Real-time hand tracking from a webcam feed
- 21-point hand landmark detection (MediaPipe HandLandmarker)
- Index-finger based drawing
- Gesture-based color control (fist to cycle colors)
- Multiple drawing colors, plus an eraser
- Canvas clearing and drawing save-to-file
- Live FPS counter

## Controls

Each finger draws in its own color when it's the *only* one extended — no
color-cycling or mode-switching required, just change which finger is up.

| Input | Action |
|---|---|
| Only index finger extended | Draw **Brown**, brush follows the index tip |
| Only middle finger extended | Draw **Blue**, brush follows the middle tip |
| Only ring finger extended | Draw **Green**, brush follows the ring tip |
| Only thumb extended | **Eraser**, follows the thumb tip |
| Fist (everything curled) | Pen lifted — nothing is drawn |
| Any other hand pose, or no hand visible | Pen lifted — nothing is drawn |
| `c` key | Clear the canvas |
| `s` key | Save the current drawing to `saved_drawings/` |
| `q` or `Esc` | Quit |

The pinky isn't used by any gesture, and thumb detection is deliberately
lenient (it just checks the thumb sticks out from the palm), so a relaxed
pointing hand with the thumb slightly out still counts as "index only."

## How it works

1. **Capture** — OpenCV reads frames from the webcam and mirrors them for natural,
   selfie-style interaction.
2. **Landmark detection** — Each frame is passed to MediaPipe's `HandLandmarker`,
   which returns 21 `(x, y, z)` points describing the hand's skeleton (wrist, and
   base/middle/tip joints for each finger).
3. **Gesture detection** — Simple geometry on those points decides the gesture:
   a finger counts as "extended" if its tip sits above its middle knuckle in the
   image; the thumb is checked separately by comparing its distance from the palm.
   Whichever of index/middle/ring/thumb is extended *on its own* selects that
   finger's color and brush position; everything curled is a fist (no drawing).
4. **Virtual canvas** — A persistent black image the same size as the frame
   accumulates strokes across frames. While drawing, a line is drawn from the
   fingertip's previous position to its current one each frame, so fast motion
   still produces a continuous stroke instead of dots.
5. **Compositing** — The canvas is masked and blended on top of the live webcam
   frame, so strokes appear to float over the video instead of replacing it.
   The eraser works by "drawing" in the canvas's own background color, which the
   mask treats as transparent.

## Requirements

- Python 3.9+ (tested on 3.13)
- A webcam
- macOS, Linux, or Windows — run directly on the host OS, **not** inside Docker.
  Docker containers don't get a working path to the host's camera device, so
  `cv2.VideoCapture(0)` fails there. A local virtual environment avoids that.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the hand landmark model (already included in this repo under models/,
#    but if it's missing, fetch it with):
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Usage

```bash
source venv/bin/activate
python3 air_canvas.py
```

A window will open showing your webcam feed with the drawing overlay. macOS/Windows
will prompt for camera permission the first time you run it — allow it, or the
capture will fail.

Saved drawings land in `saved_drawings/` as `drawing_<unix-timestamp>.png`.

## Project structure

```
project/
├── air_canvas.py          # Main application
├── requirements.txt       # Pinned dependencies
├── models/
│   └── hand_landmarker.task   # MediaPipe hand landmark model
├── saved_drawings/        # Saved canvases (created/used at runtime)
├── venv/                  # Local virtual environment
└── task.txt               # Original project brief
```

## Troubleshooting

- **Black/frozen window or crash on startup**: another app may be holding the
  camera, or camera permission was denied — check your OS's privacy settings for
  Terminal/Python.
- **No hand detected**: make sure your hand is well-lit and fully in frame;
  `min_hand_detection_confidence` in `air_canvas.py` can be lowered slightly if
  detection feels too strict.
- **Low FPS**: the model runs on CPU by default for broad compatibility; this is
  usually still fast enough for smooth drawing, but very old machines may lag.
