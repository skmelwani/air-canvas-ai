# Air Canvas AI

A real-time, touchless drawing system. A webcam feed is analyzed with MediaPipe's
hand-landmark model to track your hand, and whichever finger you extend becomes a
virtual brush of its own color — no mouse, stylus, or touchscreen required.

**Try it live: [skmelwani.github.io/air-canvas-ai](https://skmelwani.github.io/air-canvas-ai/)**
— runs entirely in your browser, nothing to install.

This repo has two implementations of the same idea:

| | [`/` (web)](index.html) | [`python/`](python/) |
|---|---|---|
| Runs in | Any modern browser | Local Python process |
| Setup | None — just open the page | Virtual environment + pip install |
| Hand tracking | MediaPipe Tasks Vision (WebAssembly) | MediaPipe Tasks Vision (Python) |
| Best for | Trying it instantly, sharing a link | Hacking on the CV/gesture logic locally |

```
Webcam -> Hand-tracking model (MediaPipe) -> 21 Hand Landmarks
       -> Gesture Detection -> Virtual Canvas -> Composited Output
```

## Features

- Real-time hand tracking from a webcam feed
- 21-point hand landmark detection (MediaPipe HandLandmarker)
- Per-finger color drawing (no mode-switching required)
- Dedicated eraser gesture
- Canvas clearing and drawing save-to-file
- Live FPS counter

## Controls

Each finger draws in its own color when it's the *only* one extended — no
color-cycling or mode-switching required, just change which finger is up.
Identical in both the web and Python versions.

| Input | Action |
|---|---|
| Only index finger extended | Draw **Brown**, brush follows the index tip |
| Only middle finger extended | Draw **Blue**, brush follows the middle tip |
| Only ring finger extended | Draw **Green**, brush follows the ring tip |
| Only thumb extended | **Eraser**, follows the thumb tip |
| Fist (everything curled) | Pen lifted — nothing is drawn |
| Any other hand pose, or no hand visible | Pen lifted — nothing is drawn |

The pinky isn't used by any gesture, and thumb detection is deliberately
lenient (it just checks the thumb sticks out from the palm), so a relaxed
pointing hand with the thumb slightly out still counts as "index only."

## How it works

1. **Capture** — the webcam frame is mirrored for natural, selfie-style interaction.
2. **Landmark detection** — each frame is passed to MediaPipe's `HandLandmarker`,
   which returns 21 `(x, y, z)` points describing the hand's skeleton (wrist, and
   base/middle/tip joints for each finger).
3. **Gesture detection** — simple geometry on those points decides the gesture:
   a finger counts as "extended" if its tip sits above its middle knuckle in the
   image; the thumb is checked separately by comparing its distance from the palm.
   Whichever of index/middle/ring/thumb is extended *on its own* selects that
   finger's color and brush position; everything curled is a fist (no drawing).
4. **Virtual canvas** — a persistent drawing layer the same size as the frame
   accumulates strokes across frames. While drawing, a line is drawn from the
   fingertip's previous position to its current one each frame, so fast motion
   still produces a continuous stroke instead of dots.
5. **Compositing** — the drawing layer is blended on top of the live webcam
   frame, so strokes appear to float over the video instead of replacing it.

The web version (`app.js`) and the Python version (`python/air_canvas.py`) implement
this pipeline independently but with matching gesture logic — the web version uses
native canvas alpha transparency for the eraser (`destination-out` compositing),
while the Python version fakes transparency by drawing in the canvas's background
color and masking it out, since OpenCV has no native alpha canvas.

---

## Web version

Nothing to install — hand tracking runs on-device via WebAssembly, and video never
leaves your browser tab.

**Live:** https://skmelwani.github.io/air-canvas-ai/

To run it locally instead:

```bash
# any static file server works, e.g.:
python3 -m http.server 8000
# then open http://localhost:8000
```

A plain `file://` open may also work in some browsers, but a local server is more
reliable since camera access requires a "secure context" (HTTPS, or `localhost`).

Saved drawings download as a transparent-background PNG via the browser's normal
download flow.

**Files:** [`index.html`](index.html), [`style.css`](style.css), [`app.js`](app.js) — no build step, no dependencies to install.

---

## Python version

```bash
cd python

# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the hand landmark model (already included in this repo under
#    python/models/, but if it's missing, fetch it with):
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# 4. Run it
python3 air_canvas.py
```

A window will open showing your webcam feed with the drawing overlay. macOS/Windows
will prompt for camera permission the first time you run it — allow it, or the
capture will fail. Run this directly on the host OS, **not** inside Docker — Docker
containers don't get a working path to the host's camera device, so
`cv2.VideoCapture(0)` fails there.

Keyboard shortcuts (window must be focused): `c` clear canvas, `s` save drawing to
`python/saved_drawings/`, `q` / `Esc` quit.

## Project structure

```
project/
├── index.html              # Web app entry point
├── style.css                # Web app styling
├── app.js                    # Web app logic (hand tracking, gestures, canvas)
├── python/
│   ├── air_canvas.py         # Python desktop app
│   ├── requirements.txt      # Pinned dependencies
│   ├── models/
│   │   └── hand_landmarker.task   # MediaPipe hand landmark model
│   ├── saved_drawings/       # Saved canvases (created/used at runtime)
│   └── venv/                 # Local virtual environment (not tracked in git)
└── task.txt                   # Original project brief
```

## Troubleshooting

- **Web: page stuck on "Requesting camera access…"**: check the browser's
  address-bar camera permission icon; if denied, reset it and reload.
- **Web: "Failed to load the hand-tracking model"**: it fetches from a CDN and
  Google's model storage at runtime, so this usually means no internet access.
- **Python: black/frozen window or crash on startup**: another app may be holding
  the camera, or camera permission was denied — check your OS's privacy settings
  for Terminal/Python.
- **No hand detected (either version)**: make sure your hand is well-lit and fully
  in frame.
- **Low FPS (either version)**: hand tracking runs on CPU by default for broad
  compatibility; usually still smooth enough for drawing, but very old machines
  or phones may lag.
