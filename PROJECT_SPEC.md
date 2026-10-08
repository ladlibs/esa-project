# PROJECT SPEC: Vision-Based Road Curvature & Sharp Turn Warning System

This document is written for an AI coding agent (or a new human contributor)
to pick up this project with zero prior context and continue it correctly.
Read this whole file before writing or changing any code.

---

## 1. Project Goal

Build a software pipeline that:
1. Takes road video/images as input
2. Detects lane lines
3. Calculates the road's curvature (radius, in meters)
4. Issues a "SHARP TURN AHEAD" warning **only when there is genuinely a
   sharp turn** — not on straight roads, not flickering on/off due to
   noise, not falsely triggering from shadows/lighting/detection errors

This is a monocular (single camera), vision-only pipeline. No IMU, no
vehicle speed input, no sensor fusion — deliberately, by design decision
(see Section 7 for why this was chosen over a visual-inertial approach).

**Definition of "done" / success criteria:**
- Runs on the provided sample dataset (Section 3) without crashing
- Produces a curvature number that is large (1000+ m) on straight roads
  and small (under a few hundred m) on genuinely curved roads
- Produces a stable, non-flickering warning on video — the warning
  should not turn on/off multiple times per second on a road with
  constant curvature
- Produces an output video with the detected lane area drawn on it in
  the correct (unwarped) perspective, plus the curvature value and
  warning text overlaid
- All hardcoded, camera-specific values are clearly commented and
  isolated so they can be recalibrated for different footage (see
  Section 6 — this is critical, do not skip)

---

## 2. Pipeline Architecture (what exists, in order)

The pipeline is a sequence of independent, testable stages. Each stage
is its own file in `src/`, each can be run standalone on a single image
for debugging, and each later stage imports functions from earlier ones.

```
src/
  threshold.py       Stage 1: color thresholding -> binary lane mask
  perspective.py     Stage 2: perspective warp -> bird's-eye view
  curvature.py       Stage 3: sliding window search + polynomial fit
                                + curvature calculation
  video_pipeline.py  Stage 4: applies stages 1-3 to every video frame
```

  warning.py         Stage 5: turns a curvature number into a stable,
                                debounced sharp-turn warning
```

### 2.1 `threshold.py` — Color Thresholding

Purpose: convert a raw color photo into a binary (black/white) image
where lane-line pixels are white.

Key functions:
- `hls_s_threshold(img, thresh=(120,255))` — thresholds the Saturation
  channel of HLS colorspace. Lane paint tends to be more saturated than
  plain road.
- `yellow_white_mask(img)` — explicit HSV color range masks for yellow
  and white paint.
- `combined_binary(img)` — ORs both methods together.
- `region_of_interest(binary_img)` — crops to a trapezoid region
  covering only the road ahead, removing sky/trees/opposite-lane noise.
  **This trapezoid is defined as PERCENTAGES of image width/height**
  (0.10, 0.45, 0.55, 0.60, 0.95), so it scales automatically to any
  image resolution. This one is NOT hardcoded to pixel values, unlike
  perspective.py below — a deliberate difference, see Section 6.

### 2.2 `perspective.py` — Perspective Transform (Bird's-Eye View)

Purpose: warp the road region so lane lines that are parallel in the
real world appear as parallel vertical lines in the image, instead of
converging toward a vanishing point. This is required because curvature
math only works correctly in this top-down space.

Key functions:
- `get_perspective_points(img_shape)` — returns 4 `src` points (a
  trapezoid) and 4 `dst` points (a rectangle). **THE SRC POINTS ARE
  HARDCODED PIXEL COORDINATES**: `[190, height], [596, 447], [685, 447],
  [1125, height]`. These were manually picked by visually inspecting
  `straight_lines1.jpg` (a 1280x720 image) and finding where the lane
  lines actually sit. THIS IS THE #1 THING THAT BREAKS ON NEW FOOTAGE.
  See Section 6.
- `warp_image(img)` — computes the transform matrix `M` (and its
  inverse `Minv`) via `cv2.getPerspectiveTransform`, then applies it
  via `cv2.warpPerspective`.

Calibration check: run on `straight_lines1.jpg`. Output
`_warped_color.jpg` MUST show both lane lines as vertical, parallel
lines. If they lean or converge, the src points are wrong for this
footage and must be redone (procedure in Section 6).

### 2.3 `curvature.py` — Lane Pixel Detection + Curvature Calculation

Purpose: find exact lane pixel coordinates in the warped binary image,
fit a curve through them, and compute a real-world curvature radius.

Key functions:
- `find_lane_pixels(binary_warped, n_windows=9, margin=100, minpix=50)`
  — histogram peak-finding on the bottom half to get starting x
  positions, then 9 stacked sliding windows climbing the image,
  re-centering on detected pixel clusters as they go (this is how it
  follows a curve).
- `fit_polynomial(leftx, lefty, rightx, righty, img_shape)` — fits
  `x = A*y^2 + B*y + C` via `np.polyfit`, in BOTH pixel space (for
  drawing) and meter space (for curvature math), using:
  - `YM_PER_PIX = 30 / 720` (meters per pixel vertically — assumes
    ~30m of road is visible in the warped image's height)
  - `XM_PER_PIX = 3.7 / 700` (meters per pixel horizontally — assumes
    a standard 3.7m US lane width mapped to ~700px in the warped image)
  **THESE TWO CONSTANTS ARE ALSO HARDCODED AND RESOLUTION/CAMERA
  DEPENDENT.** See Section 6.
- `measure_curvature(fit_result, img_shape)` — applies the standard
  radius-of-curvature formula at the point nearest the vehicle (bottom
  of image): `((1 + (2*A*y_eval + B)**2)**1.5) / abs(2*A)`, evaluated
  separately for left and right lane, then averaged.

### 2.4 `video_pipeline.py` — Video Loop

Purpose: apply the full pipeline to every frame of a video file.

Key functions:
- `process_frame(frame)` — runs threshold -> warp -> find_lane_pixels
  -> fit_polynomial -> measure_curvature, then draws a green filled
  lane-area polygon in the warped space, warps it BACK to the original
  perspective using `Minv`, and blends it onto the original frame with
  `cv2.addWeighted`. Also overlays curvature text and a placeholder
  sharp-turn warning (`avg_curverad < 400` -> red text).
- `process_video(input_path, output_path, max_frames=None)` — reads
  frame by frame with `cv2.VideoCapture`, calls `process_frame`, writes
  with `cv2.VideoWriter`. Wraps each frame in try/except so one bad
  frame doesn't crash the whole video (falls back to passing the raw
  frame through unannotated).

**KNOWN LIMITATIONS OF THIS FILE THAT MUST BE FIXED (this is core to
"not buggy" per the user's requirement):**
1. **No frame-to-frame smoothing.** Every frame is detected completely
   independently via full sliding-window search. This means curvature
   numbers can jump around noisily frame to frame, which will cause a
   naive threshold-based warning to flicker on/off rapidly on borderline
   curvature. THIS MUST BE FIXED — see Section 8, this is Stage 5's job.
2. **No re-use of previous frame's fit.** Real implementations search
   in a narrow margin around the previous frame's polynomial instead of
   re-running the full histogram+windows search every frame. This is a
   performance concern, not a correctness one — full search every frame
   is CORRECT, just slow. Only optimize this after correctness/stability
   (Section 8) is solid.
3. **The `avg_curverad < 400` sharp-turn check is a placeholder with no
   justification and no debounce.** It must be replaced per Section 8.
4. **No handling for "only one lane detected"** (e.g. one line occluded
   by glare or another vehicle) beyond whatever `measure_curvature`
   does (falls back to the single available curverad). This should be
   explicitly tested against — see Section 9 test checklist.

---

## 3. Sample Dataset In Use

All current testing uses the Udacity Self-Driving Car Nanodegree
"Advanced Lane Finding" dataset:
https://github.com/udacity/CarND-Advanced-Lane-Lines

Specifically:
- `test_images/straight_lines1.jpg`, `test1.jpg`, `test2.jpg` — used
  for per-image pipeline testing
- `test_images/project_video.mp4` — used for video pipeline testing
  (1280x720 resolution, confirmed by direct inspection)
- Also available in this dataset but not yet tested against:
  `challenge_video.mp4`, `harder_challenge_video.mp4` (explicitly
  called out by Udacity as harder cases with shadows/lighting changes
  that break naive thresholding — good stress tests once the main
  pipeline is stable, not before)

**All hardcoded calibration values in this codebase (Section 6) were
derived from this dataset's camera at 1280x720 resolution.** If the
final project uses different footage (a different dashcam, a phone
camera, a different dataset like TuSimple/CULane, or footage recorded
by the team), these values are very likely wrong and MUST be
recalibrated using the exact procedure in Section 6 before the pipeline
can be trusted on that footage. Do not assume it will "probably still
work" — the perspective transform especially will silently produce a
skewed bird's-eye view that quietly corrupts every curvature number
downstream without throwing any error.

---

## 4. Environment / Dependencies

```
pip install opencv-python numpy matplotlib
```
(`opencv-python-headless` also works and is preferred in headless/CI
environments with no GUI libraries available.)

Python 3.9+. No GPU, no deep learning framework required for the
current pipeline (Stage 1-4). If a rain/weather classifier is added
later (see Section 10, stretch goal), that would need PyTorch or
TensorFlow plus a pretrained MobileNet/ResNet.

---

## 5. Folder Structure

```
lane_project/
  src/
    threshold.py
    perspective.py
    curvature.py
    video_pipeline.py
    warning.py            <- NOT YET BUILT, see Section 8
  test_images/             <- sample images/videos, gitignored
  output_images/           <- generated outputs, gitignored
  README.md                <- setup + usage instructions for humans
  PROJECT_SPEC.md           <- this file
```

---

## 6. CRITICAL: Hardcoded, Camera/Resolution-Dependent Values

This section is the most important one if the dataset, camera, or
resolution changes. Every value below was manually calibrated against
1280x720 footage from a centrally-mounted dashcam. **Changing the input
source without redoing this calibration is the single most likely
cause of the pipeline silently producing wrong curvature numbers.**

### 6.1 `perspective.py` -> `get_perspective_points()` -> `src` points

```python
src = np.float32([
    [190, height],
    [596, 447],
    [685, 447],
    [1125, height]
])
```

These are **absolute pixel coordinates**, not percentages, unlike the
ROI trapezoid in `threshold.py`. They were found by opening
`straight_lines1.jpg` and visually identifying where the yellow (left)
and white (right) lane lines cross two horizontal reference lines: the
bottom of the image (closest to the car) and roughly 60% up the image
(far ahead, near where the lines start converging).

**Recalibration procedure required whenever the camera, mounting
position, resolution, or field of view changes:**
1. Obtain a frame from the new footage showing a straight, empty
   stretch of road (no other vehicles obscuring the lane lines).
2. Open it in an image viewer that shows pixel coordinates on hover
   (or write a small script using `matplotlib.pyplot.imshow` with
   `plt.ginput()` to click 4 points and print their coordinates).
3. Identify 4 points: bottom-left and bottom-right where the lane
   lines meet the bottom edge of the image (or just above the hood/
   dashboard if visible), and top-left/top-right at a consistent
   height further up the image (not too close to the horizon, or
   the far-field pixels become too sparse/noisy after warping —
   roughly 55-65% of the image height from the top is a reasonable
   starting point, matching the existing `447` value at `720*0.62`).
4. Replace the 4 numbers in `src`.
5. Re-run `python src/perspective.py <the_straight_road_image>` and
   confirm `_warped_color.jpg` shows vertical, parallel lane lines.
   If not, adjust and repeat. Do not proceed to Stage 3+ testing until
   this passes.
6. Also sanity-check `_src_overlay.jpg` — the drawn trapezoid should
   sit tightly along both lane lines in the original (unwarped) image.

### 6.2 `curvature.py` -> `YM_PER_PIX` and `XM_PER_PIX`

```python
YM_PER_PIX = 30 / 720   # meters per pixel, vertical
XM_PER_PIX = 3.7 / 700  # meters per pixel, horizontal
```

These convert pixel measurements in the WARPED image into real-world
meters, and directly scale the final curvature-in-meters output. They
assume:
- The warped image is 720px tall and this corresponds to about 30
  meters of real road distance (i.e. how far ahead the `src` trapezoid
  in `perspective.py` reaches).
- The two lane lines are separated by ~700px in the warped image, and
  a real lane is ~3.7m wide (US standard; adjust if the target region
  uses a different standard, e.g. many countries use ~3.0-3.75m).

**These constants MUST be re-derived any time `get_perspective_points()`
is recalibrated (Section 6.1), because they depend on exactly how much
real-world distance the new `src`/`dst` points correspond to.**
Recalibration procedure:
1. After fixing `src` points in `perspective.py`, warp a straight-road
   test image and measure (in pixel space, on the warped output) the
   horizontal distance between the two lane lines. Update
   `XM_PER_PIX = <real_lane_width_meters> / <that_pixel_distance>`.
2. Estimate how far ahead (in real meters) the top of your `src`
   trapezoid actually reaches — this requires either a known reference
   in the scene (e.g. lane dash markings are a standard ~3m long with
   ~9m gaps in the US, useful for triangulating distance) or physically
   measuring against the real camera setup. Update
   `YM_PER_PIX = <that_distance_meters> / <warped_image_height_px>`.
3. If this estimation can't be done precisely, it is acceptable to
   leave these approximate — the RELATIVE curvature comparisons
   (sharp vs. not sharp) will still be roughly consistent even if the
   absolute meter value has some error. Flag this assumption explicitly
   in the project writeup/report if left approximate.

### 6.3 `threshold.py` -> `region_of_interest()` trapezoid percentages

```python
polygon = np.array([[
    (int(width * 0.10), height),
    (int(width * 0.45), int(height * 0.60)),
    (int(width * 0.55), int(height * 0.60)),
    (int(width * 0.95), height)
]], dtype=np.int32)
```

Unlike Section 6.1, these ARE percentage-based and will scale to any
resolution automatically. However, they still encode an assumption
about camera mounting (centered, roughly level) and field of view.
If the new camera is mounted off-center, angled significantly
up/down, or has a much wider/narrower field of view, these percentages
may crop out real lane pixels or include too much non-road area.
Verify with `_roi_overlay.jpg` any time the camera changes, even though
no code change is strictly required by a resolution change alone.

### 6.4 `threshold.py` -> color threshold values

```python
hls_s_threshold(img, thresh=(120, 255))
lower_yellow = np.array([15, 80, 100]); upper_yellow = np.array([35, 255, 255])
lower_white  = np.array([0, 0, 200]);   upper_white  = np.array([180, 30, 255])
```

These are lighting/camera-sensor dependent, not resolution dependent.
A different camera's color response, or footage shot in different
lighting (overcast, dusk, artificial lighting, a different country's
lane paint color/condition) may need these ranges adjusted. Verify with
`_roi_binary.jpg` on a handful of representative frames from the new
footage; if lane lines aren't showing up clearly or there's excessive
noise, adjust these ranges before trusting anything downstream.

### 6.5 `warning.py` -> sharp-turn thresholds

```python
self.enter_threshold = 300
self.exit_threshold = 450
```

These values define the hysteresis for triggering and clearing the sharp turn warning. They operate on the already-converted meter value from Section 6.2. 
- The warning turns ON when the smoothed curvature drops below 300m for a consecutive number of frames.
- The warning turns OFF only when the smoothed curvature rises above 450m for a consecutive number of frames.

They are dependent on `YM_PER_PIX`/`XM_PER_PIX` being reasonably accurate — a wrong meter conversion will make these thresholds meaningless. Tune these numbers based on empirical testing with target footage.

---

## 7. Design Decisions Already Made (do not relitigate without reason)

- **Vision-only, no IMU/sensor fusion.** A visual-inertial approach
  (referenced paper: Alrazouk et al., "Efficient Real-Time Road
  Curvature Estimation: Visual-Inertial Approach", IFAC 2023) was
  considered and explicitly rejected for this project because it
  targets motorcycles (models roll dynamics), requires an IMU and
  vehicle speed as inputs, and was validated in a vehicle simulator —
  all out of scope for a monocular, software-only project. Do not
  introduce IMU/speed dependencies unless the project scope itself
  changes.
- **Sliding window + polynomial fit over Hough Transform.** An initial
  simpler Hough Transform approach was superseded by the sliding-window
  histogram approach because it handles curves far more robustly.
  Don't regress to Hough-only lane finding.
- **Color thresholding (HLS S-channel + HSV yellow/white masks) over
  Sobel gradients.** Following the Udacity reference project's own
  documented finding that gradient-based thresholds are more easily
  confused by shadows and road texture than direct color-based
  thresholds. If extending threshold.py, prefer improving/expanding
  color-based methods over reintroducing gradient-based ones, unless
  specifically addressing a color-thresholding failure case.
- **Camera calibration (chessboard distortion correction) was
  deliberately skipped** as out of scope/low value for this project's
  grading focus, per explicit user decision. It can be added later
  (Section 10) but is not required for "done."

---

## 8. NEXT REQUIRED WORK: Stage 5 — `warning.py`

This is the most important remaining piece for meeting the user's
explicit requirement of a warning that "only [fires] if there is a
sharp turn and not be buggy." Build this as a new file,
`src/warning.py`, and integrate it into `video_pipeline.py`, replacing
the current inline `avg_curverad < 400` check.

Required behavior:

1. **Curvature smoothing.** Maintain a rolling buffer (e.g. the last
   5-10 frames) of `avg_curverad` values. Use a smoothed value (simple
   moving average, or median to be more robust to single-frame
   outliers) for all warning decisions instead of the raw per-frame
   value. This directly fixes the flicker problem in Section 2.4.

2. **Hysteresis / distinct enter and exit thresholds.** Do not use one
   threshold for both triggering and clearing the warning — this causes
   rapid on/off flicker exactly at the boundary. Use two thresholds,
   e.g. trigger below `SHARP_TURN_ENTER = 300` meters, but only clear
   the warning once curvature rises back above
   `SHARP_TURN_EXIT = 450` meters (exact numbers should be tuned
   empirically against the test videos, these are starting points).

3. **Frame-persistence / debounce.** Require the smoothed curvature to
   stay below the enter-threshold for N consecutive frames (e.g. 5-8)
   before actually triggering the warning, and similarly require it to
   stay above the exit-threshold for N consecutive frames before
   clearing it. This is standard state-machine debouncing and is
   required — do not skip it even if smoothing (item 1) is implemented,
   they solve different problems (smoothing reduces noise in the
   number; debounce prevents a genuinely borderline number from
   flapping the on/off state).

4. **Graceful handling of missed detections.** If a frame produces no
   usable curvature (both lanes undetected), do NOT feed that as a data
   point into the smoothing buffer as if it were a real curvature
   value, and do NOT immediately clear or trigger a warning based on a
   missing frame. Hold the previous state and previous smoothed value
   steady, and only act once real detections resume. Log/count how
   often this happens per video as a diagnostic (high frequency
   indicates the threshold.py color thresholds need tuning for that
   footage, per Section 6.4).

5. **Suggested implementation shape** (an agent building this should
   feel free to adjust specifics, but this is the expected structure):

```python
class SharpTurnWarningSystem:
    def __init__(self, enter_threshold=300, exit_threshold=450,
                 smoothing_window=8, debounce_frames=5):
        ...
    def update(self, curverad_or_none):
        """Call once per frame with the current frame's avg_curverad
        (or None if detection failed). Returns True/False: whether the
        warning should currently be displayed."""
        ...
```

6. **Testing requirement for this stage specifically:** run the full
   video pipeline with this integrated on `project_video.mp4` (mild,
   mostly gradual curvature) end to end, and visually confirm in the
   output video that the warning:
   - Does NOT trigger during the straight/gently-curving sections
   - DOES trigger during the tightest-curvature section(s)
   - Does not flicker on/off within a single continuous curve
   - Turns off cleanly again once the road straightens

   Do this by eye, by scrubbing through the output video, not just by
   trusting that the code compiles and runs.

---

## 9. Testing Checklist (apply after Stage 5, and after ANY future change)

Run through all of these before considering the pipeline "working
correctly," not just a subset:

- [ ] `threshold.py` on `test1.jpg`, `test2.jpg`, `straight_lines1.jpg`
      — `_roi_binary.jpg` shows clean lane lines, minimal noise
- [ ] `perspective.py` on `straight_lines1.jpg` — warped lines are
      vertical and parallel (the core calibration check)
- [ ] `perspective.py` on a curved test image — warped lines show a
      smooth curve, not straight lines and not wild distortion
- [ ] `curvature.py` on `straight_lines1.jpg` — curvature reported in
      the 1000s of meters (correctly reads as "not curving")
- [ ] `curvature.py` on a curved test image — curvature noticeably
      smaller, and the debug `_lane_fit.jpg` image shows sliding
      windows correctly tracking both lanes with a sensible polynomial
      fit overlaid
- [ ] `video_pipeline.py` (with Stage 5 integrated) on the full
      `project_video.mp4` — lane overlay tracks correctly for the
      whole video, curvature number stays reasonably stable
      frame-to-frame (no wild jumps), warning fires only on genuinely
      sharp sections, no crashes for the full video duration
- [ ] Spot-check a handful of frames where another vehicle partially
      occludes a lane line — confirm the pipeline degrades gracefully
      (uses the other lane's fit, or holds last known state) rather
      than producing a wildly wrong curvature spike
- [ ] (Stretch, do only once the above all pass) Run against
      `challenge_video.mp4` and note failure modes for the report —
      this video is explicitly designed by Udacity to stress-test
      shadow/lighting handling and is expected to reveal weaknesses in
      the current color-thresholding approach; document what breaks
      rather than trying to achieve perfect results on it

---

## 10. Explicitly Out of Scope / Stretch Goals (do not build unless asked)

- Camera calibration / chessboard distortion correction (deliberately
  skipped, Section 7)
- Visual-inertial fusion / IMU input (deliberately rejected, Section 7)
- Rain/weather detection classifier (MobileNet/ResNet transfer
  learning) — was floated early in project scoping as a "combined"
  project idea with a teammate, but is a separate stretch feature, not
  required for the core curvature/warning pipeline to be considered
  complete
- Real-time performance optimization (search-around-previous-fit
  instead of full sliding window every frame) — valid future work, but
  explicitly lower priority than correctness/stability (Section 8) per
  current project priorities

---

## 11. Instructions For An AI Agent Continuing This Project

1. Read this entire document first.
2. Do not modify the hardcoded values in Section 6 unless you are
   actively recalibrating for new footage — if you do, update BOTH the
   code AND this document's Section 6 with the new values and the
   footage/resolution they apply to.
3. Build Stage 5 (`warning.py`) next, per Section 8, exactly.
4. After Stage 5 is integrated, run the full Section 9 checklist before
   declaring anything "done." Do not skip steps in that checklist even
   if earlier stages were already verified once before — regressions
   are possible when integrating new code.
5. If you change or add a dataset/camera source, follow Section 6's
   recalibration procedures BEFORE running Section 9's checklist
   against that new source — do not test against uncalibrated
   perspective points, the results will be meaningless.
6. Keep this document up to date as the source of truth: if you add a
   new hardcoded/environment-dependent value anywhere in the codebase,
   document it in Section 6 in the same format (what it is, what it
   assumes, exact recalibration steps). If you complete Stage 5,
   update Section 2's "NOT YET BUILT" note and Section 6.5's
   description to reflect the real implementation, not the placeholder.