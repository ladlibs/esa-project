# esa-project
Vision based road curvature and Sharp turn warning


A software pipeline that detects lane lines in road video/images,
calculates the road's curvature, and issues a warning before sharp turns.

## What this project does (high level)

1. Take a road image/video frame
2. Find the lane line pixels (color-based detection)
3. "Unwarp" the road into a top-down (bird's-eye) view
4. Fit a curve to the lane pixels and calculate curvature
5. Warn the driver if the curve is sharp

We're currently on **Step 3 of 6** (see Project Status below).

---

## Getting Started (first time setup)

### 1. Clone the repo

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Install Python dependencies

You need Python 3.9+ installed. Then run:

```bash
pip install opencv-python numpy matplotlib
```

If `pip` doesn't work, try `pip3` instead.

### 3. Check the folder structure

```
lane_project/
  src/                   <- all pipeline code goes here
    threshold.py         <- Step 2: color thresholding (lane pixel detection)
    perspective.py        <- Step 3: perspective transform (bird's-eye view)
  test_images/            <- sample road images for testing (not auto-included, see below)
  output_images/          <- results get saved here automatically when you run a script
  README.md               <- this file
```

### 4. Get test images

`test_images/` is where you drop sample road photos to test the code on.
We're using the standard Udacity lane-finding dataset images. Download
these into `test_images/`:

- https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/test_images/test1.jpg
- https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/test_images/test2.jpg
- https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/test_images/straight_lines1.jpg

(`straight_lines1.jpg` is important — it's used to verify the perspective
transform is calibrated correctly. Don't skip it.)

### 5. Run the pipeline stages

Each script can be run directly on one image to test that stage in isolation:

```bash
# Step 2: color thresholding
python src/threshold.py test_images/test1.jpg

# Step 3: perspective transform (also runs step 2 internally)
python src/perspective.py test_images/straight_lines1.jpg
```

Check the `output_images/` folder after running — each script saves a few
labeled output files so you can visually verify the result.

---

## How to know if it's working

### Step 2 (`threshold.py`) outputs:
- `<name>_binary.jpg` — full-frame result, will look messy (includes sky/trees) — this is expected, ignore it
- `<name>_roi_binary.jpg` — **this is the one that matters.** Should show clean white lane lines on a black background
- `<name>_roi_overlay.jpg` — original photo with a green trapezoid drawn on it, showing the crop region. Lane lines should stay inside the trapezoid, but don't need to hug it exactly

### Step 3 (`perspective.py`) outputs:
- `<name>_src_overlay.jpg` — green trapezoid should sit tightly on both lane lines
- `<name>_warped_color.jpg` — **run this on `straight_lines1.jpg` first.** Lane lines should appear vertical and roughly parallel. If they lean or converge, the calibration points in `get_perspective_points()` need adjusting (see below)
- `<name>_warped_binary.jpg` — same warp applied to the binary mask; this feeds into the next step

---

## Important: camera calibration is footage-specific

The 4 pixel coordinates in `perspective.py` → `get_perspective_points()` are
hardcoded and were manually picked to match the Udacity dataset's camera
(1280x720, centrally mounted dashcam). If you swap in different footage
(a different dashcam, different resolution, different mounting position),
these points **will be wrong** and the bird's-eye warp will look skewed.

To recalibrate:
1. Get a screenshot of a straight stretch of road from your new footage
2. Open it and note the pixel (x, y) coordinates of each lane line at two
   heights: close to the camera (bottom of image) and far ahead (near
   where the lines converge)
3. Update the 4 points in `src = np.float32([...])` inside
   `get_perspective_points()`
4. Re-run `python src/perspective.py <your_straight_road_image>` and check
   that `_warped_color.jpg` shows vertical parallel lines

---

## Project Status

- [x] Step 1: Environment setup
- [x] Step 2: Color thresholding (lane pixel detection) — `src/threshold.py`
- [x] Step 3: Perspective transform (bird's-eye view) — `src/perspective.py`
- [ ] Step 4: Sliding-window lane detection + polynomial fit + curvature calculation
- [ ] Step 5: Sharp-turn warning logic
- [ ] Step 6: Testing, tuning, and report writeup

---

## Git workflow for teammates

1. Pull latest changes before starting work: `git pull`
2. Create a branch for your part (optional but recommended):
   `git checkout -b your-name-feature`
3. Don't commit large files — `test_images/`, `output_images/`, and any
   `.mp4` video files should NOT be pushed to the repo (see `.gitignore`)
4. Commit with clear messages describing which step/function you touched:
   `git commit -m "Step 4: add sliding window search"`
5. If you change calibration points, camera assumptions, or thresholds,
   leave a comment in the code AND mention it in your commit message —
   these values are easy to silently break for everyone else

```

We don't commit test images/outputs because they're either easy to
re-download (see step 4 above) or generated automatically by running
the scripts — keeping them out of git keeps the repo small.
