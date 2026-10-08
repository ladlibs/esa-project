"""
Step 4: Sliding Window Lane Detection + Polynomial Fit + Curvature
--------------------------------------------------------------------
Goal: take the warped (bird's-eye) binary image from Step 3 and:
  1. Find which pixels belong to the LEFT lane line vs the RIGHT lane line
  2. Fit a curve (2nd-degree polynomial) through each set of pixels
  3. Calculate the real-world curvature (in meters) of the road
  4. Draw the detected lane back onto the original image for a sanity check

This plugs directly into threshold.py (Step 2) and perspective.py (Step 3).

Run this file directly to test on a single image:
    python src/lane_fit.py test_images/test1.jpg
"""

import cv2
import numpy as np
import sys
import os

from threshold import combined_binary   # Step 2
from perspective import warp_image      # Step 3


# ---------------------------------------------------------------------
# Real-world scale: these convert "pixels in the warped image" into
# "meters on the actual road". You WILL need to tune these for your own
# footage/camera -- the values below are standard starting guesses from
# the Udacity dataset (lane is ~3.7m wide, ~30m of road fits in the
# warped image vertically).
# ---------------------------------------------------------------------
YM_PER_PIX = 30 / 720   # meters per pixel in y (vertical/forward) direction
XM_PER_PIX = 3.7 / 700  # meters per pixel in x (horizontal/lateral) direction


def find_lane_pixels_sliding_window(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    Slide small rectangular "windows" up the image from bottom to top,
    following the lane line as it curves, and collect all the white
    pixels that fall inside those windows.

    Returns (leftx, lefty, rightx, righty, out_img) -- the pixel
    coordinates belonging to each lane line, plus a visualization image.
    """
    # Take a histogram of the bottom half of the image to find a
    # starting x-position for each lane line (where white pixels peak).
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = binary_warped.shape[0] // nwindows

    # Positions of all non-zero (white) pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                       (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                       (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Which white pixels fall inside each window?
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If we found enough pixels, recenter the next window on their mean
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    """
    Find lane pixels, fit a 2nd-degree polynomial (x = ay^2 + by + c) to
    each lane line, and draw the result for a visual sanity check.

    Returns (left_fit, right_fit, out_img) where left_fit/right_fit are
    the [a, b, c] coefficients in PIXEL space.
    """
    leftx, lefty, rightx, righty, out_img = find_lane_pixels_sliding_window(binary_warped)

    if len(leftx) == 0 or len(rightx) == 0:
        raise ValueError("Not enough lane pixels found -- check Step 2/3 output quality.")

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Colour in the detected pixels for the visualization
    out_img[lefty, leftx] = [255, 0, 0]     # left lane = red
    out_img[righty, rightx] = [0, 0, 255]   # right lane = blue

    # Draw the fitted curve as a yellow line
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    for y, lx, rx in zip(ploty.astype(int), left_fitx.astype(int), right_fitx.astype(int)):
        if 0 <= lx < out_img.shape[1]:
            cv2.circle(out_img, (lx, y), 2, (0, 255, 255), -1)
        if 0 <= rx < out_img.shape[1]:
            cv2.circle(out_img, (rx, y), 2, (0, 255, 255), -1)

    return left_fit, right_fit, out_img


def measure_curvature_from_fit(fit, y_eval=719):
    """
    Directly calculate curvature radius (in meters) from a pixel-space polynomial fit.
    x = a*y^2 + b*y + c  ->  x_m = a_m * y_m^2 + b_m * y_m + c_m
    """
    a_m = fit[0] * XM_PER_PIX / (YM_PER_PIX ** 2)
    b_m = fit[1] * XM_PER_PIX / YM_PER_PIX
    y_m = y_eval * YM_PER_PIX
    curverad = ((1 + (2 * a_m * y_m + b_m) ** 2) ** 1.5) / (np.absolute(2 * a_m) + 1e-6)
    return curverad


def measure_curvature_real(binary_warped, left_fit_pix, right_fit_pix):
    """
    Calculate real-world curvature (in meters) evaluated at the bottom of the image.
    Uses pixel coordinates if available; falls back to direct polynomial conversion.
    """
    try:
        leftx, lefty, rightx, righty, _ = find_lane_pixels_sliding_window(binary_warped)
        if len(leftx) > 50 and len(rightx) > 50:
            y_eval = binary_warped.shape[0] - 1
            y_eval_m = y_eval * YM_PER_PIX

            left_fit_cr = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
            right_fit_cr = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval_m + left_fit_cr[1]) ** 2) ** 1.5) / \
                             (np.absolute(2 * left_fit_cr[0]) + 1e-6)
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval_m + right_fit_cr[1]) ** 2) ** 1.5) / \
                              (np.absolute(2 * right_fit_cr[0]) + 1e-6)
            avg_curverad = (left_curverad + right_curverad) / 2
            return left_curverad, right_curverad, avg_curverad
    except Exception:
        pass

    # Direct calculation fallback
    y_eval = binary_warped.shape[0] - 1
    left_curverad = measure_curvature_from_fit(left_fit_pix, y_eval)
    right_curverad = measure_curvature_from_fit(right_fit_pix, y_eval)
    avg_curverad = (left_curverad + right_curverad) / 2
    return left_curverad, right_curverad, avg_curverad


class RobustLaneTracker:
    """
    Temporal tracker across video frames:
      - Searches in margin around previous polynomial (fast & rejects road cracks/shadows)
      - Checks geometric sanity: lane width (450-900 px) across the entire height
      - Handles single-line occlusions by projecting the opposite line
      - Smooths polynomial fits over a moving average buffer
    """
    def __init__(self, buffer_size=10, standard_lane_width_px=690):
        self.buffer_size = buffer_size
        self.standard_lane_width_px = standard_lane_width_px
        self.recent_left_fits = []
        self.recent_right_fits = []
        self.best_left_fit = None
        self.best_right_fit = None

    def search_around_poly(self, binary_warped, left_fit, right_fit, margin=65):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
        return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    def is_valid_pair(self, left_fit, right_fit):
        if left_fit is None or right_fit is None:
            return False
        # Test width at 3 heights: bottom (719), mid (360), top (100)
        widths = []
        for y in [719, 360, 100]:
            lx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
            rx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
            widths.append(rx - lx)
        if any(w < 450 or w > 900 for w in widths):
            return False
        return True

    def update(self, binary_warped):
        left_fit = None
        right_fit = None

        # 1. Search around prior fit if available
        if self.best_left_fit is not None and self.best_right_fit is not None:
            lx, ly, rx, ry = self.search_around_poly(binary_warped, self.best_left_fit, self.best_right_fit)
            l_ok = len(lx) >= 200
            r_ok = len(rx) >= 150
            if l_ok and r_ok:
                l_cand = np.polyfit(ly, lx, 2)
                r_cand = np.polyfit(ry, rx, 2)
                if self.is_valid_pair(l_cand, r_cand):
                    left_fit, right_fit = l_cand, r_cand
            elif l_ok and not r_ok:
                # Left line is clean, project right line
                l_cand = np.polyfit(ly, lx, 2)
                r_cand = np.copy(l_cand)
                r_cand[2] += self.standard_lane_width_px
                left_fit, right_fit = l_cand, r_cand
            elif r_ok and not l_ok:
                # Right line is clean, project left line
                r_cand = np.polyfit(ry, rx, 2)
                l_cand = np.copy(r_cand)
                l_cand[2] -= self.standard_lane_width_px
                left_fit, right_fit = l_cand, r_cand

        # 2. Fall back to sliding window if not found
        if left_fit is None or right_fit is None:
            try:
                l_cand, r_cand, _ = fit_polynomial(binary_warped)
                if self.is_valid_pair(l_cand, r_cand):
                    left_fit, right_fit = l_cand, r_cand
            except Exception:
                pass

        if left_fit is not None and right_fit is not None:
            self.recent_left_fits.append(left_fit)
            self.recent_right_fits.append(right_fit)
            if len(self.recent_left_fits) > self.buffer_size:
                self.recent_left_fits.pop(0)
                self.recent_right_fits.pop(0)
            self.best_left_fit = np.mean(self.recent_left_fits, axis=0)
            self.best_right_fit = np.mean(self.recent_right_fits, axis=0)
            return self.best_left_fit, self.best_right_fit, True
        else:
            return self.best_left_fit, self.best_right_fit, False


def process_image(path, save_dir="output_images"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    binary = combined_binary(img)
    warped_binary, M, Minv = warp_image(binary)

    left_fit, right_fit, fit_vis = fit_polynomial(warped_binary)
    left_curverad, right_curverad, avg_curverad = measure_curvature_real(
        warped_binary, left_fit, right_fit)

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    cv2.imwrite(os.path.join(save_dir, f"{base}_lane_fit.jpg"), fit_vis)

    print(f"[{base}] Left radius:  {left_curverad:.1f} m")
    print(f"[{base}] Right radius: {right_curverad:.1f} m")
    print(f"[{base}] Avg radius:   {avg_curverad:.1f} m")
    print(f"Saved: {base}_lane_fit.jpg -> {save_dir}/")

    return avg_curverad


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/lane_fit.py <path_to_image>")
        sys.exit(1)

    process_image(sys.argv[1])