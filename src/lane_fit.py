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


def measure_curvature_real(binary_warped, left_fit_pix, right_fit_pix):
    """
    Convert the pixel-space polynomial fits into a real-world curvature
    (in meters), evaluated at the bottom of the image (closest to the car).

    Smaller radius = sharper turn. A straight road gives a very large
    (near-infinite) radius.
    """
    leftx, lefty, rightx, righty, _ = find_lane_pixels_sliding_window(binary_warped)

    # Re-fit in REAL-WORLD units (meters) instead of pixels
    left_fit_cr = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

    y_eval = binary_warped.shape[0] - 1  # bottom of image = closest to car
    y_eval_m = y_eval * YM_PER_PIX

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval_m + left_fit_cr[1]) ** 2) ** 1.5) / \
                     np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval_m + right_fit_cr[1]) ** 2) ** 1.5) / \
                      np.absolute(2 * right_fit_cr[0])

    avg_curverad = (left_curverad + right_curverad) / 2
    return left_curverad, right_curverad, avg_curverad


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