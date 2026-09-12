"""
Step 2: Color Thresholding for Lane Detection
-----------------------------------------------
Goal: take a raw road image and produce a binary (black/white) image
where lane-line pixels are white (255) and everything else is black (0).

Why color thresholding instead of plain Canny edges?
The Udacity reference project found that Sobel-gradient-only thresholds
get confused by shadows, tree lines, and road texture. Isolating the
S-channel (HLS colorspace) plus explicit yellow/white color masks gives
a much cleaner result for lane lines specifically.

Run this file directly to test on a single image:
    python src/threshold.py test_images/your_image.jpg
"""

import cv2
import numpy as np
import sys
import os


def hls_s_threshold(img, thresh=(120, 255)):
    """
    Isolate the S (Saturation) channel from HLS colorspace and threshold it.
    Lane lines (yellow/white) tend to have high saturation compared to
    the grey road surface, so this channel highlights them well.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary = np.zeros_like(s_channel)
    binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 255
    return binary


def yellow_white_mask(img):
    """
    Explicit color masks for yellow and white lane markings using HSV.
    This directly targets the two colors lane lines actually are,
    which is more robust than gradient-based methods alone.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellow lane lines
    lower_yellow = np.array([15, 80, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # White lane lines (low saturation, high brightness)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    combined = cv2.bitwise_or(yellow_mask, white_mask)
    return combined


def combined_binary(img):
    """
    Combine the S-channel threshold with the yellow/white color mask.
    A pixel is kept if EITHER method flags it as a lane pixel.
    """
    s_binary = hls_s_threshold(img)
    color_mask = yellow_white_mask(img)

    combined = np.zeros_like(s_binary)
    combined[(s_binary == 255) | (color_mask == 255)] = 255
    return combined


def region_of_interest(binary_img):
    """
    Mask out everything except a trapezoid region covering the road
    ahead of the vehicle. This removes sky, trees, and other cars from
    the binary image before we try to fit lane lines.

    NOTE: these trapezoid points are a starting guess for a typical
    dashcam mounted centrally. You WILL need to tune these numbers by
    looking at your own footage — this is expected and normal.
    """
    height, width = binary_img.shape[:2]
    mask = np.zeros_like(binary_img)

    polygon = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.45), int(height * 0.60)),
        (int(width * 0.55), int(height * 0.60)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(binary_img, mask)
    return masked, polygon


def process_image(path, save_dir="output_images"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    binary = combined_binary(img)
    roi_binary, polygon = region_of_interest(binary)

    # Draw the ROI polygon on a copy of the original for visual sanity check
    overlay = img.copy()
    cv2.polylines(overlay, polygon, isClosed=True, color=(0, 255, 0), thickness=2)

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    cv2.imwrite(os.path.join(save_dir, f"{base}_binary.jpg"), binary)
    cv2.imwrite(os.path.join(save_dir, f"{base}_roi_binary.jpg"), roi_binary)
    cv2.imwrite(os.path.join(save_dir, f"{base}_roi_overlay.jpg"), overlay)

    print(f"Saved: {base}_binary.jpg, {base}_roi_binary.jpg, {base}_roi_overlay.jpg -> {save_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/threshold.py <path_to_image>")
        sys.exit(1)

    process_image(sys.argv[1])
