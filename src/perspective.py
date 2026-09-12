"""
Step 3: Perspective Transform (Bird's-Eye View)
-------------------------------------------------
Goal: warp the road region so it looks like it's viewed from directly
above ("bird's-eye view"). This matters because:

  - Lane lines that are actually parallel in the real world look like
    they converge toward a vanishing point in a normal camera image.
  - Curvature math (fitting a polynomial, computing radius) only makes
    sense in a top-down view where x/y map cleanly to real-world
    left-right/forward-back distances.

We reuse the trapezoid points from Step 2's region_of_interest() as
the SOURCE points, and map them to a rectangle (the DESTINATION) that
fills the output image. cv2.getPerspectiveTransform computes the
matrix that performs this mapping.

Run this file directly to test on a single image:
    python src/perspective.py test_images/straight_lines1.jpg

IMPORTANT: calibrate this on a STRAIGHT road image first (like
straight_lines1.jpg). If your warp is correct, the two lane lines in
the output should appear as two vertical, parallel lines. If they
lean inward/outward, your source points need adjusting.
"""

import cv2
import numpy as np
import sys
import os

from threshold import combined_binary  # Step 2


def get_perspective_points(img_shape):
    """
    Define the 4 source points (trapezoid on the road) and 4
    destination points (rectangle) used for the warp.

    These MUST match (or closely match) the region_of_interest()
    trapezoid from threshold.py, since that's the road area we
    actually want to unwarp.

    Returns (src, dst) as float32 arrays of 4 points each, in order:
    [bottom-left, top-left, top-right, bottom-right]
    """
    height, width = img_shape[:2]

    # NOTE: these points are hand-calibrated for a 1280x720 dashcam frame
    # mounted centrally (verified against straight_lines1.jpg -- the lane
    # lines come out vertical and parallel in the warp). If you switch to
    # different footage/camera, you MUST recalibrate these by picking
    # pixel coordinates off a straight-road frame from that footage.
    src = np.float32([
        [190, height],          # bottom-left  (on the yellow line)
        [596, 447],             # top-left     (on the yellow line, far ahead)
        [685, 447],             # top-right    (on the right lane line, far ahead)
        [1125, height]          # bottom-right (on the right lane line)
    ])

    # Destination: a straight rectangle filling most of the output image,
    # with a small margin on left/right so warped lines don't clip.
    margin = int(width * 0.20)
    dst = np.float32([
        [margin, height],              # bottom-left
        [margin, 0],                   # top-left
        [width - margin, 0],           # top-right
        [width - margin, height]       # bottom-right
    ])

    return src, dst


def warp_image(img):
    """
    Apply the perspective warp to a color or binary image.
    Returns (warped_image, M, Minv) where M is the forward transform
    and Minv is its inverse (needed later to warp lane overlays back
    onto the original image).
    """
    src, dst = get_perspective_points(img.shape)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    height, width = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)

    return warped, M, Minv


def process_image(path, save_dir="output_images"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Warp the original color image (for visual inspection)
    warped_color, M, Minv = warp_image(img)

    # Warp the Step 2 binary image (this is what Step 4 will actually use)
    binary = combined_binary(img)
    warped_binary, _, _ = warp_image(binary)

    # Draw the source trapezoid on the original for a sanity-check image
    src, _ = get_perspective_points(img.shape)
    overlay = img.copy()
    cv2.polylines(overlay, [src.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    cv2.imwrite(os.path.join(save_dir, f"{base}_src_overlay.jpg"), overlay)
    cv2.imwrite(os.path.join(save_dir, f"{base}_warped_color.jpg"), warped_color)
    cv2.imwrite(os.path.join(save_dir, f"{base}_warped_binary.jpg"), warped_binary)

    print(f"Saved: {base}_src_overlay.jpg, {base}_warped_color.jpg, {base}_warped_binary.jpg -> {save_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/perspective.py <path_to_image>")
        sys.exit(1)

    process_image(sys.argv[1])