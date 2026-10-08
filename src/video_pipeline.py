"""
Step 4 (video) + Step 5: Full Pipeline on Video + Sharp-Turn Warning
----------------------------------------------------------------------
Goal: run the whole lane-detection pipeline on every frame of a video
and overlay:
  - the detected lane curvature (in meters)
  - a "SHARP TURN AHEAD" warning when the curve gets tight enough

This file does NOT change threshold.py, perspective.py, or lane_fit.py.
It just imports and reuses their functions on each video frame, since a
video frame is just a single image, one after another.

Run this file directly:
    python src/video_pipeline.py test_videos/project_video.mp4
"""

import cv2
import numpy as np
import sys
import os

from threshold import combined_binary, region_of_interest       # Step 2 (friend's code, enhanced with Lab/Luv/Sobel)
from perspective import warp_image          # Step 3 (friend's code, unchanged)
from curvature import (
    fit_polynomial,
    measure_curvature_real,
    RobustLaneTracker,
    YM_PER_PIX,
    XM_PER_PIX
)  # Step 4
from warning import SharpTurnWarningSystem

# Initialize warning system with hysteresis and debounce parameters
warning_system = SharpTurnWarningSystem(
    enter_threshold=300,
    exit_threshold=450,
    smoothing_window=8,
    debounce_frames=5
)
lane_tracker = RobustLaneTracker(buffer_size=10)


def draw_lane_overlay(original_img, warped_binary, left_fit, right_fit, Minv):
    """
    Draw the detected lane as a green filled area on the warped image,
    then unwarp it back onto the original (non-bird's-eye) image so it
    looks like a normal lane highlight on the road.
    """
    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    color_warp = np.zeros_like(original_img).astype(np.uint8)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # Draw left and right lane outlines in blue (BGR: 255, 0, 0)
    cv2.polylines(color_warp, np.int_([pts_left]), isClosed=False, color=(255, 0, 0), thickness=30)
    cv2.polylines(color_warp, np.int_([pts_right]), isClosed=False, color=(255, 0, 0), thickness=30)

    # Unwarp the blue lane lines back to the original camera perspective
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 1.0, 0)
    return result


def annotate_frame(result, avg_curverad, is_sharp_turn):
    """
    Draw the curvature reading and, if needed, a sharp-turn warning
    in the top-left corner of the frame.
    """
    cv2.putText(result, f"Curve radius: {avg_curverad:.0f} m", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    if is_sharp_turn:
        cv2.putText(result, "!! SHARP TURN AHEAD !!", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)

    return result


def process_frame(frame, warning_system, tracker=lane_tracker):
    """
    Run the full pipeline on a single video frame with temporal tracking.
    """
    try:
        binary = combined_binary(frame)
        roi_binary, _ = region_of_interest(binary)
        warped_binary, M, Minv = warp_image(roi_binary)

        left_fit, right_fit, ok = tracker.update(warped_binary)

        if left_fit is not None and right_fit is not None:
            _, _, avg_curverad = measure_curvature_real(warped_binary, left_fit, right_fit)
            is_sharp_turn = warning_system.update(avg_curverad)
            
            # Display smoothed curvature value from the warning buffer for visual stability
            display_curverad = np.median(warning_system.curverad_buffer) if warning_system.curverad_buffer else avg_curverad
            result = draw_lane_overlay(frame, warped_binary, left_fit, right_fit, Minv)
            result = annotate_frame(result, display_curverad, is_sharp_turn)
            return result
        else:
            warning_system.update(None)
            cv2.putText(frame, "Lane not detected", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            return frame

    except Exception:
        warning_system.update(None)
        cv2.putText(frame, "Lane not detected", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return frame


def process_video(input_path, output_path="output_images/output_video.mp4"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = process_frame(frame, warning_system, lane_tracker)
        writer.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    writer.release()
    print(f"Done. Saved {frame_count} frames -> {output_path}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/video_pipeline.py <path_to_video>")
        sys.exit(1)

    process_video(sys.argv[1])