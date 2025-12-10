from __future__ import annotations
from typing import Dict, Any
import time
import csv
from pathlib import Path

import cv2
import numpy as np

from common.video import open_source
from .classical_ops import detect_lanes, draw_lanes, evaluate_detection_quality
from .state import estimate_pose, SpeedEstimator


def _put_label(img, text):
    """Write a readable black rectangle label on any image."""
    if img is None:
        return None
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(out, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
    cv2.putText(out, text, (6, th + 6), font, scale, (255, 255, 255), thickness)
    return out


def run(source: str, cfg: Dict[str, Any], display: bool = False) -> int:
    """
    PURE CLASSICAL PIPELINE (NO NEURAL MASK)
    Shows FULL 6-panel debug visualization with titles.
    """

    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # --- configs ---
    hsvcfg = cfg.get("hsv", {}) or {}
    canny  = cfg.get("canny", {}) or {}
    hough  = cfg.get("hough", {}) or {}
    kincfg = cfg.get("kinematics", {}) or {}

    # --- kinematics ---
    fps   = float(kincfg.get("fps", 30.0))
    mpp   = float(kincfg.get("meters_per_pixel_bottom", 0.001))
    laneW = float(kincfg.get("assumed_lane_width_m", 1.2))

    spd = SpeedEstimator(fps=fps, meters_per_pixel_bottom=mpp)

    # --- robustness counters ---
    total_frames = 0
    good_frames  = 0
    bad_dx       = 0
    bad_da       = 0

    prev_x = None
    prev_alpha = None
    prev_valid = False

    dx_thresh = 0.5
    da_thresh = np.deg2rad(10)

    # --- CSV logging ---
    csv_path = Path("data/benchmark_classical.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "x_m", "alpha_deg", "v_mps", "dt_sec"])

    prev_time = time.perf_counter()
    frame_idx = 0

    # ===================================================================
    # MAIN LOOP
    # ===================================================================
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total_frames += 1
        h_img, w_img = frame.shape[:2]

        # ------------------------
        # RUN CLASSICAL PIPELINE
        # ------------------------
        result = detect_lanes(frame, None, hsvcfg, canny, hough)
        quality = evaluate_detection_quality(result, w_img, h_img)
        print("DETECTION QUALITY:", quality)

        x_m, alpha = estimate_pose(result, frame.shape, mpp, assumed_lane_width_m=laneW)
        v_mps = spd.update(frame, None)

        # ------------------------
        # ROBUSTNESS
        # ------------------------
        valid_frame = bool(quality.get("valid_frame"))

        if valid_frame:
            good_frames += 1

        if prev_valid and valid_frame and prev_x is not None:
            dx = abs(x_m - prev_x)
            da = abs(alpha - prev_alpha)
            if dx > dx_thresh: bad_dx += 1
            if da > da_thresh: bad_da += 1

        prev_x = x_m
        prev_alpha = alpha
        prev_valid = valid_frame

        # ------------------------
        # LOGGING
        # ------------------------
        now = time.perf_counter()
        dt_sec = now - prev_time if frame_idx > 0 else 0
        prev_time = now

        csv_writer.writerow([
            frame_idx,
            f"{x_m:.6f}",
            f"{np.degrees(alpha):.3f}",
            f"{v_mps:.6f}",
            f"{dt_sec:.6f}",
        ])
        csv_file.flush()

        # -----------------------------------------------------
        # FULL 6-PANEL VISUALIZATION WITH TITLES
        # -----------------------------------------------------
        if display:

            # Raw debug outputs
            hsv_mask     = result.get("hsv_mask")
            edges        = result.get("edges")
            edges_masked = result.get("edges_masked")
            hough_vis    = result.get("hough_vis")
            lanes_vis    = draw_lanes(frame.copy(), result)
            original     = frame.copy()

            # Guarantee BGR
            def to_bgr(img):
                if img is None:
                    return np.zeros_like(frame)
                if img.ndim == 2:
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return img

            panels = [
                _put_label(to_bgr(hsv_mask), "HSV MASK"),
                _put_label(to_bgr(edges), "CANNY EDGES"),
                _put_label(to_bgr(edges_masked), "MASKED EDGES"),
                _put_label(to_bgr(hough_vis), "HOUGH LINES"),
                _put_label(lanes_vis, "FINAL LANES"),
                _put_label(to_bgr(original), "ORIGINAL FRAME"),
            ]

            # Resize all panels
            half = (w_img // 2, h_img // 2)
            panels = [cv2.resize(p, half, interpolation=cv2.INTER_AREA) for p in panels]

            grid = np.vstack([
                np.hstack(panels[0:3]),
                np.hstack(panels[3:6]),
            ])

            cv2.imshow("RDW - Classical Debug Grid", grid)

            # ESC → quit
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        frame_idx += 1

    # ===================================================================
    # END
    # ===================================================================
    cap.release()
    csv_file.close()

    # ----------------------
    # PRINT SUMMARY
    # ----------------------
    if total_frames > 0:
        print("\n========== ROBUSTNESS SUMMARY ==========")
        print(f"total_frames: {total_frames}")
        print(f"good_frames:  {good_frames}")
        print(f"bad_dx:       {bad_dx}")
        print(f"bad_da:       {bad_da}")
        print(f"good_ratio:   {good_frames/total_frames:.4f}")
        print(f"bad_dx_ratio: {bad_dx/total_frames:.4f}")
        print(f"bad_da_ratio: {bad_da/total_frames:.4f}")
        print("========================================\n")

    return total_frames
