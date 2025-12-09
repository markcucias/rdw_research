from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time

import cv2
import numpy as np

from common.video import open_source
from .classical_ops import detect_lanes, draw_lanes, evaluate_detection_quality
from .roadnet_runtime import PriorRuntime
from .state import estimate_pose, SpeedEstimator


def run(source: str, cfg: Dict[str, Any], display: bool = False) -> int:
    """
    Classical + neural-prior pipeline with:
      - Efficiency benchmark:
          * FULL pipeline (YOLOP + fusion + classical)
          * CLASSICAL-ONLY (no prior / no road mask)
      - Robustness benchmark:
          * counts large frame-to-frame deviations in pose (x, alpha)

    Per frame:
      - Prior (YOLOP) → drivable/lane probabilities, fused and pruned to a road mask
      - Classical (HSV → Canny → Hough) with *late* masking for robustness
      - Kinematics: lateral offset x (m), heading α (rad), forward speed v (m/s)
      - Optional visualization: debug grid and final lanes
    """
    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # --- config blocks ---
    pcfg   = cfg.get("prior", {}) or {}
    hsvcfg = cfg.get("hsv", {}) or {}
    canny  = cfg.get("canny", {}) or {}
    hough  = cfg.get("hough", {}) or {}
    kin    = cfg.get("kinematics", {}) or {}

    prev_prob = None

    # --- prior runtime (YOLOP) ---
    prior = PriorRuntime(
        onnx_path=pcfg.get("model"),
        input_size=tuple(pcfg.get("input_size", [640, 640])),
        threshold=float(pcfg.get("threshold", 0.65)),
        enabled=bool(pcfg.get("enabled", True)),
        letterbox=bool(pcfg.get("letterbox", True)),
    )

    # --- kinematics helpers ---
    fps   = float(kin.get("fps", 30.0))
    mpp   = float(kin.get("meters_per_pixel_bottom", 0.01))  # tune after calibration
    laneW = kin.get("assumed_lane_width_m", 3.5)
    spd   = SpeedEstimator(fps=fps, meters_per_pixel_bottom=mpp)

    # --- efficiency accumulators ---
    total_full_ms = 0.0
    total_classical_ms = 0.0
    n_full = 0
    n_classical = 0

    # --- robustness accumulators (pose-based) ---
    total_frames = 0
    good_frames = 0          # frames where detection_quality["valid_frame"] is True
    bad_dx = 0               # large jumps in lateral offset
    bad_da = 0               # large jumps in heading

    prev_x: float | None = None
    prev_alpha: float | None = None
    prev_valid: bool = False

    # thresholds for "large deviation"
    dx_thresh = 0.5                  # in meters (tune as needed)
    da_thresh = np.deg2rad(10.0)     # 10 degrees in radians

    last_vis = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total_frames += 1
        h_img, w_img = frame.shape[:2]

        # ================= FULL PIPELINE (timed) =================
        t0_full = time.perf_counter()

        # 1) prior probabilities (prefer DA+LL; fall back to binary if needed)
        if hasattr(prior, "infer_probs"):
            prob_da, prob_ll = prior.infer_probs(frame)
        else:
            # Legacy binary mask
            mask0 = prior.infer_mask(frame)
            prob_da = mask0.astype(np.float32) / 255.0
            prob_ll = np.zeros_like(prob_da, dtype=np.float32)

        # 2) fuse DA with softened lane-line map; downweight sky and grass
        lane_boost = cv2.GaussianBlur(prob_ll, (0, 0), 1.0)
        prob = np.maximum(prob_da, np.minimum(1.0, lane_boost * 1.2))

        # temporal smoothing of prior probabilities
        alpha_smooth = 0.6  # tune 0.5–0.8
        if prev_prob is None:
            prev_prob = prob.copy()
        else:
            prob = alpha_smooth * prob + (1.0 - alpha_smooth) * prev_prob
            prev_prob = prob.copy()

        # suppress sky based on horizon ratio
        horizon_ratio = float(pcfg.get("horizon_ratio", 0.35))
        horizon = int(horizon_ratio * h_img)
        prob[:horizon, :] *= 0.2

        # suppress obvious grass by HSV heuristic
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vc = cv2.split(hsv_img)
        grass = ((Hc >= 35) & (Hc <= 85) & (Sc >= 60)).astype(np.uint8) * 255
        grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        prob *= (1.0 - (grass.astype(np.float32) / 255.0) * 0.9)

        # extra sky suppression using brightness
        sky = (Vc > 180).astype(np.float32)  # tune 170–200
        prob *= (1.0 - sky * 0.8)

        # 3) threshold + tidy + morphological operations → road mask
        thr = float(pcfg.get("threshold", 0.65))
        mask = (prob >= thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        d = int(pcfg.get("mask_dilate", 11))
        if d > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        else:
            kernel = np.ones((1, 1), np.uint8)

        erosion = int(pcfg.get("mask_erode", 6))
        if erosion > 0:
            mask = cv2.erode(mask, np.ones((erosion, erosion), np.uint8), iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 4) classical lanes (late masking happens inside detect_lanes)
        result = detect_lanes(frame, mask, hsv_cfg=hsvcfg, canny_cfg=canny, hough_cfg=hough)

        # 5) detection quality (geometry-based)
        quality = evaluate_detection_quality(result, w_img, h_img)
        print("DETECTION QUALITY:", quality)

        # 6) pose (x, α, v)
        x_m, alpha = estimate_pose(result, frame.shape, mpp, assumed_lane_width_m=laneW)
        v_mps = spd.update(frame, mask)

        t1_full = time.perf_counter()
        full_ms = (t1_full - t0_full) * 1000.0
        total_full_ms += full_ms
        n_full += 1

        # ================= CLASSICAL-ONLY PIPELINE (timed) =================
        # Run the classical (HSV+Canny+Hough) pipeline without any road mask
        t0_classical = time.perf_counter()
        result_classical = detect_lanes(frame, None, hsv_cfg=hsvcfg, canny_cfg=canny, hough_cfg=hough)
        _ = estimate_pose(result_classical, frame.shape, mpp, assumed_lane_width_m=laneW)
        t1_classical = time.perf_counter()

        classical_ms = (t1_classical - t0_classical) * 1000.0
        total_classical_ms += classical_ms
        n_classical += 1

        # ================= ROBUSTNESS BENCHMARK (pose jumps) =================
        valid_frame = bool(quality.get("valid_frame", False))
        if valid_frame:
            good_frames += 1

        if prev_valid and valid_frame and prev_x is not None and prev_alpha is not None:
            dx = abs(x_m - prev_x)
            da = abs(alpha - prev_alpha)

            if dx > dx_thresh:
                bad_dx += 1
            if da > da_thresh:
                bad_da += 1

        prev_x = x_m
        prev_alpha = alpha
        prev_valid = valid_frame

        # ================= VISUALIZATION =================
        vis = draw_lanes(frame.copy(), result)

        # Draw label "FINAL LANES"
        text = "FINAL LANES"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(vis, (0, 0), (text_width + 20, text_height + 20), (0, 0, 0), thickness=-1)
        cv2.putText(vis, text, (5, text_height + 5), font, scale, (255, 255, 255), thickness)

        if display:
            # fused mask overlay with label
            color = np.zeros_like(frame)
            color[..., 1] = mask
            color[..., 2] = mask
            mask_overlay = cv2.addWeighted(frame, 0.65, color, 0.35, 0.0)

            label_text = "FUSED PRIOR MASK"
            (label_width, label_height), label_baseline = cv2.getTextSize(label_text, font, scale, thickness)
            cv2.rectangle(mask_overlay, (0, 0), (label_width + 20, label_height + 20), (0, 0, 0), thickness=-1)
            cv2.putText(mask_overlay, label_text, (5, label_height + 5), font, scale, (255, 255, 255), thickness)

            h, w = frame.shape[:2]
            blank_img = np.zeros((h, w, 3), dtype=np.uint8)

            debug_imgs = [
                mask_overlay,
                result.get("hsv_mask", blank_img),
                result.get("edges", blank_img),
                result.get("edges_masked", blank_img),
                result.get("hough_vis", blank_img),
                vis,
            ]

            # Ensure all debug images are BGR
            for i in range(len(debug_imgs)):
                img = debug_imgs[i]
                if img.ndim == 2:
                    debug_imgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Resize all images to half size
            half_size = (w // 2, h // 2)
            debug_imgs_resized = [
                cv2.resize(im, half_size, interpolation=cv2.INTER_AREA)
                for im in debug_imgs
            ]

            # Stack into 2x3 grid
            top_row = np.hstack(debug_imgs_resized[0:3])
            bottom_row = np.hstack(debug_imgs_resized[3:6])
            grid = np.vstack([top_row, bottom_row])

            cv2.imshow("RDW - Debug Grid", grid)

            wait = 0 if getattr(cap, "single_image", False) else 1
            if (cv2.waitKey(wait) & 0xFF) == 27:
                break

        last_vis = vis

        print(
            f"{source} | x={x_m:+.3f} | v={v_mps:+.3f} | alpha={np.degrees(alpha):+.2f}°"
        )

    # ================= FINAL STATS =================
    cap.release()
    if display and last_vis is not None:
        if not getattr(cap, "single_image", False):
            cv2.imshow("RDW - Classical (fusion) - last", last_vis)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    if n_full > 0:
        avg_full = total_full_ms / n_full
        fps_full = 1000.0 / avg_full if avg_full > 0 else 0.0
        print(
            f"\nAverage FULL pipeline time: {avg_full:.2f} ms ({fps_full:.1f} FPS)"
        )

    if n_classical > 0:
        avg_classical = total_classical_ms / n_classical
        fps_classical = 1000.0 / avg_classical if avg_classical > 0 else 0.0
        print(
            f"Average CLASSICAL-ONLY time: {avg_classical:.2f} ms ({fps_classical:.1f} FPS)\n"
        )

    # robustness summary
    if total_frames > 0:
        good_ratio = good_frames / total_frames
        bad_dx_ratio = bad_dx / total_frames
        bad_da_ratio = bad_da / total_frames
    else:
        good_ratio = bad_dx_ratio = bad_da_ratio = 0.0

    print("========== ROBUSTNESS SUMMARY ==========")
    print(f"total_frames: {float(total_frames):.4f}")
    print(f"good_frames: {float(good_frames):.4f}")
    print(f"bad_dx: {float(bad_dx):.4f}")
    print(f"bad_da: {float(bad_da):.4f}")
    print(f"good_ratio: {good_ratio:.4f}")
    print(f"bad_dx_ratio: {bad_dx_ratio:.4f}")
    print(f"bad_da_ratio: {bad_da_ratio:.4f}")
    print("========================================\n")

    print("Done. Frames are processed.")
    return total_frames