from __future__ import annotations
from typing import Dict, Any
import time
import csv
from pathlib import Path

import cv2
import numpy as np

from common.video import open_source
from .classical_ops import detect_lanes, draw_lanes, evaluate_detection_quality
from .roadnet_runtime import PriorRuntime
from .state import estimate_pose, SpeedEstimator


# -------------------------------------------------------------
# Helper for overlay labels
# -------------------------------------------------------------
def _put_label(img, text):
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
    cv2.putText(out, text, (6, th + 6), font, scale, (255, 255, 255), thick)
    return out


def _to_bgr(img, ref):
    """Ensure every debug image is valid BGR."""
    if img is None:
        return np.zeros_like(ref)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _build_mask(prob2: np.ndarray, pcfg: Dict[str, Any], *, thr: float, er: int, dil: int) -> np.ndarray:
    """Build binary mask from prob2 with given params (no extra YOLOP calls)."""
    mask = (prob2 >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    if er > 0:
        mask = cv2.erode(mask, np.ones((er, er), np.uint8), 1)

    if dil > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
        mask = cv2.dilate(mask, kernel, 1)

    return mask


# -------------------------------------------------------------
# MAIN PIPELINE — WITH NEURAL PRIOR FUSION
# -------------------------------------------------------------
def run(source: str, cfg: Dict[str, Any], display: bool = False) -> int:

    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {source}")

    # -------------------------------
    # CONFIG
    # -------------------------------
    pcfg   = cfg.get("prior", {}) or {}
    hsvcfg = cfg.get("hsv", {}) or {}
    canny  = cfg.get("canny", {}) or {}
    hough  = cfg.get("hough", {}) or {}
    kin    = cfg.get("kinematics", {}) or {}

    fps   = float(kin.get("fps", 30))
    mpp   = float(kin.get("meters_per_pixel_bottom", 0.001))
    laneW = float(kin.get("assumed_lane_width_m", 1.2))

    spd = SpeedEstimator(fps=fps, meters_per_pixel_bottom=mpp)

    # -------------------------------
    # PRIOR MODEL
    # -------------------------------
    prior = PriorRuntime(
        onnx_path=pcfg.get("model"),
        input_size=tuple(pcfg.get("input_size", [640, 640])),
        threshold=float(pcfg.get("threshold", 0.65)),
        enabled=bool(pcfg.get("enabled", True)),
        letterbox=bool(pcfg.get("letterbox", True)),
    )

    prev_prob = None

    # -------------------------------
    # SPEED-UP CONTROLS
    # -------------------------------
    prior_stride = int(pcfg.get("prior_stride", 4))
    infer_size = pcfg.get("prior_infer_size", [416, 416])
    infer_w = int(infer_size[0])
    infer_h = int(infer_size[1])

    # -------------------------------
    # EFFICIENCY + ROBUSTNESS METRICS
    # -------------------------------
    total_full_ms = 0.0
    total_class_ms = 0.0
    n_full = 0
    n_class = 0

    total_frames = 0
    good_frames = 0
    bad_dx = 0
    bad_da = 0

    prev_x = None
    prev_a = None
    prev_valid = False

    dx_thresh = 0.5
    da_thresh = np.deg2rad(10.0)

    # -------------------------------
    # CSV LOGGING
    # -------------------------------
    csv_path = Path("data/benchmark_classical.csv")
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "x_m", "alpha_deg", "v_mps", "dt_sec"])

    prev_time = time.perf_counter()

    frame_idx = 0

    cached_prob_da = None
    cached_prob_ll = None

    last_grid = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total_frames += 1
        h_img, w_img = frame.shape[:2]

        t0_full = time.perf_counter()

        # ------------------------------------------------------------
        # 1) PRIOR INFERENCE (strided + small infer)
        # ------------------------------------------------------------
        ran_prior = False

        if (frame_idx % prior_stride == 0) or (cached_prob_da is None) or (cached_prob_ll is None):
            small = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)

            if hasattr(prior, "infer_probs"):
                prob_da_s, prob_ll_s = prior.infer_probs(small)
            else:
                raw_s = prior.infer_mask(small)
                prob_da_s = raw_s.astype(np.float32) / 255.0
                prob_ll_s = np.zeros_like(prob_da_s)

            prob_da = cv2.resize(prob_da_s, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            prob_ll = cv2.resize(prob_ll_s, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

            cached_prob_da = prob_da.copy()
            cached_prob_ll = prob_ll.copy()
            ran_prior = True
        else:
            prob_da = cached_prob_da
            prob_ll = cached_prob_ll

        # ------------------------------------------------------------
        # 2) FUSION + TEMPORAL SMOOTHING
        # ------------------------------------------------------------
        lane_boost = cv2.GaussianBlur(prob_ll, (0, 0), 1.0)
        prob = np.maximum(prob_da, np.minimum(1.0, lane_boost * 1.2)).astype(np.float32)

        if prev_prob is None:
            prev_prob = prob.copy()
        else:
            prob = 0.6 * prob + 0.4 * prev_prob
            prev_prob = prob.copy()

        # ------------------------------------------------------------
        # 3) SUPPRESS SKY + GRASS
        # ------------------------------------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vc = cv2.split(hsv)

        horizon = int(pcfg.get("horizon_ratio", 0.35) * h_img)

        prob2 = prob.copy()
        prob2[:horizon, :] *= 0.2

        grass = ((Hc >= 35) & (Hc <= 85) & (Sc >= 60)).astype(np.uint8) * 255
        grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        prob2 *= (1.0 - 0.9 * (grass.astype(np.float32) / 255.0))

        sky = (Vc > 180).astype(np.float32)
        prob2 *= (1.0 - 0.8 * sky)

        # ------------------------------------------------------------
        # 4) MASK (default params)
        # ------------------------------------------------------------
        base_thr = float(pcfg.get("threshold", 0.65))
        base_er  = int(pcfg.get("mask_erode", 6))
        base_dil = int(pcfg.get("mask_dilate", 11))

        mask = _build_mask(prob2, pcfg, thr=base_thr, er=base_er, dil=base_dil)

        # ------------------------------------------------------------
        # 5) CLASSICAL LANES (masked)
        # ------------------------------------------------------------
        result = detect_lanes(frame, mask, hsvcfg, canny, hough)
        quality = evaluate_detection_quality(result, w_img, h_img)
        print("DETECTION QUALITY:", quality)

        # ------------------------------------------------------------
        # TURN FIX: fallback BEFORE rerunning YOLOP
        #
        # If masked result is invalid, try:
        #   (A) relaxed mask (same prob2, lower thr, more dil)
        #   (B) classical-only (no mask) and use if valid
        # ------------------------------------------------------------
        if not quality.get("valid_frame", False):
            relaxed_thr = max(0.45, base_thr - 0.08)
            relaxed_dil = int(max(base_dil, base_dil + 6))
            relaxed_er  = int(max(0, base_er - 2))

            mask_relaxed = _build_mask(prob2, pcfg, thr=relaxed_thr, er=relaxed_er, dil=relaxed_dil)
            result_relaxed = detect_lanes(frame, mask_relaxed, hsvcfg, canny, hough)
            quality_relaxed = evaluate_detection_quality(result_relaxed, w_img, h_img)

            if quality_relaxed.get("valid_frame", False):
                mask = mask_relaxed
                result = result_relaxed
                quality = quality_relaxed
                print("DETECTION QUALITY (RELAXED MASK):", quality)
            else:
                # Classical-only fallback (very cheap) for turns where mask cuts a lane.
                result_class_fallback = detect_lanes(frame, None, hsvcfg, canny, hough)
                quality_class_fallback = evaluate_detection_quality(result_class_fallback, w_img, h_img)

                if quality_class_fallback.get("valid_frame", False):
                    # Keep mask for speed estimator, but use classical-only lanes for pose/robustness.
                    result = result_class_fallback
                    quality = quality_class_fallback
                    print("DETECTION QUALITY (CLASSICAL FALLBACK):", quality)

        # ------------------------------------------------------------
        # 6) POSE + SPEED
        # ------------------------------------------------------------
        x_m, alpha = estimate_pose(result, frame.shape, mpp, laneW)
        v_mps = spd.update(frame, mask)

        t1_full = time.perf_counter()
        dt_full = (t1_full - t0_full) * 1000.0
        total_full_ms += dt_full
        n_full += 1

        # ------------------------------------------------------------
        # CLASSICAL-ONLY TIMING (unchanged)
        # ------------------------------------------------------------
        t0_class = time.perf_counter()
        result_class = detect_lanes(frame, None, hsvcfg, canny, hough)
        _ = estimate_pose(result_class, frame.shape, mpp, laneW)
        dt_class = (time.perf_counter() - t0_class) * 1000.0
        total_class_ms += dt_class
        n_class += 1

        # ------------------------------------------------------------
        # ROBUSTNESS
        # ------------------------------------------------------------
        valid = bool(quality.get("valid_frame"))
        if valid:
            good_frames += 1

        if prev_valid and valid and prev_x is not None:
            dx = abs(x_m - prev_x)
            da = abs(alpha - prev_a)
            if dx > dx_thresh:
                bad_dx += 1
            if da > da_thresh:
                bad_da += 1

        prev_x = x_m
        prev_a = alpha
        prev_valid = valid

        # ------------------------------------------------------------
        # CSV LOGGING
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # VISUALIZATION (6-panel)
        # ------------------------------------------------------------
        if display:
            color = np.zeros_like(frame)
            color[..., 1] = mask
            color[..., 2] = mask
            fused_overlay = cv2.addWeighted(frame, 0.65, color, 0.35, 0.0)

            hsv_mask     = _to_bgr(result.get("hsv_mask"), frame)
            edges        = _to_bgr(result.get("edges"), frame)
            edges_masked = _to_bgr(result.get("edges_masked"), frame)
            hough_vis    = _to_bgr(result.get("hough_vis"), frame)
            final_lanes  = draw_lanes(frame.copy(), result)

            p1 = _put_label(fused_overlay, "FUSED PRIOR MASK")
            p2 = _put_label(hsv_mask, "HSV MASK")
            p3 = _put_label(edges, "CANNY EDGES")
            p4 = _put_label(edges_masked, "MASKED EDGES")
            p5 = _put_label(hough_vis, "HOUGH LINES")
            p6 = _put_label(final_lanes, "FINAL LANES")

            panels = [p1, p2, p3, p4, p5, p6]

            half = (w_img // 2, h_img // 2)
            panels = [cv2.resize(p, half, interpolation=cv2.INTER_AREA) for p in panels]

            grid = np.vstack([
                np.hstack(panels[0:3]),
                np.hstack(panels[3:6]),
            ])

            cv2.imshow("RDW - Neural Fusion + Classical Debug Grid", grid)
            last_grid = grid

            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        frame_idx += 1

    cap.release()
    csv_file.close()

    if display and last_grid is not None:
        cv2.imshow("RDW - Neural Fusion + Classical Debug Grid (FINAL)", last_grid)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if n_full > 0:
        avg_full = total_full_ms / n_full
        print(f"\nAverage FULL pipeline time: {avg_full:.2f} ms ({1000/avg_full:.1f} FPS)")

    if n_class > 0:
        avg_class = total_class_ms / n_class
        print(f"Average CLASSICAL-ONLY time: {avg_class:.2f} ms ({1000/avg_class:.1f} FPS)\n")

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