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
# helpers for debug grid
# -------------------------------------------------------------
def _put_label(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a black box with white text in the top-left corner."""
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
    cv2.putText(out, text, (6, th + 6), font, scale, (255, 255, 255), thick)
    return out


def _to_bgr(img: np.ndarray | None, ref: np.ndarray) -> np.ndarray:
    """Ensure every debug image is 3-channel BGR and non-None."""
    if img is None:
        return np.zeros_like(ref)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# -------------------------------------------------------------
# MAIN PIPELINE — Classical + YOLOP prior
# -------------------------------------------------------------
def run(source: str, cfg: Dict[str, Any], display: bool = False) -> int:
    """
    Classical + neural-prior pipeline with:
      - Efficiency benchmark (full vs classical-only)
      - Robustness benchmark (pose jumps)
      - CSV logging (overwrites data/benchmark_classical.csv each run)
      - 6-panel debug grid when display=True

    Логика детекции/оценки позы сохраняется, мы только меняем визуализацию.
    """
    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # --- configs ---
    pcfg   = cfg.get("prior", {}) or {}
    hsvcfg = cfg.get("hsv", {}) or {}
    canny  = cfg.get("canny", {}) or {}
    hough  = cfg.get("hough", {}) or {}
    kincfg = cfg.get("kinematics", {}) or {}
    y_ref_ratio = float(kincfg.get("y_ref_ratio", 0.65))

    prev_prob = None

    # --- YOLOP prior ---
    prior = PriorRuntime(
        onnx_path=pcfg.get("model"),
        input_size=tuple(pcfg.get("input_size", [640, 640])),
        threshold=float(pcfg.get("threshold", 0.65)),
        enabled=bool(pcfg.get("enabled", True)),
        letterbox=bool(pcfg.get("letterbox", True)),
    )

    # --- kinematics ---
    fps   = float(kincfg.get("fps", 30.0))
    mpp   = float(kincfg.get("meters_per_pixel_bottom", 0.001))
    laneW = float(kincfg.get("assumed_lane_width_m", 1.2))

    # сглаживание x и alpha
    x_smooth_alpha = float(kincfg.get("x_smooth_alpha", 0.7))

    spd = SpeedEstimator(fps=fps, meters_per_pixel_bottom=mpp)

    # --- efficiency accumulators ---
    total_full_ms = 0.0
    total_classical_ms = 0.0
    n_full = 0
    n_classical = 0

    # --- robustness accumulators ---
    total_frames = 0
    good_frames  = 0
    bad_dx       = 0
    bad_da       = 0

    # --- pose history for robustness + сглаживание ---
    prev_x     = None
    prev_alpha = None
    prev_valid = False
    have_prev_pose = False

    dx_thresh = 0.5                 # м, порог для "скачков" x
    da_thresh = np.deg2rad(10.0)    # рад, порог для "скачков" угла

    # ================================================================
    # CSV FILE (rewritten every run)
    # ================================================================
    csv_path = Path("data/benchmark_classical.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "x_m", "alpha_deg", "v_mps", "dt_sec"])

    prev_time = time.perf_counter()
    frame_idx = 0
    last_grid = None

    # ================================================================
    # MAIN LOOP
    # ================================================================
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        total_frames += 1
        h_img, w_img = frame.shape[:2]

        # ============================================================
        # FULL PIPELINE TIMING
        # ============================================================
        t0_full = time.perf_counter()

        # 1) PRIOR INFERENCE
        if hasattr(prior, "infer_probs"):
            prob_da, prob_ll = prior.infer_probs(frame)
        else:
            mask0 = prior.infer_mask(frame)
            prob_da = mask0.astype(np.float32) / 255.0
            prob_ll = np.zeros_like(prob_da)

        # 2) FUSION + TEMPORAL SMOOTHING
        lane_boost = cv2.GaussianBlur(prob_ll, (0, 0), 1.0)
        prob = np.maximum(prob_da, np.minimum(1.0, lane_boost * 1.2))

        alpha_smooth = 0.6
        if prev_prob is None:
            prev_prob = prob.copy()
        else:
            prob = alpha_smooth * prob + (1.0 - alpha_smooth) * prev_prob
            prev_prob = prob.copy()

        # Horizon suppression
        horizon_ratio = float(pcfg.get("horizon_ratio", 0.35))
        horizon_y = int(horizon_ratio * h_img)
        prob[:horizon_y, :] *= 0.2

        # Grass suppression
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vc = cv2.split(hsv_img)
        grass = ((Hc >= 35) & (Hc <= 85) & (Sc >= 60)).astype(np.uint8) * 255
        grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        prob *= (1.0 - 0.9 * grass.astype(np.float32) / 255.0)

        # Sky suppression
        sky = (Vc > 180).astype(np.float32)
        prob *= (1.0 - 0.8 * sky)

        # 3) THRESHOLD → MASK
        thr = float(pcfg.get("threshold", 0.65))
        mask = (prob >= thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # erosion + dilation
        er_k = int(pcfg.get("mask_erode", 6))
        if er_k > 0:
            mask = cv2.erode(mask, np.ones((er_k, er_k), np.uint8), 1)

        dil_k = int(pcfg.get("mask_dilate", 11))
        if dil_k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
            mask = cv2.dilate(mask, kernel, 1)

        # 4) CLASSICAL LANES
        result = detect_lanes(frame, mask, hsv_cfg=hsvcfg, canny_cfg=canny, hough_cfg=hough)

        # 5) DETECTION QUALITY
        quality = evaluate_detection_quality(result, w_img, h_img)
        print("DETECTION QUALITY:", quality)

        # 6) POSE (с учётом valid_lane и сглаживания)
        x_raw, alpha_raw = estimate_pose(
            result,
            frame.shape,
            meters_per_pixel_bottom=mpp,
            assumed_lane_width_m=laneW,
            y_ref_ratio=y_ref_ratio,
        )
        valid_lane = bool(result.get("valid_lane", True))

        if valid_lane:
            if have_prev_pose:
                x_m   = x_smooth_alpha * x_raw     + (1.0 - x_smooth_alpha) * prev_x
                alpha = x_smooth_alpha * alpha_raw + (1.0 - x_smooth_alpha) * prev_alpha
            else:
                x_m   = x_raw
                alpha = alpha_raw

            prev_x = x_m
            prev_alpha = alpha
            have_prev_pose = True
        else:
            if have_prev_pose:
                x_m   = prev_x
                alpha = prev_alpha
            else:
                x_m   = x_raw
                alpha = alpha_raw

        v_mps = spd.update(frame, mask)

        # Close full-pipeline timing
        t1_full = time.perf_counter()
        dt_full_ms = (t1_full - t0_full) * 1000.0
        total_full_ms += dt_full_ms
        n_full += 1

        # ============================================================
        # CLASSICAL-ONLY TIMING
        # ============================================================
        t0_classical = time.perf_counter()
        result_classical = detect_lanes(frame, None, hsv_cfg=hsvcfg, canny_cfg=canny, hough_cfg=hough)
        _ = estimate_pose(
            result_classical,
            frame.shape,
            meters_per_pixel_bottom=mpp,
            assumed_lane_width_m=laneW,
            y_ref_ratio=y_ref_ratio,
        )
        t1_classical = time.perf_counter()

        dt_classical_ms = (t1_classical - t0_classical) * 1000.0
        total_classical_ms += dt_classical_ms
        n_classical += 1

        # ============================================================
        # ROBUSTNESS METRICS
        # ============================================================
        valid_frame = bool(quality.get("valid_frame", False))

        if valid_frame:
            good_frames += 1

        if prev_valid and valid_frame and prev_x is not None:
            dx = abs(x_m - prev_x)
            da = abs(alpha - prev_alpha)

            if dx > dx_thresh:
                bad_dx += 1
            if da > da_thresh:
                bad_da += 1

        prev_valid = valid_frame

        # ============================================================
        # CSV LOGGING
        # ============================================================
        now = time.perf_counter()
        dt_sec = now - prev_time if frame_idx > 0 else 0.0
        prev_time = now

        csv_writer.writerow([
            frame_idx,
            f"{x_m:.6f}",
            f"{np.degrees(alpha):.3f}",
            f"{v_mps:.6f}",
            f"{dt_sec:.6f}",
        ])
        csv_file.flush()

        # ============================================================
        # VISUALIZATION — 6-panel debug grid
        # ============================================================
        if display:
            # fused prior mask
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
            p2 = _put_label(hsv_mask,      "HSV MASK")
            p3 = _put_label(edges,         "CANNY EDGES")
            p4 = _put_label(edges_masked,  "MASKED EDGES")
            p5 = _put_label(hough_vis,     "HOUGH LINES")
            p6 = _put_label(final_lanes,   "FINAL LANES")

            panels = [p1, p2, p3, p4, p5, p6]
            half = (w_img // 2, h_img // 2)
            panels = [cv2.resize(p, half, interpolation=cv2.INTER_AREA) for p in panels]

            grid = np.vstack([
                np.hstack(panels[0:3]),
                np.hstack(panels[3:6]),
            ])

            cv2.imshow("RDW - Classical + Prior Debug Grid", grid)
            last_grid = grid

            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        # консольный вывод для дебага
        print(f"{source} | frame={frame_idx:04d} | x={x_m:+.3f} m | v={v_mps:+.3f} m/s | alpha={np.degrees(alpha):+.2f}°")

        frame_idx += 1

    # ============================================================
    # END — PRINT STATS
    # ============================================================
    cap.release()
    csv_file.close()

    if display and last_grid is not None:
        cv2.imshow("RDW - Classical + Prior Debug Grid (FINAL)", last_grid)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if n_full > 0:
        avg_full = total_full_ms / n_full
        print(f"\nAverage FULL pipeline time: {avg_full:.2f} ms ({1000.0 / avg_full:.1f} FPS)")

    if n_classical > 0:
        avg_cls = total_classical_ms / n_classical
        print(f"Average CLASSICAL-ONLY time: {avg_cls:.2f} ms ({1000.0 / avg_cls:.1f} FPS)\n")

    # robustness summary
    if total_frames > 0:
        print("========== ROBUSTNESS SUMMARY ==========")
        print(f"total_frames: {total_frames:.4f}")
        print(f"good_frames:  {good_frames:.4f}")
        print(f"bad_dx:       {bad_dx:.4f}")
        print(f"bad_da:       {bad_da:.4f}")
        print(f"good_ratio:   {good_frames/total_frames:.4f}")
        print(f"bad_dx_ratio: {bad_dx/total_frames:.4f}")
        print(f"bad_da_ratio: {bad_da/total_frames:.4f}")
        print("========================================\n")

    return total_frames
