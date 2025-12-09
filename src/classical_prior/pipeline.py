from __future__ import annotations
from typing import Dict, Any
import cv2
import numpy as np
import time
import csv
from pathlib import Path

from common.video import open_source
from .classical_ops import detect_lanes, draw_lanes, evaluate_detection_quality
from .roadnet_runtime import PriorRuntime
from .state import estimate_pose, SpeedEstimator


def run(source: str, cfg: Dict[str, Any], display: bool = False) -> int:
    """
    Classical + neural-prior pipeline.

    Per frame:
      1. YOLOP prior → drivable-area probabilities
      2. Mask refinement (smoothing, horizon suppression, grass/sky filtering)
      3. Threshold → road mask
      4. Classical lane extraction (HSV → Canny → Hough)
      5. Pose estimation and speed estimation
      6. Optional visualization (6-panel debug grid)
    """

    # ----------------------------------------------------------------------
    # 0. Input source
    # ----------------------------------------------------------------------
    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # ----------------------------------------------------------------------
    # 1. Load config sections
    # ----------------------------------------------------------------------
    pcfg   = cfg.get("prior", {}) or {}
    hsvcfg = cfg.get("hsv", {}) or {}
    canny  = cfg.get("canny", {}) or {}
    hough  = cfg.get("hough", {}) or {}
    kincfg = cfg.get("kinematics", {}) or {}

    prev_prob = None

    # ----------------------------------------------------------------------
    # 2. Prior model (YOLOP / ONNX)
    # ----------------------------------------------------------------------
    prior = PriorRuntime(
        onnx_path=pcfg.get("model"),
        input_size=tuple(pcfg.get("input_size", [640, 640])),
        threshold=float(pcfg.get("threshold", 0.65)),
        enabled=bool(pcfg.get("enabled", True)),
        letterbox=bool(pcfg.get("letterbox", True)),
    )

    # ----------------------------------------------------------------------
    # 3. Kinematics helpers
    # ----------------------------------------------------------------------
    fps   = float(kincfg.get("fps", 30.0))
    mpp   = float(kincfg.get("meters_per_pixel_bottom", 0.0015))
    laneW = float(kincfg.get("assumed_lane_width_m", 1.2))

    spd_estimator = SpeedEstimator(fps=fps, meters_per_pixel_bottom=mpp)

    metrics_path = Path("data/benchmark_classical.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["frame_idx", "x_m", "alpha_deg", "v_mps", "dt_sec"])
    prev_time = time.perf_counter()

    frames = 0
    last_vis = None
    
    full_times = []
    classical_times = []

    # ======================================================================
    # MAIN LOOP
    # ======================================================================
    while True:
        t0_full = time.perf_counter()
        
        ok, frame = cap.read()
        if not ok:
            break

        h_img, w_img = frame.shape[:2]

        # ================================================================
        # 4. PRIOR INFERENCE — drivable (DA) + lane-line (LL) probabilities
        # ================================================================
        if hasattr(prior, "infer_probs"):
            prob_da, prob_ll = prior.infer_probs(frame)
        else:   # fallback if only binary mask available
            mask_raw = prior.infer_mask(frame)
            prob_da = mask_raw.astype(np.float32) / 255.0
            prob_ll = np.zeros_like(prob_da, dtype=np.float32)

        # ================================================================
        # 5. FUSE PROBABILITIES — strengthen LL, smooth in time
        # ================================================================
        lane_boost = cv2.GaussianBlur(prob_ll, (0, 0), 1.0)
        prob = np.maximum(prob_da, np.minimum(1.0, lane_boost * 1.2))

        alpha_smooth = 0.6
        if prev_prob is None:
            prev_prob = prob.copy()
        else:
            prob = alpha_smooth * prob + (1 - alpha_smooth) * prev_prob
            prev_prob = prob.copy()

        # ================================================================
        # 6. SUPPRESS SKY & GRASS USING HSV
        # ================================================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vc = cv2.split(hsv)

        # Horizon suppression
        horizon_y = int(0.35 * h_img)
        prob[:horizon_y, :] *= 0.2

        # Grass suppression
        grass_mask = ((Hc >= 35) & (Hc <= 85) & (Sc >= 60)).astype(np.uint8) * 255
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        prob *= (1.0 - 0.9 * grass_mask.astype(np.float32) / 255.0)

        # Sky suppression via brightness
        sky = (Vc > 180).astype(np.float32)
        prob *= (1.0 - 0.8 * sky)

        # ================================================================
        # 7. CONVERT PROBABILITY FIELD → BINARY ROAD MASK
        # ================================================================
        thr = float(pcfg.get("threshold", 0.65))
        mask = (prob >= thr).astype(np.uint8) * 255

        # Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # Erode + dilate to stabilize boundaries
        erode_k = int(pcfg.get("mask_erode", 6))
        mask = cv2.erode(mask, np.ones((erode_k, erode_k), np.uint8), 1)

        dilate_k = int(pcfg.get("mask_dilate", 11))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, kernel, 1)

        # ================================================================
        # 8. CLASSICAL LANE EXTRACTION
        # ================================================================
        t1_full = time.perf_counter()
        t0_class = time.perf_counter()
        
        result = detect_lanes(frame, mask, hsvcfg, canny, hough)
        
        t1_full = time.perf_counter()
        full_times.append((t1_full - t0_full) * 1000)
        
        t1_class = time.perf_counter()
        classical_times.append((t1_class - t0_class) * 1000)

        # Evaluate quality (used for robustness later)
        quality = evaluate_detection_quality(result, w_img, h_img)
        print("DETECTION QUALITY:", quality)

        # ================================================================
        # 9. KINEMATICS: lateral offset x, heading angle α, speed v
        # ================================================================
        x_m, alpha = estimate_pose(result, frame.shape, mpp, assumed_lane_width_m=laneW)
        v_mps = spd_estimator.update(frame, mask)

        # ================================================================
        # 10. VISUALIZATION OF FINAL LANES
        # ================================================================
        vis = draw_lanes(frame.copy(), result)

        # Label
        label = "FINAL LANES"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.7, 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(vis, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
        cv2.putText(vis, label, (5, th + 5), font, scale, (255, 255, 255), thick)

        # ================================================================
        # 11. DEBUG GRID (optional)
        # ================================================================
        if display:
            overlay_color = np.zeros_like(frame)
            overlay_color[..., 1] = mask
            overlay_color[..., 2] = mask
            fused_mask_overlay = cv2.addWeighted(frame, 0.65, overlay_color, 0.35, 0.0)

            # Debug labels
            (lw, lh), _ = cv2.getTextSize("FUSED PRIOR MASK", font, scale, thick)
            cv2.rectangle(fused_mask_overlay, (0, 0), (lw + 20, lh + 20), (0, 0, 0), -1)
            cv2.putText(fused_mask_overlay, "FUSED PRIOR MASK", (5, lh + 5), font, scale, (255, 255, 255), thick)

            blank = np.zeros((h_img, w_img, 3), dtype=np.uint8)

            panels = [
                fused_mask_overlay,
                result.get("hsv_mask", blank),
                result.get("edges", blank),
                result.get("edges_masked", blank),
                result.get("hough_vis", blank),
                vis,
            ]

            # Ensure consistent 3-channel BGR
            for i in range(len(panels)):
                if panels[i].ndim == 2:
                    panels[i] = cv2.cvtColor(panels[i], cv2.COLOR_GRAY2BGR)

            # Resize for grid
            half = (w_img // 2, h_img // 2)
            panels = [cv2.resize(p, half, interpolation=cv2.INTER_AREA) for p in panels]

            grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
            cv2.imshow("RDW - Debug Grid", grid)

            if (cv2.waitKey(0 if getattr(cap, "single_image", False) else 1) & 0xFF) == 27:
                break

        frames += 1
        last_vis = vis

        print(f"{source} | x={x_m:+.3f} | v={v_mps:+.3f} | alpha={np.degrees(alpha):+.2f}°")
        t_now = time.perf_counter()
        dt_sec = t_now - prev_time if frames > 0 else 0.0
        prev_time = t_now

        metrics_writer.writerow([
            frames,
            f"{x_m:.6f}",
            f"{np.degrees(alpha):.3f}",
            f"{v_mps:.6f}",
            f"{dt_sec:.6f}",
        ])

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    cap.release()

    if display and last_vis is not None:
        if not getattr(cap, "single_image", False):
            cv2.imshow("RDW - Classical (fusion) - last", last_vis)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    if full_times:
        avg_full = sum(full_times) / len(full_times)
        print(f"\nAverage FULL pipeline time: {avg_full:.2f} ms ({1000/avg_full:.1f} FPS)")

    if classical_times:
        avg_class = sum(classical_times) / len(classical_times)
        print(f"Average CLASSICAL-ONLY time: {avg_class:.2f} ms ({1000/avg_class:.1f} FPS)")
    metrics_file.close()

    return frames