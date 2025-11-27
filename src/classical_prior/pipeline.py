from __future__ import annotations
from typing import Dict, Any
import cv2
import numpy as np

from common.video import open_source
from .classical_ops import detect_lanes, draw_lanes
from .roadnet_runtime import PriorRuntime
from .state import estimate_pose, SpeedEstimator


def run(source: str, cfg: Dict[str, Any], display: bool = False) -> int:
    """
    Classical + neural-prior pipeline.

    Per frame:
      - Prior (YOLOP) → drivable/lane probabilities, fused and pruned to a generous road mask
      - Classical (HSV → Canny → Hough) with *late* masking for robustness
      - Kinematics: lateral offset x (m), heading α (rad), forward speed v (m/s)
      - Optional visualization: mask overlay and final lanes (with x, v, α overlay)
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

    # --- prior runtime ---
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

    frames = 0
    last_vis = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) prior probabilities (prefer DA+LL; fall back to binary if needed)
        if hasattr(prior, "infer_probs"):
            prob_da, prob_ll = prior.infer_probs(frame)
        else:
            mask0 = prior.infer_mask(frame)
            prob_da = mask0.astype(np.float32) / 255.0
            prob_ll = np.zeros_like(prob_da, dtype=np.float32)

        # 2) fuse DA with softened lane-line map; downweight sky and grass
        lane_boost = cv2.GaussianBlur(prob_ll, (0, 0), 1.0)
        prob = np.maximum(prob_da, np.minimum(1.0, lane_boost * 1.2))

        h_img, w_img = frame.shape[:2]
        horizon = int(0.35 * h_img)  # adjust per camera tilt
        prob[:horizon, :] *= 0.2

        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Hc, Sc, _ = cv2.split(hsv_img)
        grass = ((Hc >= 35) & (Hc <= 85) & (Sc >= 60)).astype(np.uint8) * 255
        grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        prob = prob * (1.0 - (grass.astype(np.float32) / 255.0) * 0.9)

        # 3) threshold + tidy + dilate to get a generous road mask
        thr = float(pcfg.get("threshold", 0.65))
        mask = (prob >= thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        d = int(pcfg.get("mask_dilate", 11))
        if d > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # 4) classical lanes (late masking happens inside detect_lanes)
        result = detect_lanes(frame, mask, hsvcfg, canny, hough)

        # 5) kinematics (x, α, v)
        x_m, alpha = estimate_pose(result, frame.shape, mpp, assumed_lane_width_m=laneW)
        v_mps = spd.update(frame, mask)

        # 6) visualization
        vis = draw_lanes(frame.copy(), result)
        # Draw black rectangle background and white text "FINAL LANES"
        text = "FINAL LANES"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(vis, (0, 0), (text_width + 10, text_height + 10), (0, 0, 0), thickness=-1)
        cv2.putText(vis, text, (5, text_height + 5), font, scale, (255, 255, 255), thickness)

        if display:
            # fused mask overlay
            color = np.zeros_like(frame)
            color[..., 1] = mask
            color[..., 2] = mask
            mask_overlay = cv2.addWeighted(frame, 0.65, color, 0.35, 0.0)
            # Draw black rectangle background and white text "FUSED PRIOR MASK"
            text_mask = "FUSED PRIOR MASK"
            (text_width_m, text_height_m), baseline_m = cv2.getTextSize(text_mask, font, scale, thickness)
            cv2.rectangle(mask_overlay, (0, 0), (text_width_m + 10, text_height_m + 10), (0, 0, 0), thickness=-1)
            cv2.putText(mask_overlay, text_mask, (5, text_height_m + 5), font, scale, (255, 255, 255), thickness)

            cv2.namedWindow("RDW - Fused Prior Mask", cv2.WINDOW_NORMAL)
            cv2.namedWindow("RDW - Classical (fusion)", cv2.WINDOW_NORMAL)
            cv2.moveWindow("RDW - Fused Prior Mask", 100, 100)
            cv2.moveWindow("RDW - Classical (fusion)", 800, 100)

            cv2.imshow("RDW - Fused Prior Mask", mask_overlay)
            cv2.imshow("RDW - Classical (fusion)", vis)

            # Show debug images if present
            if "hsv_mask" in result:
                cv2.namedWindow("RDW - HSV Mask", cv2.WINDOW_NORMAL)
                cv2.moveWindow("RDW - HSV Mask", 100, 500)
                # Draw black rectangle background and white text "HSV MASK"
                hsv_mask = result["hsv_mask"].copy()
                text_hsv = "HSV MASK"
                (tw, th), baseline = cv2.getTextSize(text_hsv, font, scale, thickness)
                cv2.rectangle(hsv_mask, (0, 0), (tw + 10, th + 10), (0, 0, 0), thickness=-1)
                cv2.putText(hsv_mask, text_hsv, (5, th + 5), font, scale, (255, 255, 255), thickness)
                cv2.imshow("RDW - HSV Mask", hsv_mask)
            if "edges" in result:
                cv2.namedWindow("RDW - Edges", cv2.WINDOW_NORMAL)
                cv2.moveWindow("RDW - Edges", 400, 500)
                edges_img = result["edges"].copy()
                text_edges = "CANNY EDGES"
                (tw, th), baseline = cv2.getTextSize(text_edges, font, scale, thickness)
                cv2.rectangle(edges_img, (0, 0), (tw + 10, th + 10), (0, 0, 0), thickness=-1)
                cv2.putText(edges_img, text_edges, (5, th + 5), font, scale, (255, 255, 255), thickness)
                cv2.imshow("RDW - Edges", edges_img)
            if "edges_masked" in result:
                cv2.namedWindow("RDW - Masked Edges", cv2.WINDOW_NORMAL)
                cv2.moveWindow("RDW - Masked Edges", 700, 500)
                edges_masked_img = result["edges_masked"].copy()
                text_masked = "MASKED EDGES"
                (tw, th), baseline = cv2.getTextSize(text_masked, font, scale, thickness)
                cv2.rectangle(edges_masked_img, (0, 0), (tw + 10, th + 10), (0, 0, 0), thickness=-1)
                cv2.putText(edges_masked_img, text_masked, (5, th + 5), font, scale, (255, 255, 255), thickness)
                cv2.imshow("RDW - Masked Edges", edges_masked_img)
            if "hough_vis" in result:
                cv2.namedWindow("RDW - Hough Lines", cv2.WINDOW_NORMAL)
                cv2.moveWindow("RDW - Hough Lines", 1000, 500)
                hough_vis_img = result["hough_vis"].copy()
                text_hough = "HOUGH LINES"
                (tw, th), baseline = cv2.getTextSize(text_hough, font, scale, thickness)
                cv2.rectangle(hough_vis_img, (0, 0), (tw + 10, th + 10), (0, 0, 0), thickness=-1)
                cv2.putText(hough_vis_img, text_hough, (5, th + 5), font, scale, (255, 255, 255), thickness)
                cv2.imshow("RDW - Hough Lines", hough_vis_img)

            wait = 0 if getattr(cap, "single_image", False) else 1
            if (cv2.waitKey(wait) & 0xFF) == 27:
                break

        frames += 1
        last_vis = vis

        print(f"{source} | x={x_m:+.3f} | v={v_mps:+.3f} | alpha={np.degrees(alpha):+.2f}°")

    cap.release()
    if display and last_vis is not None:
        if not getattr(cap, "single_image", False):
            cv2.imshow("RDW - Classical (fusion) - last", last_vis)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frames