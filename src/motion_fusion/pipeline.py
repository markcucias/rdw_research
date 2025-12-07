from __future__ import annotations

from typing import Dict, Any
import os

import cv2
import numpy as np

from .lanenet_runtime import LaneNetRuntime
from .odometry import VisualOdometry
from .filters import KalmanPoseFilter

def open_source(source: str):

    if source.startswith("camera:"):
        idx = int(source.split(":")[1])
        cap = cv2.VideoCapture(idx)
        cap.single_image = False
        return cap

    if os.path.isfile(source):
        img = cv2.imread(source)
        if img is not None:
            class SingleImageCapture:
                def __init__(self, img_):
                    self.img = img_
                    self.done = False
                    self.single_image = True

                def isOpened(self):
                    return True

                def read(self):
                    if self.done:
                        return False, None
                    self.done = True
                    return True, self.img

                def release(self):
                    pass

            return SingleImageCapture(img)

    cap = cv2.VideoCapture(source)
    cap.single_image = False
    return cap


# ---------- измерения по полосе ----------

def compute_lane_measurements(
    lane_prob: np.ndarray,
    bottom_ratio: float = 0.4,
    threshold: float = 0.5,
    min_points: int = 500,
) -> tuple[float, float, bool, np.ndarray]:

    h, w = lane_prob.shape
    y_start = int(h * (1.0 - bottom_ratio))
    roi = lane_prob[y_start:, :]
    mask = (roi >= threshold).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if xs.size < min_points:
        return 0.0, 0.0, False, mask

    xs_full = xs
    ys_full = ys + y_start

    x_mean = float(xs_full.mean())
    x_offset_pixels = x_mean - w / 2.0
    pts = np.column_stack((xs_full, ys_full)).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)

    theta_lane = float(np.arctan2(vx, vy))

    mask_vis = (mask * 255).astype(np.uint8)

    return x_offset_pixels, theta_lane, True, mask_vis



def run(source: str, cfg: Dict[str, Any], display: bool = False) -> int:
    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    lcfg = cfg.get("lanenet", {}) or {}
    ocfg = cfg.get("odometry", {}) or {}
    kcfg = (cfg.get("filters", {}) or {}).get("kalman", {}) or {}
    kin_cfg = cfg.get("kinematics", {}) or {}

    bottom_ratio = float(kin_cfg.get("lane_bottom_ratio", 0.4))
    meters_per_pixel = float(kin_cfg.get("meters_per_pixel_bottom", 0.01))
    fps = float(kin_cfg.get("fps", 30.0))
    dt = 1.0 / fps if fps > 0 else 0.0
    thr = float(lcfg.get("threshold", 0.5))

    lanenet = LaneNetRuntime(
        model_path=lcfg.get("model", ""),
        input_size=tuple(lcfg.get("input_size", [640, 640])),
        threshold=thr,
        enabled=bool(lcfg.get("enabled", True)),
        letterbox=bool(lcfg.get("letterbox", True)),
    )

    vo = VisualOdometry(ocfg)
    kf = KalmanPoseFilter(kcfg)

    frames = 0
    last_vis = None
    prev_pose = None
    prev_speed = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        lane_prob, meta = lanenet.infer(frame)
        odom = vo.update(frame)

        dx_m = odom.dx * meters_per_pixel
        dy_m = odom.dy * meters_per_pixel

        kf.predict(dx_m, dy_m, odom.dtheta)

        x_lane_pix, theta_lane, lane_ok, roi_mask_vis = compute_lane_measurements(
            lane_prob,
            bottom_ratio=bottom_ratio,
            threshold=thr,
        )

        if lane_ok:
            kf.update_lane(x_lane_pix * meters_per_pixel, theta_lane)

        pose = kf.pose()
        x_m, y_m, theta = pose.x, pose.y, pose.theta

        if prev_pose is not None and dt > 0:
            dx_global = x_m - prev_pose.x
            dy_global = y_m - prev_pose.y
            dist_m = float(np.hypot(dx_global, dy_global))
            v_m_s = dist_m / dt
        else:
            v_m_s = 0.0

        prev_pose = pose
        prev_speed = v_m_s

        alpha_deg = float(np.degrees(theta))
        print(
            f"[frame {frames:05d}]  x = {x_m:+.3f} m   "
            f"v = {v_m_s:6.3f} m/s   alpha = {alpha_deg:+6.2f} deg"
        )

        h, w = frame.shape[:2]

        mask_bin = (lane_prob >= thr).astype(np.uint8) * 255
        mask_rgb = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
        mask_rgb[:, :, 1] = 0

        overlay = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0)

        center = (w // 2, h - 20)
        length = 120
        end = (
            int(center[0] + length * np.sin(theta)),
            int(center[1] - length * np.cos(theta)),
        )
        cv2.arrowedLine(overlay, center, end, (0, 255, 0), 3, tipLength=0.15)


        vis_original = frame.copy()
        cv2.putText(
            vis_original,
            "ORIGINAL 1",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        lane_prob_vis = (lane_prob * 255).astype(np.uint8)
        vis_prob = cv2.cvtColor(lane_prob_vis, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            vis_prob,
            "LANE PROB 2",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        vis_bin = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            vis_bin,
            "LANE BINARY MASK 3",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        vis_roi = cv2.cvtColor(roi_mask_vis, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            vis_roi,
            "LANE BOTTOM ROI 4",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        vis_overlay = overlay.copy()
        cv2.putText(
            vis_overlay,
            "MOTION FUSION 5",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        vis_vo = None
        if odom.vis_matches is not None:
            vis_vo = odom.vis_matches.copy()
            cv2.putText(
                vis_vo,
                "VO MATCHES (ORB + RANSAC)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        if display:
            cv2.imshow("RDW - Motion Fusion (final)", vis_overlay)
            cv2.imshow("RDW - Original", vis_original)
            cv2.imshow("RDW - Lane Prob", vis_prob)
            cv2.imshow("RDW - Lane Binary", vis_bin)
            cv2.imshow("RDW - Lane Bottom ROI", vis_roi)
            if vis_vo is not None:
                cv2.imshow("RDW - VO Matches", vis_vo)

            wait = 0 if getattr(cap, "single_image", False) else 1
            if (cv2.waitKey(wait) & 0xFF) == 27:  # ESC
                break

        last_vis = overlay
        frames += 1

    cap.release()

    if display:
        if last_vis is not None and not getattr(cap, "single_image", False):
            cv2.imshow("Last frame", last_vis)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frames
