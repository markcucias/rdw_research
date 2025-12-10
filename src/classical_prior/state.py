from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

def _x_at(line: Tuple[int,int,int,int], y: int) -> int:
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1:  # vertical guard
        return x1
    t = (y - y1) / (y2 - y1 + 1e-9)
    return int(x1 + t * (x2 - x1))

def _centerline(left: Optional[Tuple[int,int,int,int]],
                right: Optional[Tuple[int,int,int,int]],
                img_h: int, img_w: int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    yb = img_h - 1
    yh = int(img_h * 0.6)
    if left is not None and right is not None:
        xb = int(0.5 * (_x_at(left, yb) + _x_at(right, yb)))
        xh = int(0.5 * (_x_at(left, yh) + _x_at(right, yh)))
        return (xb, yb), (xh, yh)
    # fallback: single side → assume lane width from config later (caller handles)
    raise ValueError("need at least one fitted line")

def estimate_pose(
    result: Dict[str, Any],
    img_shape: Tuple[int,int, int],
    meters_per_pixel_bottom: float,
    assumed_lane_width_m: Optional[float] = None
) -> Tuple[float, float]:
    """
    Return (x_m, alpha_rad) where:
      x_m        : lateral offset (camera center to lane center) at image bottom
      alpha_rad  : heading of lane centerline relative to camera vertical (right-handed)
    """
    h, w = img_shape[:2]
    left = result.get("left_line")
    right = result.get("right_line")

    # по умолчанию считаем, что измерение валидно
    result["valid_lane"] = True          # <<< NEW
    result["lane_width_px"] = None       # <<< NEW

    if left is None and right is None:
        # ничего не нашли — говорим, что линия невалидна
        result["valid_lane"] = False     # <<< NEW
        return 0.0, 0.0

    lane_width_px = None                 # <<< NEW

    try:
        (xb, yb), (xh, yh) = _centerline(left, right, h, w)
        # ширина полосы в пикселях по нижней строке кадра
        if left is not None and right is not None:      # <<< NEW
            x_left_bottom  = _x_at(left,  h - 1)        # <<< NEW
            x_right_bottom = _x_at(right, h - 1)        # <<< NEW
            lane_width_px  = abs(x_right_bottom - x_left_bottom)  # <<< NEW
    except ValueError:
        # one-sided fallback using assumed lane width
        if left is not None and assumed_lane_width_m is not None:
            xb_left = _x_at(left, h - 1)
            lane_px = int(assumed_lane_width_m / max(meters_per_pixel_bottom, 1e-9))
            xb = xb_left + lane_px // 2
            xh = _x_at(left, int(h * 0.6)) + lane_px // 2
            yb, yh = h - 1, int(h * 0.6)
            lane_width_px = float(lane_px)              # <<< NEW
        elif right is not None and assumed_lane_width_m is not None:
            xb_right = _x_at(right, h - 1)
            lane_px = int(assumed_lane_width_m / max(meters_per_pixel_bottom, 1e-9))
            xb = xb_right - lane_px // 2
            xh = _x_at(right, int(h * 0.6)) - lane_px // 2
            yb, yh = h - 1, int(h * 0.6)
            lane_width_px = float(lane_px)              # <<< NEW
        else:
            result["valid_lane"] = False                # <<< NEW
            return 0.0, 0.0

    # --- sanity-check ширины полосы ---                      <<< NEW BLOCK
    if assumed_lane_width_m is not None and meters_per_pixel_bottom > 0.0 and lane_width_px is not None:
        lane_width_px_expected = assumed_lane_width_m / meters_per_pixel_bottom
        min_lane_px = 0.6 * lane_width_px_expected
        max_lane_px = 1.4 * lane_width_px_expected

        if lane_width_px < min_lane_px or lane_width_px > max_lane_px:
            # просто помечаем, но НЕ обнуляем измерение
            result["valid_lane"] = False
            result["lane_width_px"] = lane_width_px
        else:
            result["valid_lane"] = True
            result["lane_width_px"] = lane_width_px
    # --- конец sanity-check блока ---                        <<< END NEW

    cx = w * 0.5
    dx_px = cx - xb                              # right is +, left is -
    x_m   = dx_px * meters_per_pixel_bottom

    # heading: angle of lane centerline relative to vertical axis
    vx = float(xh - xb)
    vy = float(yh - yb)  # negative (upward)
    alpha = np.arctan2(vx, -vy)                  # 0 = straight, + right yaw

    return float(x_m), float(alpha)

class SpeedEstimator:
    """
    Very light forward-speed proxy from optical flow on the road mask bottom band.
    v is reported in m/s given fps and meters_per_pixel_bottom.
    """
    def __init__(self, fps: float, meters_per_pixel_bottom: float, band_ratio: float = 0.25):
        self.fps = float(fps)
        self.mpp = float(meters_per_pixel_bottom)
        self.band_ratio = float(band_ratio)
        self.prev_gray = None

    def update(self, bgr: np.ndarray, road_mask: Optional[np.ndarray]) -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        h, w = gray.shape[:2]
        y0 = int(h * (1.0 - self.band_ratio))
        roi = slice(y0, h)

        # dense flow on ROI, then keep only road pixels
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray[roi], gray[roi],
                                            None, 0.5, 3, 21, 3, 5, 1.2, 0)
        fy = flow[..., 1]  # vertical flow (pixels/frame), forward ~ negative
        if road_mask is not None:
            mask_roi = (road_mask[roi] > 0)
            if mask_roi.any():
                fy = fy[mask_roi]

        med = float(np.median(-fy))              # pixels/frame, forward positive
        v_mps = med * self.mpp * self.fps

        self.prev_gray = gray
        return v_mps
