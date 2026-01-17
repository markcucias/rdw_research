from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2


def _x_at(line: Tuple[int, int, int, int], y: int) -> int:
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1:  # почти вертикальная
        return x1
    t = (y - y1) / (y2 - y1 + 1e-9)
    return int(x1 + t * (x2 - x1))


def _centerline(
    left: Optional[Tuple[int, int, int, int]],
    right: Optional[Tuple[int, int, int, int]],
    img_h: int,
    img_w: int,
    y_ref_ratio: Optional[float] = None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Возвращает две точки центр-линии:
      (xb, yb) – ближе к камере
      (xh, yh) – чуть выше
    Если y_ref_ratio задан, работаем вокруг этой высоты.
    Если нет – старое поведение: низ кадра + 0.6*h.
    """
    if y_ref_ratio is None:
        yb = img_h - 1
        yh = int(img_h * 0.6)
    else:
        # "нижняя" точка – там, где мы хотим измерять x
        yb = int(img_h * y_ref_ratio)
        # вторая точка – немного выше (10% высоты), но не вылезаем за кадр
        dy = int(img_h * 0.1)
        yh = max(0, yb - dy)

    if left is not None and right is not None:
        xb = int(0.5 * (_x_at(left, yb) + _x_at(right, yb)))
        xh = int(0.5 * (_x_at(left, yh) + _x_at(right, yh)))
        return (xb, yb), (xh, yh)

    # если только одна линия – fallback обрабатывается в estimate_pose
    raise ValueError("need at least one fitted line")


def estimate_pose(
    result: Dict[str, Any],
    img_shape: Tuple[int, int, int],
    meters_per_pixel_bottom: float,
    assumed_lane_width_m: Optional[float] = None,
    y_ref_ratio: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Возвращает (x_m, alpha_rad), где:
      x_m       : боковое смещение центра полосы относительно центра кадра
                  на высоте y_ref (если задан) или внизу кадра (по-старому)
      alpha_rad : угол центр-линии относительно вертикали (0 = прямо, + вправо)
    """
    h, w = img_shape[:2]
    left = result.get("left_line")
    right = result.get("right_line")

    if left is None and right is None:
        return 0.0, 0.0

    # пробуем использовать обе линии
    try:
        (xb, yb), (xh, yh) = _centerline(left, right, h, w, y_ref_ratio=y_ref_ratio)
    except ValueError:
        # fallback: только одна линия + предполагаемая ширина полосы
        if assumed_lane_width_m is None or meters_per_pixel_bottom <= 0:
            return 0.0, 0.0

        lane_px = int(assumed_lane_width_m / max(meters_per_pixel_bottom, 1e-9))

        # выбираем референсную высоту
        if y_ref_ratio is None:
            yb = h - 1
            yh = int(h * 0.6)
        else:
            yb = int(h * y_ref_ratio)
            dy = int(h * 0.1)
            yh = max(0, yb - dy)

        if left is not None:
            xb_left = _x_at(left, yb)
            xb = xb_left + lane_px // 2
            xh = _x_at(left, yh) + lane_px // 2
        elif right is not None:
            xb_right = _x_at(right, yb)
            xb = xb_right - lane_px // 2
            xh = _x_at(right, yh) - lane_px // 2
        else:
            return 0.0, 0.0

    # центр камеры
    cx = w * 0.5
    dx_px = cx - xb           # вправо +, влево -
    x_m = dx_px * meters_per_pixel_bottom

    # направление центр-линии
    vx = float(xh - xb)
    vy = float(yh - yb)       # вверх отрицательное
    alpha = np.arctan2(vx, -vy)

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
