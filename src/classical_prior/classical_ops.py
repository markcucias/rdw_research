from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

# Types we need for the return values
Line = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
FitLine = Optional[Line]


def detect_lanes(
    bgr: np.ndarray,
    road_mask: Optional[np.ndarray],
    hsv_cfg: Dict[str, Any],
    canny_cfg: Dict[str, Any],
    hough_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Minimal classical stack:
      1) Apply drivable-area mask (not yet used; full frame)
      2) HSV threshold for white paint
      3) Canny edges
      4) HoughLinesP -> raw segments
      5) Split into left/right & fit one line per side (optional but helps stability)

    Returns:
      {
        "segments": List[Line],        # raw Hough segments
        "left_line": FitLine,          # fitted (x1,y1,x2,y2) or None
        "right_line": FitLine
      }
    """
    h, w = bgr.shape[:2]

    # 1) Apply road mask if provided (otherwise use full frame)
    if road_mask is not None:
        masked = cv2.bitwise_and(bgr, bgr, mask=road_mask)
    else:
        masked = bgr

    # 2) HSV threshold (white lanes as a starting point)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_cfg.get("white_lower", [0, 0, 200]), dtype=np.uint8)
    upper = np.array(hsv_cfg.get("white_upper", [180, 40, 255]), dtype=np.uint8)
    binm = cv2.inRange(hsv, lower, upper)

    # small clean-up to remove speckles
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 3) Canny edges
    low = int(canny_cfg.get("low", 80))
    high = int(canny_cfg.get("high", 160))
    edges = cv2.Canny(binm, low, high)

    # 4) Probabilistic Hough transform
    rho = float(hough_cfg.get("rho", 1.0))
    theta_deg = float(hough_cfg.get("theta_deg", 1.0))
    threshold = int(hough_cfg.get("threshold", 30))
    min_len = int(hough_cfg.get("min_len", 40))
    max_gap = int(hough_cfg.get("max_gap", 20))

    lines = cv2.HoughLinesP(
        edges,
        rho,
        np.deg2rad(theta_deg),
        threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    segments: List[Line] = [] if lines is None else [tuple(map(int, l)) for l in lines[:, 0, :]]

    # 5) Split and fit
    left_segs, right_segs = _split_left_right(segments, w, h)
    left_fit = _fit_line_from_segments(left_segs, h)
    right_fit = _fit_line_from_segments(right_segs, h)

    return {
        "segments": segments,
        "left_line": left_fit,
        "right_line": right_fit,
    }


def draw_lanes(img: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """
    Draw raw segments (thin) and fitted left/right (thick).
    """
    out = img
    # raw segments in green
    for (x1, y1, x2, y2) in result.get("segments", []):
        cv2.line(out, (x1, y1), (x2, y2), (0, 200, 0), 1, cv2.LINE_AA)

    # fitted lines in blue/red
    if result.get("left_line") is not None:
        x1, y1, x2, y2 = result["left_line"]
        cv2.line(out, (x1, y1), (x2, y2), (255, 100, 0), 3, cv2.LINE_AA)
    if result.get("right_line") is not None:
        x1, y1, x2, y2 = result["right_line"]
        cv2.line(out, (x1, y1), (x2, y2), (0, 100, 255), 3, cv2.LINE_AA)

    return out


# ---------- helpers ----------

def _split_left_right(segments: List[Line], img_w: int, img_h: int) -> Tuple[List[Line], List[Line]]:
    """
    Heuristic: keep reasonably sloped lines; classify by slope sign and x-position.
    - left lane: negative slope, lives left of image center
    - right lane: positive slope, lives right of image center
    """
    left, right = [], []
    cx = img_w * 0.5

    for x1, y1, x2, y2 in segments:
        dx = (x2 - x1)
        if abs(dx) < 1:
            m = 1e9
        else:
            m = (y2 - y1) / dx

        if abs(m) < 0.3:
            continue

        xm = 0.5 * (x1 + x2)
        if m < 0 and xm < cx:
            left.append((x1, y1, x2, y2))
        elif m > 0 and xm > cx:
            right.append((x1, y1, x2, y2))

    return left, right


def _fit_line_from_segments(segments: List[Line], img_h: int) -> FitLine:
    """
    Fit one robust line to a set of segments using cv2.fitLine.
    Return a segment spanning from bottom of image to a horizon line (60% height).
    """
    if not segments:
        return None

    pts = []
    for x1, y1, x2, y2 in segments:
        pts.append([x1, y1])
        pts.append([x2, y2])
    pts = np.asarray(pts, dtype=np.float32)

    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy, x0, y0 = [float(v) for v in line]

    # get x at two y-levels (bottom and a "horizon")
    yb = img_h - 1
    yh = int(img_h * 0.6)

    def x_at(y):
        t = (y - y0) / (vy + 1e-9)
        return int(x0 + t * vx)

    return (x_at(yb), yb, x_at(yh), yh)