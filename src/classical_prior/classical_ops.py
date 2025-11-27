from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

# Types for clarity
Line = Tuple[int, int, int, int]     # (x1, y1, x2, y2)
FitLine = Optional[Line]


def detect_lanes(
    bgr: np.ndarray,
    road_mask: Optional[np.ndarray],
    hsv_cfg: Dict[str, Any],
    canny_cfg: Dict[str, Any],
    hough_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classical lane extractor with a *late* neural prior gate.

    Steps:
      1) Color threshold on the full frame (white + optional yellow) → binary map.
      2) Canny edges on that map.
      3) Apply the road mask *after* edges, with a small dilation margin.
      4) Hough segment detection; if nothing appears, relax the mask and finally try without it.
      5) Split segments into left/right and fit one line per side.

    Returns:
      dict with raw "segments" and fitted "left_line"/"right_line" (or None),
      plus debug images: "hsv_mask", "edges", "edges_masked", "hough_vis".
    """
    h, w = bgr.shape[:2]

    # (1) white + yellow thresholding
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white_lower = np.array(hsv_cfg.get("white_lower", [0, 0, 200]), dtype=np.uint8)
    white_upper = np.array(hsv_cfg.get("white_upper", [180, 40, 255]), dtype=np.uint8)
    bin_white = cv2.inRange(hsv, white_lower, white_upper)

    yellow_lower = np.array(hsv_cfg.get("yellow_lower", [15, 60, 120]), dtype=np.uint8)
    yellow_upper = np.array(hsv_cfg.get("yellow_upper", [40, 255, 255]), dtype=np.uint8)
    bin_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    binm = cv2.bitwise_or(bin_white, bin_yellow)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Save debug hsv_mask
    hsv_mask = binm.copy()
    text = "HSV MASK"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(hsv_mask, (0, 0), (text_width + 10, text_height + 10), (0, 0, 0), thickness=-1)
    cv2.putText(hsv_mask, text, (5, text_height + 5), font, scale, (255, 255, 255), thickness)

    # (2) edges
    low = int(canny_cfg.get("low", 80))
    high = int(canny_cfg.get("high", 160))
    edges = cv2.Canny(binm, low, high)
    text = "CANNY EDGES"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(edges, (0, 0), (text_width + 10, text_height + 10), (0, 0, 0), thickness=-1)
    cv2.putText(edges, text, (5, text_height + 5), font, scale, (255, 255, 255), thickness)

    # (3) late mask gate with margin
    edges_masked = edges
    if road_mask is not None:
        margin = int(hsv_cfg.get("mask_margin", 7))
        if margin > 0:
            k = np.ones((margin, margin), np.uint8)
            rm = cv2.dilate(road_mask, k, iterations=1)
        else:
            rm = road_mask
        edges_masked = cv2.bitwise_and(edges, edges, mask=rm)
    text = "MASKED EDGES"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(edges_masked, (0, 0), (text_width + 10, text_height + 10), (0, 0, 0), thickness=-1)
    cv2.putText(edges_masked, text, (5, text_height + 5), font, scale, (255, 255, 255), thickness)

    # (4) Hough with fallbacks
    rho       = float(hough_cfg.get("rho", 1.0))
    theta_deg = float(hough_cfg.get("theta_deg", 1.0))
    thr       = int(hough_cfg.get("threshold", 30))
    min_len   = int(hough_cfg.get("min_len", 40))
    max_gap   = int(hough_cfg.get("max_gap", 20))

    def hough_run(edge_img: np.ndarray) -> List[Line]:
        lines = cv2.HoughLinesP(
            edge_img,
            rho,
            np.deg2rad(theta_deg),
            thr,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )
        return [] if lines is None else [tuple(map(int, l)) for l in lines[:, 0, :]]

    segments = hough_run(edges_masked)

    if not segments and road_mask is not None:
        size = max(11, 2 * int(hsv_cfg.get("mask_margin", 7)))
        rm2 = cv2.dilate(road_mask, np.ones((size, size), np.uint8), iterations=1)
        segments = hough_run(cv2.bitwise_and(edges, edges, mask=rm2))

    if not segments and road_mask is not None:
        segments = hough_run(edges)

    # (4b) create hough_vis image
    hough_vis = bgr.copy()
    for (x1, y1, x2, y2) in segments:
        cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    text = "HOUGH LINES"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(hough_vis, (0, 0), (text_width + 10, text_height + 10), (0, 0, 0), thickness=-1)
    cv2.putText(hough_vis, text, (5, text_height + 5), font, scale, (255, 255, 255), thickness)

    # (5) group and fit
    left_segs, right_segs = _split_left_right(segments, w, h)
    left_fit = _fit_line_from_segments(left_segs, h)
    right_fit = _fit_line_from_segments(right_segs, h)

    return {
        "segments": segments,
        "left_line": left_fit,
        "right_line": right_fit,
        "hsv_mask": hsv_mask,
        "edges": edges,
        "edges_masked": edges_masked,
        "hough_vis": hough_vis,
    }


def draw_lanes(img: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """
    Overlay raw segments (thin green) and the fitted left/right lanes (thick colored).
    """
    out = img
    for (x1, y1, x2, y2) in result.get("segments", []):
        cv2.line(out, (x1, y1), (x2, y2), (0, 200, 0), 1, cv2.LINE_AA)

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
    Separate segments by slope sign and horizontal location.
    Negative slope left of center → left; positive slope right of center → right.
    Reject nearly horizontal lines.
    """
    left, right = [], []
    cx = img_w * 0.5

    for x1, y1, x2, y2 in segments:
        dx = (x2 - x1)
        m = 1e9 if abs(dx) < 1 else (y2 - y1) / dx
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
    Fit a single robust line (cv2.fitLine) through segment endpoints and
    return it as a displayable segment from the image bottom to ~60% height.
    """
    if not segments:
        return None

    pts = []
    for x1, y1, x2, y2 in segments:
        pts.append([x1, y1]); pts.append([x2, y2])
    pts = np.asarray(pts, dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1).tolist()

    yb = img_h - 1
    yh = int(img_h * 0.6)

    def x_at(y: float) -> int:
        t = (y - y0) / (vy + 1e-9)
        return int(x0 + t * vx)

    return (x_at(yb), yb, x_at(yh), yh)