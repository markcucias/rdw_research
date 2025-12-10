from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

# Types for clarity
Line    = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
FitLine = Optional[Line]


def detect_lanes(
    bgr: np.ndarray,
    road_mask: Optional[np.ndarray],
    hsv_cfg: Dict[str, Any],
    canny_cfg: Dict[str, Any],
    hough_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classical lane extractor with optional YOLOP road-mask.
    Fixed so that classical-only mode works (no mask = no edge deletion).
    """
    h, w = bgr.shape[:2]

    # ---------------------------------------------------------
    # (1) WHITE + YELLOW HSV THRESHOLDING
    # ---------------------------------------------------------
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    white_lower = np.array(hsv_cfg.get("white_lower", [0, 0, 200]), dtype=np.uint8)
    white_upper = np.array(hsv_cfg.get("white_upper", [180, 40, 255]), dtype=np.uint8)
    bin_white = cv2.inRange(hsv, white_lower, white_upper)

    yellow_lower = np.array(hsv_cfg.get("yellow_lower", [15, 60, 120]), dtype=np.uint8)
    yellow_upper = np.array(hsv_cfg.get("yellow_upper", [40, 255, 255]), dtype=np.uint8)
    bin_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    binm = cv2.bitwise_or(bin_white, bin_yellow)

    # clean small blobs
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    clean = np.zeros_like(binm)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 200:
            clean[labels == i] = 255
    binm = clean

    # Debug HSV mask
    hsv_mask = binm.copy()
    cv2.rectangle(hsv_mask, (0, 0), (130, 30), (0, 0, 0), -1)
    cv2.putText(hsv_mask, "HSV MASK", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    # ---------------------------------------------------------
    # (2) CANNY EDGES
    # ---------------------------------------------------------
    low  = int(canny_cfg.get("low", 80))
    high = int(canny_cfg.get("high", 160))
    edges = cv2.Canny(binm, low, high)

    # Remove very small edge fragments
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
    clean_edges = np.zeros_like(edges)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 150:
            clean_edges[labels == i] = 255
    edges = clean_edges

    # annotate
    cv2.rectangle(edges, (0, 0), (160, 30), (0, 0, 0), -1)
    cv2.putText(edges, "CANNY EDGES", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    # remove sky
    horizon = int(0.35 * h)
    edges[:horizon, :] = 0

    # ---------------------------------------------------------
    # (3) MASKED EDGES — FIXED FOR CLASSICAL-ONLY MODE
    # ---------------------------------------------------------

    if road_mask is not None:
        margin = int(hsv_cfg.get("mask_margin", 7))
        if margin > 0:
            rm = cv2.dilate(road_mask, np.ones((margin, margin), np.uint8), 1)
        else:
            rm = road_mask

        edges_masked = cv2.bitwise_and(edges, edges, mask=rm)
        edges_masked[:horizon, :] = 0
    else:
        # *** FIX: previously edges were destroyed by a 7×7 OPEN ***
        # For classical-only, keep edges as-is (optional small clean).
        edges_masked = edges.copy()
        # optional gentle open:
        # edges_masked = cv2.morphologyEx(edges_masked, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    # label
    cv2.rectangle(edges_masked, (0, 0), (180, 30), (0, 0, 0), -1)
    cv2.putText(edges_masked, "MASKED EDGES", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    # ---------------------------------------------------------
    # (4) HOUGH LINES
    # ---------------------------------------------------------
    rho       = float(hough_cfg.get("rho", 1.0))
    theta_deg = float(hough_cfg.get("theta_deg", 1.0))
    thr       = int(hough_cfg.get("threshold", 30))
    min_len   = int(hough_cfg.get("min_len", 40))
    max_gap   = int(hough_cfg.get("max_gap", 20))

    def hough_run(img):
        lines = cv2.HoughLinesP(
            img,
            rho,
            np.deg2rad(theta_deg),
            thr,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )
        return [] if lines is None else [tuple(map(int, l)) for l in lines[:, 0, :]]

    segments = hough_run(edges_masked)

    # fallback if no segments and mask present
    if not segments and road_mask is not None:
        rm2 = cv2.dilate(road_mask, np.ones((15, 15), np.uint8), 1)
        segments = hough_run(cv2.bitwise_and(edges, edges, mask=rm2))

    # fallback: no mask at all
    if not segments:
        segments = hough_run(edges)

    # Draw Hough segments for debug
    hough_vis = bgr.copy()
    for (x1, y1, x2, y2) in segments:
        cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(hough_vis, (0, 0), (160, 30), (0, 0, 0), -1)
    cv2.putText(hough_vis, "HOUGH LINES", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    # ---------------------------------------------------------
    # (5) GROUP LEFT/RIGHT + FIT LINES
    # ---------------------------------------------------------
    left_segs, right_segs = _split_left_right(segments, w, h)
    left_fit  = _fit_line_from_segments(left_segs, h)
    right_fit = _fit_line_from_segments(right_segs, h)

    return {
        "segments":       segments,
        "left_segments":  left_segs,
        "right_segments": right_segs,
        "left_line":      left_fit,
        "right_line":     right_fit,
        "hsv_mask":       hsv_mask,
        "edges":          edges,
        "edges_masked":   edges_masked,
        "hough_vis":      hough_vis,
    }



def draw_lanes(img: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """
    Visualize lanes.

    - Left lane:  draw Hough segments in one color (e.g. blue).
    - Right lane: draw Hough segments in another color (e.g. orange).
    - If left/right segment groups are missing, fall back to all segments in green.

    Fitted lines (left_line/right_line) are still computed for pose and robustness,
    but we do *not* draw them anymore to avoid weird extrapolated lines.
    """
    out = img.copy()

    left_segs  = result.get("left_segments")
    right_segs = result.get("right_segments")

    if left_segs is None or right_segs is None:
        # Fallback: just draw all segments in green
        for (x1, y1, x2, y2) in result.get("segments", []):
            cv2.line(out, (x1, y1), (x2, y2), (0, 200, 0), 2, cv2.LINE_AA)
        return out

    # Left lane segments (blue-ish)
    for (x1, y1, x2, y2) in left_segs:
        cv2.line(out, (x1, y1), (x2, y2), (255, 100, 0), 3, cv2.LINE_AA)

    # Right lane segments (green/orange-ish)
    for (x1, y1, x2, y2) in right_segs:
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 100), 3, cv2.LINE_AA)

    return out


# ---------- helpers ----------

def _split_left_right(
    segments: List[Line],
    img_w: int,
    img_h: int,
) -> Tuple[List[Line], List[Line]]:
    """
    Separate segments into left/right by where they hit the *bottom* of the image.

    For each segment, take the endpoint with the larger y (closer to the bottom).
    If its x < center → left lane; else → right lane.

    This is much more stable on curves than looking at slope sign.
    """
    left: List[Line] = []
    right: List[Line] = []
    cx = img_w * 0.5

    for x1, y1, x2, y2 in segments:
        # endpoint closest to the bottom
        if y1 > y2:
            xb, yb = x1, y1
        else:
            xb, yb = x2, y2

        # ignore segments that live too high (safety)
        if yb < img_h * 0.4:
            continue

        if xb < cx:
            left.append((x1, y1, x2, y2))
        else:
            right.append((x1, y1, x2, y2))

    return left, right


    

def _fit_line_from_segments(segments: List[Line], img_h: int) -> FitLine:
    """
    Robust line fit using *all* segment endpoints.
    Always returns a line if segments are present.
    """
    if not segments:
        return None

    pts = []
    for x1, y1, x2, y2 in segments:
        pts.append([x1, y1])
        pts.append([x2, y2])

    pts_np = np.asarray(pts, dtype=np.float32)

    # Try a robust cv2.fitLine
    try:
        vx, vy, x0, y0 = cv2.fitLine(
            pts_np, cv2.DIST_L2, 0, 0.01, 0.01
        ).reshape(-1).tolist()
    except:
        # Fallback: average segment
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_mean = int(np.mean(xs))
        # draw vertical line as fallback
        return (x_mean, img_h - 1, x_mean, int(img_h * 0.6))

    # Extrapolate line to bottom and 60% height
    yb = img_h - 1
    yh = int(img_h * 0.6)

    def x_at(y):
        t = (y - y0) / (vy + 1e-9)
        return int(x0 + t * vx)

    return (x_at(yb), yb, x_at(yh), yh)




def evaluate_detection_quality(result: Dict[str, Any], img_w: int, img_h: int) -> Dict[str, bool]:
    """
    A detection is valid if:
      - left_segments exist
      - right_segments exist
      - left lane is left of right lane
    """
    left_segs  = result.get("left_segments", [])
    right_segs = result.get("right_segments", [])

    valid_left  = len(left_segs)  > 0
    valid_right = len(right_segs) > 0

    crossing = False

    if valid_left and valid_right:
        # Use fitted lines
        left_line  = result.get("left_line")
        right_line = result.get("right_line")

        if left_line and right_line:
            lx1, ly1, lx2, ly2 = left_line
            rx1, ry1, rx2, ry2 = right_line

            # Check bottom ordering: left must be to the left
            if lx1 > rx1:
                crossing = True

            # Check mid-height
            mid_y = img_h // 2

            def x_at(line, y):
                x1, y1, x2, y2 = line
                t = (y - y1) / (y2 - y1 + 1e-9)
                return x1 + t * (x2 - x1)

            if x_at(left_line, mid_y) > x_at(right_line, mid_y):
                crossing = True

    valid_frame = valid_left and valid_right and not crossing

    return {
        "valid_left":  valid_left,
        "valid_right": valid_right,
        "crossing":    crossing,
        "valid_frame": valid_frame,
    }