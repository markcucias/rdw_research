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
    Classical lane extractor with a *late* neural prior gate.

    Steps:
      1) Color threshold on the full frame (white + optional yellow) → binary map.
      2) Canny edges on that map.
      3) Apply the road mask *after* edges, with a small dilation margin.
      4) Hough segment detection; if nothing appears, relax the mask and finally try without it.
      5) Split segments into left/right (by where they hit the bottom of the frame) and
         fit one straight line per side for pose estimation.

    Returns:
      dict with:
        - "segments":       all Hough segments
        - "left_segments":  segments assigned to the left lane
        - "right_segments": segments assigned to the right lane
        - "left_line":      fitted line for left lane (or None)
        - "right_line":     fitted line for right lane (or None)
        - debug images: "hsv_mask", "edges", "edges_masked", "hough_vis"
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

    # Remove isolated dots, then connect broken lane pixels
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN,  np.ones((5, 5),  np.uint8))
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((7, 7),  np.uint8))

    # Remove tiny blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    min_area = 200  # tune: 100–400 depending on image size

    clean = np.zeros_like(binm)
    for i in range(1, num_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255
    binm = clean

    # hsv_mask debug image
    hsv_mask = binm.copy()
    text = "HSV MASK"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(hsv_mask, (0, 0), (tw + 10, th + 10), (0, 0, 0), thickness=-1)
    cv2.putText(hsv_mask, text, (5, th + 5), font, scale, (255, 255, 255), thickness)

    # (2) edges
    low  = int(canny_cfg.get("low", 80))
    high = int(canny_cfg.get("high", 160))
    edges = cv2.Canny(binm, low, high)

    # Remove small edge fragments
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    clean_edges = np.zeros_like(edges)
    min_edge_area = 150  # tune 100–300

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_edge_area:
            clean_edges[labels == i] = 255
    edges = clean_edges

    text = "CANNY EDGES"
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(edges, (0, 0), (tw + 10, th + 10), (0, 0, 0), thickness=-1)
    cv2.putText(edges, text, (5, th + 5), font, scale, (255, 255, 255), thickness)

    # Remove edges above horizon (~35% height)
    horizon = int(0.35 * h)
    edges[:horizon, :] = 0

    # (3) late mask gate with margin
    edges_masked = edges.copy()
    edges_masked = cv2.morphologyEx(edges_masked, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    if road_mask is not None:
        margin = int(hsv_cfg.get("mask_margin", 7))
        if margin > 0:
            k = np.ones((margin, margin), np.uint8)
            rm = cv2.dilate(road_mask, k, iterations=1)
        else:
            rm = road_mask
        edges_masked = cv2.bitwise_and(edges, edges, mask=rm)
        edges_masked[:horizon, :] = 0

    text = "MASKED EDGES"
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(edges_masked, (0, 0), (tw + 10, th + 10), (0, 0, 0), thickness=-1)
    cv2.putText(edges_masked, text, (5, th + 5), font, scale, (255, 255, 255), thickness)

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
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(hough_vis, (0, 0), (tw + 20, th + 20), (0, 0, 0), thickness=-1)
    cv2.putText(hough_vis, text, (5, th + 5), font, scale, (255, 255, 255), thickness)

    # (5) group and fit
    left_segs, right_segs = _split_left_right(segments, w, h)

    # First-pass fits (used only to score/clean segments)
    left_fit0  = _fit_line_from_segments(left_segs, h)
    right_fit0 = _fit_line_from_segments(right_segs, h)

    # --- anti-clutter filtering that does NOT kill turns ---
    tol_px = int(hsv_cfg.get("inlier_tol_px", 16))            # distance-to-fit tolerance
    max_keep = int(hsv_cfg.get("inlier_max_keep", 10))        # keep only top-N per side
    yb_strict = float(hsv_cfg.get("bottom_keep_ratio", 0.62)) # strict: keep segments whose bottom endpoint is low
    yb_relax  = float(hsv_cfg.get("bottom_relax_ratio", 0.48))# relax for turns if strict removes too much

    left_segs  = _keep_best_segments(left_segs,  left_fit0,  tol_px, max_keep, h, yb_strict, yb_relax)
    right_segs = _keep_best_segments(right_segs, right_fit0, tol_px, max_keep, h, yb_strict, yb_relax)

    # Final fits after cleaning
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
        for (x1, y1, x2, y2) in result.get("segments", []):
            cv2.line(out, (x1, y1), (x2, y2), (0, 200, 0), 2, cv2.LINE_AA)
        return out

    for (x1, y1, x2, y2) in left_segs:
        cv2.line(out, (x1, y1), (x2, y2), (255, 100, 0), 3, cv2.LINE_AA)

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
        if y1 > y2:
            xb, yb = x1, y1
        else:
            xb, yb = x2, y2

        if yb < img_h * 0.4:
            continue

        if xb < cx:
            left.append((x1, y1, x2, y2))
        else:
            right.append((x1, y1, x2, y2))

    return left, right


def _keep_best_segments(
    segs: List[Line],
    fit: FitLine,
    tol_px: int,
    max_keep: int,
    img_h: int,
    bottom_keep_ratio_strict: float,
    bottom_keep_ratio_relax: float,
) -> List[Line]:
    """
    Stops straight-road clutter:
      1) Prefer segments whose LOWER endpoint is near the bottom (anchored).
      2) Score by distance-to-fit and length.
      3) If strict anchoring removes too much (turns), relax it.
    """
    if not segs or fit is None:
        return segs

    def seg_bottom_y(s: Line) -> int:
        x1, y1, x2, y2 = s
        return y1 if y1 > y2 else y2

    # Precompute fit line distance helper
    x1, y1, x2, y2 = fit
    vx = float(x2 - x1)
    vy = float(y2 - y1)
    denom = (vx * vx + vy * vy) ** 0.5 + 1e-9

    def point_line_dist(px: float, py: float) -> float:
        return abs(vy * px - vx * py + x2 * y1 - y2 * x1) / denom

    def score_segments(candidates: List[Line]) -> List[Line]:
        scored = []
        for ax, ay, bx, by in candidates:
            dx = bx - ax
            dy = by - ay
            length = (dx * dx + dy * dy) ** 0.5

            # reject near-horizontal junk (helps a lot on straight dataset)
            if abs(dy) < 0.22 * abs(dx):
                continue

            d = 0.5 * (point_line_dist(ax, ay) + point_line_dist(bx, by))
            if d <= tol_px:
                scored.append((d, -length, (ax, ay, bx, by)))

        if not scored:
            return candidates

        scored.sort(key=lambda t: (t[0], t[1]))
        return [t[2] for t in scored[:max_keep]]

    y_strict = int(bottom_keep_ratio_strict * img_h)
    y_relax  = int(bottom_keep_ratio_relax * img_h)

    strict = [s for s in segs if seg_bottom_y(s) >= y_strict]
    kept = score_segments(strict)

    # If we lost too much (common on turns), relax bottom constraint
    if len(kept) < 2:
        relax = [s for s in segs if seg_bottom_y(s) >= y_relax]
        kept2 = score_segments(relax)
        if len(kept2) >= 2:
            return kept2

    # Final safety: if still too few, do no bottom anchoring but still score
    if len(kept) < 2:
        kept3 = score_segments(segs)
        return kept3 if len(kept3) >= 2 else segs

    return kept


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

    try:
        vx, vy, x0, y0 = cv2.fitLine(
            pts_np, cv2.DIST_L2, 0, 0.01, 0.01
        ).reshape(-1).tolist()
    except:
        xs = [p[0] for p in pts]
        x_mean = int(np.mean(xs))
        return (x_mean, img_h - 1, x_mean, int(img_h * 0.6))

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
        left_line  = result.get("left_line")
        right_line = result.get("right_line")

        if left_line and right_line:
            lx1, ly1, lx2, ly2 = left_line
            rx1, ry1, rx2, ry2 = right_line

            if lx1 > rx1:
                crossing = True

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