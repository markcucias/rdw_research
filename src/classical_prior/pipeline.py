from __future__ import annotations
from typing import Dict, Any
import cv2
import numpy as np

from common.video import open_source
from .classical_ops import detect_lanes, draw_lanes

def run(source: str, cfg: Dict[str, Any], display: bool = False) -> int:
    """
    Reads frames from SOURCE, runs classical_ops with a full-frame mask, and (optionally) displays.
    Returns number of processed frames.
    """
    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    frames = 0
    last_vis = None

    hsv   = cfg.get("hsv", {})
    canny = cfg.get("canny", {})
    hough = cfg.get("hough", {})

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # full-frame mask for now (no neural prior yet)
        road_mask = None

        result = detect_lanes(frame, road_mask, hsv, canny, hough)
        vis = draw_lanes(frame.copy(), result)

        if display:
            cv2.imshow("RDW - Classical (ops)", vis)
            wait = 0 if getattr(cap, "single_image", False) else 1
            if cv2.waitKey(wait) & 0xFF == 27:
                break

        frames += 1
        last_vis = vis

    cap.release()
    if display and last_vis is not None:
        if not getattr(cap, "single_image", False):
            cv2.imshow("RDW - Classical (last)", last_vis)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frames