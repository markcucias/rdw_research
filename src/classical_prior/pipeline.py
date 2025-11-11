from __future__ import annotations
from typing import Dict, Any
import cv2
import numpy as np

from common.video import open_source
from .classical_ops import detect_lanes, draw_lanes
from .roadnet_runtime import PriorRuntime


def _vis_mask(frame, mask):
    import cv2, numpy as np
    # colorize mask (cyan) and alpha-blend on top of the frame
    color = np.zeros_like(frame)
    color[:, :, 1] = mask  # G
    color[:, :, 2] = mask  # R
    return cv2.addWeighted(frame, 0.65, color, 0.35, 0.0)



def run(source: str, cfg: dict, display: bool = False) -> int:
    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    prior_cfg = cfg.get("prior", {})
    prior = PriorRuntime(
        onnx_path=prior_cfg.get("model"),
        input_size=tuple(prior_cfg.get("input_size", [640, 640])),
        threshold=float(prior_cfg.get("threshold", 0.5)),
        enabled=bool(prior_cfg.get("enabled", True)),
    )

    hsv   = cfg.get("hsv", {})
    canny = cfg.get("canny", {})
    hough = cfg.get("hough", {})

    frames = 0
    last_vis = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        road_mask = prior.infer_mask(frame)

        if display:
            import cv2
            mask_preview = _vis_mask(frame, road_mask)
            cv2.imshow("RDW - Prior Mask", mask_preview)
            # single image → wait; sequences → short wait; Esc skips to end
            wait = 0 if getattr(cap, "single_image", False) else 1
            if cv2.waitKey(wait) & 0xFF == 27:
                break

        result = detect_lanes(frame, road_mask, hsv, canny, hough)
        vis = draw_lanes(frame.copy(), result)

        if display:
            cv2.imshow("RDW - Classical (fusion)", vis)
            wait = 0 if getattr(cap, "single_image", False) else 1
            if cv2.waitKey(wait) & 0xFF == 27:
                break

        frames += 1
        last_vis = vis

    cap.release()
    if display and last_vis is not None:
        if not getattr(cap, "single_image", False):
            cv2.imshow("RDW - Classical (fusion) - last", last_vis)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frames


