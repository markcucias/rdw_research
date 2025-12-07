from __future__ import annotations

from typing import Tuple, Dict, Any, List
import os

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # if onnxruntime is not installed, we just disable the network
    ort = None


class LaneNetRuntime:
    """
    Runtime for YOLOP (or similar models) that outputs a lane-line probability map
    in the original frame size.

    The class is intentionally generic: it tries to find drivable-area and lane-line
    outputs by name; if that fails, it falls back to using the last two outputs.
    """

    def __init__(
        self,
        model_path: str | None,
        input_size: Tuple[int, int] = (640, 640),  # (W, H)
        threshold: float = 0.5,
        enabled: bool = True,
        letterbox: bool = True,
    ) -> None:
        self.onnx_path = model_path or ""
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.threshold = float(threshold)
        self.letterbox = bool(letterbox)

        self._session: "ort.InferenceSession | None" = None
        self._iname: str | None = None
        self._outnames: List[str] = []

        if enabled and ort is not None and os.path.exists(self.onnx_path):
            try:
                self._session = ort.InferenceSession(
                    self.onnx_path,
                    providers=["CPUExecutionProvider"],
                )
                self._iname = self._session.get_inputs()[0].name
                self._outnames = [o.name for o in self._session.get_outputs()]
            except Exception:
                # If something goes wrong, just disable the network
                self._session = None

    @property
    def enabled(self) -> bool:
        return self._session is not None

    def _infer_raw(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Internal method: returns (prob_da, prob_ll) as float32 in [0,1]
        resized back to the original frame.
        """
        h, w = bgr.shape[:2]
        if self._session is None:
            # If the model is unavailable, pretend everything is drivable and there are no lanes.
            return np.ones((h, w), np.float32), np.zeros((h, w), np.float32)

        iw, ih = self.input_size
        if self.letterbox:
            img_resized, meta = _letterbox_resize(bgr, (iw, ih))
        else:
            img_resized = cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
            meta = None

        x = img_resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
        x = np.transpose(x, (2, 0, 1))[None, ...]               # NCHW

        outs = self._session.run(None, {self._iname: x})
        names_l = [n.lower() for n in self._outnames]

        # Try to find output indices by name
        da_idx = None
        ll_idx = None
        for i, name in enumerate(names_l):
            if da_idx is None and ("da" in name or "drive" in name or "road" in name):
                da_idx = i
            if ll_idx is None and ("ll" in name or "lane" in name):
                ll_idx = i

        # Fallback: just use the last two outputs
        if da_idx is None or ll_idx is None:
            if len(outs) >= 2:
                da_idx, ll_idx = len(outs) - 2, len(outs) - 1
            else:
                return np.ones((h, w), np.float32), np.zeros((h, w), np.float32)

        prob_da_small = _to_prob_map(outs[da_idx])
        prob_ll_small = _to_prob_map(outs[ll_idx])

        if self.letterbox and meta is not None:
            pa = _unletterbox_to_size(prob_da_small, meta, (w, h))
            pl = _unletterbox_to_size(prob_ll_small, meta, (w, h))
        else:
            pa = cv2.resize(prob_da_small, (w, h), interpolation=cv2.INTER_LINEAR)
            pl = cv2.resize(prob_ll_small, (w, h), interpolation=cv2.INTER_LINEAR)

        return pa.astype(np.float32), pl.astype(np.float32)

    def infer(self, bgr: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Public method used by the pipeline.

        Returns:
          lane_prob : HxW float32 [0,1] — lane-line probability
          meta      : dict with a downscaled probability map for debug display
        """
        h, w = bgr.shape[:2]
        prob_da, prob_ll = self._infer_raw(bgr)

        lane_prob = prob_ll if prob_ll is not None else prob_da
        lane_prob = lane_prob.astype(np.float32)

        small_w, small_h = max(1, w // 4), max(1, h // 4)
        prob_small = cv2.resize(lane_prob, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        meta = {
            "enabled": self.enabled,
            "prob_small": prob_small,
        }
        return lane_prob, meta


def _to_prob_map(y: np.ndarray) -> np.ndarray:
    """
    Convert a raw network output to a single-channel probability map in [0,1].

    Supports typical shapes:
      - (H, W)
      - (1, H, W)
      - (2, H, W)  -> argmax over channels
      - (1, 1, H, W)
      - (1, 2, H, W) -> argmax over channels
    """
    arr = np.squeeze(y)
    if arr.ndim == 3:
        # (C, H, W)
        if arr.shape[0] == 2:
            arr = np.argmax(arr, axis=0).astype(np.float32)
        else:
            arr = arr[0, ...].astype(np.float32)

    arr = arr.astype(np.float32)

    # If values look like logits (large magnitude), apply sigmoid
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    if max_abs > 6.0:
        arr = 1.0 / (1.0 + np.exp(-arr))

    # Normalize to [0,1] if out of range
    mn, mx = float(arr.min()), float(arr.max())
    if mx > 1.0 or mn < 0.0:
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr[:] = 0.0

    return np.clip(arr, 0.0, 1.0)


def _letterbox_resize(img: np.ndarray, target_wh: Tuple[int, int]):
    """
    Resize with preserved aspect ratio and zero padding.

    Returns (padded_image, meta) where meta contains enough info to
    later unpad/resize back to the original resolution.
    """
    tw, th = target_wh
    h, w = img.shape[:2]

    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=img.dtype)

    sx = (tw - nw) // 2
    sy = (th - nh) // 2
    canvas[sy:sy + nh, sx: sx + nw] = resized

    meta = (sx, sy, scale, (w, h))
    return canvas, meta


def _unletterbox_to_size(prob_small: np.ndarray, meta, out_wh: Tuple[int, int]) -> np.ndarray:
    """
    Remove letterbox padding and resize a small probability map back to (out_w, out_h).
    """
    sx, sy, scale, (orig_w, orig_h) = meta
    nh = int(orig_h * scale)
    nw = int(orig_w * scale)

    cropped = prob_small[sy: sy + nh, sx: sx + nw]
    out_w, out_h = out_wh
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
