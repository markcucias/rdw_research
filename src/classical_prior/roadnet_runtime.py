from __future__ import annotations
import os
from typing import Tuple, List
import numpy as np
import cv2


class PriorRuntime:
    """
    YOLOP-based prior for drivable area (and lane-line) inferences.

    Design goals:
    - If the ONNX model is missing or disabled, keep the rest of the pipeline running.
    - Return probabilities in the original image size; caller decides thresholding/expansion.
    - Support letterbox resize to match training preprocessing and unmap back correctly.
    """

    def __init__(
        self,
        onnx_path: str | None,
        input_size: Tuple[int, int] = (640, 640),  # (W, H)
        threshold: float = 0.65,                   # default threshold used by infer_mask
        enabled: bool = True,
        letterbox: bool = True,
    ):
        self.onnx_path = onnx_path or ""
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.threshold = float(threshold)
        self.letterbox = bool(letterbox)

        self._session = None
        self._iname: str | None = None
        self._outnames: List[str] = []

        if enabled and os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
                self._iname = self._session.get_inputs()[0].name
                self._outnames = [o.name for o in self._session.get_outputs()]
            except Exception:
                self._session = None  # fall back quietly

    def infer_probs(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (prob_da, prob_ll) as float32 in [0,1], resized back to the original frame.
        prob_da is drivable-area; prob_ll is lane-line. If the model is unavailable,
        returns (ones, zeros) to keep downstream logic alive.
        """
        h, w = bgr.shape[:2]
        if self._session is None:
            return np.ones((h, w), np.float32), np.zeros((h, w), np.float32)

        iw, ih = self.input_size
        if self.letterbox:
            img_resized, meta = _letterbox_resize(bgr, (iw, ih))
        else:
            img_resized = cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)

        x = img_resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
        x = np.transpose(x, (2, 0, 1))[None, ...]               # NCHW

        outs = self._session.run(None, {self._iname: x})
        names_l = [n.lower() for n in self._outnames]

        # pick DA head and LL head by name if present; otherwise fall back to a reasonable guess
        da_idx = names_l.index("drive_area_seg") if "drive_area_seg" in names_l else _pick_output_index(self._outnames)
        ll_idx = names_l.index("lane_line_seg")  if "lane_line_seg"  in names_l else da_idx

        prob_da_small = _to_prob_map(outs[da_idx])
        prob_ll_small = _to_prob_map(outs[ll_idx])

        if self.letterbox:
            pa = _unletterbox_to_size(prob_da_small, meta, (w, h))
            pl = _unletterbox_to_size(prob_ll_small, meta, (w, h))
        else:
            pa = cv2.resize(prob_da_small, (w, h), interpolation=cv2.INTER_LINEAR)
            pl = cv2.resize(prob_ll_small, (w, h), interpolation=cv2.INTER_LINEAR)

        return pa.astype(np.float32), pl.astype(np.float32)

    def infer_mask(self, bgr: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper when a binary prior is enough.
        Thresholds the drivable-area probabilities at self.threshold
        and returns a uint8 mask (0/255) in original size.
        """
        prob_da, _ = self.infer_probs(bgr)
        mask = (prob_da >= self.threshold).astype(np.uint8) * 255
        return mask


def _to_prob_map(y: np.ndarray) -> np.ndarray:
    """
    Convert a raw network output to a single-channel probability map in [0,1].
    Accepts shapes (H,W) or (C,H,W) with C in {1,2}. Applies sigmoid if values look like logits.
    """
    arr = np.squeeze(y)
    if arr.ndim == 3 and arr.shape[0] in (1, 2):
        ch = 1 if arr.shape[0] == 2 else 0
        arr = arr[ch]
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _pick_output_index(outnames: List[str]) -> int:
    """
    Heuristic: choose a segmentation-like head by name, otherwise take the last output.
    """
    low = [n.lower() for n in outnames]
    for i, n in enumerate(low):
        if "da" in n or "drive" in n or "drivable" in n or "seg" in n:
            return i
    return len(outnames) - 1


def _letterbox_resize(img: np.ndarray, target_wh: Tuple[int, int]):
    """
    Letterbox-resize to target (W,H) while preserving aspect ratio.
    Returns (padded_image, meta) where meta carries pad and scale to invert later.
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

    meta = (sx, sy, scale, (w, h))  # pad_x, pad_y, scale, (orig_w, orig_h)
    return canvas, meta


def _unletterbox_to_size(prob_small: np.ndarray, meta, out_wh: Tuple[int, int]) -> np.ndarray:
    """
    Remove letterbox padding and resize a small probability map back to (out_w, out_h).
    """
    sx, sy, scale, (orig_w, orig_h) = meta
    nh = int(orig_h * scale)
    nw = int(orig_w * scale)

    # crop padding areas away (note: prob_small is HxW float map)
    cropped = prob_small[sy: sy + nh, sx: sx + nw]
    out_w, out_h = out_wh
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)