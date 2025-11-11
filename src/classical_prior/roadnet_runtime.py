from __future__ import annotations
import os
import cv2
import numpy as np

class PriorRuntime:
    """
    Returns a 0/255 drivable-area mask. If model missing or disabled, returns full-frame mask.
    """

    def __init__(self, onnx_path: str | None, input_size: tuple[int, int], threshold: float, enabled: bool):
        self.enabled = enabled and onnx_path and os.path.exists(onnx_path)
        self.input_size = input_size
        self.threshold = float(threshold)
        self._session = None
        self._iname = None
        self._onames = None

        if self.enabled:
            import onnxruntime as ort
            self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._iname = self._session.get_inputs()[0].name
            # YOLOP typically has multiple outputs; DA is usually the last or named accordingly.
            self._onames = [o.name for o in self._session.get_outputs()]

    def infer_mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if not self.enabled or self._session is None:
            return np.full((h, w), 255, dtype=np.uint8)

        ih, iw = self.input_size
        img = cv2.resize(bgr, (iw, ih))
        x = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]
        x = np.transpose(x, (2, 0, 1))[None, ...]       # NCHW

        outs = self._session.run(None, {self._iname: x})  # get all outputs once
        # Pick the segmentation-like output: smallest #channels map or 2-class map
        da = None
        cand_msgs = []
        for idx, y in enumerate(outs):
            arr = np.squeeze(y)  # drop batch
            cand_msgs.append(f"o{idx} shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f}")
            # Cases: (H,W), (C,H,W) where C in {1,2}
            if arr.ndim == 2:
                da = arr
                da_src = f"o{idx}_HW"
                break
            if arr.ndim == 3 and arr.shape[0] in (1, 2):
                # if 2 classes, guess channel 1 is 'drivable'; if 1, take channel 0
                ch = 1 if arr.shape[0] == 2 else 0
                da = arr[ch]
                da_src = f"o{idx}_C{arr.shape[0]}[ch={ch}]"
                break

        # Debug once:
        if not hasattr(self, "_debugged"):
            print("[prior] candidates:", " | ".join(cand_msgs))
            print("[prior] chosen:", da_src if da is not None else "NONE")
            self._debugged = True

        if da is None:
            # Couldn’t identify a segmentation head → fallback full mask
            return np.full((h, w), 255, dtype=np.uint8)

        # If values aren’t in [0,1], apply sigmoid (typical for logits)
        da_min, da_max = float(da.min()), float(da.max())
        if da_min < 0 or da_max > 1.0:
            da = 1.0 / (1.0 + np.exp(-da))

        mask_small = (da > self.threshold).astype(np.uint8) * 255
        return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)