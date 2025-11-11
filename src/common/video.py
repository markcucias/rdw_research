import os
from typing import List, Tuple, Optional

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in IMAGE_EXTS


def _list_images_sorted(folder: str) -> List[str]:
    """
    Tries to sort image files in natural order (e.g., img2 before img10).
    Takes the folder path and returns a list of full paths to image files.
    """
    entries = [n for n in os.listdir(folder) if _is_image(n)]

    def key(name: str) -> Tuple[str, int]:
        base, _ = os.path.splitext(name)
        i, digits = len(base) - 1, ""
        while i >= 0 and base[i].isdigit():
            digits = base[i] + digits
            i -= 1
        return (base[: i + 1], int(digits) if digits else -1)

    entries.sort(key=key)
    return [os.path.join(folder, n) for n in entries]


class _ImageFolderCapture:
    """OpenCV-like capture for a folder of images."""

    def __init__(self, folder: str):
        self.paths = _list_images_sorted(folder)
        self._i = 0

    def isOpened(self) -> bool:
        return len(self.paths) > 0

    def read(self):
        if self._i >= len(self.paths):
            return False, None
        path = self.paths[self._i]
        self._i += 1
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return False, None
        return True, img

    def release(self):
        pass


class _SingleImageCapture:
    """OpenCV-like capture for a single image that yields exactly one frame."""

    def __init__(self, path: str):
        self.path = path
        self._done = False
        self.single_image = True

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self._done:
            return False, None
        img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        self._done = True
        if img is None:
            return False, None
        return True, img

    def release(self):
        pass


def open_source(source: str):
    """
    Returns an object with .isOpened(), .read(), .release() like cv2.VideoCapture.
    Accepts:
      - 'camera:N'  → webcam at index N
      - directory   → iterates all images in natural order
      - image file  → yields the single image
      - video file  → cv2.VideoCapture on that file
    """
    if source.startswith("camera:"):
        idx = int(source.split(":")[1])
        return cv2.VideoCapture(idx)

    if os.path.isdir(source):
        return _ImageFolderCapture(source)

    if os.path.isfile(source) and _is_image(source):
        return _SingleImageCapture(source)

    # fallback: assume it's a video file path
    return cv2.VideoCapture(source)