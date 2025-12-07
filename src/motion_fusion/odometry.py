from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import cv2
import numpy as np


@dataclass
class OdomResult:
    dx: float
    dy: float
    dtheta: float
    n_matches: int
    vis_matches: Optional[np.ndarray]


class VisualOdometry:
    """
    Очень простой VO:
      - ORB-фичи между соседними кадрами
      - BFMatcher
      - AffinePartial2D + RANSAC
      - возвращаем dx, dy, dtheta в ПИКСЕЛЯХ (конвертим в метры уже в pipeline)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.orb = cv2.ORB_create(
            nfeatures=int(cfg.get("orb_nfeatures", 1000))
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.ransac_thresh = float(cfg.get("ransac_thresh", 3.0))
        self.min_matches = int(cfg.get("min_matches", 60))

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp = None
        self.prev_des = None

        # Накопленная поза (пока нам нужна только для отладки)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def update(self, frame_bgr: np.ndarray) -> OdomResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, None)
        dx = dy = dtheta = 0.0
        n_matches = 0
        vis = None

        if self.prev_des is not None and des is not None and len(kp) > 0:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda m: m.distance)
            n_matches = len(matches)

            if n_matches >= self.min_matches:
                pts_prev = np.float32(
                    [self.prev_kp[m.queryIdx].pt for m in matches]
                ).reshape(-1, 1, 2)
                pts_curr = np.float32(
                    [kp[m.trainIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                M, inliers = cv2.estimateAffinePartial2D(
                    pts_prev,
                    pts_curr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_thresh,
                )

                if M is not None:
                    dx = float(M[0, 2])
                    dy = float(M[1, 2])
                    dtheta = float(np.arctan2(M[1, 0], M[0, 0]))

                    # интегрируем 'глобальную' позу только для отладки
                    self.theta += dtheta
                    self.x += dx * np.cos(self.theta) - dy * np.sin(self.theta)
                    self.y += dx * np.sin(self.theta) + dy * np.cos(self.theta)

                    vis = cv2.drawMatches(
                        self.prev_gray,
                        self.prev_kp,
                        gray,
                        kp,
                        matches[:80],
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    )

        # обновляем состояние
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        return OdomResult(
            dx=dx,
            dy=dy,
            dtheta=dtheta,
            n_matches=n_matches,
            vis_matches=vis,
        )

    def pose(self):
        return self.x, self.y, self.theta
