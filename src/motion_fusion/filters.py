from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Pose:
    x: float
    y: float
    theta: float


class KalmanPoseFilter:
    """
    Простейший Калман:
      состояние: [x, y, theta]^T
      предсказание: добавляем dx, dy, dtheta из одометрии
      измерение: [x_lane, theta_lane]
    """

    def __init__(self, cfg: dict) -> None:
        q = float(cfg.get("process_noise", 0.05))
        r_lane = float(cfg.get("meas_noise_lane", 0.2))
        init_cov = float(cfg.get("initial_cov", 1.0))

        self.x = np.zeros((3, 1), dtype=np.float32)
        self.P = np.eye(3, dtype=np.float32) * init_cov

        self.Q = np.eye(3, dtype=np.float32) * q
        self.R_lane = np.eye(2, dtype=np.float32) * r_lane

    def predict(self, dx: float, dy: float, dtheta: float) -> None:
        F = np.eye(3, dtype=np.float32)
        u = np.array([[dx], [dy], [dtheta]], dtype=np.float32)

        self.x = F @ self.x + u
        self.P = F @ self.P @ F.T + self.Q

    def update_lane(self, x_lane: float, theta_lane: float) -> None:
        # z = [x_lane, theta_lane]
        H = np.array([[1, 0, 0],
                      [0, 0, 1]], dtype=np.float32)
        z = np.array([[x_lane],
                      [theta_lane]], dtype=np.float32)

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_lane
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(3, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

    def pose(self) -> Pose:
        return Pose(
            float(self.x[0]),
            float(self.x[1]),
            float(self.x[2]),
        )
