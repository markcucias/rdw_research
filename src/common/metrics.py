from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import csv
import math


def compute_lateral_deviation_cm(
    x_left: float,
    x_right: float,
    img_width: int,
    lane_width_m: float = 3.5,
) -> Optional[float]:
    """
    Classical формула из отчёта Ашокана.

    x_left, x_right  – координаты пересечения ЛЕВОЙ и ПРАВОЙ границы полосы
                       с опорной горизонтальной линией (в пикселях).

    img_width        – ширина изображения в пикселях.

    lane_width_m     – реальная ширина полосы в метрах (примерно 3.5 м).

    Возвращает отклонение 'машины' (центр нижнего края кадра) от центра полосы
    в САНТИМЕТРАХ:
        dev_cm > 0  -> машина смещена вправо от центра полосы
        dev_cm < 0  -> смещена влево
    """
    lane_width_px = abs(x_right - x_left)
    if lane_width_px < 1:
        # нет адекватной измеримой полосы
        return None

    # центр полосы в пикселях
    x_lane = 0.5 * (x_left + x_right)

    # считаем, что проекция машины – центр нижнего края кадра
    kart_x = img_width / 2.0

    dev_px = kart_x - x_lane

    # перевод пикселей в сантиметры
    lane_width_cm = lane_width_m * 100.0
    px_to_cm = lane_width_cm / lane_width_px

    dev_cm = dev_px * px_to_cm
    return dev_cm


@dataclass
class FrameMetrics:
    frame_idx: int
    filename: str
    scenario: str
    deviation_cm: Optional[float]
    dt: float  # время обработки кадра в секундах


@dataclass
class BenchmarkLogger:
    """
    Простенький агрегатор:
    - копит метрики по кадрам,
    - умеет считать средние,
    - умеет сохранять в CSV.
    """
    records: List[FrameMetrics] = field(default_factory=list)

    def add(
        self,
        frame_idx: int,
        filename: str,
        scenario: str,
        deviation_cm: Optional[float],
        dt: float,
    ):
        self.records.append(
            FrameMetrics(
                frame_idx=frame_idx,
                filename=filename,
                scenario=scenario,
                deviation_cm=deviation_cm,
                dt=dt,
            )
        )

    def summary(self) -> Dict[str, float]:
        vals = [r.deviation_cm for r in self.records if r.deviation_cm is not None]
        times = [r.dt for r in self.records]

        if not vals or not times:
            return {}

        mae = sum(abs(v) for v in vals) / len(vals)
        max_dev = max(abs(v) for v in vals)
        avg_dt = sum(times) / len(times)
        fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

        return {
            "mae_dev_cm": mae,
            "max_dev_cm": max_dev,
            "avg_dt_ms": avg_dt * 1000.0,
            "fps": fps,
        }

    def save_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_idx",
                    "filename",
                    "scenario",
                    "deviation_cm",
                    "dt_sec",
                ]
            )
            for r in self.records:
                writer.writerow(
                    [
                        r.frame_idx,
                        r.filename,
                        r.scenario,
                        "" if r.deviation_cm is None else f"{r.deviation_cm:.3f}",
                        f"{r.dt:.6f}",
                    ]
                )
