from __future__ import annotations

import csv
import math
from pathlib import Path
import argparse
from collections import defaultdict


def load_gt(path: Path):
    """Читает GT: frame_idx,x_gt_m,scenario."""
    gt = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            x_gt_m = float(row["x_gt_m"])
            scenario = row.get("scenario", "unknown") or "unknown"
            gt[frame_idx] = {
                "x_gt_m": x_gt_m,
                "scenario": scenario,
                "filename": row.get("filename", ""),
            }
    return gt


def load_bench(path: Path):
    """Читает benchmark_classical.csv: frame_idx,x_m,alpha_deg,v_mps,dt_sec."""
    bench = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            x_pred_m = float(row["x_m"])
            bench[frame_idx] = {
                "x_pred_m": x_pred_m,
                "alpha_deg": float(row.get("alpha_deg", 0.0)),
                "v_mps": float(row.get("v_mps", 0.0)),
            }
    return bench


def main():
    parser = argparse.ArgumentParser(
        description="Compare Classical+Prior predictions (x_m) with manual ground truth (x_gt_m)."
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="data/gt_classical_x.csv",
        help="Path to ground-truth CSV (frame_idx,filename,x_gt_m,scenario).",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default="data/benchmark_classical.csv",
        help="Path to benchmark CSV produced by rdw-classical.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/accuracy_classical.csv",
        help="Where to save joined per-frame results.",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt)
    bench_path = Path(args.bench)
    out_path = Path(args.out)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {bench_path}")

    gt = load_gt(gt_path)
    bench = load_bench(bench_path)

    records = []
    abs_errors_all = []
    per_scenario_errors = defaultdict(list)

    for frame_idx, gt_row in gt.items():
        if frame_idx not in bench:
            print(f"[WARN] frame_idx {frame_idx} not found in benchmark, skipping.")
            continue

        x_gt = gt_row["x_gt_m"]
        scenario = gt_row["scenario"]
        filename = gt_row["filename"]
        x_pred = bench[frame_idx]["x_pred_m"]

        err = x_pred - x_gt
        abs_err = abs(err)

        abs_errors_all.append(abs_err)
        per_scenario_errors[scenario].append(abs_err)

        records.append(
            {
                "frame_idx": frame_idx,
                "filename": filename,
                "scenario": scenario,
                "x_gt_m": x_gt,
                "x_pred_m": x_pred,
                "err_m": err,
                "abs_err_m": abs_err,
            }
        )

    if not records:
        print("No overlapping frames between GT and benchmark. Check frame_idx numbering.")
        return

    # --- глобальные метрики ---
    n = len(abs_errors_all)
    mae = sum(abs_errors_all) / n
    rmse = math.sqrt(sum(e * e for e in abs_errors_all) / n)
    max_err = max(abs_errors_all)

    print("\n=== Classical + Prior accuracy (x) ===")
    print(f"Frames used   = {n}")
    print(f"MAE(x)        = {mae:.3f} m")
    print(f"RMSE(x)       = {rmse:.3f} m")
    print(f"Max |error|   = {max_err:.3f} m")

    # --- метрики по сценариям (straight / curve / ...) ---
    print("\nPer-scenario MAE:")
    for scenario, errs in per_scenario_errors.items():
        if not errs:
            continue
        mae_s = sum(errs) / len(errs)
        print(f"  {scenario:12s} : MAE = {mae_s:.3f} m  (N={len(errs)})")

    # --- сохранить объединённый CSV ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "filename",
                "scenario",
                "x_gt_m",
                "x_pred_m",
                "err_m",
                "abs_err_m",
            ],
        )
        writer.writeheader()
        for r in sorted(records, key=lambda r: r["frame_idx"]):
            writer.writerow(r)

    print("\nPer-frame accuracy saved to", out_path)


if __name__ == "__main__":
    main()
