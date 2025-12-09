from pathlib import Path
import csv

out_path = Path("data/gt_center_130.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_idx", "filename", "x_gt_m", "scenario"])
    for i in range(130):
        filename = f"frame_{i:06d}.png"  # просто заглушка
        writer.writerow([i, filename, 0.01, "straight"])

print("saved to", out_path)
