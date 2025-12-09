from __future__ import annotations

import cv2
from pathlib import Path
import csv
import argparse


def annotate_folder(
    img_dir: Path,
    out_csv: Path,
    lane_width_m: float = 3.5,
):
    img_paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )
    if not img_paths:
        print("No images found in", img_dir)
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f_out = out_csv.open("w", newline="")
    writer = csv.writer(f_out)
    writer.writerow(["frame_idx", "filename", "x_gt_m", "scenario"])

    print("Instructions:")
    print("  - Left click: mark LEFT lane boundary at bottom region")
    print("  - Right click: mark RIGHT lane boundary at bottom region")
    print("  - Press 's' to skip frame")
    print("  - Press 'q' to quit")
    print("  - After left+right clicks, you will be asked scenario in terminal.")
    print()

    state = {
        "x_left": None,
        "x_right": None,
    }

    def reset_state():
        state["x_left"] = None
        state["x_right"] = None

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            # первый клик -> левая полоса
            if state["x_left"] is None:
                state["x_left"] = x
                print(f"  Left lane x_left = {x}")
            # второй клик -> правая полоса
            elif state["x_right"] is None:
                state["x_right"] = x
                print(f"  Right lane x_right = {x}")
            else:
                # если вдруг кликнул третий раз — просто перезапишем правую
                state["x_right"] = x
                print(f"  Right lane updated x_right = {x}")

    for frame_idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print("Cannot read", img_path)
            continue

        h, w = img.shape[:2]

        reset_state()
        win_name = "Annotate GT x (left-click=left lane, right-click=right lane)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, on_mouse)

        while True:
            vis = img.copy()
            # рисуем подсказку про нижнюю часть изображения
            cv2.putText(
                vis,
                f"Frame {frame_idx} - {img_path.name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                "Click LEFT lane (LMB) and RIGHT lane (RMB) near bottom; 's' skip, 'q' quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # если уже кликнули – рисуем вертикальные линии
            if state["x_left"] is not None:
                cv2.line(vis, (state["x_left"], int(h * 0.6)), (state["x_left"], h - 1), (0, 255, 0), 2)
            if state["x_right"] is not None:
                cv2.line(vis, (state["x_right"], int(h * 0.6)), (state["x_right"], h - 1), (0, 0, 255), 2)

            cv2.imshow(win_name, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                print("Stopping annotation.")
                f_out.close()
                cv2.destroyAllWindows()
                return
            if key == ord("s"):
                print(f"Skipping frame {frame_idx}")
                break

            # если обе границы есть – считаем x_gt и спрашиваем сценарий
            if state["x_left"] is not None and state["x_right"] is not None:
                x_left = state["x_left"]
                x_right = state["x_right"]
                lane_width_px = abs(x_right - x_left)
                if lane_width_px < 5:
                    print("Lane width too small, re-click.")
                    reset_state()
                    continue

                x_lane_px = 0.5 * (x_left + x_right)
                kart_x_px = w / 2.0
                x_gt_px = kart_x_px - x_lane_px

                lane_width_m = float(lane_width_m)
                lane_width_cm = lane_width_m * 100.0
                px_to_cm = lane_width_cm / lane_width_px
                x_gt_cm = x_gt_px * px_to_cm
                x_gt_m = x_gt_cm / 100.0

                print(f"  lane_width_px = {lane_width_px}")
                print(f"  x_lane_px     = {x_lane_px:.1f}")
                print(f"  x_gt_px       = {x_gt_px:.1f}")
                print(f"  x_gt_m        = {x_gt_m:.3f} m")

                scenario_raw = input("  Enter scenario label (e.g. straight/curve/...): ").strip() or "unknown"

                # убираем все странные/непечатаемые символы, оставляем только нормальные
                scenario = "".join(ch for ch in scenario_raw if ch.isprintable())

                writer.writerow(
                    [
                        frame_idx,
                        img_path.name,
                        f"{x_gt_m:.6f}",
                        scenario,
                    ]
                )

                print(f"Saved GT for frame {frame_idx}\n")
                break  # к следующему кадру

    f_out.close()
    cv2.destroyAllWindows()
    print("Annotation finished. GT saved to", out_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive annotation of ground-truth lateral offset x using lane boundaries."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Folder with 100 frames (png/jpg). Filenames should be in frame order.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/gt_classical_x.csv",
        help="Output CSV for ground truth.",
    )
    parser.add_argument(
        "--lane-width-m",
        type=float,
        default=3.5,
        help="Assumed real lane width in meters.",
    )
    args = parser.parse_args()

    img_dir = Path(args.images)
    out_csv = Path(args.out)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    annotate_folder(img_dir, out_csv, lane_width_m=args.lane_width_m)


if __name__ == "__main__":
    main()
