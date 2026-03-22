import argparse
import csv
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference+tracking and export raw tracking data.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained .pt weights")
    parser.add_argument("--source", type=str, required=True, help="Image folder/video path/camera index")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg.lower() != "auto":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))

    headers = ["frame_idx", "image_name", "track_id", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"]

    t0 = time.perf_counter()
    frame_count = 0
    det_count = 0

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        results = model.track(
            source=args.source,
            stream=True,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            classes=args.classes,
            save=False,
            verbose=False,
        )

        for frame_idx, result in enumerate(results):
            frame_count += 1
            image_name = Path(result.path).name if result.path else f"frame_{frame_idx:06d}"

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            ids = None
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                class_id = int(clss[i])
                class_name = model.names.get(class_id, str(class_id))
                track_id = int(ids[i]) if ids is not None else -1

                writer.writerow(
                    [
                        frame_idx,
                        image_name,
                        track_id,
                        class_id,
                        class_name,
                        float(confs[i]),
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ]
                )
                det_count += 1

    elapsed = time.perf_counter() - t0
    fps = frame_count / elapsed if elapsed > 0 else 0.0

    summary = {
        "weights": str(args.weights.resolve()),
        "source": args.source,
        "frames_processed": frame_count,
        "detections_exported": det_count,
        "elapsed_seconds": elapsed,
        "fps": fps,
        "tracker": args.tracker,
        "conf": args.conf,
        "iou": args.iou,
    }

    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Tracking CSV saved: {args.output_csv}")
    print(f"Summary saved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
