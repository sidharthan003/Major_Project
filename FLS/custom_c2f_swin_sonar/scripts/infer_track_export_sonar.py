import argparse
import csv
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from src.tracking.payload import build_phuman_payload
from src.tracking.sonar_box_expander import SonarBoxExpander


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLO-Swin inference+tracking and export.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--tracker", type=str, default="bytetrack.yaml")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--classes", type=int, nargs="*", default=None)
    p.add_argument("--expand-ratio", type=float, default=3.0)
    p.add_argument("--sensor-id", type=str, default="sonar_01")
    return p.parse_args()


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
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    expander = SonarBoxExpander(ratio=args.expand_ratio)

    headers = ["frame_idx", "image_name", "track_id", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2", "ex1", "ey1", "ex2", "ey2"]

    t0 = time.perf_counter()
    frame_count, det_count = 0, 0

    with args.output_csv.open("w", newline="", encoding="utf-8") as fcsv, args.output_jsonl.open("w", encoding="utf-8") as fj:
        writer = csv.writer(fcsv)
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
                fj.write(json.dumps(build_phuman_payload(frame_idx, int(time.time() * 1000), args.sensor_id, [])) + "\n")
                continue

            xyxy = boxes.xyxy.cpu()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            h, w = result.orig_shape
            expanded = expander.expand_xyxy(xyxy, w=w, h=h).cpu().numpy()
            xyxy_np = xyxy.numpy()

            tracks = []
            for i in range(len(xyxy_np)):
                x1, y1, x2, y2 = xyxy_np[i].tolist()
                ex1, ey1, ex2, ey2 = expanded[i].tolist()
                class_id = int(clss[i])
                class_name = model.names.get(class_id, str(class_id))
                track_id = int(ids[i]) if ids is not None else -1
                conf = float(confs[i])

                writer.writerow([frame_idx, image_name, track_id, class_id, class_name, conf, x1, y1, x2, y2, ex1, ey1, ex2, ey2])
                det_count += 1

                tracks.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "expanded_bbox_xyxy": [ex1, ey1, ex2, ey2],
                })

            fj.write(json.dumps(build_phuman_payload(frame_idx, int(time.time() * 1000), args.sensor_id, tracks)) + "\n")

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
        "expand_ratio": args.expand_ratio,
        "output_csv": str(args.output_csv.resolve()),
        "output_jsonl": str(args.output_jsonl.resolve()),
    }

    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()