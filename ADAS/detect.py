"""
detect.py
---------
Run inference with a trained YOLOv8 model on:
  • a single image / directory of images
  • a video file
  • a live webcam / RTSP stream

Usage examples
--------------
# Single image
python detect.py --weights runs/detect/thermal_person/weights/best.pt \
                 --source path/to/image.jpg

# Directory of images
python detect.py --weights best.pt --source images_thermal_val/data

# Video file
python detect.py --weights best.pt --source path/to/video.mp4 --save-video

# Webcam (index 0)
python detect.py --weights best.pt --source 0

# RTSP stream
python detect.py --weights best.pt --source rtsp://user:pass@ip:port/stream

Outputs are saved to runs/detect/predict_<timestamp>/
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


# ── default weights path ──────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "runs/detect/thermal_person/weights/best.pt"


# ── argument parser ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Thermal human detection — inference"
    )
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="Path to trained .pt weights")
    p.add_argument("--source",  required=True,
                   help="Image / directory / video / webcam index / RTSP URL")
    p.add_argument("--conf",    type=float, default=0.35,
                   help="Confidence threshold (0–1)")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--imgsz",   type=int,   default=640,
                   help="Inference image size")
    p.add_argument("--device",  default="",
                   help="Device: gpu index (0) or 'cpu'")
    p.add_argument("--save-video", action="store_true",
                   help="Write output video to disk")
    p.add_argument("--no-show", action="store_true",
                   help="Disable live preview window")
    p.add_argument("--output",  default="",
                   help="Output directory (auto-generated if empty)")
    p.add_argument("--max-det", type=int, default=300,
                   help="Maximum detections per frame")
    return p.parse_args()


# ── detection logic ───────────────────────────────────────────────────────────
def run_detection(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"[ERROR] Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    # Determine output directory
    out_dir = Path(args.output) if args.output else (
        Path("runs/detect") / f"predict_{int(time.time())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    source: str | int = args.source
    # Treat numeric strings as webcam indices
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    _is_image_source = isinstance(source, str) and (
        Path(source).is_file() and
        Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    )
    _is_dir_source = isinstance(source, str) and Path(source).is_dir()
    _is_video_or_stream = not _is_image_source and not _is_dir_source

    print(f"\n[INFO] Model   : {weights_path}")
    print(f"[INFO] Source  : {source}")
    print(f"[INFO] Conf    : {args.conf}  |  IoU: {args.iou}")
    print(f"[INFO] Output  : {out_dir}\n")

    # ── single image / directory  ────────────────────────────────────────────
    if _is_image_source or _is_dir_source:
        results = model.predict(
            source  = source,
            conf    = args.conf,
            iou     = args.iou,
            imgsz   = args.imgsz,
            device  = args.device if args.device else None,
            max_det = args.max_det,
            save    = True,
            save_txt= True,
            project = str(out_dir.parent),
            name    = out_dir.name,
            show    = not args.no_show,
        )
        _print_summary(results)
        return

    # ── video / stream / webcam ──────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open source: {source}")

    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        out_video = out_dir / "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
        print(f"[INFO] Saving video → {out_video}")

    frame_idx  = 0
    person_cls = 0   # our only class

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source  = frame,
            conf    = args.conf,
            iou     = args.iou,
            imgsz   = args.imgsz,
            device  = args.device if args.device else None,
            max_det = args.max_det,
            verbose = False,
        )

        # Annotate detections manually for full control
        annotated = frame.copy()
        n_persons  = 0
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id != person_cls:
                    continue
                n_persons += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Draw bounding box (green)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"person {conf:.2f}"
                cv2.putText(
                    annotated, label,
                    (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

        # HUD — person count or "No human detected"
        if n_persons == 0:
            hud_text  = "No human detected"
            hud_color = (0, 0, 255)    # red
        else:
            hud_text  = f"Humans detected: {n_persons}"
            hud_color = (0, 255, 0)    # green

        # Semi-transparent background strip for readability
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)

        cv2.putText(
            annotated, hud_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, hud_color, 2
        )

        if writer:
            writer.write(annotated)

        if not args.no_show:
            cv2.imshow("Thermal Person Detection  [q to quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Processed {frame_idx} frames.  Output: {out_dir}")


# ── summary helper ────────────────────────────────────────────────────────────
def _print_summary(results) -> None:
    """
    For each image print whether a human was detected.
    Also saves an annotated copy with 'No human detected' text when applicable.
    """
    import numpy as np

    total_det  = 0
    no_human   = 0

    for r in results:
        n = len(r.boxes) if r.boxes else 0
        total_det += n

        img_name = Path(r.path).name if r.path else "image"

        if n == 0:
            no_human += 1
            print(f"  [{img_name}]  --> No human detected")

            # Stamp the saved annotated image with the 'No human detected' banner
            saved_path = r.save_path if hasattr(r, "save_path") and r.save_path else None
            if saved_path and Path(saved_path).exists():
                img = cv2.imread(str(saved_path))
                if img is not None:
                    ih, iw = img.shape[:2]
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (iw, 50), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                    cv2.putText(
                        img, "No human detected",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2
                    )
                    cv2.imwrite(str(saved_path), img)
        else:
            print(f"  [{img_name}]  --> {n} human(s) detected")

    print(f"\n[DONE] {len(results)} image(s) processed.")
    print(f"       Humans found     : {len(results) - no_human} image(s), {total_det} total detection(s)")
    print(f"       No human detected: {no_human} image(s)")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_detection(parse_args())
