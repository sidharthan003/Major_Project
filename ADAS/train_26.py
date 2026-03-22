"""
train_26.py
-----------
Fine-tunes YOLO26 on the same thermal person-detection dataset.

Steps:
  1.  python prepare_data.py   (run once)
  2.  python train_26.py

Checkpoints and metrics are saved to runs/detect/thermal_person_26/
"""

import argparse
from pathlib import Path

# Config
DATASET_YAML = Path(__file__).parent / "thermal_person.yaml"

# YOLO26 weights options (example names):
#   yolo26n.pt  - nano
#   yolo26s.pt  - small
#   yolo26m.pt  - medium
#   yolo26l.pt  - large
#   yolo26x.pt  - xlarge
DEFAULT_WEIGHTS = "yolo26n.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_PROJECT = "runs/detect"
DEFAULT_NAME = "thermal_person_26"
DEFAULT_PATIENCE = 15
DEFAULT_WORKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26 for thermal human detection")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Pretrained weights or path to a .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="Input image size (square)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help="Early-stopping patience")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint in runs/detect/<name>/weights/last.pt")
    parser.add_argument("--device", default="",
                        help="cuda device (0, 0,1, cpu) - empty = auto-select")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] ultralytics not installed.\n"
            "        Run: pip install ultralytics"
        ) from exc

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"[ERROR] Dataset YAML not found: {DATASET_YAML}\n"
            "        Run python prepare_data.py first."
        )

    if args.resume:
        resume_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not resume_ckpt.exists():
            raise FileNotFoundError(
                f"[ERROR] Resume checkpoint not found: {resume_ckpt}"
            )
        weights = str(resume_ckpt)
        print(f"[INFO] Resuming training from {weights}")
    else:
        weights = args.weights

    model = YOLO(weights)

    print(f"\n{'=' * 60}")
    print(f"  Training  : {weights}")
    print(f"  Dataset   : {DATASET_YAML}")
    print(f"  Epochs    : {args.epochs}  (patience={args.patience})")
    print(f"  Img size  : {args.imgsz}x{args.imgsz}")
    print(f"  Batch     : {args.batch}")
    print(f"  Output    : {args.project}/{args.name}")
    print(f"{'=' * 60}\n")

    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        device=args.device if args.device else None,
        resume=args.resume,
        # Thermal-oriented augmentation
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        flipud=0.1,
        mosaic=1.0,
        mixup=0.1,
        # Optimizer
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=1e-4,
        warmup_epochs=3,
        # Validation thresholds
        conf=0.001,
        iou=0.6,
        save=True,
        save_period=10,
        plots=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n[DONE] Training complete.")
    print(f"       Best weights -> {best}")
    print(f"       Run inference with: python detect.py --weights {best}")


if __name__ == "__main__":
    main()
