"""
train.py
--------
Fine-tunes YOLOv8 on the thermal person-detection dataset.

Steps:
  1.  python prepare_data.py   (run once)
  2.  python train.py

Checkpoints and metrics are saved to runs/detect/thermal_person/
"""

import argparse
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

DATASET_YAML = Path(__file__).parent / "thermal_person.yaml"

# Backbone weights:
#   yolov8n.pt  – nano   (fastest, lowest accuracy)
#   yolov8s.pt  – small
#   yolov8m.pt  – medium (good balance for ADAS use-cases)
#   yolov8l.pt  – large
#   yolov8x.pt  – xlarge (highest accuracy, slowest)
DEFAULT_WEIGHTS  = "yolov8s.pt"
DEFAULT_EPOCHS   = 50
DEFAULT_IMGSZ    = 640
DEFAULT_BATCH    = 16          # reduce to 8 or 4 if you run out of VRAM
DEFAULT_PROJECT  = "runs/detect"
DEFAULT_NAME     = "thermal_person"
DEFAULT_PATIENCE = 15          # early-stopping patience (epochs without improvement)
DEFAULT_WORKERS  = 4


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 for thermal human detection")
    p.add_argument("--weights",  default=DEFAULT_WEIGHTS,
                   help="Pretrained weights or path to a .pt checkpoint")
    p.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz",    type=int,   default=DEFAULT_IMGSZ,
                   help="Input image size (square)")
    p.add_argument("--batch",    type=int,   default=DEFAULT_BATCH)
    p.add_argument("--workers",  type=int,   default=DEFAULT_WORKERS)
    p.add_argument("--project",  default=DEFAULT_PROJECT)
    p.add_argument("--name",     default=DEFAULT_NAME)
    p.add_argument("--patience", type=int,   default=DEFAULT_PATIENCE,
                   help="Early-stopping patience")
    p.add_argument("--resume",   action="store_true",
                   help="Resume from last checkpoint in runs/detect/<name>/weights/last.pt")
    p.add_argument("--device",   default="",
                   help="cuda device (0, 0,1, cpu) – empty = auto-select")
    return p.parse_args()


# ── training ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "[ERROR] ultralytics not installed.\n"
            "        Run:  pip install ultralytics"
        )

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"[ERROR] Dataset YAML not found: {DATASET_YAML}\n"
            "        Run  python prepare_data.py  first."
        )

    # ── resume path ──────────────────────────────────────────────────────────
    if args.resume:
        resume_ckpt = (
            Path(args.project) / args.name / "weights" / "last.pt"
        )
        if not resume_ckpt.exists():
            raise FileNotFoundError(
                f"[ERROR] Resume checkpoint not found: {resume_ckpt}"
            )
        weights = str(resume_ckpt)
        print(f"[INFO] Resuming training from {weights}")
    else:
        weights = args.weights

    model = YOLO(weights)

    print(f"\n{'='*60}")
    print(f"  Training  : {weights}")
    print(f"  Dataset   : {DATASET_YAML}")
    print(f"  Epochs    : {args.epochs}  (patience={args.patience})")
    print(f"  Img size  : {args.imgsz}x{args.imgsz}")
    print(f"  Batch     : {args.batch}")
    print(f"  Output    : {args.project}/{args.name}")
    print(f"{'='*60}\n")

    # ── Training augmentation strategy for thermal imagery ──────────────────
    # • hsv_h / hsv_s are zeroed because thermal images are single-channel grey
    #   (even though stored as 3-ch JPG they contain no hue/saturation info)
    # • flipud is enabled because pedestrians can appear at varying vertical pos
    # • degrees / translate / scale provide robustness to camera motion

    results = model.train(
        data      = str(DATASET_YAML),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        workers   = args.workers,
        project   = args.project,
        name      = args.name,
        patience  = args.patience,
        device    = args.device if args.device else None,
        resume    = args.resume,
        # augmentation
        hsv_h     = 0.0,    # no hue shift (greyscale thermal)
        hsv_s     = 0.0,    # no saturation shift
        hsv_v     = 0.3,    # brightness variation is valid for thermal
        degrees   = 5.0,    # small rotation for minor camera tilt
        translate = 0.1,
        scale     = 0.4,
        fliplr    = 0.5,    # horizontal flip
        flipud    = 0.1,    # occasional vertical flip
        mosaic    = 1.0,    # mosaic augmentation (helps small objects)
        mixup     = 0.1,
        # optimiser
        optimizer = "AdamW",
        lr0       = 1e-3,
        lrf       = 0.01,
        weight_decay = 1e-4,
        warmup_epochs = 3,
        # confidence thresholds used during val
        conf      = 0.001,
        iou       = 0.6,
        save      = True,
        save_period = 10,   # save checkpoint every N epochs
        plots     = True,
    )

    print("\n[DONE] Training complete.")
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"       Best weights → {best}")
    print(f"       Run inference with:  python detect.py --weights {best}")


if __name__ == "__main__":
    main()
