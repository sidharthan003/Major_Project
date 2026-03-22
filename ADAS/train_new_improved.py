"""
train_thermal_yolov8s.py
------------------------
Fine-tunes YOLOv8s on the thermal human detection dataset.

Goals:
- High recall for rescue detection
- Stable F1 score
- Real-time inference compatibility
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# Dataset configuration
DATASET_YAML = Path(__file__).parent / "thermal_person.yaml"

# Default parameters
DEFAULT_WEIGHTS = "yolov8s.pt"
DEFAULT_EPOCHS = 120
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_PROJECT = "runs/detect"
DEFAULT_NAME = "thermal_person_v8s"
DEFAULT_PATIENCE = 20
DEFAULT_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8s for thermal human detection")

    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--device", default="")

    return parser.parse_args()


def main():

    args = parse_args()

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}"
        )

    model = YOLO(args.weights)

    print("\n==============================")
    print("YOLOv8s Thermal Human Training")
    print("==============================")
    print("Dataset :", DATASET_YAML)
    print("Epochs  :", args.epochs)
    print("Image   :", args.imgsz)
    print("Batch   :", args.batch)
    print("Output  :", args.project + "/" + args.name)
    print("==============================\n")

    model.train(

        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device if args.device else None,

        project=args.project,
        name=args.name,

        patience=args.patience,

        # -------------------------
        # Augmentations (thermal)
        # -------------------------
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,

        degrees=5.0,
        translate=0.15,
        scale=0.5,

        fliplr=0.5,
        flipud=0.1,

        mosaic=1.0,
        mixup=0.1,

        # -------------------------
        # Detection optimization
        # -------------------------
        box=8.5,
        cls=0.4,
        dfl=1.5,

        # -------------------------
        # Optimizer
        # -------------------------
        optimizer="AdamW",

        lr0=1e-3,
        lrf=0.01,

        weight_decay=1e-4,

        warmup_epochs=3,

        # -------------------------
        # Validation thresholds
        # -------------------------
        conf=0.001,
        iou=0.5,

        # -------------------------
        # Performance
        # -------------------------
        amp=True,
        cache=True,

        save=True,
        save_period=10,
        plots=True
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"

    print("\nTraining complete.")
    print("Best weights saved at:")
    print(best)

    print("\nExample inference command:\n")
    print(
        f"yolo detect predict model={best} source=test_images conf=0.18 iou=0.5 imgsz=640 half=True"
    )


if __name__ == "__main__":
    main()