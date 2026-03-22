import argparse
import json
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standard YOLOv8 baseline on sonar data.")
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="sonar_baseline")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg.lower() != "auto":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def extract_metric(value):
    if isinstance(value, list):
        return float(value[0]) if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def train_with_fallback(args: argparse.Namespace, device: str) -> tuple[YOLO, Path, bool]:
    train_kwargs = {
        "data": str(args.data_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": device,
        "workers": args.workers,
        "patience": args.patience,
        "pretrained": True,
    }

    model = YOLO(args.model)
    try:
        model.train(project=args.project, name=args.name, **train_kwargs)
        return model, Path(args.project) / args.name, True
    except ValueError as exc:
        if "I/O operation on closed file" not in str(exc):
            raise

        # Retry once with checkpoint saving disabled to bypass intermittent torch zip save failures.
        fallback_project = Path(tempfile.gettempdir()) / "yolo_runs_fallback"
        fallback_name = f"{args.name}_nosave"
        print("Checkpoint save failed due to a PyTorch file I/O issue. Retrying with save=False...")

        model = YOLO(args.model)
        model.train(project=str(fallback_project), name=fallback_name, save=False, **train_kwargs)
        return model, fallback_project / fallback_name, False


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data_yaml}")

    model, run_dir, checkpoints_saved = train_with_fallback(args, device)

    metrics = model.val(data=str(args.data_yaml), imgsz=args.imgsz, device=device)

    summary = {
        "model": args.model,
        "data_yaml": str(args.data_yaml.resolve()),
        "mAP50": extract_metric(metrics.box.map50),
        "mAP50_95": extract_metric(metrics.box.map),
        "precision": extract_metric(metrics.box.mp),
        "recall": extract_metric(metrics.box.mr),
        "run_dir": str(run_dir.resolve()),
        "checkpoints_saved": checkpoints_saved,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / "baseline_metrics.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Validation summary saved:")
    print(json.dumps(summary, indent=2))
    print(f"Metrics path: {out_file}")


if __name__ == "__main__":
    main()
