import argparse
import json
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO
from src.models.register_ultralytics import register_custom_modules


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO-Swin sonar model.")
    p.add_argument("--data-yaml", type=Path, required=True)
    p.add_argument("--model-yaml", type=Path, default=Path("configs/yolov8n-sonar-swin.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="sonar_swin")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30)
    return p.parse_args()


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


def extract_speed(metrics):
    speed = getattr(metrics, "speed", None)
    if not isinstance(speed, dict):
        return None, None
    inference_ms = speed.get("inference")
    try:
        inference_ms = float(inference_ms)
    except (TypeError, ValueError):
        return None, None
    fps = 1000.0 / inference_ms if inference_ms > 0 else None
    return inference_ms, fps


def train_with_fallback(args: argparse.Namespace, device: str):
    train_kwargs = {
        "data": str(args.data_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": device,
        "workers": args.workers,
        "patience": args.patience,
        "pretrained": False,
    }

    register_custom_modules()
    model = YOLO(str(args.model_yaml))
    try:
        model.train(project=args.project, name=args.name, **train_kwargs)
        return model, Path(args.project) / args.name, True
    except ValueError as exc:
        if "I/O operation on closed file" not in str(exc):
            raise
        fallback_project = Path(tempfile.gettempdir()) / "yolo_runs_fallback"
        fallback_name = f"{args.name}_nosave"
        print("Checkpoint save failed. Retrying with save=False...")
        register_custom_modules()
        model = YOLO(str(args.model_yaml))
        model.train(project=str(fallback_project), name=fallback_name, save=False, **train_kwargs)
        return model, fallback_project / fallback_name, False


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data_yaml}")
    if not args.model_yaml.exists():
        raise FileNotFoundError(f"Model YAML not found: {args.model_yaml}")

    model, run_dir, checkpoints_saved = train_with_fallback(args, device)
    metrics = model.val(data=str(args.data_yaml), imgsz=args.imgsz, device=device)
    inference_ms, fps = extract_speed(metrics)

    summary = {
        "model_yaml": str(args.model_yaml.resolve()),
        "data_yaml": str(args.data_yaml.resolve()),
        "mAP50": extract_metric(metrics.box.map50),
        "mAP50_95": extract_metric(metrics.box.map),
        "precision": extract_metric(metrics.box.mp),
        "recall": extract_metric(metrics.box.mr),
        "inference_ms": inference_ms,
        "fps": fps,
        "run_dir": str(run_dir.resolve()),
        "checkpoints_saved": checkpoints_saved,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / "sonar_swin_metrics.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Metrics path: {out_file}")


if __name__ == "__main__":
    main()