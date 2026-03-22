import argparse
import json
import sys
import tempfile
from pathlib import Path
import warnings

import torch
from ultralytics import YOLO

# Ensure local imports work when running this file from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_YAML = PROJECT_ROOT.parent / "baseline_sonar_yolo" / "data" / "yolo_sonar" / "data.yaml"
DEFAULT_MODEL = "configs/yolov8n-sonar-swin.yaml"

from src.models.register_ultralytics import register_custom_modules


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO-Swin sonar model.")
    p.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="sonar_swin")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=50)
    return p.parse_args()


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_model_spec(model_spec: str) -> str:
    # Keep built-in specs (e.g. yolov8n.pt) unless an actual file exists.
    candidate = Path(model_spec)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return str(project_candidate)

    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return model_spec


def is_custom_swin_model(model_spec: str) -> bool:
    model_path = Path(model_spec)
    if model_path.suffix.lower() in {".yaml", ".yml"} and model_path.exists():
        try:
            return "C2f_Swin" in model_path.read_text(encoding="utf-8")
        except OSError:
            return False
    return False


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

    model = YOLO(args.model)
    try:
        model.train(project=args.project, name=args.name, **train_kwargs)
        return model, Path(args.project) / args.name, True
    except ValueError as exc:
        if "I/O operation on closed file" not in str(exc):
            raise
        fallback_project = Path(tempfile.gettempdir()) / "yolo_runs_fallback"
        fallback_name = f"{args.name}_nosave"
        print("Checkpoint save failed. Retrying with save=False...")
        model = YOLO(args.model)
        model.train(project=str(fallback_project), name=fallback_name, save=False, **train_kwargs)
        return model, fallback_project / fallback_name, False


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    args.data_yaml = resolve_input_path(args.data_yaml)
    args.model = resolve_model_spec(args.model)

    register_custom_modules()

    if not is_custom_swin_model(args.model):
        warnings.warn(
            "Selected model is not the custom Swin YAML architecture. "
            "For README-aligned custom training, use --model configs/yolov8n-sonar-swin.yaml."
        )

    if not args.data_yaml.exists():
        raise FileNotFoundError(
            f"Data YAML not found: {args.data_yaml}\n"
            f"Default expected path: {DEFAULT_DATA_YAML}\n"
            "Pass --data-yaml with your dataset config file if it is elsewhere."
        )

    model, run_dir, checkpoints_saved = train_with_fallback(args, device)
    metrics = model.val(data=str(args.data_yaml), imgsz=args.imgsz, device=device)
    inference_ms, fps = extract_speed(metrics)

    summary = {
        "model": args.model,
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