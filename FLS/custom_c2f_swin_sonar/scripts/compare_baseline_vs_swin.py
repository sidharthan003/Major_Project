import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def delta(a, b, key):
    va, vb = a.get(key), b.get(key)
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return vb - va
    return None


def main():
    p = argparse.ArgumentParser(description="Compare baseline and YOLO-Swin summaries.")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--swin", type=Path, required=True)
    args = p.parse_args()

    base = load_json(args.baseline)
    swin = load_json(args.swin)

    report = {
        "baseline": str(args.baseline.resolve()),
        "swin": str(args.swin.resolve()),
        "delta_mAP50": delta(base, swin, "mAP50"),
        "delta_mAP50_95": delta(base, swin, "mAP50_95"),
        "delta_precision": delta(base, swin, "precision"),
        "delta_recall": delta(base, swin, "recall"),
        "delta_fps": delta(base, swin, "fps"),
        "delta_inference_ms": delta(base, swin, "inference_ms"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()