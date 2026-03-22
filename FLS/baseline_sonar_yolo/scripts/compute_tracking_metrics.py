import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Frag Ratio and ID switches from tracking CSV.")
    parser.add_argument("--tracking-csv", type=Path, required=True)
    parser.add_argument("--target-class-id", type=int, default=0, help="Class ID to evaluate (default: victim=0)")
    parser.add_argument(
        "--ground-truth-presence",
        type=Path,
        default=None,
        help="Optional CSV with columns: frame,present (1 or 0)",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def compute_id_switches(track_ids: list[int | None]) -> int:
    switches = 0
    prev_id = None

    for tid in track_ids:
        if tid is None:
            continue
        if prev_id is None:
            prev_id = tid
            continue
        if tid != prev_id:
            switches += 1
            prev_id = tid

    return switches


def pick_frame_track_ids(df: pd.DataFrame) -> pd.Series:
    # Choose one target per frame (highest confidence) for continuity analysis.
    sorted_df = df.sort_values(["frame_idx", "conf"], ascending=[True, False])
    top_per_frame = sorted_df.drop_duplicates(subset=["frame_idx"], keep="first")
    return top_per_frame.set_index("frame_idx")["track_id"]


def main() -> None:
    args = parse_args()

    if not args.tracking_csv.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {args.tracking_csv}")

    df = pd.read_csv(args.tracking_csv)
    required_cols = {"frame_idx", "track_id", "class_id", "conf"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in tracking CSV: {sorted(missing)}")

    target_df = df[df["class_id"] == args.target_class_id].copy()
    if target_df.empty:
        result = {
            "target_class_id": args.target_class_id,
            "id_switches": 0,
            "frag_ratio": 1.0,
            "frag_ratio_mode": "no_detections",
            "notes": "No detections found for target class.",
        }
    else:
        frame_track = pick_frame_track_ids(target_df)

        min_frame = int(df["frame_idx"].min())
        max_frame = int(df["frame_idx"].max())
        all_frames = pd.Index(range(min_frame, max_frame + 1), name="frame_idx")

        aligned = frame_track.reindex(all_frames)
        ids = [None if pd.isna(v) or int(v) < 0 else int(v) for v in aligned.tolist()]

        id_switches = compute_id_switches(ids)

        if args.ground_truth_presence is not None:
            if not args.ground_truth_presence.exists():
                raise FileNotFoundError(f"GT presence CSV not found: {args.ground_truth_presence}")

            gt = pd.read_csv(args.ground_truth_presence)
            if not {"frame", "present"}.issubset(gt.columns):
                raise ValueError("GT presence CSV must include columns: frame,present")

            gt = gt.set_index("frame").reindex(all_frames).fillna({"present": 0})
            gt_present = gt["present"].astype(int).clip(lower=0, upper=1)

            total_present_frames = int(gt_present.sum())
            missed_frames = int(((gt_present == 1) & pd.Series([tid is None for tid in ids], index=all_frames)).sum())

            frag_ratio = (missed_frames / total_present_frames) if total_present_frames > 0 else 0.0
            frag_mode = "ground_truth"
        else:
            # Proxy mode if GT is unavailable: frame gaps where target class has no tracked ID.
            total_frames = len(ids)
            missed_frames = sum(1 for tid in ids if tid is None)
            frag_ratio = (missed_frames / total_frames) if total_frames > 0 else 0.0
            frag_mode = "proxy_no_gt"

        result = {
            "target_class_id": args.target_class_id,
            "frames_evaluated": len(ids),
            "id_switches": int(id_switches),
            "frag_ratio": float(frag_ratio),
            "frag_ratio_mode": frag_mode,
        }

    if args.output_json is None:
        output_json = args.tracking_csv.with_suffix(".tracking_metrics.json")
    else:
        output_json = args.output_json

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
