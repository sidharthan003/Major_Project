# Baseline Sonar Detection Framework (Standard YOLOv8n)

This package sets up a standard, unmodified YOLO baseline for sonar target detection and tracking.

## What This Baseline Covers

- Backbone/Neck/Head: standard Ultralytics YOLOv8n architecture.
- Tracker adapter: raw YOLO boxes directly passed to tracker (`ByteTrack`) with no expansion.
- Outputs for fusion layer input: bounding boxes, class IDs, confidence scores.
- Baseline metrics: mAP@0.5, recall, FPS, Frag Ratio, ID switches.

## Folder Layout

- `scripts/prepare_yolo_dataset.py`: Converts Pascal VOC XML to YOLO format and creates split files.
- `scripts/train_baseline.py`: Trains and validates `yolov8n.pt` on your sonar dataset.
- `scripts/infer_track_export.py`: Runs detection+tracking and exports per-frame tracking CSV.
- `scripts/compute_tracking_metrics.py`: Computes Frag Ratio and ID switch metrics from tracking output.

## 1) Setup

```powershell
cd "e:\workout_programs\MAJOR PROJECT\FLS\baseline_sonar_yolo"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Prepare YOLO Dataset

```powershell
python scripts/prepare_yolo_dataset.py `
  --dataset-root "e:\workout_programs\MAJOR PROJECT\FLS\FLS_Detection_Dataset" `
  --output-root "e:\workout_programs\MAJOR PROJECT\FLS\baseline_sonar_yolo\data\yolo_sonar" `
  --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```

Expected outputs:

- `data/yolo_sonar/images/{train,val,test}`
- `data/yolo_sonar/labels/{train,val,test}`
- `data/yolo_sonar/data.yaml`

## 3) Train Baseline YOLOv8n

```powershell
python scripts/train_baseline.py `
  --data-yaml "e:\workout_programs\MAJOR PROJECT\FLS\baseline_sonar_yolo\data\yolo_sonar\data.yaml" `
  --model yolov8n.pt `
  --imgsz 640 --epochs 100 --batch 16 --device auto
```

The script also runs validation and stores a baseline metrics JSON.

## 4) Inference + Tracking + Export

```powershell
python scripts/infer_track_export.py `
  --weights "e:\workout_programs\MAJOR PROJECT\FLS\baseline_sonar_yolo\runs\detect\sonar_baseline\weights\best.pt" `
  --source "e:\workout_programs\MAJOR PROJECT\FLS\FLS_Detection_Dataset\JPEGImages" `
  --output-csv "e:\workout_programs\MAJOR PROJECT\FLS\baseline_sonar_yolo\outputs\tracking_baseline.csv" `
  --tracker bytetrack.yaml --conf 0.25 --iou 0.5
```

CSV fields:

- `frame_idx, image_name, track_id, class_id, class_name, conf, x1, y1, x2, y2`

Also writes an FPS summary JSON next to the CSV.

## 5) Compute Frag Ratio and ID Switches

If you have frame-level ground truth presence for the class of interest:

```csv
frame,present
0,1
1,1
2,0
```

Run:

```powershell
python scripts/compute_tracking_metrics.py `
  --tracking-csv "e:\workout_programs\MAJOR PROJECT\FLS\baseline_sonar_yolo\outputs\tracking_baseline.csv" `
  --target-class-id 0 `
  --ground-truth-presence "e:\workout_programs\MAJOR PROJECT\FLS\baseline_sonar_yolo\outputs\gt_presence.csv"
```

If ground truth presence is not provided, the script reports a proxy Frag Ratio based on observed frame gaps.

## Metrics Alignment to Baseline Spec

- Detection accuracy: mAP@0.5 from YOLO validation.
- Life-critical sensitivity: recall from YOLO validation.
- Real-time throughput: FPS from `infer_track_export.py`.
- Tracking continuity: Frag Ratio and ID switches from `compute_tracking_metrics.py`.
- Heatmap focus: use your preferred Grad-CAM workflow as a separate analysis pass on the trained model.

## Notes

- Class order is sourced from `label_list.txt` (current: `victim`, `boat`, `plane`).
- This baseline intentionally keeps architecture and tracking handoff standard (no custom blocks, no box inflation).
