# Magnum Inceptum

Multi-module computer vision project for human/target detection and tracking across two sensing domains:

- ADAS: thermal human detection on FLIR ADAS imagery (YOLO-based)
- FLS: forward-looking sonar target detection/tracking (baseline YOLO and custom Swin-enhanced variants)

## Repository Layout

```text
Magnum_Inceptum/
├── ADAS/
│   ├── detect.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── train_26.py
│   ├── train_new_improved.py
│   ├── thermal_person.yaml
│   ├── requirements.txt
│   ├── README.md
│   ├── images_* (raw dataset folders)
│   ├── dataset/ (generated YOLO-format data)
│   └── runs/ (training and inference outputs)
│
└── FLS/
    ├── baseline_sonar_yolo/
    │   ├── scripts/
    │   │   ├── prepare_yolo_dataset.py
    │   │   ├── train_baseline.py
    │   │   ├── infer_track_export.py
    │   │   └── compute_tracking_metrics.py
    │   ├── data/
    │   ├── runs/
    │   ├── requirements.txt
    │   └── README.md
    │
    ├── custom_c2f_swin_sonar/
    │   ├── configs/
    │   ├── scripts/
    │   ├── src/
    │   ├── tests/
    │   ├── requirements.txt
    │   ├── pytest.ini
    │   └── README.md
    │
    └── FLS_Detection_Dataset/
        ├── Annotations/
        ├── JPEGImages/
        ├── label_list.txt
        └── labelmap.txt
```

## Module 1: ADAS Thermal Human Detection

Purpose:
- Detect humans in thermal (IR) frames/video
- Train YOLO models on FLIR ADAS thermal annotations (person-only by default)

Main files:
- `prepare_data.py`: COCO JSON to YOLO labels conversion
- `train.py`: standard thermal training entry point
- `train_26.py`: YOLO26 variant training
- `train_new_improved.py`: YOLOv8s-focused training profile
- `detect.py`: inference on images, video, webcam, RTSP streams

### ADAS Setup

From repo root:

```powershell
cd ADAS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### ADAS Data Preparation

```powershell
python prepare_data.py
```

This generates YOLO-format labels/images under `ADAS/dataset/`.

### ADAS Training

```powershell
python train.py
```

Common variants:

```powershell
python train.py --epochs 100 --batch 8 --weights yolov8s.pt
python train_26.py --epochs 50 --weights yolo26n.pt
python train_new_improved.py
```

### ADAS Inference

```powershell
python detect.py --weights runs/detect/thermal_person/weights/best.pt --source images_thermal_val/data
```

## Module 2: FLS Sonar Detection and Tracking

FLS includes two tracks:

1. Baseline sonar pipeline (`FLS/baseline_sonar_yolo`)
- Standard YOLOv8n architecture
- ByteTrack pipeline
- Tracking export and continuity metrics scripts

2. Custom architecture track (`FLS/custom_c2f_swin_sonar`)
- Swin-enhanced backbone experiments
- Sonar-specific model/tracking components and tests

### FLS Baseline Setup

```powershell
cd FLS\baseline_sonar_yolo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### FLS Baseline Workflow

Prepare YOLO dataset from Pascal VOC style annotations:

```powershell
python scripts/prepare_yolo_dataset.py --dataset-root "..\FLS_Detection_Dataset" --output-root "data\yolo_sonar" --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```

Train baseline model:

```powershell
python scripts/train_baseline.py --data-yaml "data\yolo_sonar\data.yaml" --model yolov8n.pt --imgsz 640 --epochs 100 --batch 16 --device auto
```

Run detection and tracking export:

```powershell
python scripts/infer_track_export.py --weights "runs\detect\sonar_baseline\weights\best.pt" --source "..\FLS_Detection_Dataset\JPEGImages" --output-csv "outputs\tracking_baseline.csv"
```

Compute tracking metrics:

```powershell
python scripts/compute_tracking_metrics.py --tracking-csv "outputs\tracking_baseline.csv" --target-class-id 0
```

## Datasets

- ADAS uses FLIR thermal data splits under `ADAS/images_thermal_*` and related index/annotation files.
- FLS uses dataset assets under `FLS/FLS_Detection_Dataset` (VOC XML in `Annotations`, images in `JPEGImages`).

## Outputs

Typical generated outputs:

- `runs/` directories for training/inference artifacts
- `results.csv` and plots for training metrics
- tracking CSV/JSON files for FLS baseline metrics

## Environment

Recommended:

- Python 3.10+
- GPU-enabled PyTorch for training performance
- Windows PowerShell commands shown above (project currently organized for Windows paths)

## Notes

- For module-specific details and advanced parameters, refer to:
  - `ADAS/README.md`
  - `FLS/baseline_sonar_yolo/README.md`
  - `FLS/custom_c2f_swin_sonar/README.md`
- Keep large raw datasets and model artifacts managed carefully to avoid oversized commits.
