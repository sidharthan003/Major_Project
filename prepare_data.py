"""
prepare_data.py
---------------
Converts the COCO JSON annotations from the thermal splits into YOLO-format .txt
label files, keeping only the 'person' class.

Directory layout expected:
    images_thermal_train/
        coco.json
        data/   <-- JPG frames
    images_thermal_val/
        coco.json
        data/   <-- JPG frames

Output (created next to this script):
    dataset/
        images/
            train/   (symlinks or copies to original JPGs)
            val/
        labels/
            train/   (YOLO .txt files)
            val/
"""

import json
import os
import shutil
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TRAIN_JSON = BASE_DIR / "images_thermal_train" / "coco.json"
VAL_JSON   = BASE_DIR / "images_thermal_val"   / "coco.json"
TRAIN_IMGS = BASE_DIR / "images_thermal_train" / "data"
VAL_IMGS   = BASE_DIR / "images_thermal_val"   / "data"

OUT_DIR    = BASE_DIR / "dataset"

TARGET_CLASS_NAME = "person"   # change to None to keep ALL classes


# ── helpers ────────────────────────────────────────────────────────────────────

def convert_coco_to_yolo(coco_json_path: Path,
                          src_img_dir: Path,
                          out_img_dir: Path,
                          out_lbl_dir: Path,
                          target_class: str | None = "person") -> None:
    """
    Reads a COCO-format JSON, writes one YOLO label file per image,
    and copies (or symlinks) images into out_img_dir.

    If target_class is set, only annotations for that class are written and
    the YOLO class id is always 0.  If None, all classes are kept and their
    original category order is used.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Loading {coco_json_path} …")
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build category lookup
    categories: list[dict] = coco["categories"]
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    if target_class:
        # Map the target category id → yolo class 0
        target_ids = {c["id"] for c in categories if c["name"] == target_class}
        if not target_ids:
            raise ValueError(f"Class '{target_class}' not found in {coco_json_path}")
        cat_id_to_yolo = {cid: 0 for cid in target_ids}
        print(f"[INFO] Filtering to '{target_class}' (category ids: {target_ids})")
    else:
        # Keep all classes; sort by id for a stable mapping
        sorted_ids = sorted(cat_id_to_name.keys())
        cat_id_to_yolo = {cid: idx for idx, cid in enumerate(sorted_ids)}
        print(f"[INFO] Keeping all {len(sorted_ids)} classes")

    # Build image id → filename map
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image id
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in cat_id_to_yolo:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    copied = 0
    skipped = 0
    labels_written = 0

    for img_id, ann_list in anns_by_image.items():
        img_info = img_id_to_info[img_id]
        filename = img_info["file_name"]
        img_w    = img_info["width"]
        img_h    = img_info["height"]

        src_path = src_img_dir / filename
        if not src_path.exists():
            # Some datasets store bare filenames; try stripping sub-dirs
            src_path = src_img_dir / Path(filename).name
        if not src_path.exists():
            skipped += 1
            continue

        # Copy image
        dst_img = out_img_dir / src_path.name
        if not dst_img.exists():
            shutil.copy2(src_path, dst_img)
            copied += 1

        # Write YOLO label (one row per annotation)
        stem     = src_path.stem
        lbl_path = out_lbl_dir / f"{stem}.txt"
        lines    = []
        for ann in ann_list:
            x, y, bw, bh = ann["bbox"]          # COCO: [x_min, y_min, w, h]
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            cls = cat_id_to_yolo[ann["category_id"]]
            # Clamp to [0, 1] to guard against any annotation overshoot
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(lbl_path, "w") as lf:
            lf.write("\n".join(lines))
        labels_written += 1

    print(f"[INFO] Images copied : {copied}")
    print(f"[INFO] Images skipped: {skipped}  (source file not found)")
    print(f"[INFO] Label files   : {labels_written}")


def write_image_list(img_dir: Path, out_txt: Path) -> None:
    """Write a plain-text list of image paths (used by some YOLO trainers)."""
    paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    out_txt.write_text("\n".join(str(p.resolve()) for p in paths))
    print(f"[INFO] Image list → {out_txt}  ({len(paths)} images)")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Thermal Person Detection — Data Preparation")
    print("=" * 60)

    # Train split
    convert_coco_to_yolo(
        coco_json_path = TRAIN_JSON,
        src_img_dir    = TRAIN_IMGS,
        out_img_dir    = OUT_DIR / "images" / "train",
        out_lbl_dir    = OUT_DIR / "labels" / "train",
        target_class   = TARGET_CLASS_NAME,
    )

    # Validation split
    convert_coco_to_yolo(
        coco_json_path = VAL_JSON,
        src_img_dir    = VAL_IMGS,
        out_img_dir    = OUT_DIR / "images" / "val",
        out_lbl_dir    = OUT_DIR / "labels" / "val",
        target_class   = TARGET_CLASS_NAME,
    )

    # Convenience image-list .txt files
    write_image_list(OUT_DIR / "images" / "train", OUT_DIR / "train.txt")
    write_image_list(OUT_DIR / "images" / "val",   OUT_DIR / "val.txt")

    print("\n[DONE] Dataset prepared at:", OUT_DIR)
    print("       Run  python train.py  to begin training.")


if __name__ == "__main__":
    main()
