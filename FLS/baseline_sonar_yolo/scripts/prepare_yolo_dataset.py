import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC sonar dataset to YOLO format.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to FLS_Detection_Dataset")
    parser.add_argument("--output-root", type=Path, required=True, help="Output folder for YOLO dataset")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_class_names(dataset_root: Path) -> list[str]:
    label_file = dataset_root / "label_list.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"Missing label list: {label_file}")
    classes = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError("No classes found in label_list.txt")
    return classes


def convert_bbox(size_w: float, size_h: float, xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float, float, float]:
    x_center = ((xmin + xmax) / 2.0) / size_w
    y_center = ((ymin + ymax) / 2.0) / size_h
    width = (xmax - xmin) / size_w
    height = (ymax - ymin) / size_h
    return x_center, y_center, width, height


def parse_voc(xml_path: Path, class_to_idx: dict[str, int]) -> tuple[str, list[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    if not filename:
        filename = f"{xml_path.stem}.jpg"

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing size tag in {xml_path}")
    width = float(size_node.findtext("width", "0"))
    height = float(size_node.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    yolo_lines: list[str] = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name", "").strip()
        if class_name not in class_to_idx:
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))

        x_center, y_center, bw, bh = convert_bbox(width, height, xmin, ymin, xmax, ymax)
        if bw <= 0 or bh <= 0:
            continue

        class_idx = class_to_idx[class_name]
        yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    return filename, yolo_lines


def split_items(items: list[Path], train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must equal 1.0")

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def ensure_layout(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_and_write(
    split_name: str,
    xml_files: list[Path],
    annotations_dir: Path,
    images_dir: Path,
    output_root: Path,
    class_to_idx: dict[str, int],
) -> int:
    copied = 0
    for xml_path in xml_files:
        full_xml = annotations_dir / xml_path.name
        image_name, yolo_lines = parse_voc(full_xml, class_to_idx)

        src_image = images_dir / image_name
        if not src_image.exists():
            alt = images_dir / f"{full_xml.stem}.jpg"
            if alt.exists():
                src_image = alt
            else:
                continue

        dst_img = output_root / "images" / split_name / src_image.name
        dst_lbl = output_root / "labels" / split_name / f"{src_image.stem}.txt"

        shutil.copy2(src_image, dst_img)
        dst_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")
        copied += 1

    return copied


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_root = args.dataset_root
    annotations_dir = dataset_root / "Annotations"
    images_dir = dataset_root / "JPEGImages"

    if not annotations_dir.exists() or not images_dir.exists():
        raise FileNotFoundError("Dataset root must contain Annotations/ and JPEGImages/")

    classes = load_class_names(dataset_root)
    class_to_idx = {name: i for i, name in enumerate(classes)}

    output_root = args.output_root
    ensure_layout(output_root)

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {annotations_dir}")

    random.shuffle(xml_files)
    train_xml, val_xml, test_xml = split_items(xml_files, args.train_ratio, args.val_ratio, args.test_ratio)

    n_train = copy_and_write("train", train_xml, annotations_dir, images_dir, output_root, class_to_idx)
    n_val = copy_and_write("val", val_xml, annotations_dir, images_dir, output_root, class_to_idx)
    n_test = copy_and_write("test", test_xml, annotations_dir, images_dir, output_root, class_to_idx)

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": classes,
    }

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print("YOLO dataset prepared.")
    print(f"Classes: {classes}")
    print(f"Train images: {n_train}")
    print(f"Val images: {n_val}")
    print(f"Test images: {n_test}")
    print(f"Data YAML: {yaml_path}")


if __name__ == "__main__":
    main()
