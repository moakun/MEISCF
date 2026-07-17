"""Dataset preparation: VisDrone / UAVDT -> YOLO format + dataset statistics.

The original code assumed an already-converted dataset; it never converted the
native VisDrone annotation format. This module does the conversion and also
emits dataset statistics (class distribution, object-size distribution) that
feed the visualization stage.

VisDrone-DET annotation format (one .txt per image, one object per line):
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<trunc>,<occ>
category: 0=ignored 1=pedestrian 2=people 3=bicycle 4=car 5=van 6=truck
          7=tricycle 8=awning-tricycle 9=bus 10=motor 11=others
We keep categories 1-10 -> YOLO class ids 0-9, dropping 'ignored' and 'others'.

Expected raw VisDrone layout (as downloaded):
    <root>/VisDrone2019-DET-train/images/*.jpg
    <root>/VisDrone2019-DET-train/annotations/*.txt
    <root>/VisDrone2019-DET-val/...
    <root>/VisDrone2019-DET-test-dev/...   (optional)
"""

import json
import logging
from pathlib import Path
from collections import Counter

import yaml

logger = logging.getLogger(__name__)

VISDRONE_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                  'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

# VisDrone category id -> YOLO class id (1..10 -> 0..9). 0 and 11 are dropped.
_VISDRONE_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def _read_image_size(img_path):
    """Return (width, height) for an image, using PIL (no full decode)."""
    from PIL import Image
    with Image.open(img_path) as im:
        return im.size  # (w, h)


def _convert_split(raw_split_dir, out_img_dir, out_lbl_dir, link=True):
    """Convert one VisDrone split. Returns per-class object counts + size list."""
    raw_split_dir = Path(raw_split_dir)
    img_src = raw_split_dir / 'images'
    ann_src = raw_split_dir / 'annotations'
    if not img_src.is_dir():
        raise FileNotFoundError(f"Missing images dir: {img_src}")
    if not ann_src.is_dir():
        raise FileNotFoundError(f"Missing annotations dir: {ann_src}")

    out_img_dir = Path(out_img_dir); out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir = Path(out_lbl_dir); out_lbl_dir.mkdir(parents=True, exist_ok=True)

    class_counts = Counter()
    obj_sizes = []          # sqrt(w*h) in pixels, for size-distribution analysis
    n_images = n_objects = 0

    images = [p for p in sorted(img_src.iterdir()) if p.suffix.lower() in _IMG_EXTS]
    for img_path in images:
        ann_path = ann_src / (img_path.stem + '.txt')
        try:
            W, H = _read_image_size(img_path)
        except Exception as e:
            logger.warning(f"Skipping unreadable image {img_path.name}: {e}")
            continue

        lines_out = []
        if ann_path.exists():
            for raw in ann_path.read_text().splitlines():
                raw = raw.strip().rstrip(',')
                if not raw:
                    continue
                parts = raw.split(',')
                if len(parts) < 6:
                    continue
                try:
                    x, y, w, h = (float(parts[0]), float(parts[1]),
                                  float(parts[2]), float(parts[3]))
                    cat = int(parts[5])
                except ValueError:
                    continue
                if cat not in _VISDRONE_MAP:
                    continue                      # ignored region / 'others'
                if w <= 0 or h <= 0:
                    continue
                cls = _VISDRONE_MAP[cat]
                # Clip to image bounds, then normalize to YOLO cxcywh.
                x2, y2 = min(x + w, W), min(y + h, H)
                x, y = max(x, 0.0), max(y, 0.0)
                bw, bh = x2 - x, y2 - y
                if bw <= 1 or bh <= 1:
                    continue
                cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
                nw, nh = bw / W, bh / H
                lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                class_counts[cls] += 1
                obj_sizes.append((bw * bh) ** 0.5)
                n_objects += 1

        # Write label (empty file = background image, valid for YOLO).
        (out_lbl_dir / (img_path.stem + '.txt')).write_text("\n".join(lines_out))
        _place_image(img_path, out_img_dir / img_path.name, link)
        n_images += 1

    logger.info(f"  {raw_split_dir.name}: {n_images} images, {n_objects} objects")
    return {'class_counts': dict(class_counts), 'obj_sizes': obj_sizes,
            'n_images': n_images, 'n_objects': n_objects}


def _place_image(src, dst, link=True):
    """Symlink (default) or copy an image into the YOLO tree to save disk space."""
    import os
    import shutil
    if dst.exists():
        return
    if link:
        try:
            os.symlink(src.resolve(), dst)
            return
        except (OSError, NotImplementedError):
            pass  # symlink not permitted (e.g. some filesystems) -> fall back
    shutil.copy2(src, dst)


def convert_visdrone(raw_root, out_root='VisDrone_YOLO', link_images=True,
                     splits=('train', 'val')):
    """Convert a raw VisDrone download to YOLO format and write VisDrone.yaml.

    Parameters
    ----------
    raw_root    : directory containing VisDrone2019-DET-{train,val,test-dev}
    out_root    : output YOLO dataset root
    link_images : symlink images instead of copying (Linux server: recommended)
    splits      : which splits to convert ('train','val','test')

    Returns the path to the generated dataset YAML.
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    split_dirs = {
        'train': 'VisDrone2019-DET-train',
        'val':   'VisDrone2019-DET-val',
        'test':  'VisDrone2019-DET-test-dev',
    }

    # Fail fast with an actionable message rather than crashing later when the
    # output yaml can't be written (the usual cause: config.yaml still has the
    # placeholder path '/path/to/server/VisDrone').
    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Raw dataset root does not exist: '{raw_root}'.\n"
            f"  -> Edit paths.raw_dataset in config.yaml to your actual VisDrone "
            f"download (the folder containing VisDrone2019-DET-train/, "
            f"VisDrone2019-DET-val/, each with images/ and annotations/)."
        )

    logger.info(f"Converting VisDrone: {raw_root} -> {out_root}")
    stats = {}
    for split in splits:
        raw_split = raw_root / split_dirs[split]
        if not raw_split.is_dir():
            logger.warning(f"Split '{split}' not found at {raw_split}; skipping.")
            continue
        stats[split] = _convert_split(
            raw_split,
            out_root / 'images' / split,
            out_root / 'labels' / split,
            link=link_images,
        )

    if not stats:
        found = sorted(p.name for p in raw_root.iterdir() if p.is_dir()) \
            if raw_root.is_dir() else []
        raise FileNotFoundError(
            f"No VisDrone splits found under '{raw_root}'. Expected one of "
            f"{list(split_dirs.values())}, each containing images/ and "
            f"annotations/.\n  Subfolders actually present: {found or '(none)'}\n"
            f"  -> Check paths.raw_dataset in config.yaml, or point it at your "
            f"already-converted YOLO dataset (with images/train + labels/train)."
        )

    yaml_path = write_dataset_yaml(out_root, splits=[s for s in splits if s in stats])
    (out_root / 'dataset_stats.json').write_text(json.dumps(stats, indent=2))
    logger.info(f"Conversion complete. dataset.yaml -> {yaml_path}")
    return yaml_path


def write_dataset_yaml(out_root, splits=('train', 'val'), names=None):
    """Write the Ultralytics dataset YAML for a converted YOLO dataset."""
    out_root = Path(out_root)
    names = names or VISDRONE_NAMES
    cfg = {
        'path': str(out_root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(names),
        'names': names,
    }
    if 'test' in splits:
        cfg['test'] = 'images/test'
    yaml_path = out_root / 'VisDrone.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
    return yaml_path


def dataset_already_yolo(path):
    """Heuristic: does `path` already look like a converted YOLO dataset?"""
    path = Path(path)
    return (path / 'images' / 'train').is_dir() and (path / 'labels' / 'train').is_dir()


def ensure_dataset(raw_or_yolo_root, out_root='VisDrone_YOLO', link_images=True):
    """Idempotent entry point: convert if needed, else reuse existing YOLO tree.

    Returns the dataset YAML path.
    """
    src = Path(raw_or_yolo_root)
    # Case 1: already a YOLO dataset with a yaml -> reuse.
    if dataset_already_yolo(src):
        existing = src / 'VisDrone.yaml'
        if existing.exists():
            logger.info(f"Using existing YOLO dataset: {existing}")
            return existing
        return write_dataset_yaml(src)
    # Case 2: output already converted from a previous run.
    if dataset_already_yolo(Path(out_root)):
        existing = Path(out_root) / 'VisDrone.yaml'
        if existing.exists():
            logger.info(f"Using previously converted dataset: {existing}")
            return existing
    # Case 3: raw VisDrone -> convert.
    return convert_visdrone(src, out_root=out_root, link_images=link_images)
