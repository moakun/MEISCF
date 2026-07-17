"""Tiled (SAHI-style) inference + evaluation for small-object recall.

VisDrone images are ~2000x1500 px but objects are tiny (63% < 32 px). Running
the detector on the whole image (even at 1280) shrinks those objects below the
P3 stride-8 grid, so they are missed (see the confusion matrix: 38-54% of small
classes land in 'background'). Slicing the image into overlapping tiles and
detecting on each tile makes small objects effectively larger, then we merge all
tile detections back into full-image coordinates with class-aware NMS.

This module is self-contained:
  * sliced prediction via Ultralytics ``model.predict`` on each tile,
  * cross-tile merge with torchvision ``batched_nms``,
  * a numpy COCO-style mAP (IoU 0.50:0.95, 101-point interpolation) computed
    directly against the YOLO-format ground-truth labels -- no pycocotools and
    no COCO-json conversion required.

To trust the metric, run it once with ``slice=None`` (full-image only) and check
the number matches Ultralytics ``model.val`` at the same imgsz (within ~0.5 pp);
the metric code is then validated and the tiled number is comparable.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .registry import register_meiscf_modules
from .data_prep import VISDRONE_NAMES

logger = logging.getLogger(__name__)

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
_IOU_THR = np.arange(0.5, 1.0, 0.05)          # COCO 0.50:0.05:0.95 (10 thresholds)

# np.trapz was renamed to np.trapezoid in NumPy 2.x (np.trapz removed).
_TRAPZ = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _box_iou(b1, b2):
    """IoU matrix between two sets of xyxy boxes. b1 [N,4], b2 [M,4] -> [N,M]."""
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)), dtype=np.float32)
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-16)


def _tile_windows(W, H, slice_size, overlap):
    """Return a list of (x0, y0, x1, y1) tile windows covering a W x H image."""
    if slice_size is None or (W <= slice_size and H <= slice_size):
        return [(0, 0, W, H)]
    step = max(int(slice_size * (1 - overlap)), 1)

    def origins(total):
        if total <= slice_size:
            return [0]
        pts = list(range(0, total - slice_size + 1, step))
        if pts[-1] != total - slice_size:
            pts.append(total - slice_size)        # ensure full coverage to edge
        return pts

    xs, ys = origins(W), origins(H)
    return [(x, y, min(x + slice_size, W), min(y + slice_size, H))
            for y in ys for x in xs]


# ---------------------------------------------------------------------------
# Sliced prediction
# ---------------------------------------------------------------------------
def sliced_predict(model, img_rgb, slice_size=640, overlap=0.25, full_image=True,
                   conf=0.001, nms_iou=0.6, max_det=1000, imgsz_full=1280,
                   per_tile_max_det=300):
    """Detect on overlapping tiles (+ optional full image), merge to global coords.

    Returns an [N,6] array: x1, y1, x2, y2, conf, cls (pixel coords on img_rgb).
    """
    H, W = img_rgb.shape[:2]
    do_tiles = slice_size is not None
    windows = _tile_windows(W, H, slice_size, overlap) if do_tiles else []
    all_boxes, all_scores, all_cls = [], [], []

    def _collect(res, ox, oy):
        b = res.boxes
        if b is None or b.shape[0] == 0:
            return
        xyxy = b.xyxy.cpu().numpy().copy()
        xyxy[:, [0, 2]] += ox
        xyxy[:, [1, 3]] += oy
        all_boxes.append(xyxy)
        all_scores.append(b.conf.cpu().numpy())
        all_cls.append(b.cls.cpu().numpy())

    # Tiles
    for (x0, y0, x1, y1) in windows:
        crop = img_rgb[y0:y1, x0:x1]
        r = model.predict(crop, imgsz=slice_size, conf=conf, iou=nms_iou,
                          max_det=per_tile_max_det, verbose=False)[0]
        _collect(r, x0, y0)

    # Full-image pass (SAHI "standard prediction"): recovers large objects a tile
    # may crop in half. Always run when not tiling (slice_size=None baseline).
    if full_image or not do_tiles:
        r = model.predict(img_rgb, imgsz=imgsz_full, conf=conf, iou=nms_iou,
                          max_det=per_tile_max_det, verbose=False)[0]
        _collect(r, 0, 0)

    if not all_boxes:
        return np.zeros((0, 6), dtype=np.float32)

    boxes = np.concatenate(all_boxes, 0)
    scores = np.concatenate(all_scores, 0)
    cls = np.concatenate(all_cls, 0)

    # Class-aware NMS merge across all tiles.
    from torchvision.ops import batched_nms
    keep = batched_nms(
        torch.from_numpy(boxes).float(),
        torch.from_numpy(scores).float(),
        torch.from_numpy(cls).long(), float(nms_iou)).numpy()
    keep = keep[:max_det]
    return np.concatenate(
        [boxes[keep], scores[keep, None], cls[keep, None]], 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Ground truth + metric
# ---------------------------------------------------------------------------
def _label_path_for(img_path):
    """YOLO convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt."""
    p = str(img_path)
    if '/images/' in p.replace('\\', '/'):
        p = p.replace('\\', '/').replace('/images/', '/labels/', 1)
    else:
        p = str(Path(img_path).parent.parent / 'labels' / Path(img_path).name)
    return Path(p).with_suffix('.txt')


def _read_gt(img_path, W, H):
    """Read YOLO label -> [M,5] array cls,x1,y1,x2,y2 in pixel coords."""
    lp = _label_path_for(img_path)
    if not lp.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in lp.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = (int(float(parts[0])), float(parts[1]), float(parts[2]),
                           float(parts[3]), float(parts[4]))
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        rows.append([c, x1, y1, x2, y2])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), np.float32)


def _ap(recall, precision):
    """COCO-style 101-point interpolated AP (matches Ultralytics 'interp')."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))   # precision envelope
    x = np.linspace(0, 1, 101)
    return float(_TRAPZ(np.interp(x, mrec, mpre), x))


def compute_map(per_image, nc, names=None):
    """COCO mAP from per-image (preds[N,6], gts[M,5]) pairs.

    Returns dict with mAP50, mAP50-95 and per-class AP50 / AP50-95.
    """
    names = names or [str(i) for i in range(nc)]
    T = len(_IOU_THR)
    confs = {c: [] for c in range(nc)}
    tps = {c: [] for c in range(nc)}      # each entry [n_det, T]
    n_gt = {c: 0 for c in range(nc)}

    for preds, gts in per_image:
        for c in range(nc):
            pc = preds[preds[:, 5] == c]
            gc = gts[gts[:, 0] == c]
            n_gt[c] += len(gc)
            if len(pc) == 0:
                continue
            pc = pc[np.argsort(-pc[:, 4])]            # sort by confidence desc
            tp = np.zeros((len(pc), T), dtype=np.float32)
            if len(gc):
                ious = _box_iou(pc[:, :4], gc[:, 1:5])
                for ti, thr in enumerate(_IOU_THR):
                    matched = np.zeros(len(gc), dtype=bool)
                    for pi in range(len(pc)):
                        row = ious[pi].copy()
                        row[matched] = -1.0
                        gi = int(np.argmax(row))
                        if row[gi] >= thr:
                            tp[pi, ti] = 1.0
                            matched[gi] = True
            confs[c].append(pc[:, 4])
            tps[c].append(tp)

    ap = np.full((nc, T), np.nan, dtype=np.float32)
    for c in range(nc):
        if n_gt[c] == 0:
            continue
        if not confs[c]:
            ap[c, :] = 0.0
            continue
        conf = np.concatenate(confs[c])
        tp = np.concatenate(tps[c], 0)
        order = np.argsort(-conf)
        tp = tp[order]
        for ti in range(T):
            tpc = np.cumsum(tp[:, ti])
            fpc = np.cumsum(1.0 - tp[:, ti])
            recall = tpc / (n_gt[c] + 1e-16)
            precision = tpc / (tpc + fpc + 1e-16)
            ap[c, ti] = _ap(recall, precision)

    per_class = {}
    for c in range(nc):
        per_class[names[c]] = {
            'AP50': (float(ap[c, 0]) if not np.isnan(ap[c, 0]) else None),
            'AP50-95': (float(np.nanmean(ap[c])) if not np.isnan(ap[c]).all() else None),
        }
    return {
        'mAP50': float(np.nanmean(ap[:, 0])),
        'mAP50-95': float(np.nanmean(ap)),
        'per_class': per_class,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _val_images(data_yaml):
    import yaml as _yaml
    cfg = _yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg.get('path', '.'))
    val = cfg.get('val', 'images/val')
    vdir = (root / val) if not Path(val).is_absolute() else Path(val)
    if not vdir.is_dir():
        vdir = Path(val)
    imgs = [p for p in sorted(vdir.iterdir()) if p.suffix.lower() in _IMG_EXTS]
    names = cfg.get('names', VISDRONE_NAMES)
    return imgs, names


def tiled_evaluate(model_path, data_yaml, slice_size=640, overlap=0.25,
                   full_image=True, conf=0.001, nms_iou=0.6, max_det=1000,
                   imgsz_full=1280, out_json=None, limit=None):
    """Run tiled inference over the val split and compute COCO mAP.

    Set slice_size=None to run full-image-only through the SAME metric (use this
    to validate the metric against Ultralytics model.val).
    """
    import cv2
    from ultralytics import YOLO
    register_meiscf_modules()
    model = YOLO(str(model_path))

    imgs, names = _val_images(data_yaml)
    if limit:
        imgs = imgs[:limit]
    nc = len(names)
    mode = 'full-image only' if slice_size is None else \
        f'tiled slice={slice_size} overlap={overlap} full_image={full_image}'
    logger.info(f"Tiled eval ({mode}) over {len(imgs)} val images ...")

    per_image = []
    for i, img_path in enumerate(imgs):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        preds = sliced_predict(model, rgb, slice_size=slice_size, overlap=overlap,
                               full_image=full_image, conf=conf, nms_iou=nms_iou,
                               max_det=max_det, imgsz_full=imgsz_full)
        gts = _read_gt(img_path, W, H)
        per_image.append((preds, gts))
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(imgs)} images processed")

    metrics = compute_map(per_image, nc, names=names)
    metrics['config'] = {
        'slice': slice_size, 'overlap': overlap, 'full_image': full_image,
        'conf': conf, 'nms_iou': nms_iou, 'max_det': max_det,
        'imgsz_full': imgsz_full, 'n_images': len(per_image),
    }
    logger.info(f"Tiled eval: mAP@50={metrics['mAP50']:.4f} "
                f"mAP@50-95={metrics['mAP50-95']:.4f}")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(metrics, indent=2))
        logger.info(f"Saved tiled evaluation -> {out_json}")
    return metrics
