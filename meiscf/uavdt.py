"""UAVDT -> YOLO conversion and zero-shot evaluation of VisDrone-trained models.

Expected layout of the official download:

    <root>/UAV-benchmark-M/M0101/img000001.jpg ...
    <root>/UAV-benchmark-MOTD_v1.0/GT/M0101_gt_whole.txt
    <root>/UAV-benchmark-MOTD_v1.0/GT/M0101_gt_ignore.txt
    <root>/M_attr/{train,test}/M0101_attr.txt            (optional; gives the split)

Ground truth (``*_gt_whole.txt``), one object per line:

    frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
    out_of_view, occlusion, object_category

with object_category 1=car, 2=truck, 3=bus. ``*_gt_ignore.txt`` uses the same
first six fields and marks regions of vehicles too small to annotate; the
benchmark excludes them from evaluation.

Labels are written with VisDrone class ids (car 3, truck 5, bus 8) so that a
VisDrone-trained detector can be validated directly: Ultralytics scores only the
classes that occur in the ground truth, so its other seven classes are ignored.
Two adjustments make the zero-shot comparison fair, both applied at evaluation
time rather than by editing images or labels:

  * detections that lie mostly inside an ignore region are dropped;
  * UAVDT has no van class, so VisDrone "van" detections are relabelled as car
    and non-maximum suppression is re-applied to the merged class.
"""

import json
import logging
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

from .registry import register_meiscf_modules
from .significance import _identity

logger = logging.getLogger(__name__)

# UAVDT category id -> VisDrone/YOLO class id used by our detectors.
UAVDT_TO_VISDRONE = {1: 3, 2: 5, 3: 8}       # car, truck, bus
UAVDT_NAMES = {3: 'car', 5: 'truck', 8: 'bus'}
VAN_CLASS, CAR_CLASS = 4, 3

# The 20 detection test sequences of the official split (the other 30 M
# sequences are the training set). Used when the M_attr folder is absent.
TEST_SEQUENCES = (
    'M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606',
    'M0701', 'M0801', 'M0802', 'M1001', 'M1004', 'M1007', 'M1009', 'M1101',
    'M1301', 'M1302', 'M1303', 'M1401',
)


def _find_dirs(raw_root):
    raw_root = Path(raw_root)
    img_root = next((raw_root / n for n in ('UAV-benchmark-M', 'UAVDT_Benchmark_M',
                                            'UAV-benchmark-M/UAV-benchmark-M')
                     if (raw_root / n).is_dir()), None)
    gt_root = next((p for p in (raw_root / 'UAV-benchmark-MOTD_v1.0' / 'GT',
                                raw_root / 'GT') if p.is_dir()), None)
    if img_root is None or gt_root is None:
        present = sorted(p.name for p in raw_root.iterdir()) if raw_root.is_dir() else []
        raise FileNotFoundError(
            f"UAVDT not found under '{raw_root}'. Expected UAV-benchmark-M/ with one "
            f"folder per sequence and UAV-benchmark-MOTD_v1.0/GT/ with *_gt_whole.txt.\n"
            f"  Present: {present or '(nothing)'}")
    return img_root, gt_root


def _split_sequences(raw_root, img_root, split):
    """Sequence ids for a split: from M_attr if present, else the official list.

    split='all' takes every sequence present. Nothing here is trained on UAVDT,
    so the official train/test division does not constrain a zero-shot evaluation,
    and using every sequence widens the coverage of trucks, buses and object sizes.
    """
    attr = Path(raw_root) / 'M_attr' / split
    available = sorted(p.name for p in img_root.iterdir() if p.is_dir())
    if split == 'all':
        logger.info(f"Using all {len(available)} sequences present: {' '.join(available)}")
        return available
    if attr.is_dir():
        seqs = sorted(p.name.split('_attr')[0] for p in attr.glob('*_attr.txt'))
        if seqs:
            logger.info(f"Split '{split}' from {attr}: {len(seqs)} sequences.")
            return [s for s in seqs if s in available]
    seqs = ([s for s in available if s in TEST_SEQUENCES] if split == 'test'
            else [s for s in available if s not in TEST_SEQUENCES])
    logger.info(f"No M_attr folder; using the published {split} list: {len(seqs)} of "
                f"{len(available)} sequences present ({' '.join(seqs)}). Sequences "
                f"available but not in this split: "
                f"{' '.join(s for s in available if s not in seqs) or 'none'}")
    return seqs


def _parse_gt(path):
    """frame index -> [(visdrone_cls, x, y, w, h), ...] from *_gt_whole.txt."""
    out = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.strip().rstrip(',').split(',')
        if len(parts) < 9:
            continue
        try:
            frame = int(parts[0])
            x, y, w, h = (float(v) for v in parts[2:6])
            cat = int(parts[8])
        except ValueError:
            continue
        if cat in UAVDT_TO_VISDRONE and w > 0 and h > 0:
            out[frame].append((UAVDT_TO_VISDRONE[cat], x, y, w, h))
    return out


def _parse_ignore(path):
    """frame index -> [[x1, y1, x2, y2], ...] from *_gt_ignore.txt."""
    out = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.strip().rstrip(',').split(',')
        if len(parts) < 6:
            continue
        try:
            frame = int(parts[0])
            x, y, w, h = (float(v) for v in parts[2:6])
        except ValueError:
            continue
        if w > 0 and h > 0:
            out[frame].append([x, y, x + w, y + h])
    return out


def _ioa(boxes, regions):
    """Intersection over box area, for each box against every region."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros((len(boxes), max(len(regions), 1)), dtype=np.float32)
    b = np.asarray(boxes, dtype=np.float64)[:, None, :]
    r = np.asarray(regions, dtype=np.float64)[None, :, :]
    iw = np.clip(np.minimum(b[..., 2], r[..., 2]) - np.maximum(b[..., 0], r[..., 0]), 0, None)
    ih = np.clip(np.minimum(b[..., 3], r[..., 3]) - np.maximum(b[..., 1], r[..., 1]), 0, None)
    area = np.clip((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]), 1e-9, None)
    return (iw * ih) / area


def convert_uavdt(raw_root, out_root='UAVDT_YOLO', split='test', frame_stride=5,
                  link_images=True, ignore_ioa=0.5):
    """Convert UAVDT to a YOLO dataset with VisDrone class ids.

    frame_stride : keep every n-th annotated frame (UAVDT is 30 fps video, so
                   consecutive frames are nearly identical).
    """
    from .data_prep import _place_image, VISDRONE_NAMES
    from PIL import Image
    import yaml as _yaml

    raw_root = Path(raw_root)
    img_root, gt_root = _find_dirs(raw_root)
    sequences = _split_sequences(raw_root, img_root, split)
    if not sequences:
        raise FileNotFoundError(f"No sequences found for split '{split}' under {img_root}")

    out_root = Path(out_root)
    out_img = out_root / 'images' / split
    out_lbl = out_root / 'labels' / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    sizes, ignore_map, per_seq = [], {}, {}
    n_images = n_boxes = n_dropped = n_ignore_regions = 0

    for seq in sequences:
        frames = sorted((img_root / seq).glob('img*.jpg'))
        gt = _parse_gt(gt_root / f'{seq}_gt_whole.txt')
        ig = _parse_ignore(gt_root / f'{seq}_gt_ignore.txt')
        if not gt:
            logger.warning(f"{seq}: no ground truth in {gt_root}; skipping sequence.")
            continue
        kept = 0
        for frame_path in frames[::frame_stride]:
            try:
                idx = int(frame_path.stem[3:])
            except ValueError:
                continue
            with Image.open(frame_path) as im:
                W, H = im.size
            regions = [[max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)]
                       for x1, y1, x2, y2 in ig.get(idx, [])]
            lines = []
            for cls, x, y, w, h in gt.get(idx, []):
                x1, y1 = max(x, 0.0), max(y, 0.0)
                x2, y2 = min(x + w, float(W)), min(y + h, float(H))
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1 or bh <= 1:
                    continue
                if regions and _ioa([[x1, y1, x2, y2]], regions).max() >= ignore_ioa:
                    n_dropped += 1          # annotated inside an ignore region
                    continue
                lines.append(f"{cls} {(x1 + bw / 2) / W:.6f} {(y1 + bh / 2) / H:.6f} "
                             f"{bw / W:.6f} {bh / H:.6f}")
                counts[cls] += 1
                sizes.append(float((bw * bh) ** 0.5))
                n_boxes += 1
            stem = f"{seq}_{idx:06d}"
            (out_lbl / f"{stem}.txt").write_text("\n".join(lines))
            _place_image(frame_path, out_img / f"{stem}.jpg", link_images)
            if regions:
                ignore_map[stem] = [[round(v, 2) for v in r] for r in regions]
                n_ignore_regions += len(regions)
            n_images += 1
            kept += 1
        per_seq[seq] = {'frames_total': len(frames), 'frames_kept': kept}

    yaml_path = out_root / 'UAVDT.yaml'
    cfg = {'path': str(out_root.resolve()), 'train': f'images/{split}',
           'val': f'images/{split}', 'nc': len(VISDRONE_NAMES), 'names': VISDRONE_NAMES}
    with open(yaml_path, 'w') as f:
        _yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
    ignore_path = out_root / f'ignore_regions_{split}.json'
    ignore_path.write_text(json.dumps(ignore_map))
    stats = {'split': split, 'sequences': len(per_seq), 'frame_stride': frame_stride,
             'images': n_images, 'boxes': n_boxes,
             'class_counts': {UAVDT_NAMES[c]: n for c, n in sorted(counts.items())},
             'boxes_dropped_in_ignore_regions': n_dropped,
             'images_with_ignore_regions': len(ignore_map),
             'ignore_regions': n_ignore_regions,
             'object_size_px': {
                 'median': float(np.median(sizes)) if sizes else None,
                 'share_below_32px': float(np.mean(np.array(sizes) < 32)) if sizes else None,
                 'share_below_16px': float(np.mean(np.array(sizes) < 16)) if sizes else None},
             'per_sequence': per_seq}
    (out_root / f'uavdt_stats_{split}.json').write_text(json.dumps(stats, indent=2))
    logger.info(f"UAVDT {split}: {len(per_seq)} sequences, {n_images} images, {n_boxes} boxes "
                f"({stats['class_counts']}), {n_ignore_regions} ignore regions, "
                f"{n_dropped} boxes dropped inside them -> {yaml_path}")
    return yaml_path, ignore_path, stats


class _UavdtEvalHooks:
    """Drops detections inside ignore regions and merges van into car."""

    def __init__(self, ignore_map, merge_van=True, nms_iou=0.5, ioa_thr=0.5):
        self.ignore = ignore_map
        self.merge_van = merge_van
        self.nms_iou = nms_iou
        self.ioa_thr = ioa_thr
        self.n_dropped = self.n_merged = self.n_images = 0
        self.stems = []          # dataloader order, for grouping images by sequence

    def __call__(self, validator):
        orig_update = validator.update_metrics

        def update_metrics(preds, batch):
            for si, pred in enumerate(preds):
                keep = self._image_mask(pred, batch, si)
                if keep is not None:
                    _select(pred, keep)
                self.stems.append(Path(str(batch['im_file'][si])).stem)
                self.n_images += 1
            return orig_update(preds, batch)

        validator.update_metrics = update_metrics

    def _image_mask(self, pred, batch, si):
        import torch
        boxes, conf, cls = _unpack(pred)
        if boxes is None or len(boxes) == 0:
            return None
        if self.merge_van:
            van = cls == VAN_CLASS
            if van.any():
                cls[van] = CAR_CLASS
                self.n_merged += int(van.sum())
                car = torch.nonzero(cls == CAR_CLASS).flatten()
                from torchvision.ops import nms
                keep_car = car[nms(boxes[car].float(), conf[car].float(), self.nms_iou)]
                mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
                mask[car] = False
                mask[keep_car] = True
            else:
                mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
        else:
            mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)

        stem = Path(str(batch['im_file'][si])).stem
        regions = self.ignore.get(stem)
        if regions:
            gain = float(batch['ratio_pad'][si][0][0])
            pad_w, pad_h = (float(v) for v in batch['ratio_pad'][si][1])
            r = np.asarray(regions, dtype=np.float64) * gain
            r[:, [0, 2]] += pad_w
            r[:, [1, 3]] += pad_h
            ioa = _ioa(boxes.detach().cpu().numpy().astype(np.float64), r)
            inside = torch.as_tensor(ioa.max(axis=1) >= self.ioa_thr, device=boxes.device)
            self.n_dropped += int((mask & inside).sum())
            mask &= ~inside
        return mask


def _unpack(pred):
    """(boxes, conf, cls) views for Ultralytics 8.3 tensors and 8.4 dicts."""
    if isinstance(pred, dict):
        if 'bboxes' not in pred or pred['bboxes'] is None:
            return None, None, None
        return pred['bboxes'], pred['conf'], pred['cls']
    if pred is None or len(pred) == 0:
        return None, None, None
    return pred[:, :4], pred[:, 4], pred[:, 5]


def _select(pred, keep):
    if isinstance(pred, dict):
        for k, v in list(pred.items()):
            if hasattr(v, 'shape') and len(getattr(v, 'shape', ())) and v.shape[0] == keep.shape[0]:
                pred[k] = v[keep]
    else:
        pred.data = pred[keep]


def evaluate_uavdt(models, data_yaml, ignore_json, out_dir, imgszs=(1280,), conf=0.001,
                   iou=0.5, max_det=600, device=None, workers=4, batch=4, merge_van=True,
                   capture_stats=False):
    """Zero-shot evaluation of VisDrone-trained checkpoints on UAVDT.

    capture_stats : also store per-image detection statistics (batch size 1) so
                    that differences between models can be given confidence
                    intervals. UAVDT is video, so frames within a sequence are
                    highly correlated and the resampling unit has to be the
                    sequence; the image-to-sequence mapping is saved alongside.
    """
    from ultralytics import YOLO
    from .significance import _PerImageCapture, save_stats, _identity as _ident

    ignore_map = json.loads(Path(ignore_json).read_text()) if ignore_json else {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    register_meiscf_modules()
    rows, report = [], {}

    for imgsz in imgszs:
        for label, weights in models.items():
            logger.info(f"UAVDT: {label} @ {imgsz}px (merge_van={merge_van})")
            model = YOLO(str(weights))
            hooks = _UavdtEvalHooks(ignore_map, merge_van=merge_van, nms_iou=iou)
            model.add_callback('on_val_start', hooks)
            capture = None
            if capture_stats:
                capture = _PerImageCapture()
                model.add_callback('on_val_batch_end', capture)
            m = model.val(data=str(data_yaml), imgsz=imgsz,
                          batch=1 if capture_stats else batch, conf=conf,
                          iou=iou, max_det=max_det, plots=False, save_json=False,
                          verbose=False, device=device, workers=workers)
            if capture is not None:
                records = capture.records()
                if len(records) != len(hooks.stems):
                    raise RuntimeError(
                        f"{label}: captured {len(records)} per-image stats for "
                        f"{len(hooks.stems)} images; per-image capture needs batch=1.")
                save_stats(records, {'mAP50': float(m.box.map50),
                                     'mAP50-95': float(m.box.map), 'imgsz': imgsz},
                           10, out_dir / f'stats_{label}_{imgsz}.npz',
                           identity=_ident(weights, data_yaml, imgsz=imgsz, conf=conf,
                                           iou=iou, max_det=max_det, merge_van=merge_van))
                (out_dir / f'image_order_{label}_{imgsz}.json').write_text(
                    json.dumps(hooks.stems))
            per_class = {}
            for i, c in enumerate(m.box.ap_class_index):
                per_class[UAVDT_NAMES.get(int(c), str(int(c)))] = {
                    'AP50': float(m.box.ap50[i]), 'AP50-95': float(m.box.ap[i])}
            rec = {'label': label, 'weights': str(weights), 'imgsz': imgsz,
                   'merge_van': merge_van, 'mAP50': float(m.box.map50),
                   'mAP50-95': float(m.box.map), 'precision': float(m.box.mp),
                   'recall': float(m.box.mr), 'per_class': per_class,
                   'images': hooks.n_images,
                   'detections_dropped_in_ignore_regions': hooks.n_dropped,
                   'van_detections_merged_into_car': hooks.n_merged,
                   'identity': _identity(weights, data_yaml, imgsz=imgsz, conf=conf,
                                         iou=iou, max_det=max_det, merge_van=merge_van)}
            report[f'{label}_{imgsz}'] = rec
            rows.append(rec)
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    (out_dir / f"uavdt_results{'_merged' if merge_van else '_raw'}.json").write_text(
        json.dumps(report, indent=2))
    head = f"{'model':22s}{'imgsz':>7s}{'mAP@50':>9s}{'mAP@50:95':>11s}" + \
           "".join(f"{n:>9s}" for n in ('car', 'truck', 'bus'))
    lines = ["", "=" * len(head), f"UAVDT zero-shot (van merged into car: {merge_van})",
             "=" * len(head), head]
    for r in rows:
        cells = "".join(f"{100 * r['per_class'][n]['AP50']:9.1f}" if n in r['per_class']
                        else f"{'-':>9s}" for n in ('car', 'truck', 'bus'))
        lines.append(f"{r['label']:22s}{r['imgsz']:7d}{100 * r['mAP50']:9.2f}"
                     f"{100 * r['mAP50-95']:11.2f}{cells}")
    text = "\n".join(lines)
    print(text)
    (out_dir / 'uavdt_summary.txt').write_text(text)
    return report
