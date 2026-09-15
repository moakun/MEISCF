"""AP by object size, using the Ultralytics validator's own box matching.

For every validation image (batch=1) the capture stores the detections, the
ground truth and, taken from inside ``validator.match_predictions``, the index of
the ground-truth box each detection was matched to at every IoU threshold. The
matching is therefore exactly the validator's; the wrapper confirms this on every
image by comparing its matched/unmatched flags with the TP matrix the validator
returns.

AP for a size range follows the COCO convention: ground truth outside the range
is ignored, detections matched to ignored ground truth are ignored, and unmatched
detections whose own size lies outside the range are ignored. Unlike pycocotools,
matching is done once over all ground truth (as the validator does) and AP comes
from the installed Ultralytics ``ap_per_class``, so the all-sizes row reproduces
``model.val()``. Object size is sqrt(w*h), in original-image pixels or in
network-input pixels at the evaluated input size.
"""

import json
import logging
from pathlib import Path

import numpy as np

from .registry import register_meiscf_modules
from .significance import _identity

logger = logging.getLogger(__name__)

# (name, low, high) on sqrt(w*h); a box belongs to a range when low <= size < high.
SCHEMES = {
    # COCO area ranges (32^2 and 96^2) in original-image pixels.
    'coco_native': (('all', 0.0, np.inf), ('small', 0.0, 32.0),
                    ('medium', 32.0, 96.0), ('large', 96.0, np.inf)),
    # Finer split of COCO "small", original-image pixels.
    'small_native': (('<8', 0.0, 8.0), ('8-16', 8.0, 16.0), ('16-32', 16.0, 32.0)),
    # Size in network-input pixels at the evaluated input size: the quantity the
    # sampling argument is about (one stride-4 cell = 4 px, one stride-8 cell = 8 px).
    'input_px': (('<4', 0.0, 4.0), ('4-8', 4.0, 8.0), ('8-16', 8.0, 16.0),
                 ('16-32', 16.0, 32.0), ('>=32', 32.0, np.inf)),
}
_SCHEME_UNIT = {'coco_native': 'native', 'small_native': 'native', 'input_px': 'input'}


def _np(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy().copy()
    return np.array(x, copy=True)


def _match_indices(pred_classes, true_classes, iou, thresholds):
    """Replicates Ultralytics BaseValidator.match_predictions (non-scipy path),
    additionally returning which ground-truth row each detection matched."""
    correct_class = true_classes[:, None] == pred_classes
    iou = (iou * correct_class).cpu().numpy()
    n_det = iou.shape[1]
    correct = np.zeros((n_det, len(thresholds)), dtype=bool)
    matched = np.full((n_det, len(thresholds)), -1, dtype=np.int32)
    for i, threshold in enumerate(thresholds):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
            matched[matches[:, 1].astype(int), i] = matches[:, 0]
    return correct, matched


class _BoxCapture:
    """Hooks installed on the validator at on_val_start (requires batch=1)."""

    def __init__(self):
        self.images = []
        self.n_checked = 0
        self.n_mismatch = 0
        self._matched = None

    def __call__(self, validator):
        thresholds = validator.iouv.cpu().tolist()
        orig_match = validator.match_predictions
        orig_update = validator.update_metrics

        def match_predictions(pred_classes, true_classes, iou, use_scipy=False):
            correct = orig_match(pred_classes, true_classes, iou, use_scipy)
            mine, matched = _match_indices(pred_classes, true_classes, iou, thresholds)
            self.n_checked += 1
            if not np.array_equal(mine, _np(correct).astype(bool)):
                self.n_mismatch += 1
            self._matched = matched
            return correct

        def update_metrics(preds, batch):
            if len(preds) != 1:
                raise RuntimeError("size_eval capture requires batch=1")
            rec = self._record(preds[0], batch, len(thresholds))
            self._matched = None
            out = orig_update(preds, batch)
            if self._matched is not None:
                if self._matched.shape[0] != len(rec['pconf']):
                    raise RuntimeError("detections and matched indices are misaligned")
                rec['matched'] = self._matched
            self.images.append(rec)
            return out

        validator.match_predictions = match_predictions
        validator.update_metrics = update_metrics

    @staticmethod
    def _record(pred, batch, n_thr):
        if isinstance(pred, dict):                        # Ultralytics 8.4
            pbox, pconf, pcls = pred['bboxes'], pred['conf'], pred['cls']
        else:                                             # 8.3: [x1, y1, x2, y2, conf, cls]
            pbox, pconf, pcls = pred[:, :4], pred[:, 4], pred[:, 5]
        pbox = _np(pbox).reshape(-1, 4).astype(np.float32)
        idx = _np(batch['batch_idx']).reshape(-1) == 0
        gxywh = _np(batch['bboxes']).reshape(-1, 4)[idx].astype(np.float64)
        gcls = _np(batch['cls']).reshape(-1)[idx]
        h, w = (int(s) for s in batch['img'].shape[2:])
        h0, w0 = (int(s) for s in batch['ori_shape'][0])
        try:
            rh, rw = (float(r) for r in batch['ratio_pad'][0][0])
        except Exception:                                 # no ratio_pad: assume plain letterbox
            rh = rw = min(h / h0, w / w0)
        gw, gh = gxywh[:, 2] * w, gxywh[:, 3] * h         # letterboxed-input pixels
        pw = np.clip(pbox[:, 2] - pbox[:, 0], 0, None)
        ph = np.clip(pbox[:, 3] - pbox[:, 1], 0, None)
        n_det = len(pbox)
        return {
            'im_file': str(batch['im_file'][0]),
            'pconf': _np(pconf).reshape(-1).astype(np.float32),
            'pcls': _np(pcls).reshape(-1).astype(np.float32),
            'gcls': gcls.astype(np.float32),
            'psize_input': np.sqrt(pw * ph).astype(np.float32),
            'psize_native': np.sqrt((pw / rw) * (ph / rh)).astype(np.float32),
            'gsize_input': np.sqrt(gw * gh).astype(np.float32),
            'gsize_native': np.sqrt((gw / rw) * (gh / rh)).astype(np.float32),
            'matched': np.full((n_det, n_thr), -1, dtype=np.int32),
            'shape': (h0, w0, h, w, rh, rw),
        }


def _concat(images):
    """Per-image records -> flat arrays; matched indices become global GT rows."""
    n_pred = np.array([len(r['pconf']) for r in images], dtype=np.int64)
    n_gt = np.array([len(r['gcls']) for r in images], dtype=np.int64)
    p_start = np.concatenate([[0], np.cumsum(n_pred)[:-1]])
    g_start = np.concatenate([[0], np.cumsum(n_gt)[:-1]])
    matched = []
    for r, gs in zip(images, g_start):
        m = r['matched'].astype(np.int64)
        matched.append(np.where(m >= 0, m + gs, -1))
    n_thr = images[0]['matched'].shape[1] if images else 10
    cat = lambda k, dt: (np.concatenate([r[k] for r in images]).astype(dt)
                         if images else np.zeros(0, dt))
    return {
        'pconf': cat('pconf', np.float32), 'pcls': cat('pcls', np.float32),
        'psize_input': cat('psize_input', np.float32),
        'psize_native': cat('psize_native', np.float32),
        'gcls': cat('gcls', np.float32),
        'gsize_input': cat('gsize_input', np.float32),
        'gsize_native': cat('gsize_native', np.float32),
        'matched': (np.concatenate(matched).astype(np.int32) if matched
                    else np.zeros((0, n_thr), np.int32)),
        'p_start': p_start, 'n_pred': n_pred, 'g_start': g_start, 'n_gt': n_gt,
        'shapes': np.array([r['shape'] for r in images], dtype=np.float64),
        'im_file': np.array([r['im_file'] for r in images]),
    }


def _check_sizes(cap):
    """Guard against silent coordinate-space errors in the size computation.

    Ground-truth sizes must agree with the label files on disk, and detections
    matched at IoU>=0.5 must be about as large as their ground truth; otherwise
    the batch or prediction format has changed and every size bin would be wrong.
    """
    from ultralytics.data.utils import img2label_paths
    worst = 0.0
    for i, f in enumerate(cap['im_file']):
        lb = Path(img2label_paths([str(f)])[0])
        rows = (np.loadtxt(lb, ndmin=2)[:, :5] if lb.exists() and lb.stat().st_size
                else np.zeros((0, 5)))
        if len(rows):
            rows = np.unique(rows, axis=0)                 # Ultralytics drops duplicate labels
        start, n = int(cap['g_start'][i]), int(cap['n_gt'][i])
        if len(rows) != n:
            raise RuntimeError(f"{f}: {n} boxes in the validator batch, {len(rows)} in {lb}")
        if n == 0:
            continue
        h0, w0 = cap['shapes'][i][:2]
        ref = np.sort(np.sqrt(rows[:, 3] * w0 * rows[:, 4] * h0))
        got = np.sort(cap['gsize_native'][start:start + n].astype(np.float64))
        worst = max(worst, float(np.max(np.abs(got - ref) / np.maximum(ref, 1e-6))))
    if worst > 0.01:
        raise RuntimeError(f"ground-truth sizes differ from the label files by up to {worst:.1%}")
    m = cap['matched'][:, 0]
    d = np.where(m >= 0)[0]
    med = None
    if len(d):
        med = float(np.median(cap['psize_native'][d] / np.maximum(cap['gsize_native'][m[d]], 1e-6)))
        if not 0.9 < med < 1.1:
            raise RuntimeError("detections and ground truth are in different coordinate spaces "
                               f"(median size ratio of matched pairs {med:.3f})")
    return worst, med


def save_capture(cap, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, _meta=np.array(json.dumps(meta)), **cap)
    logger.info(f"Saved box capture -> {path}")


def load_capture(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['_meta']))
    cap = {k: z[k] for k in z.files if k != '_meta'}
    return cap, meta


def _rows(start, length, idx):
    """Row indices of the images in idx (duplicates allowed, for bootstrapping)."""
    lens = length[idx]
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(start[idx] - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return np.arange(total) + offsets


def size_ap(cap, schemes=SCHEMES, img_idx=None, ap_fn=None, names=None):
    """AP@50 and AP@50:95 per size range. img_idx selects (or resamples) images."""
    if ap_fn is None:
        from ultralytics.utils.metrics import ap_per_class as ap_fn
    n_img = len(cap['n_pred'])
    img_idx = np.arange(n_img) if img_idx is None else np.asarray(img_idx)
    P = _rows(cap['p_start'], cap['n_pred'], img_idx)
    G = _rows(cap['g_start'], cap['n_gt'], img_idx)
    conf, pcls, matched = cap['pconf'][P], cap['pcls'][P], cap['matched'][P]
    n_thr = matched.shape[1]
    out = {}
    for scheme, ranges in schemes.items():
        unit = _SCHEME_UNIT.get(scheme, 'native')
        gsize_all = cap[f'gsize_{unit}']
        dsize = cap[f'psize_{unit}'][P]
        out[scheme] = {}
        for name, lo, hi in ranges:
            g_in_all = (gsize_all >= lo) & (gsize_all < hi)
            tcls = cap['gcls'][G][g_in_all[G]]
            rec = {'n_gt': int(len(tcls))}
            if len(tcls) == 0:
                rec.update(AP50=None, **{'AP50-95': None})
                out[scheme][name] = rec
                continue
            classes = np.unique(tcls).astype(int)
            det_in = (dsize >= lo) & (dsize < hi)
            ap = np.zeros((len(classes), n_thr))
            for t in range(n_thr):
                m = matched[:, t]
                has = m >= 0
                tp = np.zeros(len(m), dtype=bool)
                tp[has] = g_in_all[m[has]]
                keep = ~((has & ~tp) | (~has & ~det_in))
                if keep.sum() == 0:
                    continue                                   # AP stays 0 for every class
                res = ap_fn(tp[keep][:, None], conf[keep], pcls[keep], tcls, plot=False)
                ap[:, t] = res[5][:, 0]
            rec['AP50'] = float(ap[:, 0].mean())
            rec['AP50-95'] = float(ap.mean())
            rec['per_class_AP50'] = {(names[c] if names else str(c)): float(a)
                                     for c, a in zip(classes, ap[:, 0])}
            rec['per_class_n_gt'] = {(names[c] if names else str(c)): int((tcls == c).sum())
                                     for c in classes}
            out[scheme][name] = rec
    return out


def capture_boxes(model_path, data_yaml, imgsz=1280, conf=0.001, iou=0.5,
                  max_det=600, device=None, workers=2, cache_npz=None):
    """Validate at batch=1 and capture detections, ground truth and matching."""
    ident = _identity(model_path, data_yaml, imgsz=imgsz, conf=conf, iou=iou,
                      max_det=max_det, kind='size_eval')
    if cache_npz and Path(cache_npz).exists():
        cap, meta = load_capture(cache_npz)
        if meta.get('identity') == ident:
            logger.info(f"Reusing cached box capture: {cache_npz}")
            return cap, meta
        logger.warning(f"{cache_npz} was built from another checkpoint or settings; recomputing.")

    from ultralytics import YOLO
    register_meiscf_modules()
    model = YOLO(str(model_path))
    capture = _BoxCapture()
    model.add_callback('on_val_start', capture)
    m = model.val(data=str(data_yaml), imgsz=imgsz, batch=1, conf=conf, iou=iou,
                  max_det=max_det, plots=False, save_json=False, verbose=False,
                  device=device, workers=workers)
    names = {int(k): v for k, v in model.names.items()}
    if capture.n_mismatch:
        raise RuntimeError(
            f"Matching replication differed from the validator on {capture.n_mismatch} "
            f"of {capture.n_checked} images; size-stratified AP would not be exact.")
    cap = _concat(capture.images)
    size_err, size_ratio = _check_sizes(cap)
    meta = {'identity': ident, 'imgsz': imgsz, 'max_det': max_det,
            'names': [names[i] for i in sorted(names)],
            'reported': {'mAP50': float(m.box.map50), 'mAP50-95': float(m.box.map)},
            'n_images': len(capture.images), 'matching_checked_images': capture.n_checked,
            'gt_size_max_rel_error': size_err, 'matched_size_ratio_median': size_ratio}

    full = size_ap(cap, {'coco_native': SCHEMES['coco_native'][:1]})['coco_native']['all']
    ok = (abs(full['AP50'] - meta['reported']['mAP50']) < 1e-3 and
          abs(full['AP50-95'] - meta['reported']['mAP50-95']) < 1e-3)
    if not ok:
        raise RuntimeError(
            f"All-sizes AP from the capture ({full['AP50']:.5f}/{full['AP50-95']:.5f}) "
            f"does not reproduce the validator ({meta['reported']['mAP50']:.5f}/"
            f"{meta['reported']['mAP50-95']:.5f}).")
    logger.info(f"Captured {meta['n_images']} images; matching identical to the validator on "
                f"{capture.n_checked} images; all-sizes mAP@50 {full['AP50']*100:.2f} reproduces "
                f"model.val(); ground-truth sizes within {size_err:.2%} of the label files.")
    if cache_npz:
        save_capture(cap, cache_npz, meta)
    return cap, meta


def run_size_ap(models, data_yaml, out_dir, imgszs=(1280,), conf=0.001, iou=0.5,
                max_det=600, device=None, workers=2):
    """Capture + size-stratified AP for {label: weights} at each input size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for imgsz in imgszs:
        for label, weights in models.items():
            logger.info(f"Size-stratified AP: {label} @ {imgsz}px")
            cap, meta = capture_boxes(weights, data_yaml, imgsz=imgsz, conf=conf, iou=iou,
                                      max_det=max_det, device=device, workers=workers,
                                      cache_npz=out_dir / f'boxes_{label}_{imgsz}.npz')
            res = size_ap(cap, names=meta['names'])
            doc = {'label': label, 'weights': str(weights), 'imgsz': imgsz,
                   'reported': meta['reported'], 'results': res}
            (out_dir / f'size_ap_{label}_{imgsz}.json').write_text(json.dumps(doc, indent=2))
            table.append((label, imgsz, res))

    lines = ["", "=" * 96, "AP@50 (%) by object size  [n ground-truth boxes]", "=" * 96]
    for scheme, ranges in SCHEMES.items():
        unit = 'original px' if _SCHEME_UNIT[scheme] == 'native' else 'input px'
        lines.append(f"{scheme} (sqrt(w*h), {unit})")
        head = f"  {'model':18s}" + "".join(f"{name:>14s}" for name, _, _ in ranges)
        lines.append(head)
        for label, imgsz, res in table:
            cells = []
            for name, _, _ in ranges:
                r = res[scheme][name]
                ap = '-' if r['AP50'] is None else f"{100 * r['AP50']:.1f}"
                cells.append(f"{ap} [{r['n_gt']}]".rjust(14))
            model_col = f"{label} @{imgsz}"
            lines.append(f"  {model_col:18s}" + "".join(cells))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (out_dir / 'size_ap_summary.txt').write_text(text)
    return table
