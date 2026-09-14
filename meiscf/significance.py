"""Statistical significance testing for detector comparisons.

Addresses the reviewer request for "a proper significance test (e.g., a paired
t-test across all validation images)". Two complementary tests are provided,
because they answer different questions:

  * IMAGE-LEVEL (this module): paired bootstrap + paired permutation test over
    the validation images. Answers "would this gap survive a different sample of
    validation images?". Many evaluation units (548 on VisDrone) -> tight CIs.

  * SEED-LEVEL (meiscf.multiseed): paired comparison across independent
    retrainings. Answers "would this gap survive retraining?". This is the
    question that matters for a claim about a METHOD, and it is typically the
    dominant source of variance.

Correctness note
----------------
Per-image statistics are captured from Ultralytics' OWN validator (via the
``on_val_batch_end`` callback, with batch=1 so one callback == one image), and
mAP is recomputed with Ultralytics' OWN ``ap_per_class``. This guarantees the
bootstrap operates on exactly the quantities behind the reported mAP -- we do
not reimplement matching or AP. ``collect_per_image_stats`` asserts that mAP
recomputed over all images reproduces the validator's reported value; if that
assertion fails the capture is unreliable and we refuse to return results.
"""

import json
import logging
from pathlib import Path

import numpy as np

from .registry import register_meiscf_modules

logger = logging.getLogger(__name__)

_STAT_KEYS = ('tp', 'conf', 'pred_cls', 'target_cls')

# Two-sided t critical values at 95% for small samples, indexed by n (df = n-1).
# Avoids a hard scipy dependency for the seed-level interval.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262}


def _to_numpy(x):
    """Detached numpy COPY (Ultralytics clears its stat lists after validation)."""
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy().copy()
    return np.array(x, copy=True)


def _stats_store(validator):
    """Locate the validator's per-image stat lists across Ultralytics versions.

    8.3.x : ``validator.stats``          (dict of lists of torch tensors; images
                                          with no preds AND no labels are skipped)
    8.4.x : ``validator.metrics.stats``  (dict of lists of numpy arrays; every
                                          image is appended). In 8.4.x
                                          ``validator.stats`` still exists but is
                                          never populated, so check metrics FIRST.
    """
    for owner in (getattr(validator, 'metrics', None), validator):
        store = getattr(owner, 'stats', None) if owner is not None else None
        if isinstance(store, dict) and all(k in store for k in _STAT_KEYS):
            return store
    import ultralytics
    raise RuntimeError(
        "Cannot find per-image detection stats on the Ultralytics validator "
        f"(checked validator.metrics.stats and validator.stats; ultralytics "
        f"{ultralytics.__version__}). Its internals changed again; update "
        "meiscf/significance.py::_stats_store.")


class _PerImageCapture:
    """on_val_batch_end callback: snapshot the newest per-image stat entry.

    Requires batch=1 so that one callback == one image. Ultralytics runs
    update_metrics() before on_val_batch_end in BaseValidator.__call__.
    """

    def __init__(self):
        self.raw, self._prev, self.niou = [], 0, None

    def __call__(self, validator):
        store = _stats_store(validator)
        cur = len(store['conf'])
        if cur - self._prev > 1:
            raise RuntimeError(
                f"{cur - self._prev} images appended in one batch; per-image "
                "capture requires batch=1.")
        if cur > self._prev:
            rec = {k: _to_numpy(store[k][-1]) for k in _STAT_KEYS}
            if rec['tp'].ndim == 2:
                self.niou = rec['tp'].shape[1]
        else:
            rec = None      # 8.3.x only: image with no predictions AND no labels
        self._prev = cur
        self.raw.append(rec)

    def records(self):
        niou = self.niou or 10
        empty = {'tp': np.zeros((0, niou), dtype=bool),
                 'conf': np.zeros(0, dtype=np.float32),
                 'pred_cls': np.zeros(0, dtype=np.float32),
                 'target_cls': np.zeros(0, dtype=np.float32)}
        return [r if r is not None else {k: v.copy() for k, v in empty.items()}
                for r in self.raw]


# ---------------------------------------------------------------------------
# Per-image statistic capture
# ---------------------------------------------------------------------------
def collect_per_image_stats(model_path, data_yaml, imgsz=1536, conf=0.001,
                            iou=0.5, max_det=600, device=None, augment=False,
                            cache_npz=None, workers=2):
    """Validate a model at batch=1, capturing per-image TP/conf/cls statistics.

    Returns (records, reported, nc) where records is a list of per-image dicts
    aligned with the validation image order.
    """
    ident = _identity(model_path, data_yaml, imgsz=imgsz, conf=conf, iou=iou,
                      max_det=max_det, augment=bool(augment))
    if cache_npz and Path(cache_npz).exists():
        records, reported, nc, cached = load_stats(cache_npz, with_identity=True)
        if cached == ident:
            logger.info(f"Reusing cached per-image stats: {cache_npz}")
            return records, reported, nc
        logger.warning(f"Cache {cache_npz} was built from a different checkpoint or "
                       "settings; recomputing.")

    from ultralytics import YOLO
    register_meiscf_modules()
    model = YOLO(str(model_path))

    capture = _PerImageCapture()
    model.add_callback('on_val_batch_end', capture)
    m = model.val(data=str(data_yaml), imgsz=imgsz, batch=1, conf=conf, iou=iou,
                  max_det=max_det, plots=False, save_json=False, verbose=False,
                  augment=augment, device=device, workers=workers)
    records = capture.records()

    reported = {'mAP50': float(m.box.map50), 'mAP50-95': float(m.box.map),
                'imgsz': imgsz, 'max_det': max_det, 'augment': bool(augment)}
    nc = int(max((r['pred_cls'].max(initial=-1) for r in records), default=-1) + 1)
    nc = max(nc, int(max((r['target_cls'].max(initial=-1) for r in records),
                         default=-1) + 1))

    # Validate the capture: recomputing over ALL images must reproduce the
    # validator's own numbers, otherwise the per-image stats are not usable.
    chk50, chk5095 = map_from_indices(records, np.arange(len(records)))
    if not (abs(chk50 - reported['mAP50']) < 1e-3
            and abs(chk5095 - reported['mAP50-95']) < 1e-3):
        raise RuntimeError(
            "Per-image stat capture failed validation: recomputed "
            f"mAP@50={chk50:.5f}/mAP@50-95={chk5095:.5f} vs reported "
            f"{reported['mAP50']:.5f}/{reported['mAP50-95']:.5f}. The "
            "Ultralytics validator internals likely changed; do not trust "
            "significance results until this matches.")
    logger.info(f"Captured {len(records)} per-image stats; recomputed mAP "
                f"matches validator (mAP@50={chk50:.4f}).")

    if cache_npz:
        save_stats(records, reported, nc, cache_npz, identity=ident)
    return records, reported, nc


def _identity(model_path, data_yaml, **settings):
    """What a cached stats file was computed from: exact checkpoint + settings.

    Prevents silently reusing e.g. seed-0 stats for a seed-42 run that was given
    the same --label into the same output folder.
    """
    w = Path(model_path).resolve()
    st = w.stat()
    ident = {'weights': str(w), 'weights_size': int(st.st_size),
             'weights_mtime': int(st.st_mtime),
             'data': str(Path(data_yaml).resolve()), **settings}
    return json.loads(json.dumps(ident))      # normalise types as JSON round-trips


def save_stats(records, reported, nc, path, identity=None):
    """Persist per-image stats so bootstraps can rerun without inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for i, r in enumerate(records):
        for k in _STAT_KEYS:
            flat[f'{i}_{k}'] = r[k]
    meta = {'n': len(records), 'reported': reported, 'nc': nc, 'identity': identity}
    np.savez_compressed(path, _meta=np.array(json.dumps(meta)), **flat)
    logger.info(f"Saved per-image stats -> {path}")
    return path


def load_stats(path, with_identity=False):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['_meta']))
    records = [{k: z[f'{i}_{k}'] for k in _STAT_KEYS} for i in range(meta['n'])]
    if with_identity:
        return records, meta['reported'], meta['nc'], meta.get('identity')
    return records, meta['reported'], meta['nc']


# ---------------------------------------------------------------------------
# mAP recomputation over an arbitrary (possibly resampled) image subset
# ---------------------------------------------------------------------------
def map_from_indices(records, idx):
    """mAP@50 and mAP@50-95 over the images named by idx, via ap_per_class."""
    from ultralytics.utils.metrics import ap_per_class

    tp = np.concatenate([records[i]['tp'] for i in idx]) if len(idx) else np.zeros((0, 10))
    conf = np.concatenate([records[i]['conf'] for i in idx]) if len(idx) else np.zeros(0)
    pcls = np.concatenate([records[i]['pred_cls'] for i in idx]) if len(idx) else np.zeros(0)
    tcls = np.concatenate([records[i]['target_cls'] for i in idx]) if len(idx) else np.zeros(0)
    if len(tcls) == 0 or len(tp) == 0:
        return float('nan'), float('nan')
    ap = ap_per_class(tp, conf, pcls, tcls, plot=False)[5]   # [n_cls, n_iou]
    return float(ap[:, 0].mean()), float(ap.mean())


def _check_pairing(recA, recB):
    """Both runs must see identical images in identical order."""
    if len(recA) != len(recB):
        raise ValueError(f"Image count differs: {len(recA)} vs {len(recB)}")
    for i, (a, b) in enumerate(zip(recA, recB)):
        if a['target_cls'].shape != b['target_cls'].shape or \
           not np.array_equal(np.sort(a['target_cls']), np.sort(b['target_cls'])):
            raise ValueError(
                f"Ground truth mismatch at image index {i}: the two runs are not "
                "aligned (different dataset order or dataset). Pairing invalid.")


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------
def paired_bootstrap(recA, recB, n_boot=1000, seed=0, alpha=0.05):
    """Paired bootstrap over images: CI and p-value for mAP(A) - mAP(B)."""
    _check_pairing(recA, recB)
    n = len(recA)
    rng = np.random.default_rng(seed)
    full = np.arange(n)
    obsA = map_from_indices(recA, full)
    obsB = map_from_indices(recB, full)
    obs = {'mAP50': (obsA[0] - obsB[0]) * 100,
           'mAP50-95': (obsA[1] - obsB[1]) * 100}

    d50, d5095 = [], []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        a50, a95 = map_from_indices(recA, idx)
        b50, b95 = map_from_indices(recB, idx)
        d50.append((a50 - b50) * 100)
        d5095.append((a95 - b95) * 100)
        if (b + 1) % 200 == 0:
            logger.info(f"  bootstrap {b + 1}/{n_boot}")
    out = {}
    for name, d, o in (('mAP50', d50, obs['mAP50']),
                       ('mAP50-95', d5095, obs['mAP50-95'])):
        d = np.array(d, dtype=float)
        lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        # Bootstrap p-value: proportion of resamples on the other side of zero.
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out[name] = {'observed_delta_pp': o, 'ci_low_pp': float(lo),
                     'ci_high_pp': float(hi), 'p_value': float(min(p, 1.0)),
                     'boot_mean_pp': float(d.mean()), 'boot_std_pp': float(d.std(ddof=1)),
                     'n_boot': n_boot}
    return out


def paired_permutation(recA, recB, n_perm=1000, seed=0):
    """Paired randomization test: exchange A/B per image under H0 of no effect."""
    _check_pairing(recA, recB)
    n = len(recA)
    rng = np.random.default_rng(seed)
    full = np.arange(n)
    obs = ((map_from_indices(recA, full)[0] - map_from_indices(recB, full)[0]) * 100)

    exceed = 0
    for p in range(n_perm):
        swap = rng.random(n) < 0.5
        mixA = [recB[i] if swap[i] else recA[i] for i in range(n)]
        mixB = [recA[i] if swap[i] else recB[i] for i in range(n)]
        d = (map_from_indices(mixA, full)[0] - map_from_indices(mixB, full)[0]) * 100
        exceed += abs(d) >= abs(obs)
        if (p + 1) % 200 == 0:
            logger.info(f"  permutation {p + 1}/{n_perm}")
    return {'observed_delta_pp': float(obs),
            'p_value': float((exceed + 1) / (n_perm + 1)), 'n_perm': n_perm}


def seed_level_summary(deltas_pp):
    """Mean, std and 95% CI of paired per-seed deltas (one-sample t)."""
    d = np.asarray([x for x in deltas_pp if x is not None], dtype=float)
    n = len(d)
    if n < 2:
        return {'n': int(n), 'mean_pp': float(d.mean()) if n else None,
                'note': 'need >=2 seeds for an interval'}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = _T95.get(n, 1.96)
    t_stat = mean / sem if sem > 0 else float('inf')
    return {'n': int(n), 'mean_pp': mean, 'std_pp': sd, 'sem_pp': float(sem),
            'ci_low_pp': mean - tcrit * sem, 'ci_high_pp': mean + tcrit * sem,
            't_stat': float(t_stat), 'df': int(n - 1),
            'significant_at_95': bool((mean - tcrit * sem) * (mean + tcrit * sem) > 0),
            'per_seed_pp': d.tolist()}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def compare_models(weights_a, weights_b, data_yaml, out_dir, imgsz=1536,
                   label_a='A', label_b='B', conf=0.001, iou=0.5, max_det=600,
                   n_boot=1000, n_perm=1000, device=None, seed_deltas=None):
    """Full image-level significance comparison of two checkpoints."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting per-image stats for {label_a} @ {imgsz}px ...")
    recA, repA, _ = collect_per_image_stats(
        weights_a, data_yaml, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
        device=device, cache_npz=out_dir / f'stats_{label_a}_{imgsz}.npz')
    logger.info(f"Collecting per-image stats for {label_b} @ {imgsz}px ...")
    recB, repB, _ = collect_per_image_stats(
        weights_b, data_yaml, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
        device=device, cache_npz=out_dir / f'stats_{label_b}_{imgsz}.npz')

    logger.info(f"Paired bootstrap ({n_boot} resamples) ...")
    boot = paired_bootstrap(recA, recB, n_boot=n_boot)
    logger.info(f"Paired permutation test ({n_perm} permutations) ...")
    perm = paired_permutation(recA, recB, n_perm=n_perm)

    report = {'model_a': {'label': label_a, 'weights': str(weights_a), **repA},
              'model_b': {'label': label_b, 'weights': str(weights_b), **repB},
              'n_images': len(recA), 'imgsz': imgsz,
              'image_level_bootstrap': boot, 'image_level_permutation': perm}
    if seed_deltas:
        report['seed_level'] = seed_level_summary(seed_deltas)

    (out_dir / f'significance_{label_a}_vs_{label_b}_{imgsz}.json').write_text(
        json.dumps(report, indent=2))

    b = boot['mAP50']
    print("\n" + "=" * 72)
    print(f"IMAGE-LEVEL PAIRED TEST  ({label_a} - {label_b})  @{imgsz}px, "
          f"n={len(recA)} images")
    print(f"  {label_a}: mAP@50 = {repA['mAP50']*100:.2f}   "
          f"{label_b}: mAP@50 = {repB['mAP50']*100:.2f}")
    print(f"  delta mAP@50 = {b['observed_delta_pp']:+.2f} pp   "
          f"95% CI [{b['ci_low_pp']:+.2f}, {b['ci_high_pp']:+.2f}]   "
          f"bootstrap p = {b['p_value']:.4f}")
    print(f"  permutation p = {perm['p_value']:.4f}")
    if seed_deltas:
        s = report['seed_level']
        print(f"SEED-LEVEL PAIRED TEST  n={s['n']} seeds: "
              f"{s['mean_pp']:+.2f} +/- {s['std_pp']:.2f} pp, "
              f"95% CI [{s['ci_low_pp']:+.2f}, {s['ci_high_pp']:+.2f}], "
              f"significant={s['significant_at_95']}")
    print("=" * 72 + "\n")
    return report
