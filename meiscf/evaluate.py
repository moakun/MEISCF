"""Evaluation: multi-resolution mAP, per-class AP, FPS benchmark, params/FLOPs.

Every result is written to JSON so the visualization stage can render figures
and tables without re-running inference.
"""

import json
import time
import logging
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from .registry import register_meiscf_modules
from .data_prep import VISDRONE_NAMES

logger = logging.getLogger(__name__)


def _load(model_path):
    register_meiscf_modules()
    return YOLO(str(model_path))


def evaluate_multi_resolution(model_path, data_yaml, imgszs=(640, 800, 1024, 1280),
                              out_json=None, conf=0.001, iou=0.5, max_det=300,
                              batch=4, names=None, augment=False):
    """Validate a model at several resolutions. Returns dict keyed by imgsz.

    augment: enable test-time augmentation (multi-scale + flips). Slower but
    typically +0.5-1.5 pp; the result is no longer a single-pass inference.
    """
    model = _load(model_path)
    names = names or VISDRONE_NAMES
    results = {}
    for imgsz in imgszs:
        logger.info(f"Validating at {imgsz}px{' (TTA)' if augment else ''} ...")
        m = model.val(data=str(data_yaml), imgsz=imgsz, batch=batch,
                      conf=conf, iou=iou, max_det=max_det, plots=False,
                      save_json=False, verbose=False, augment=augment)
        per_class = {}
        try:
            for i, ap50 in zip(m.box.ap_class_index, m.box.ap50):
                per_class[names[int(i)]] = float(ap50)
        except Exception:
            pass
        results[str(imgsz)] = {
            'mAP50': float(m.box.map50),
            'mAP50-95': float(m.box.map),
            'precision': float(m.box.mp),
            'recall': float(m.box.mr),
            'per_class_AP50': per_class,
        }
        logger.info(f"  {imgsz}px: mAP@50={m.box.map50:.4f} mAP@50-95={m.box.map:.4f}")
        # Release cached activations before the next (larger) resolution,
        # otherwise the 1280 -> 1536 step can OOM on an otherwise-fine GPU.
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(results, indent=2))
        logger.info(f"Saved multi-resolution evaluation -> {out_json}")
    return results


def per_class_ap(model_path, data_yaml, imgsz=1280, out_json=None, names=None,
                 conf=0.001, iou=0.5, max_det=300, batch=4, augment=False):
    """Per-class AP@50 and AP@50-95 at a fixed resolution (paper uses 1280)."""
    model = _load(model_path)
    names = names or VISDRONE_NAMES
    m = model.val(data=str(data_yaml), imgsz=imgsz, batch=batch, conf=conf,
                  iou=iou, max_det=max_det, plots=False, verbose=False,
                  augment=augment)
    out = {'imgsz': imgsz, 'mAP50': float(m.box.map50),
           'mAP50-95': float(m.box.map), 'per_class': {}}
    for idx, ci in enumerate(m.box.ap_class_index):
        out['per_class'][names[int(ci)]] = {
            'AP50': float(m.box.ap50[idx]),
            'AP50-95': float(m.box.ap[idx]),
        }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=2))
    return out


def benchmark_fps(model_path, imgsz=1024, warmup=20, iters=100, device=None,
                  half=True, out_json=None):
    """Measure pure inference FPS + latency breakdown estimate on GPU."""
    model = _load(model_path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.model.to(device).eval()
    if half and device != 'cpu':
        net = net.half()
    dtype = torch.float16 if (half and device != 'cpu') else torch.float32
    x = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        if device != 'cpu':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            net(x)
        if device != 'cpu':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    latency_ms = elapsed / iters * 1000
    fps = iters / elapsed
    result = {'imgsz': imgsz, 'device': str(device), 'half': bool(half),
              'fps': round(fps, 2), 'latency_ms': round(latency_ms, 3),
              'iters': iters}
    logger.info(f"FPS @ {imgsz}px ({device}, half={half}): {fps:.1f} "
                f"({latency_ms:.2f} ms/img)")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=2))
    return result


def _gpu_state():
    """What the GPUs are doing right now, so a reported speed can be trusted."""
    import subprocess
    state = {}
    queries = {
        'gpus': ['--query-gpu=index,name,utilization.gpu,memory.used,memory.total,'
                 'clocks.current.sm,temperature.gpu', '--format=csv,noheader,nounits'],
        'compute_apps': ['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'],
    }
    for key, args in queries.items():
        try:
            out = subprocess.run(['nvidia-smi'] + args, capture_output=True, text=True, timeout=30)
            state[key] = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
        except Exception as exc:                      # nvidia-smi missing or unreadable
            state[key] = f"unavailable: {exc}"
    state['visible_devices'] = __import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'all')
    return state


def benchmark_fps_table(models, imgszs=(1280, 1536), batches=(1,), warmup_s=3.0,
                        repeat_s=1.5, repeats=5, half=True, device=None, fuse=True,
                        out_json=None):
    """Inference speed for several models, input sizes and batch sizes.

    No pre- or post-processing, convolution and batch-normalization layers fused.
    Each configuration is warmed up for ``warmup_s`` seconds first: a GPU that has
    been idle starts at a low clock and the first measurements come out too slow,
    which in testing made a larger input look faster than a smaller one. The timed
    loop then runs ``repeats`` times and the median is reported, with the spread
    kept so drift stays visible. The state of every GPU is recorded before and
    after, because a benchmark that shared the card with a training job would be
    meaningless.

    One image per batch is what a latency-sensitive deployment sees, but on a fast
    card a small model can finish a layer sooner than the host can submit the next
    one, so the loop measures the host instead of the model: the giveaway is a
    latency that does not grow with the input size. Pass several ``batches`` to
    check for this -- with enough images in flight the GPU becomes the limit again,
    and the throughput comparison between models is then a comparison of the models.
    """
    if device in (None, '', 'auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif isinstance(device, int) or str(device).isdigit():
        device = f'cuda:{device}'                     # config allows "0" / 0
    device = str(device)
    dtype = torch.float16 if (half and device != 'cpu') else torch.float32
    before = _gpu_state()
    busy = [a for a in (before.get('compute_apps') or []) if isinstance(a, str)]
    if busy:
        logger.warning(f"Other processes are using a GPU right now: {busy}. "
                       "Run this on an idle card or the numbers will be wrong.")
    if device != 'cpu':
        torch.backends.cudnn.benchmark = True

    report, rows = {'gpu_state_before': before, 'settings': {
        'batch': 1, 'half': bool(half), 'fused': bool(fuse), 'warmup_seconds': warmup_s,
        'seconds_per_repeat': repeat_s, 'repeats': repeats,
        'torch': torch.__version__, 'device': device,
        'device_name': torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu'}}, []

    sync = (lambda: torch.cuda.synchronize()) if device != 'cpu' else (lambda: None)
    key = lambda label, imgsz, batch: f'{label}_{imgsz}_b{batch}'
    report['settings']['batches'] = list(batches)

    for label, weights in models.items():
        model = _load(weights)
        if fuse:
            model.fuse()
        net = model.model.to(device).eval()
        if dtype == torch.float16:
            net = net.half()
        for batch in batches:
            for imgsz in imgszs:
                rec = {'label': label, 'weights': str(weights), 'imgsz': imgsz,
                       'batch': batch}
                try:
                    x = torch.zeros(batch, 3, imgsz, imgsz, device=device, dtype=dtype)
                    per_repeat = []
                    with torch.no_grad():
                        deadline, n_warm = time.perf_counter() + warmup_s, 0
                        while time.perf_counter() < deadline or n_warm < 20:
                            net(x)
                            n_warm += 1
                            if n_warm % 10 == 0:
                                sync()
                        sync()
                        t0 = time.perf_counter()       # latency estimate sets the loop length
                        for _ in range(10):
                            net(x)
                        sync()
                        latency = (time.perf_counter() - t0) / 10
                        iters = max(20, int(round(repeat_s / max(latency, 1e-6))))
                        for _ in range(repeats):
                            t0 = time.perf_counter()
                            for _ in range(iters):
                                net(x)
                            sync()
                            per_repeat.append(batch * iters / (time.perf_counter() - t0))
                    del x
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if 'out of memory' not in str(exc).lower():
                        raise
                    rec['out_of_memory'] = str(exc).splitlines()[0]
                    report[key(label, imgsz, batch)] = rec
                    logger.warning(f"{label} @ {imgsz}px batch {batch}: out of memory, skipped")
                    torch.cuda.empty_cache()
                    continue
                f = np.array(per_repeat)
                rec.update({
                    'fps_median': float(np.median(f)), 'fps_mean': float(f.mean()),
                    'fps_std': float(f.std(ddof=1)) if len(f) > 1 else 0.0,
                    'fps_min': float(f.min()), 'fps_max': float(f.max()),
                    'latency_ms_per_batch': float(1000 * batch / np.median(f)),
                    'ms_per_image': float(1000 / np.median(f)),
                    'warmup_iters': n_warm, 'iters_per_repeat': iters,
                    'fps_per_repeat': [float(v) for v in f]})
                report[key(label, imgsz, batch)] = rec
                rows.append(rec)
                logger.info(f"{label} @ {imgsz}px batch {batch}: {rec['fps_median']:.1f} img/s "
                            f"median (range {rec['fps_min']:.1f}-{rec['fps_max']:.1f}, "
                            f"{rec['latency_ms_per_batch']:.1f} ms/batch, "
                            f"{iters} iters x {repeats})")
            # A larger input has more pixels to convolve, so it must take longer. When
            # it does not, the loop is not measuring the model: it is measuring how
            # fast this host can submit kernels, either because something else is
            # using the machine or because one image at a time cannot keep the card
            # busy. Only the second cause is fixed by adding images to the batch.
            sizes = sorted(imgszs)
            for small, large in zip(sizes, sizes[1:]):
                lo, hi = report[key(label, small, batch)], report[key(label, large, batch)]
                if 'fps_median' not in lo or 'fps_median' not in hi:
                    continue
                pixels = (large / small) ** 2         # convolution cost grows with pixels
                slowdown = lo['fps_median'] / hi['fps_median']
                # How much of the implied extra work actually showed up in the clock.
                efficiency = (slowdown - 1) / (pixels - 1)
                hi[f'fps_ratio_to_{small}px'] = float(hi['fps_median'] / lo['fps_median'])
                hi[f'compute_scaling_vs_{small}px'] = float(efficiency)
                hi['host_bound'] = bool(efficiency < 0.5)
                if hi['host_bound']:
                    logger.warning(
                        f"{label} batch {batch}: {large}px has {pixels:.2f}x the pixels of "
                        f"{small}px but took only {slowdown:.2f}x as long "
                        f"({hi['fps_median']:.1f} vs {lo['fps_median']:.1f} img/s), "
                        f"{100 * efficiency:.0f}% of the implied cost. The loop is bound by the "
                        "host, not the model. Check that the machine is idle, then measure a "
                        "larger batch.")
        del model, net
        if device != 'cpu':
            torch.cuda.empty_cache()

    report['gpu_state_after'] = _gpu_state()
    lines = ["", "Throughput in images/s: median of the repeats, observed range in brackets",
             f"{'model':22s}{'batch':>6s}" + "".join(f"{'img/s @' + str(z):>26s}" for z in imgszs)]
    for label in models:
        for batch in batches:
            cells = ""
            for z in imgszs:
                r = report[key(label, z, batch)]
                cells += (f"{r['fps_median']:12.1f}  [{r['fps_min']:5.1f}, {r['fps_max']:5.1f}]"
                          if 'fps_median' in r else f"{'out of memory':>26s}")
            lines.append(f"{label:22s}{batch:6d}{cells}")
    text = "\n".join(lines)
    print(text)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, indent=2))
        Path(out_json).with_suffix('.txt').write_text(text)
    return report


def model_complexity(model_path, imgsz=640, out_json=None):
    """Report parameter count and GFLOPs (uses Ultralytics' built-in profiler)."""
    model = _load(model_path)
    n_params = sum(p.numel() for p in model.model.parameters())
    info = {'params': int(n_params)}
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        info['params'] = int(get_num_params(model.model))
        info['gflops'] = round(float(get_flops(model.model, imgsz)), 2)
    except Exception as e:
        logger.warning(f"FLOPs profiling failed ({e}); reporting param count only.")
    info['params_millions'] = round(info['params'] / 1e6, 3)
    info['imgsz'] = imgsz
    logger.info(f"Complexity @ {imgsz}px: {info['params_millions']}M params, "
                f"{info.get('gflops', 'n/a')} GFLOPs")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(info, indent=2))
    return info


def full_report(model_path, data_yaml, out_dir, imgszs=(640, 800, 1024, 1280),
                per_class_imgsz=1280, fps_imgsz=1024, max_det=300, augment=False,
                batch=4):
    """Run the complete evaluation suite and dump every artifact to out_dir.

    max_det: max detections per image kept before metric computation. VisDrone
    averages ~248 objects/image (dense scenes exceed 500), so the YOLO default
    of 300 caps recall; raise it (e.g. 600) for dense aerial scenes.
    augment: enable test-time augmentation (TTA) for the mAP passes.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = {'model': str(model_path), 'max_det': max_det, 'tta': augment,
              'batch': batch}
    report['multi_resolution'] = evaluate_multi_resolution(
        model_path, data_yaml, imgszs=imgszs, max_det=max_det, augment=augment,
        batch=batch, out_json=out_dir / 'multi_resolution.json')
    report['per_class'] = per_class_ap(
        model_path, data_yaml, imgsz=per_class_imgsz, max_det=max_det, augment=augment,
        batch=batch, out_json=out_dir / 'per_class_ap.json')
    report['fps'] = benchmark_fps(
        model_path, imgsz=fps_imgsz, out_json=out_dir / 'fps.json')
    report['complexity'] = model_complexity(
        model_path, imgsz=640, out_json=out_dir / 'complexity.json')
    (out_dir / 'full_report.json').write_text(json.dumps(report, indent=2))
    logger.info(f"Full evaluation report -> {out_dir / 'full_report.json'}")
    return report
