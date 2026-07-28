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
                per_class_imgsz=1280, fps_imgsz=1024, max_det=300, augment=False):
    """Run the complete evaluation suite and dump every artifact to out_dir.

    max_det: max detections per image kept before metric computation. VisDrone
    averages ~248 objects/image (dense scenes exceed 500), so the YOLO default
    of 300 caps recall; raise it (e.g. 600) for dense aerial scenes.
    augment: enable test-time augmentation (TTA) for the mAP passes.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = {'model': str(model_path), 'max_det': max_det, 'tta': augment}
    report['multi_resolution'] = evaluate_multi_resolution(
        model_path, data_yaml, imgszs=imgszs, max_det=max_det, augment=augment,
        out_json=out_dir / 'multi_resolution.json')
    report['per_class'] = per_class_ap(
        model_path, data_yaml, imgsz=per_class_imgsz, max_det=max_det, augment=augment,
        out_json=out_dir / 'per_class_ap.json')
    report['fps'] = benchmark_fps(
        model_path, imgsz=fps_imgsz, out_json=out_dir / 'fps.json')
    report['complexity'] = model_complexity(
        model_path, imgsz=640, out_json=out_dir / 'complexity.json')
    (out_dir / 'full_report.json').write_text(json.dumps(report, indent=2))
    logger.info(f"Full evaluation report -> {out_dir / 'full_report.json'}")
    return report
