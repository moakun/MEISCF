"""
MEISCF-YOLO: paper-faithful architecture + four-phase progressive training.

This file ACTUALLY builds and trains the MEISCF architecture described in the
paper (MEIS + Cross-scale Sandwich Fusion + FRM integrated into YOLOv11), in
contrast to the previous version which defined the modules but trained a stock
yolo11s.pt.

How the integration works (the part that was broken before):
  1. The three modules are registered into the Ultralytics model parser namespace
     (`ultralytics.nn.tasks`).
  2. `parse_model` is patched (regex, version-tolerant) so the parser knows how to
     compute input/output channels for these modules and constructs them eagerly
     with the CORRECT channel counts. Eager construction matters: parameters must
     exist before the optimizer is created, otherwise they never train.
  3. A custom model YAML (yolo11s-meiscf.yaml) places the modules in the graph.
     The backbone (layers 0-10) is identical to stock YOLO11 so the COCO-pretrained
     backbone transfers via model.load('yolo11s.pt').

Verify it builds before training:  python trainMEISCF_MULTIPHASE.py --smoke

NOTE ON EXPECTED RESULTS: realistic VisDrone-val mAP@50 for a ~12M-param model with
this pipeline is in the mid-40s to low-50s. Treat the paper's headline numbers as
targets to be re-measured with this code, and always report the baseline at the
SAME evaluation resolution (see README notes the assistant provided).
"""

import os
import re
import json
import inspect
import logging
import argparse
from pathlib import Path
from datetime import datetime

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 1. PAPER-FAITHFUL MODULES
# ============================================================================

class MEIS(nn.Module):
    """Multi-scale Edge Information Selection (paper Eqs. 1-5).

    Dilated depthwise-separable convolutions at dilation rates (1, 2, 4) extract
    edges at three receptive-field scales; a per-input channel-wise gate weights
    each scale; the gated sum is added back residually. Channel-preserving.
    """
    def __init__(self, c1, dilations=(1, 2, 4), reduction=16):
        super().__init__()
        r = max(c1 // reduction, 8)
        self.branches = nn.ModuleList()
        self.gates = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(c1, c1, 3, 1, d, dilation=d, groups=c1, bias=False),  # depthwise (dilated)
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c1, 1, bias=False),                                # pointwise
                nn.BatchNorm2d(c1),
                nn.SiLU(),
            ))
            self.gates.append(nn.Sequential(  # alpha_s = sigma(FC2(ReLU(FC1(GAP(E_s)))))
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, r, 1), nn.ReLU(inplace=True),
                nn.Conv2d(r, c1, 1), nn.Sigmoid(),
            ))

    def forward(self, x):
        e = 0
        for branch, gate in zip(self.branches, self.gates):
            es = branch(x)
            e = e + gate(es) * es     # element-wise channel-wise gating
        return x + e                  # residual integration (Eq. 5)


class SandwichFusion(nn.Module):
    """Cross-scale Sandwich Fusion (paper Eqs. 6-8).

    Multi-input. The FIRST input is the "current" level and defines the output
    resolution and channel count; remaining inputs are auxiliary levels that are
    channel-projected and resized to the current level, then combined by a
    normalized learnable-weighted sum, followed by a depthwise-separable fuse.
    """
    def __init__(self, ch_list, eps=1e-4):
        super().__init__()
        self.eps = eps
        c_out = ch_list[0]
        self.proj = nn.ModuleList()
        for c in ch_list:
            if c == c_out:
                self.proj.append(nn.Identity())
            else:
                self.proj.append(nn.Sequential(
                    nn.Conv2d(c, c_out, 1, bias=False),
                    nn.BatchNorm2d(c_out),
                ))
        self.weight = nn.Parameter(torch.ones(len(ch_list)))   # learnable w_i
        self.fuse = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, 1, 1, groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.Conv2d(c_out, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )

    def forward(self, xs):
        target = xs[0]
        H, W = target.shape[2:]
        w = F.relu(self.weight)
        w = w / (w.sum() + self.eps)          # normalized weights (Eq. 8)
        out = 0
        for i, x in enumerate(xs):
            x = self.proj[i](x)
            if x.shape[2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode='nearest')
            out = out + w[i] * x
        return self.fuse(out)


class FRM(nn.Module):
    """Feature Recalibration Module (paper Eqs. 12-21).

    Structurally identical to CBAM; applied AFTER fusion. Channel attention
    (avg+max shared MLP) then spatial attention (7x7 over channel-pooled maps),
    with a residual blend (alpha = 0.5). Channel-preserving.
    """
    def __init__(self, c1, reduction=16, alpha=0.5):
        super().__init__()
        r = max(c1 // reduction, 8)
        self.alpha = alpha
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, r, 1), nn.ReLU(inplace=True), nn.Conv2d(r, c1, 1),
        )
        self.spatial = nn.Conv2d(2, 1, 7, 1, 3, bias=False)

    def forward(self, x):
        # Channel attention
        wc = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1))
                           + self.mlp(F.adaptive_max_pool2d(x, 1)))
        fc = x * wc
        # Spatial attention
        sa = torch.cat([fc.mean(1, keepdim=True), fc.max(1, keepdim=True)[0]], dim=1)
        ws = torch.sigmoid(self.spatial(sa))
        fs = fc * ws
        # Residual blend (Eq. 21)
        return x + self.alpha * (fc - x) + self.alpha * (fs - fc)


# ============================================================================
# 2. REGISTER MODULES + PATCH THE PARSER (the previously-missing wiring)
# ============================================================================

def register_meiscf_modules():
    """Make MEIS/SandwichFusion/FRM usable inside a YOLO model YAML.

    Adds the classes to the parser namespace and inserts channel-handling
    branches into parse_model so each module is constructed eagerly with the
    correct (width-scaled) channel counts.
    """
    from ultralytics.nn import tasks

    for cls in (MEIS, SandwichFusion, FRM):
        setattr(tasks, cls.__name__, cls)

    # Already patched in this interpreter? Check the sentinel BEFORE inspecting
    # source: once patched, parse_model is an exec'd function with no source file
    # on disk, so inspect.getsource() would raise "could not get source code".
    if getattr(tasks, '_MEISCF_PATCHED', False):
        return

    src = inspect.getsource(tasks.parse_model)

    pattern = re.compile(r"\n(?P<indent> +)else:\n +c2 = ch\[f\]\n")
    m = pattern.search(src)
    if not m:
        raise RuntimeError(
            "Could not patch parse_model: Ultralytics internals differ from the "
            "expected 8.3.x layout. Inspect tasks.parse_model and adjust the patch."
        )
    ind = m.group('indent')          # indentation of the if/elif chain
    body = ind + '    '              # one level deeper
    inject = (
        f"{ind}elif m in (MEIS, FRM):  # MEISCF_PATCH (channel-preserving)\n"
        f"{body}c1 = ch[f]\n"
        f"{body}c2 = c1\n"
        f"{body}args = [c1, *args]\n"
        f"{ind}elif m is SandwichFusion:  # MEISCF_PATCH (multi-input)\n"
        f"{body}c2 = ch[f[0]]\n"
        f"{body}args = [[ch[x] for x in f], *args]\n"
    )
    insert_at = m.start() + 1        # just after the leading newline
    src = src[:insert_at] + inject + src[insert_at:]
    exec(compile(src, '<meiscf_parse_model>', 'exec'), tasks.__dict__)
    tasks._MEISCF_PATCHED = True
    logger.info("parse_model patched: MEIS, SandwichFusion, FRM registered.")


# ============================================================================
# 3. MEISCF MODEL YAML
# ============================================================================

MEISCF_YAML = """
# MEISCF-YOLOv11s. Backbone (0-10) identical to stock YOLO11 for pretrained transfer.
nc: 10
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]            # 0  P1/2
  - [-1, 1, Conv, [128, 3, 2]]           # 1  P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]    # 2
  - [-1, 1, Conv, [256, 3, 2]]           # 3  P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]    # 4  -> P3 feature
  - [-1, 1, Conv, [512, 3, 2]]           # 5  P4/16
  - [-1, 2, C3k2, [512, True]]           # 6  -> P4 feature
  - [-1, 1, Conv, [1024, 3, 2]]          # 7  P5/32
  - [-1, 2, C3k2, [1024, True]]          # 8
  - [-1, 1, SPPF, [1024, 5]]             # 9
  - [-1, 2, C2PSA, [1024]]               # 10 -> P5 feature

head:
  - [4,  1, MEIS, []]                     # 11 MEIS @ P3
  - [6,  1, MEIS, []]                     # 12 MEIS @ P4
  - [10, 1, MEIS, []]                     # 13 MEIS @ P5

  - [[12, 11, 13], 1, SandwichFusion, []] # 14 3-level fuse @ P4 (current=P4)
  - [-1, 2, C3k2, [512, False]]           # 15 P4 refined

  - [[11, 15], 1, SandwichFusion, []]     # 16 fuse @ P3 (current=P3, aux=P4n)
  - [-1, 2, C3k2, [256, False]]           # 17 P3 refined

  - [[13, 15], 1, SandwichFusion, []]     # 18 fuse @ P5 (current=P5, aux=P4n)
  - [-1, 2, C3k2, [1024, True]]           # 19 P5 refined

  - [17, 1, FRM, []]                      # 20 FRM post-fusion @ P3
  - [15, 1, FRM, []]                      # 21 FRM post-fusion @ P4
  - [19, 1, FRM, []]                      # 22 FRM post-fusion @ P5

  - [[20, 21, 22], 1, Detect, [nc]]       # 23 Detect(P3, P4, P5)
"""


def ensure_meiscf_yaml(nc=10, path='yolo11s-meiscf.yaml'):
    """Write the MEISCF model YAML to disk (filename carries the 's' scale)."""
    path = Path(path)
    cfg = yaml.safe_load(MEISCF_YAML)
    cfg['nc'] = nc
    with open(path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=None)
    return path


def build_meiscf(nc=10, pretrained='yolo11s.pt'):
    """Build the MEISCF model and transfer the pretrained backbone if available."""
    register_meiscf_modules()
    yaml_path = ensure_meiscf_yaml(nc=nc)
    model = YOLO(str(yaml_path))
    if pretrained:
        try:
            model.load(pretrained)   # transfers matching layers (full stock backbone)
            logger.info(f"Loaded pretrained weights from {pretrained} (backbone transferred).")
        except Exception as e:
            logger.warning(f"Could not load pretrained {pretrained}: {e}")
    return model


# ============================================================================
# 4. MULTI-PHASE TRAINER
# ============================================================================

class MultiPhaseTrainer:
    """Four-phase progressive training (640 -> 800 -> 1024 -> 1280 px)."""

    def __init__(self, data_yaml, experiment_name='meiscf_multiphase', nc=10):
        self.data_yaml = data_yaml
        self.experiment_name = experiment_name
        self.nc = nc
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            logger.warning("CUDA not available -> training on CPU (very slow). "
                           "Fix the GPU driver / torch-CUDA mismatch before real runs.")
        self.project_dir = Path('runs/meiscf_multiphase')
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = {f'phase{i}': {} for i in range(1, 5)}

        logger.info("=" * 80)
        logger.info("MEISCF-YOLO: Multi-Phase Training")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device} | Experiment: {experiment_name}")

    def get_base_config(self):
        return {
            'data': self.data_yaml,
            'device': self.device,
            'workers': 8,
            'project': str(self.project_dir),
            'exist_ok': True,
            'patience': 200,          # >= epochs: run full schedule (matches paper protocol)
            'save': True,
            'save_period': 10,
            'plots': True,
            'val': True,
            'verbose': True,
            'optimizer': 'AdamW',
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_bias_lr': 0.1,
            'cos_lr': True,
            # Validation/NMS settings (match paper: conf 0.001, NMS IoU 0.5)
            'conf': 0.001,
            'iou': 0.5,
            'max_det': 300,
        }

    def _run_phase(self, phase, model, **overrides):
        config = self.get_base_config()
        config['name'] = f'{self.experiment_name}_phase{phase}'
        config.update(overrides)
        logger.info("\n" + "=" * 80)
        logger.info(f"PHASE {phase}: imgsz={config['imgsz']} batch={config['batch']} "
                    f"epochs={config['epochs']} lr0={config['lr0']}")
        logger.info("=" * 80)
        results = model.train(**config)
        self.metrics_history[f'phase{phase}'] = self._extract_metrics(results)
        self._log_phase_results(phase, self.metrics_history[f'phase{phase}'])
        best = self.project_dir / config['name'] / 'weights' / 'best.pt'
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best if best.exists() else None

    def phase1(self):
        model = build_meiscf(nc=self.nc, pretrained='yolo11s.pt')
        return self._run_phase(
            1, model, epochs=100, imgsz=640, batch=16,
            lr0=0.01, lrf=0.01, warmup_epochs=5.0, warmup_momentum=0.8,
            box=7.5, cls=0.5, dfl=1.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1,
            scale=0.9, shear=2.0, fliplr=0.5, mosaic=1.0, mixup=0.15,
            copy_paste=0.3, erasing=0.4, close_mosaic=20,
        )

    def phase2(self, prev):
        model = self._load_or_warn(prev)
        return self._run_phase(
            2, model, epochs=100, imgsz=800, batch=12,
            lr0=0.005, lrf=0.05, warmup_epochs=3.0, warmup_momentum=0.8,
            box=8.0, cls=0.6, dfl=1.6,
            hsv_h=0.012, hsv_s=0.6, hsv_v=0.35, degrees=7.0, translate=0.08,
            scale=0.8, shear=1.5, fliplr=0.5, mosaic=0.8, mixup=0.10,
            copy_paste=0.4, erasing=0.3, close_mosaic=15,
        )

    def phase3(self, prev):
        model = self._load_or_warn(prev)
        if model is None:
            return None
        return self._run_phase(
            3, model, epochs=100, imgsz=1024, batch=8,
            lr0=0.002, lrf=0.08, warmup_epochs=2.0, warmup_momentum=0.85,
            box=8.5, cls=0.7, dfl=1.7,
            hsv_h=0.010, hsv_s=0.5, hsv_v=0.3, degrees=5.0, translate=0.05,
            scale=0.7, shear=1.0, fliplr=0.5, mosaic=0.5, mixup=0.05,
            copy_paste=0.2, erasing=0.2, close_mosaic=10,
        )

    def phase4(self, prev):
        model = self._load_or_warn(prev)
        if model is None:
            return None
        return self._run_phase(
            4, model, epochs=50, imgsz=1280, batch=6,
            lr0=0.0005, lrf=0.1, warmup_epochs=1.0, warmup_momentum=0.9,
            box=9.0, cls=0.8, dfl=1.8,
            hsv_h=0.005, hsv_s=0.3, hsv_v=0.2, degrees=2.0, translate=0.02,
            scale=0.6, shear=0.5, fliplr=0.5, mosaic=0.0, mixup=0.0,
            copy_paste=0.1, erasing=0.1, close_mosaic=0,
        )

    def _load_or_warn(self, prev):
        if prev and Path(prev).exists():
            logger.info(f"Loading previous phase model: {prev}")
            return YOLO(str(prev))   # custom modules already registered this process
        logger.warning("Previous phase model not found.")
        return None

    def train_all_phases(self):
        start = datetime.now()
        p1 = self.phase1()
        p2 = self.phase2(p1) if p1 else None
        p3 = self.phase3(p2) if p2 else None
        p4 = self.phase4(p3) if p3 else None
        best = self._find_best_model()
        duration = datetime.now() - start
        logger.info("\n" + "=" * 80)
        logger.info(f"TRAINING COMPLETE in {duration} | Best: {best}")
        logger.info("=" * 80)
        self._save_metrics_summary(best, duration)
        return best

    def evaluate_best_model(self, model_path=None):
        model_path = model_path or self._find_best_model()
        if not model_path or not Path(model_path).exists():
            logger.error("No model found for evaluation.")
            return None
        register_meiscf_modules()
        model = YOLO(str(model_path))
        eval_results = {}
        for imgsz in [640, 800, 1024, 1280]:
            logger.info(f"Evaluating at {imgsz}px...")
            metrics = model.val(data=self.data_yaml, batch=4, imgsz=imgsz,
                                conf=0.001, iou=0.5, max_det=300,
                                plots=False, save_json=True)
            eval_results[imgsz] = {
                'mAP50': float(metrics.box.map50),
                'mAP50-95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
            }
            logger.info(f"  mAP@50={metrics.box.map50:.4f}  mAP@50-95={metrics.box.map:.4f}")
        with open(self.project_dir / 'final_evaluation.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        return eval_results

    def _find_best_model(self):
        for phase in (4, 3, 2, 1):
            p = self.project_dir / f'{self.experiment_name}_phase{phase}' / 'weights' / 'best.pt'
            if p.exists():
                logger.info(f"Best model: Phase {phase} - {p}")
                return p
        return None

    def _extract_metrics(self, results):
        try:
            if hasattr(results, 'results_dict'):
                return {
                    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                }
        except Exception:
            pass
        return {'mAP50': 0, 'mAP50-95': 0}

    def _log_phase_results(self, phase, metrics):
        logger.info(f"Phase {phase}: mAP@50={metrics['mAP50']:.4f}  "
                    f"mAP@50-95={metrics['mAP50-95']:.4f}")

    def _save_metrics_summary(self, best_model, duration):
        summary = {
            'experiment': self.experiment_name,
            'best_model': str(best_model),
            'duration': str(duration),
            'phases': self.metrics_history,
        }
        with open(self.project_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# ============================================================================
# 5. DATASET + SMOKE TEST + MAIN
# ============================================================================

def create_visdrone_yaml(dataset_path):
    dataset_path = Path(dataset_path)
    config = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 10,
        'names': ['pedestrian', 'people', 'bicycle', 'car', 'van',
                  'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'],
    }
    yaml_path = dataset_path / 'VisDrone.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Created dataset.yaml at {yaml_path}")
    return yaml_path


def smoke_test():
    """Build the model and run one forward pass. ~10s. No training."""
    logger.info("SMOKE TEST: building MEISCF and running a forward pass...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"torch.cuda.is_available()={torch.cuda.is_available()} -> using device '{device}'")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    model = build_meiscf(nc=10, pretrained='yolo11s.pt')
    net = model.model.to(device).eval()
    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"Total parameters: {n_params / 1e6:.2f}M")
    x = torch.zeros(1, 3, 640, 640, device=device)
    with torch.no_grad():
        out = net(x)
    feats = out[1] if isinstance(out, (list, tuple)) and len(out) == 2 else out
    try:
        shapes = [tuple(f.shape) for f in (feats if isinstance(feats, (list, tuple)) else [feats])]
        logger.info(f"Forward OK. Output tensor shapes: {shapes}")
    except Exception:
        logger.info("Forward OK.")
    logger.info("SMOKE TEST PASSED. The architecture builds and runs.")


def main():
    parser = argparse.ArgumentParser(description="MEISCF-YOLO training")
    parser.add_argument('--smoke', action='store_true',
                        help="Build model + forward pass only (no training).")
    parser.add_argument('--data', default='VisDrone_YOLO',
                        help="Path to YOLO-format VisDrone dataset root.")
    args = parser.parse_args()

    register_meiscf_modules()

    if args.smoke:
        smoke_test()
        return

    dataset_path = Path(args.data)
    yaml_file = dataset_path / 'VisDrone.yaml'
    data_yaml = str(yaml_file) if yaml_file.exists() else str(create_visdrone_yaml(dataset_path))

    experiment_name = f"meiscf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = MultiPhaseTrainer(data_yaml, experiment_name, nc=10)
    best_model = trainer.train_all_phases()
    if best_model:
        trainer.evaluate_best_model(best_model)
        logger.info(f"PIPELINE COMPLETE. Best model: {best_model}")
    else:
        logger.error("Training produced no model.")


if __name__ == "__main__":
    main()
