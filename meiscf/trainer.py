"""Four-phase progressive trainer (640 -> 800 -> 1024 -> 1280 px) + single-phase.

Implements the paper's curriculum: resolution increases and learning rate /
augmentation strength decrease across phases. Each phase resumes from the best
checkpoint of the previous phase. Single-phase training is also provided for the
"baseline (640px, single-phase)" ablation control row.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

from .model_builder import build_model
from .registry import register_meiscf_modules

logger = logging.getLogger(__name__)


# Per-phase hyperparameters (paper Sec. 4).  SGD with paper LRs.
# NOTE: these LRs (0.01, 0.005, 0.002, 0.0005) are SGD-scale values. AdamW
# with these LRs and the paper's momentum=0.937 diverges to NaN/Inf.
# The original code used SGD (Ultralytics default) with these exact settings.
PHASE_CONFIGS = {
    1: dict(imgsz=640,  epochs=100, batch=16, lr0=0.01,   lrf=0.01, warmup_epochs=5.0,
            warmup_momentum=0.8,  box=7.5, cls=0.5, dfl=1.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  degrees=10.0, translate=0.1,
            scale=0.9, shear=2.0, fliplr=0.5, mosaic=1.0, mixup=0.15,
            copy_paste=0.3, erasing=0.4, close_mosaic=20),
    2: dict(imgsz=800,  epochs=100, batch=12, lr0=0.005,  lrf=0.05, warmup_epochs=3.0,
            warmup_momentum=0.8,  box=8.0, cls=0.6, dfl=1.6,
            hsv_h=0.012, hsv_s=0.6, hsv_v=0.35, degrees=7.0, translate=0.08,
            scale=0.8, shear=1.5, fliplr=0.5, mosaic=0.8, mixup=0.10,
            copy_paste=0.4, erasing=0.3, close_mosaic=15),
    3: dict(imgsz=1024, epochs=100, batch=8,  lr0=0.002,  lrf=0.08, warmup_epochs=2.0,
            warmup_momentum=0.85, box=8.5, cls=0.7, dfl=1.7,
            hsv_h=0.010, hsv_s=0.5, hsv_v=0.3,  degrees=5.0, translate=0.05,
            scale=0.7, shear=1.0, fliplr=0.5, mosaic=0.5, mixup=0.05,
            copy_paste=0.2, erasing=0.2, close_mosaic=10),
    4: dict(imgsz=1280, epochs=50,  batch=4,  lr0=0.0005, lrf=0.1,  warmup_epochs=1.0,
            warmup_momentum=0.9,  box=9.0, cls=0.8, dfl=1.8,
            hsv_h=0.005, hsv_s=0.3, hsv_v=0.2,  degrees=2.0, translate=0.02,
            scale=0.6, shear=0.5, fliplr=0.5, mosaic=0.0, mixup=0.0,
            copy_paste=0.1, erasing=0.1, close_mosaic=0),
}


class MultiPhaseTrainer:
    """Progressive trainer for a single model variant."""

    def __init__(self, data_yaml, variant='full', experiment_name=None, nc=10,
                 scale='s', pretrained='yolo11s.pt', frm_alpha=0.5,
                 project_dir='runs/meiscf', device=None, workers=8, seed=0,
                 batch_scale=1.0, amp=False, optimizer='AdamW', momentum=0.9):
        self.data_yaml = str(data_yaml)
        self.variant = variant
        self.nc = nc
        self.scale = scale
        self.pretrained = pretrained
        self.frm_alpha = frm_alpha
        self.workers = workers
        self.seed = seed
        self.batch_scale = batch_scale       # global multiplier to fit GPU memory
        self.amp = amp
        self.optimizer = optimizer
        self.momentum = momentum
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cpu':
            logger.warning("CUDA not available -> CPU training (very slow). "
                           "Fix the GPU/torch-CUDA setup before real runs.")
        self.experiment_name = experiment_name or \
            f"{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Absolute path: Ultralytics 8.4.x relocates *relative* project paths
        # under runs/detect, which broke cross-phase checkpoint discovery.
        self.project_dir = (Path(project_dir) / self.experiment_name).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = {}
        self.phase_dirs = {}          # phase -> directory Ultralytics actually used
        register_meiscf_modules()
        logger.info("=" * 80)
        logger.info(f"MEISCF trainer | variant={variant} scale={scale} "
                    f"device={self.device} seed={seed} optimizer={self.optimizer} "
                    f"momentum={self.momentum}")
        logger.info(f"Experiment dir: {self.project_dir}")
        logger.info("=" * 80)

    # ---- config helpers ----------------------------------------------------
    def _base_config(self):
        cfg = {
            'data': self.data_yaml,
            'device': self.device,
            'workers': self.workers,
            'project': str(self.project_dir),
            'exist_ok': True,
            'seed': self.seed,
            'deterministic': False,
            'patience': 200,            # >= max epochs: never early-stop (paper protocol)
            'save': True, 'save_period': 10,
            'plots': True, 'val': True, 'verbose': True,
            'optimizer': self.optimizer,
            'weight_decay': 0.0005, 'warmup_bias_lr': 0.1,
            'cos_lr': True,
            'conf': 0.001, 'iou': 0.5, 'max_det': 300,   # paper val/NMS settings
            'amp': self.amp,
        }
        # SGD momentum: 0.937 (paper value). AdamW was attempted but diverges
        # to NaN/Inf at Phase 2 with the paper's LRs; the original code used SGD.
        if self.optimizer.lower() == 'sgd':
            cfg['momentum'] = 0.937
        elif self.optimizer.lower() == 'adamw':
            cfg['momentum'] = self.momentum   # 0.9 is safe for AdamW; not used now.
        return cfg

    def _phase_overrides(self, phase, epochs_override=None):
        cfg = dict(PHASE_CONFIGS[phase])
        if self.batch_scale != 1.0:
            cfg['batch'] = max(1, int(round(cfg['batch'] * self.batch_scale)))
        if epochs_override:
            cfg['epochs'] = epochs_override
        return cfg

    # ---- phase execution ---------------------------------------------------
    def _run_phase(self, phase, model, epochs_override=None):
        cfg = self._base_config()
        cfg.update(self._phase_overrides(phase, epochs_override))
        cfg['name'] = f'phase{phase}'
        logger.info("\n" + "=" * 80)
        logger.info(f"PHASE {phase}: imgsz={cfg['imgsz']} batch={cfg['batch']} "
                    f"epochs={cfg['epochs']} lr0={cfg['lr0']}")
        logger.info("=" * 80)
        results = model.train(**cfg)
        self.metrics_history[f'phase{phase}'] = self._extract_metrics(results)
        m = self.metrics_history[f'phase{phase}']
        logger.info(f"Phase {phase} done: mAP@50={m['mAP50']:.4f} "
                    f"mAP@50-95={m['mAP50-95']:.4f}")
        # Read the directory Ultralytics actually wrote to, rather than
        # reconstructing it (version-robust: handles the runs/detect relocation).
        save_dir = Path(getattr(model.trainer, 'save_dir',
                                self.project_dir / f'phase{phase}'))
        self.phase_dirs[phase] = save_dir
        best = save_dir / 'weights' / 'best.pt'
        if not best.exists():
            logger.error(f"Expected checkpoint not found at {best}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best if best.exists() else None

    def _load_prev(self, prev):
        if prev and Path(prev).exists():
            logger.info(f"Resuming from previous phase: {prev}")
            return YOLO(str(prev))      # custom modules already registered
        logger.warning("Previous-phase checkpoint missing; cannot continue.")
        return None

    def train(self, phases=(1, 2, 3, 4), epochs_per_phase=None, resume_from=None):
        """Run the requested phases.

        epochs_per_phase : optional dict {phase: n} for quick/smoke runs.
        resume_from      : optional checkpoint to start the FIRST listed phase
                           from, instead of building fresh + loading the COCO
                           backbone. Use with phases=(3, 4) to redo only the
                           high-res phases from a good Phase-2 best.pt.
        """
        epochs_per_phase = epochs_per_phase or {}
        start = datetime.now()
        prev = None
        for i, phase in enumerate(phases):
            if i == 0 and resume_from:
                logger.info(f"Resuming phase {phase} from checkpoint: {resume_from}")
                model = self._load_prev(resume_from)
                if model is None:
                    logger.error(f"resume_from checkpoint not found: {resume_from}")
                    break
            elif i == 0:
                model, _ = build_model(self.variant, nc=self.nc, scale=self.scale,
                                       frm_alpha=self.frm_alpha,
                                       pretrained=self.pretrained)
            else:
                model = self._load_prev(prev)
                if model is None:
                    break
            prev = self._run_phase(phase, model, epochs_per_phase.get(phase))
            if prev is None:
                logger.error(f"Phase {phase} produced no checkpoint; stopping.")
                break
        best = self._find_best(phases)
        duration = datetime.now() - start
        logger.info("\n" + "=" * 80)
        logger.info(f"TRAINING COMPLETE in {duration} | best: {best}")
        logger.info("=" * 80)
        self._save_summary(best, duration, phases)
        return best

    def train_single_phase(self, imgsz=640, epochs=100, batch=16, lr0=0.01,
                           name='single'):
        """Single-resolution training (baseline control / quick experiments)."""
        model, _ = build_model(self.variant, nc=self.nc, scale=self.scale,
                               frm_alpha=self.frm_alpha, pretrained=self.pretrained)
        cfg = self._base_config()
        cfg.update(self._phase_overrides(1))   # phase-1 augmentation as the base
        cfg.update(dict(imgsz=imgsz, epochs=epochs,
                        batch=max(1, int(round(batch * self.batch_scale))),
                        lr0=lr0, name=name))
        logger.info(f"SINGLE-PHASE: imgsz={imgsz} epochs={epochs} batch={cfg['batch']} lr0={lr0}")
        start = datetime.now()
        results = model.train(**cfg)
        duration = datetime.now() - start
        self.metrics_history[name] = self._extract_metrics(results)
        save_dir = Path(getattr(model.trainer, 'save_dir', self.project_dir / name))
        self.phase_dirs[name] = save_dir
        best = save_dir / 'weights' / 'best.pt'
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._save_summary(best, duration, [name])
        return best if best.exists() else None

    # ---- bookkeeping --------------------------------------------------------
    def _find_best(self, phases):
        for phase in sorted(phases, reverse=True):
            d = self.phase_dirs.get(phase)
            if d:
                p = Path(d) / 'weights' / 'best.pt'
                if p.exists():
                    return p
        return None

    @staticmethod
    def _extract_metrics(results):
        try:
            d = getattr(results, 'results_dict', {}) or {}
            return {
                'mAP50': float(d.get('metrics/mAP50(B)', 0.0)),
                'mAP50-95': float(d.get('metrics/mAP50-95(B)', 0.0)),
                'precision': float(d.get('metrics/precision(B)', 0.0)),
                'recall': float(d.get('metrics/recall(B)', 0.0)),
            }
        except Exception:
            return {'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0}

    def _save_summary(self, best, duration, phases):
        summary = {
            'experiment': self.experiment_name,
            'variant': self.variant,
            'scale': self.scale,
            'seed': self.seed,
            'phases_run': list(phases),
            'best_model': str(best) if best else None,
            'duration': str(duration) if duration else None,
            'metrics_per_phase': self.metrics_history,
            'phase_dirs': {str(k): str(v) for k, v in self.phase_dirs.items()},
        }
        (self.project_dir / 'training_summary.json').write_text(
            json.dumps(summary, indent=2))
        return summary
