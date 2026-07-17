"""Automated ablation experiment runner.

Reproduces (and extends) every ablation in the paper:

  Table 2  - cumulative module ablation under progressive training:
             baseline(single-phase) -> +progressive -> +MEIS -> +SF -> +FRM
  Sec 3.3  - simultaneous 3-level vs sequential 2-level Sandwich Fusion
  Sec 3.4  - FRM pre-fusion vs post-fusion placement
  Eq  21   - FRM residual-blend alpha sweep (0.3 / 0.5 / 0.7)
  Sec 5.6  - multi-seed statistical validation (seeds 42, 123, 7)

Each experiment trains a variant, evaluates it, and records mAP. Because full
350-epoch runs are expensive, the runner accepts a reduced `epochs_per_phase`
for fast smoke/preview runs while keeping the same code path as full runs.
Everything is checkpointed to JSON so a crashed run can be resumed.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from .trainer import MultiPhaseTrainer
from .evaluate import evaluate_multi_resolution, model_complexity, benchmark_fps

logger = logging.getLogger(__name__)


class AblationRunner:
    def __init__(self, data_yaml, out_dir='runs/ablation', nc=10, scale='s',
                 pretrained='yolo11s.pt', eval_imgsz=1280, device=None,
                 batch_scale=1.0, workers=8, amp=False, optimizer='AdamW',
                 momentum=0.9):
        self.data_yaml = str(data_yaml)
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.nc = nc
        self.scale = scale
        self.pretrained = pretrained
        self.eval_imgsz = eval_imgsz
        self.device = device
        self.batch_scale = batch_scale
        self.workers = workers
        self.amp = amp
        self.optimizer = optimizer
        self.momentum = momentum
        self.results_path = self.out_dir / 'ablation_results.json'
        self.results = self._load_results()

    # ---- persistence --------------------------------------------------------
    def _load_results(self):
        if self.results_path.exists():
            try:
                return json.loads(self.results_path.read_text())
            except Exception:
                pass
        return {}

    def _save(self):
        self.results_path.write_text(json.dumps(self.results, indent=2))

    # ---- single experiment --------------------------------------------------
    def run_experiment(self, exp_id, variant, phases=(1, 2, 3, 4), seed=0,
                       frm_alpha=0.5, single_phase=False, epochs_per_phase=None,
                       force=False):
        """Train + evaluate one variant; cache by exp_id. Returns the record."""
        if exp_id in self.results and not force:
            logger.info(f"[skip] '{exp_id}' already complete "
                        f"(mAP@50={self.results[exp_id].get('mAP50')}). Use force=True to rerun.")
            return self.results[exp_id]

        logger.info("\n" + "#" * 80)
        logger.info(f"# ABLATION '{exp_id}': variant={variant} seed={seed} "
                    f"alpha={frm_alpha} single_phase={single_phase} "
                    f"optimizer={self.optimizer} momentum={self.momentum}")
        logger.info("#" * 80)

        trainer = MultiPhaseTrainer(
            self.data_yaml, variant=variant, experiment_name=exp_id, nc=self.nc,
            scale=self.scale, pretrained=self.pretrained, frm_alpha=frm_alpha,
            project_dir=str(self.out_dir), device=self.device,
            workers=self.workers, seed=seed, batch_scale=self.batch_scale,
            amp=self.amp, optimizer=self.optimizer, momentum=self.momentum)

        if single_phase:
            best = trainer.train_single_phase(imgsz=640, epochs=100, batch=16,
                                              lr0=0.01, name='single')
        else:
            best = trainer.train(phases=phases, epochs_per_phase=epochs_per_phase)

        record = {'variant': variant, 'seed': seed, 'frm_alpha': frm_alpha,
                  'single_phase': single_phase, 'phases': list(phases),
                  'best_model': str(best) if best else None,
                  'timestamp': datetime.now().isoformat()}

        if best:
            ev = evaluate_multi_resolution(
                best, self.data_yaml, imgszs=(self.eval_imgsz,),
                out_json=trainer.project_dir / 'eval_multi_res.json')
            r = ev[str(self.eval_imgsz)]
            record.update({'mAP50': r['mAP50'], 'mAP50-95': r['mAP50-95'],
                           'precision': r['precision'], 'recall': r['recall'],
                           'per_class_AP50': r['per_class_AP50']})
            try:
                record['complexity'] = model_complexity(best, imgsz=640)
                record['fps'] = benchmark_fps(best, imgsz=self.eval_imgsz)['fps']
            except Exception as e:
                logger.warning(f"complexity/fps failed for {exp_id}: {e}")

        self.results[exp_id] = record
        self._save()
        return record

    # ---- experiment suites --------------------------------------------------
    def run_cumulative_module_study(self, epochs_per_phase=None):
        """Paper Table 2: cumulative architectural ablation."""
        # Control row: baseline at 640px single-phase (no progressive curriculum).
        self.run_experiment('A0_baseline_singlephase', 'baseline',
                            single_phase=True)
        # Progressive baseline (no custom modules).
        self.run_experiment('A1_baseline_progressive', 'baseline',
                            epochs_per_phase=epochs_per_phase)
        # + MEIS
        self.run_experiment('A2_meis', 'meis', epochs_per_phase=epochs_per_phase)
        # + Sandwich Fusion
        self.run_experiment('A3_meis_sf', 'meis_sf', epochs_per_phase=epochs_per_phase)
        # + FRM (full model)
        self.run_experiment('A4_full', 'full', epochs_per_phase=epochs_per_phase)
        return self.results

    def run_fusion_study(self, epochs_per_phase=None):
        """3-level (full) vs 2-level (sf_sequential) Sandwich Fusion."""
        self.run_experiment('F_3level_full', 'full', epochs_per_phase=epochs_per_phase)
        self.run_experiment('F_2level_sequential', 'sf_sequential',
                            epochs_per_phase=epochs_per_phase)
        return self.results

    def run_frm_placement_study(self, epochs_per_phase=None):
        """FRM post-fusion (full) vs pre-fusion placement."""
        self.run_experiment('P_postfusion_full', 'full', epochs_per_phase=epochs_per_phase)
        self.run_experiment('P_prefusion', 'frm_prefusion',
                            epochs_per_phase=epochs_per_phase)
        return self.results

    def run_frm_alpha_sweep(self, alphas=(0.3, 0.5, 0.7), epochs_per_phase=None):
        """FRM residual-blend coefficient sweep (Eq. 21)."""
        for a in alphas:
            self.run_experiment(f'Alpha_{a}', 'full', frm_alpha=a,
                                epochs_per_phase=epochs_per_phase)
        return self.results

    def run_multiseed_study(self, seeds=(42, 123, 7), epochs_per_phase=None):
        """Statistical validation: full model vs progressive baseline across seeds."""
        for s in seeds:
            self.run_experiment(f'S_full_seed{s}', 'full', seed=s,
                                epochs_per_phase=epochs_per_phase)
            self.run_experiment(f'S_baseline_seed{s}', 'baseline', seed=s,
                                epochs_per_phase=epochs_per_phase)
        return self.results

    def run_all(self, epochs_per_phase=None, include_multiseed=False):
        """Run every ablation suite. Set include_multiseed for the 6-extra runs."""
        self.run_cumulative_module_study(epochs_per_phase)
        self.run_fusion_study(epochs_per_phase)
        self.run_frm_placement_study(epochs_per_phase)
        self.run_frm_alpha_sweep(epochs_per_phase=epochs_per_phase)
        if include_multiseed:
            self.run_multiseed_study(epochs_per_phase=epochs_per_phase)
        self._write_summary_table()
        return self.results

    # ---- reporting ----------------------------------------------------------
    def _write_summary_table(self):
        """Emit a human-readable cumulative ablation table (CSV + markdown)."""
        order = ['A0_baseline_singlephase', 'A1_baseline_progressive',
                 'A2_meis', 'A3_meis_sf', 'A4_full']
        labels = {
            'A0_baseline_singlephase': 'YOLOv11 baseline (640px, single-phase)',
            'A1_baseline_progressive': '+ Progressive training',
            'A2_meis': '+ MEIS',
            'A3_meis_sf': '+ MEIS + Sandwich Fusion',
            'A4_full': '+ MEIS + Sandwich Fusion + FRM (full MEISCF)',
        }
        rows, prev, base = [], None, None
        for k in order:
            if k not in self.results:
                continue
            m = self.results[k].get('mAP50')
            if m is None:
                continue
            m100 = m * 100
            base = m100 if base is None else base
            delta_prev = '' if prev is None else f"{m100 - prev:+.1f}"
            delta_base = f"{m100 - base:+.1f}"
            rows.append((labels[k], f"{m100:.1f}", delta_prev, delta_base))
            prev = m100

        # CSV
        csv = "Configuration,mAP@50 (%),Delta vs prev,Delta vs baseline\n"
        csv += "\n".join(",".join(r) for r in rows)
        (self.out_dir / 'ablation_table.csv').write_text(csv)
        # Markdown
        md = "| Configuration | mAP@50 (%) | Δ prev | Δ baseline |\n"
        md += "|---|---|---|---|\n"
        md += "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)
        (self.out_dir / 'ablation_table.md').write_text(md)
        logger.info(f"Ablation summary table -> {self.out_dir / 'ablation_table.md'}")
        return rows
