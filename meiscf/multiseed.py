"""Multi-seed training runs + statistical aggregation.

Trains the requested variants across several seeds (cached/resumable: a
(variant, seed) pair whose evaluation JSON already exists is skipped), then
aggregates every matching run under runs_dir into mean +/- std tables.

Aggregation is name-independent: it reads each experiment's
training_summary.json (which records 'variant' and 'seed'), so previously
completed timestamped runs (e.g. the seed-0 meis_p2/baseline_p2 runs) are
included automatically -- no renaming needed. If a seed appears twice for a
variant, the most recently modified run wins.

The paired delta (variant A - variant B on the SAME seed) is reported when both
variants share seeds; pairing removes seed-to-seed variance and is the correct
statistic for the "module contribution" claim.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_multiseed(data_yaml, variants, seeds, runs_dir='runs/meiscf', nc=10,
                  scale='s', pretrained='yolo11s.pt', device=None, workers=8,
                  batch_scale=1.0, amp=False, optimizer='SGD',
                  eval_imgszs=(1280, 1536), phases=(1, 2, 3, 4),
                  epochs_per_phase=None, eval_batch=2):
    """Train + evaluate each (variant, seed); skip pairs already evaluated."""
    from .trainer import MultiPhaseTrainer
    from .evaluate import evaluate_multi_resolution

    runs_dir = Path(runs_dir)
    for variant in variants:
        for seed in seeds:
            exp = f"{variant}_seed{seed}"
            exp_dir = runs_dir / exp
            done_marker = exp_dir / 'evaluation' / 'multi_resolution.json'
            if done_marker.exists():
                logger.info(f"[skip] {exp}: already trained + evaluated.")
                continue

            logger.info("\n" + "#" * 80)
            logger.info(f"# MULTISEED RUN: variant={variant} seed={seed}")
            logger.info("#" * 80)
            trainer = MultiPhaseTrainer(
                data_yaml, variant=variant, experiment_name=exp, nc=nc,
                scale=scale, pretrained=pretrained, project_dir=str(runs_dir),
                device=device, workers=workers, seed=seed,
                batch_scale=batch_scale, amp=amp, optimizer=optimizer)
            best = trainer.train(phases=tuple(phases),
                                 epochs_per_phase=epochs_per_phase)
            if not best:
                logger.error(f"{exp}: training produced no checkpoint; continuing.")
                continue
            # The trainer creates the experiment dir but not the evaluation
            # subdir; create it before writing the results JSON.
            done_marker.parent.mkdir(parents=True, exist_ok=True)
            # Free the training graph before high-res validation (P2 @1536 is
            # memory-hungry; leftover cache from training can trigger OOM).
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            evaluate_multi_resolution(
                best, data_yaml, imgszs=tuple(eval_imgszs), batch=eval_batch,
                max_det=600, out_json=done_marker)

    return aggregate_multiseed(runs_dir, variants)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _collect_runs(runs_dir, variants):
    """Scan runs_dir for completed experiments of the given variants.

    Returns {variant: {seed: {'mAP50': {imgsz: v}, 'mAP50-95': {imgsz: v},
                              'dir': str}}}, deduped by newest run per seed.
    """
    runs_dir = Path(runs_dir)
    found = {v: {} for v in variants}
    for summary_path in runs_dir.glob('*/training_summary.json'):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        variant = summary.get('variant')
        seed = summary.get('seed')
        if variant not in found or seed is None:
            continue
        # Only aggregate multi-phase (progressive) runs. A single-resolution
        # control records phases_run like ['single1280'] under the SAME variant
        # and seed, and would otherwise silently replace the progressive run.
        phases_run = summary.get('phases_run') or []
        if any(isinstance(p, str) and p.startswith('single') for p in phases_run):
            continue

        # Merge every evaluation*/multi_resolution.json in the experiment dir
        # (covers both the auto 'evaluation/' and manual 'evaluation_1536/').
        exp_dir = summary_path.parent
        map50, map5095 = {}, {}
        for mr_path in sorted(exp_dir.glob('evaluation*/multi_resolution.json')):
            try:
                mr = json.loads(mr_path.read_text())
            except Exception:
                continue
            for imgsz, rec in mr.items():
                if isinstance(rec, dict) and 'mAP50' in rec:
                    map50[imgsz] = rec['mAP50']
                    map5095[imgsz] = rec.get('mAP50-95')
        if not map50:
            continue

        rec = {'mAP50': map50, 'mAP50-95': map5095, 'dir': str(exp_dir),
               'mtime': summary_path.stat().st_mtime}
        prev = found[variant].get(seed)
        if prev is None or rec['mtime'] > prev['mtime']:
            found[variant][seed] = rec
    return found


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    arr = np.array(vals, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def aggregate_multiseed(runs_dir, variants, imgszs=('1280', '1536'),
                        out_name='multiseed_summary'):
    """Aggregate all runs into per-variant mean+/-std and paired deltas."""
    runs_dir = Path(runs_dir)
    found = _collect_runs(runs_dir, variants)

    report = {'generated': datetime.now().isoformat(), 'variants': {}}
    md = ["# Multi-seed summary", ""]

    for variant in variants:
        runs = found.get(variant, {})
        seeds = sorted(runs)
        vrec = {'seeds': seeds, 'runs': {str(s): runs[s]['dir'] for s in seeds},
                'per_seed': {}, 'stats': {}}
        md.append(f"## {variant}  (seeds: {seeds or 'none found'})")
        md.append("")
        md.append("| seed | " + " | ".join(f"mAP50@{z}" for z in imgszs) + " |")
        md.append("|---" * (len(imgszs) + 1) + "|")
        for s in seeds:
            row = [f"{runs[s]['mAP50'].get(z, float('nan'))*100:.2f}"
                   if runs[s]['mAP50'].get(z) is not None else "-" for z in imgszs]
            md.append(f"| {s} | " + " | ".join(row) + " |")
            vrec['per_seed'][str(s)] = {z: runs[s]['mAP50'].get(z) for z in imgszs}
        for z in imgszs:
            m, sd = _mean_std([runs[s]['mAP50'].get(z) for s in seeds])
            if m is not None:
                vrec['stats'][z] = {'mean_mAP50': m, 'std_mAP50': sd, 'n': len(seeds)}
        if vrec['stats']:
            md.append("| **mean+/-std** | " + " | ".join(
                f"**{vrec['stats'][z]['mean_mAP50']*100:.2f} +/- "
                f"{vrec['stats'][z]['std_mAP50']*100:.2f}**"
                if z in vrec['stats'] else "-" for z in imgszs) + " |")
        md.append("")
        report['variants'][variant] = vrec

    # Paired delta between the first two variants (e.g. meis_p2 - baseline_p2).
    if len(variants) >= 2:
        a, b = variants[0], variants[1]
        common = sorted(set(found.get(a, {})) & set(found.get(b, {})))
        if common:
            md.append(f"## Paired delta: {a} - {b}  (same-seed pairs: {common})")
            md.append("")
            md.append("| seed | " + " | ".join(f"d mAP50@{z} (pp)" for z in imgszs) + " |")
            md.append("|---" * (len(imgszs) + 1) + "|")
            deltas = {z: [] for z in imgszs}
            for s in common:
                row = []
                for z in imgszs:
                    va, vb = found[a][s]['mAP50'].get(z), found[b][s]['mAP50'].get(z)
                    if va is not None and vb is not None:
                        d = (va - vb) * 100
                        deltas[z].append(d)
                        row.append(f"{d:+.2f}")
                    else:
                        row.append("-")
                md.append(f"| {s} | " + " | ".join(row) + " |")
            stats_row, pair_stats = [], {}
            for z in imgszs:
                m, sd = _mean_std(deltas[z])
                if m is not None:
                    pair_stats[z] = {'mean_delta_pp': m, 'std_delta_pp': sd,
                                     'n_pairs': len(deltas[z])}
                    stats_row.append(f"**{m:+.2f} +/- {sd:.2f}**")
                else:
                    stats_row.append("-")
            md.append("| **mean+/-std** | " + " | ".join(stats_row) + " |")
            md.append("")
            # Seed-level paired t-interval: is the module effect distinguishable
            # from retraining noise? (reviewer request for a significance test)
            from .significance import seed_level_summary
            seed_tests = {}
            for z in imgszs:
                if deltas[z]:
                    seed_tests[z] = seed_level_summary(deltas[z])
            if seed_tests:
                md.append("### Seed-level paired test (t-interval on per-seed deltas)")
                md.append("")
                md.append("| res | n | mean (pp) | 95% CI | significant at 95% |")
                md.append("|---|---|---|---|---|")
                for z, s in seed_tests.items():
                    if 'ci_low_pp' in s:
                        md.append(f"| {z} | {s['n']} | {s['mean_pp']:+.2f} | "
                                  f"[{s['ci_low_pp']:+.2f}, {s['ci_high_pp']:+.2f}] | "
                                  f"**{s['significant_at_95']}** |")
                md.append("")
            report['paired_delta'] = {'a': a, 'b': b, 'seeds': common,
                                      'stats': pair_stats, 'seed_level': seed_tests}

    out_json = runs_dir / f'{out_name}.json'
    out_md = runs_dir / f'{out_name}.md'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    out_md.write_text("\n".join(md), encoding='utf-8')
    logger.info(f"Multi-seed summary -> {out_md}")
    print("\n".join(md))
    return report
