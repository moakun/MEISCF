"""Research-grade figures: training curves, ablation bars, per-class bars,
class/size distribution pies, multi-resolution lines, and a SOTA comparison.

All functions consume the JSON / CSV artifacts produced by the trainer,
evaluator, and ablation runner, so figures can be regenerated without retraining.
Uses a non-interactive matplotlib backend (safe on headless GPU servers).
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200, 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True,
})

VISDRONE_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                  'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
_C = plt.cm.tab10(np.linspace(0, 1, 10))


def _save(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved figure -> {out_path}")
    return out_path


def _read_csv(path):
    """Read an Ultralytics results.csv into a dict of column -> np.array."""
    import csv
    path = Path(path)
    with open(path) as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    cols = {h: [] for h in header}
    for r in rows[1:]:
        if not r:
            continue
        for h, v in zip(header, r):
            try:
                cols[h].append(float(v))
            except ValueError:
                cols[h].append(np.nan)
    return {h: np.array(v) for h, v in cols.items()}


# ---------------------------------------------------------------------------
# 1. Training curves across the four phases (paper Fig. 5)
# ---------------------------------------------------------------------------
def plot_training_progression(phase_csvs, out_path, phase_epochs=(100, 100, 100, 50)):
    """Stitch per-phase results.csv files into the full-training-progression
    figure: (a) losses, (b) mAP, (c) precision/recall, (d) LR schedule."""
    data = []
    offset = 0
    boundaries = []
    for csv_path in phase_csvs:
        if not Path(csv_path).exists():
            logger.warning(f"Missing results.csv: {csv_path}")
            continue
        d = _read_csv(csv_path)
        n = len(next(iter(d.values())))
        d['_global_epoch'] = np.arange(offset, offset + n)
        offset += n
        boundaries.append(offset)
        data.append(d)
    if not data:
        logger.warning("No phase CSVs found; skipping training-progression figure.")
        return None

    def cat(key):
        out = []
        for d in data:
            out.append(d.get(key, np.full(len(d['_global_epoch']), np.nan)))
        return np.concatenate(out)

    ep = cat('_global_epoch')
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # (a) losses
    for key, lab in [('train/box_loss', 'box'), ('train/cls_loss', 'cls'),
                     ('train/dfl_loss', 'dfl')]:
        y = cat(key)
        if np.isfinite(y).any():
            axs[0, 0].plot(ep, y, label=lab)
    axs[0, 0].set_title('(a) Training Loss Components')
    axs[0, 0].set_xlabel('Epoch'); axs[0, 0].set_ylabel('Loss'); axs[0, 0].legend()

    # (b) mAP
    for key, lab in [('metrics/mAP50(B)', 'mAP@50'),
                     ('metrics/mAP50-95(B)', 'mAP@50-95')]:
        y = cat(key)
        if np.isfinite(y).any():
            axs[0, 1].plot(ep, y, label=lab, linewidth=2)
    axs[0, 1].set_title('(b) Validation mAP'); axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('mAP'); axs[0, 1].legend()

    # (c) precision / recall
    for key, lab in [('metrics/precision(B)', 'precision'),
                     ('metrics/recall(B)', 'recall')]:
        y = cat(key)
        if np.isfinite(y).any():
            axs[1, 0].plot(ep, y, label=lab, linewidth=2)
    axs[1, 0].set_title('(c) Precision & Recall'); axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Score'); axs[1, 0].legend()

    # (d) LR schedule
    y = cat('lr/pg0')
    if np.isfinite(y).any():
        axs[1, 1].plot(ep, y, color='tab:red', linewidth=2)
    axs[1, 1].set_title('(d) Learning Rate Schedule'); axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('LR')

    # phase boundary markers
    for ax in axs.flat:
        for b in boundaries[:-1]:
            ax.axvline(b, color='gray', linestyle='--', alpha=0.5)
    fig.suptitle('MEISCF Training Progression Across Four Resolution Phases',
                 fontsize=14, y=1.01)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Cumulative ablation bar chart (paper Table 2 visualized)
# ---------------------------------------------------------------------------
def plot_ablation_bars(ablation_results, out_path):
    """Bar chart of cumulative mAP@50 across the module-ablation ladder."""
    order = [('A0_baseline_singlephase', 'Baseline\n(single-phase)'),
             ('A1_baseline_progressive', '+ Progressive'),
             ('A2_meis', '+ MEIS'),
             ('A3_meis_sf', '+ Sandwich\nFusion'),
             ('A4_full', '+ FRM\n(full)')]
    labels, vals = [], []
    for k, lab in order:
        if k in ablation_results and ablation_results[k].get('mAP50') is not None:
            labels.append(lab); vals.append(ablation_results[k]['mAP50'] * 100)
    if not vals:
        logger.warning("No ablation results to plot."); return None
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, vals, color=plt.cm.viridis(np.linspace(0.2, 0.85, len(vals))))
    for b, v, prev in zip(bars, vals, [None] + vals[:-1]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                ha='center', fontweight='bold')
        if prev is not None:
            ax.text(b.get_x() + b.get_width() / 2, v / 2, f"+{v - prev:.1f}",
                    ha='center', color='white', fontweight='bold')
    ax.set_ylabel('mAP@50 (%)'); ax.set_title('Cumulative Module Ablation (VisDrone val)')
    ax.set_ylim(0, max(vals) * 1.12)
    return _save(fig, out_path)


def plot_module_contribution_pie(ablation_results, out_path):
    """Pie of how each component contributes to the total gain over baseline."""
    need = ['A0_baseline_singlephase', 'A1_baseline_progressive', 'A2_meis',
            'A3_meis_sf', 'A4_full']
    if not all(k in ablation_results and ablation_results[k].get('mAP50') is not None
               for k in need):
        logger.warning("Incomplete ablation results; skipping contribution pie.")
        return None
    v = {k: ablation_results[k]['mAP50'] * 100 for k in need}
    parts = {
        'Progressive training': v['A1_baseline_progressive'] - v['A0_baseline_singlephase'],
        'MEIS': v['A2_meis'] - v['A1_baseline_progressive'],
        'Sandwich Fusion': v['A3_meis_sf'] - v['A2_meis'],
        'FRM': v['A4_full'] - v['A3_meis_sf'],
    }
    parts = {k: max(val, 0) for k, val in parts.items()}
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(list(parts.values()), labels=list(parts.keys()),
           autopct=lambda p: f"{p:.1f}%\n(+{p / 100 * sum(parts.values()):.1f}pp)",
           colors=plt.cm.Set2(np.linspace(0, 1, len(parts))), startangle=90)
    ax.set_title(f'Contribution to Total Gain (+{sum(parts.values()):.1f} pp over baseline)')
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Per-class AP comparison (paper Fig. 6)
# ---------------------------------------------------------------------------
def plot_per_class_comparison(baseline_json, meiscf_json, out_path, names=None):
    """Grouped bar chart: baseline vs MEISCF per-class AP@50."""
    names = names or VISDRONE_NAMES
    base = json.loads(Path(baseline_json).read_text())['per_class']
    ours = json.loads(Path(meiscf_json).read_text())['per_class']
    b = [base.get(n, {}).get('AP50', 0) * 100 for n in names]
    o = [ours.get(n, {}).get('AP50', 0) * 100 for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w / 2, b, w, label='YOLOv11 baseline', color='tab:gray')
    ax.bar(x + w / 2, o, w, label='MEISCF (ours)', color='tab:green')
    for i, (bb, oo) in enumerate(zip(b, o)):
        ax.text(i + w / 2, oo + 0.5, f"+{oo - bb:.1f}", ha='center',
                fontsize=8, color='darkgreen')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('AP@50 (%)'); ax.legend()
    ax.set_title('Per-Class AP@50: MEISCF vs Baseline')
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Multi-resolution line chart
# ---------------------------------------------------------------------------
def plot_multi_resolution(multi_res_json, out_path, label='MEISCF'):
    """Line chart of mAP vs evaluation resolution."""
    d = json.loads(Path(multi_res_json).read_text())
    imgszs = sorted(int(k) for k in d)
    m50 = [d[str(s)]['mAP50'] * 100 for s in imgszs]
    m5095 = [d[str(s)]['mAP50-95'] * 100 for s in imgszs]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(imgszs, m50, 'o-', label='mAP@50', linewidth=2)
    ax.plot(imgszs, m5095, 's-', label='mAP@50-95', linewidth=2)
    for s, y in zip(imgszs, m50):
        ax.annotate(f"{y:.1f}", (s, y), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=9)
    ax.set_xlabel('Evaluation resolution (px)'); ax.set_ylabel('mAP (%)')
    ax.set_title(f'{label}: Performance vs Resolution'); ax.legend()
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Dataset distribution figures (class pie + object-size histogram)
# ---------------------------------------------------------------------------
def plot_dataset_distribution(dataset_stats_json, out_dir, split='train', names=None):
    """Class-distribution pie + bar and object-size histogram from convert stats."""
    names = names or VISDRONE_NAMES
    stats = json.loads(Path(dataset_stats_json).read_text())
    if split not in stats:
        split = next(iter(stats))
    cc = stats[split]['class_counts']
    counts = [cc.get(str(i), cc.get(i, 0)) for i in range(len(names))]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Pie
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(counts, labels=names, autopct='%1.1f%%', colors=_C, startangle=90,
           textprops={'fontsize': 9})
    ax.set_title(f'VisDrone {split}: Class Distribution ({sum(counts):,} objects)')
    paths.append(_save(fig, out_dir / f'class_distribution_pie_{split}.png'))

    # Bar (log scale, class imbalance is large)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(names)), counts, color=_C)
    ax.set_yscale('log'); ax.set_ylabel('object count (log)')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title(f'VisDrone {split}: Objects per Class')
    paths.append(_save(fig, out_dir / f'class_distribution_bar_{split}.png'))

    # Object-size histogram
    sizes = stats[split].get('obj_sizes', [])
    if sizes:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(np.clip(sizes, 0, 200), bins=50, color='tab:blue', alpha=0.8)
        ax.axvline(32, color='red', linestyle='--', label='32px (small-object threshold)')
        small = np.mean(np.array(sizes) < 32) * 100
        ax.set_xlabel('object size = sqrt(w*h) (px)'); ax.set_ylabel('count')
        ax.set_title(f'VisDrone {split}: Object-Size Distribution '
                     f'({small:.0f}% below 32px)'); ax.legend()
        paths.append(_save(fig, out_dir / f'object_size_hist_{split}.png'))
    return paths


# ---------------------------------------------------------------------------
# 6. SOTA comparison (mAP vs FPS bubble, bubble size = params)
# ---------------------------------------------------------------------------
def plot_sota_comparison(out_path, extra=None):
    """Scatter of published methods (from the paper) + your measured MEISCF."""
    methods = {
        'Faster R-CNN': (28.3, 12, 41.5), 'SSD512': (23.5, 28, 26.3),
        'YOLOv5s': (33.5, 182, 7.2), 'YOLOv7': (35.8, 48, 36.9),
        'TPH-YOLOv5': (40.3, 73, 13.7), 'YOLOv8s': (37.2, 65, 11.1),
        'YOLOv11s': (37.5, 149, 9.4), 'CF-YOLO': (44.9, 52, 10.8),
        'MEISCF (ours)': (55.1, 105, 12.0),
    }
    if extra:
        methods.update(extra)
    fig, ax = plt.subplots(figsize=(11, 7))
    for name, (mAP, fps, params) in methods.items():
        ours = 'ours' in name.lower()
        ax.scatter(fps, mAP, s=params * 25, alpha=0.7,
                   color='tab:red' if ours else 'tab:blue',
                   edgecolors='black', zorder=3 if ours else 2)
        ax.annotate(name, (fps, mAP), textcoords='offset points',
                    xytext=(6, 6), fontsize=9,
                    fontweight='bold' if ours else 'normal')
    ax.set_xlabel('FPS (RTX 3060 @ 1024px)'); ax.set_ylabel('mAP@50 (%)')
    ax.set_title('Accuracy vs Speed on VisDrone (bubble size ∝ params)')
    return _save(fig, out_path)


def plot_alpha_sweep(ablation_results, out_path):
    """Line chart of FRM alpha vs mAP@50 (Eq. 21 sweep)."""
    pts = []
    for k, rec in ablation_results.items():
        if k.startswith('Alpha_') and rec.get('mAP50') is not None:
            pts.append((rec.get('frm_alpha', float(k.split('_')[1])), rec['mAP50'] * 100))
    if len(pts) < 2:
        logger.warning("Not enough alpha-sweep points; skipping."); return None
    pts.sort()
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, 'o-', linewidth=2, markersize=9)
    for x, y in pts:
        ax.annotate(f"{y:.1f}", (x, y), textcoords='offset points', xytext=(0, 9),
                    ha='center')
    ax.set_xlabel('FRM residual-blend coefficient α'); ax.set_ylabel('mAP@50 (%)')
    ax.set_title('FRM α Sweep (Eq. 21)')
    return _save(fig, out_path)


def generate_all(run_dir, ablation_json=None, dataset_stats=None,
                 baseline_per_class=None, meiscf_per_class=None,
                 multi_res_json=None, phase_csvs=None, out_dir=None):
    """Best-effort: produce every figure for which inputs are available."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir or run_dir / 'figures'); out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    if phase_csvs:
        p = plot_training_progression(phase_csvs, out_dir / 'training_progression.png')
        if p: made.append(p)
    if ablation_json and Path(ablation_json).exists():
        res = json.loads(Path(ablation_json).read_text())
        for fn, name in [(plot_ablation_bars, 'ablation_bars.png'),
                         (plot_module_contribution_pie, 'contribution_pie.png'),
                         (plot_alpha_sweep, 'alpha_sweep.png')]:
            p = fn(res, out_dir / name)
            if p: made.append(p)
    if dataset_stats and Path(dataset_stats).exists():
        made += plot_dataset_distribution(dataset_stats, out_dir)
    if baseline_per_class and meiscf_per_class and \
       Path(baseline_per_class).exists() and Path(meiscf_per_class).exists():
        p = plot_per_class_comparison(baseline_per_class, meiscf_per_class,
                                      out_dir / 'per_class_comparison.png')
        if p: made.append(p)
    if multi_res_json and Path(multi_res_json).exists():
        p = plot_multi_resolution(multi_res_json, out_dir / 'multi_resolution.png')
        if p: made.append(p)
    made.append(plot_sota_comparison(out_dir / 'sota_comparison.png'))
    logger.info(f"Generated {len(made)} figures in {out_dir}")
    return made
