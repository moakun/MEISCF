#!/usr/bin/env python
"""MEISCF-YOLO unified CLI.

Subcommands (all read defaults from config.yaml; CLI flags override):

  python run.py smoke                 Build every variant + forward pass (no training)
  python run.py prepare               Convert VisDrone -> YOLO format + dataset stats
  python run.py train                 Four-phase progressive training of one variant
  python run.py ablation              Run the full ablation suite (+ optional multiseed)
  python run.py evaluate  --weights W Multi-res mAP, per-class AP, FPS, params/FLOPs
  python run.py heatmaps  --weights W [--baseline B]  Comparative saliency figures
  python run.py visualize             (Re)generate all figures from existing JSON/CSV
  python run.py all                   prepare -> train -> evaluate -> heatmaps -> visualize

The whole pipeline is driven by config.yaml. Set the three server paths there
(pretrained yolo11s.pt, raw dataset, output dirs) and you are ready to go.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

os.environ.setdefault('PYTHONWARNINGS', 'ignore')
import warnings
warnings.filterwarnings('ignore')

import yaml

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('meiscf')


# ---------------------------------------------------------------------------
def load_config(path='config.yaml', overrides=None):
    cfg = yaml.safe_load(Path(path).read_text())
    if overrides:
        for dotted, val in overrides.items():
            d = cfg
            keys = dotted.split('.')
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val
    return cfg


def _p(cfg, key):
    return cfg['paths'][key]


# ---------------------------------------------------------------------------
def cmd_smoke(cfg, args):
    """Build every variant and run a forward pass to prove the graph compiles."""
    import torch
    from meiscf.model_builder import build_model, VARIANTS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"cuda.is_available={torch.cuda.is_available()} -> device '{device}'")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    pre = _p(cfg, 'pretrained')
    pre = pre if Path(pre).exists() else None
    x = torch.zeros(1, 3, 640, 640, device=device)
    for v in VARIANTS:
        model, ypath = build_model(v, nc=cfg['model']['nc'],
                                   scale=cfg['model']['scale'],
                                   frm_alpha=cfg['model']['frm_alpha'],
                                   pretrained=pre)
        net = model.model.to(device).eval()
        n = sum(p.numel() for p in net.parameters()) / 1e6
        with torch.no_grad():
            out = net(x)
        feats = out[1] if isinstance(out, (list, tuple)) and len(out) == 2 else out
        shapes = [tuple(f.shape) for f in feats] if isinstance(feats, (list, tuple)) else 'ok'
        logger.info(f"[OK] variant='{v:14s}' params={n:5.2f}M yaml={ypath.name} out={shapes}")
        del model, net
        if device == 'cuda':
            torch.cuda.empty_cache()
    logger.info("SMOKE TEST PASSED: all variants build and run.")


def cmd_prepare(cfg, args):
    from meiscf.data_prep import ensure_dataset
    yaml_path = ensure_dataset(_p(cfg, 'raw_dataset'),
                               out_root=_p(cfg, 'yolo_dataset'),
                               link_images=not args.copy_images)
    logger.info(f"Dataset ready: {yaml_path}")
    print(yaml_path)
    return yaml_path


def _resolve_data_yaml(cfg):
    """Return a usable dataset YAML, converting from raw if necessary."""
    from meiscf.data_prep import ensure_dataset
    yroot = Path(_p(cfg, 'yolo_dataset'))
    existing = yroot / 'VisDrone.yaml'
    if existing.exists():
        return existing
    return ensure_dataset(_p(cfg, 'raw_dataset'), out_root=str(yroot))


def cmd_train(cfg, args):
    from meiscf.trainer import MultiPhaseTrainer
    data_yaml = _resolve_data_yaml(cfg)
    pre = _p(cfg, 'pretrained')
    t = cfg['train']
    trainer = MultiPhaseTrainer(
        data_yaml, variant=args.variant or t['variant'],
        nc=cfg['model']['nc'], scale=cfg['model']['scale'],
        pretrained=pre, frm_alpha=cfg['model']['frm_alpha'],
        project_dir=_p(cfg, 'runs_dir'), device=t['device'],
        workers=t['workers'], seed=t['seed'], batch_scale=t['batch_scale'],
        amp=t.get('amp', False), optimizer=t.get('optimizer', 'AdamW'),
        momentum=t.get('momentum', 0.9))
    if args.single_phase:
        # Reviewer-requested control: train at ONE fixed resolution (the final
        # resolution of the progressive schedule) so the curriculum itself can
        # be isolated from the benefit of high-resolution inputs.
        sp = dict(imgsz=1280, epochs=350, batch=4, lr0=0.01)
        sp.update(t.get('single_phase', {}) or {})
        logger.info(f"SINGLE-PHASE control: {sp}")
        best = trainer.train_single_phase(
            imgsz=sp['imgsz'], epochs=sp['epochs'], batch=sp['batch'],
            lr0=sp['lr0'], name=f"single{sp['imgsz']}")
    else:
        epp = t.get('epochs_per_phase')
        epp = {int(k): v for k, v in epp.items()} if epp else None
        phases = tuple(t['phases'])
        if args.start_phase:
            phases = tuple(p for p in phases if p >= args.start_phase)
        best = trainer.train(phases=phases, epochs_per_phase=epp,
                             resume_from=args.resume_from)
    logger.info(f"Best model: {best}")
    if best:
        from meiscf.evaluate import full_report
        full_report(best, data_yaml, Path(trainer.project_dir) / 'evaluation',
                    imgszs=tuple(cfg['eval']['imgszs']),
                    per_class_imgsz=cfg['eval']['per_class_imgsz'],
                    fps_imgsz=cfg['eval']['fps_imgsz'],
                    max_det=cfg['eval'].get('max_det', 300),
                    batch=cfg['eval'].get('batch', 2))
    print(best)
    return best


def cmd_ablation(cfg, args):
    from meiscf.ablation import AblationRunner
    data_yaml = _resolve_data_yaml(cfg)
    a = cfg['ablation']; t = cfg['train']
    runner = AblationRunner(
        data_yaml, out_dir=_p(cfg, 'ablation_dir'), nc=cfg['model']['nc'],
        scale=cfg['model']['scale'], pretrained=_p(cfg, 'pretrained'),
        eval_imgsz=a['eval_imgsz'], device=t['device'],
        batch_scale=t['batch_scale'], workers=t['workers'],
        amp=t.get('amp', False), optimizer=t.get('optimizer', 'AdamW'),
        momentum=t.get('momentum', 0.9))
    epp = t.get('epochs_per_phase')
    epp = {int(k): v for k, v in epp.items()} if epp else None
    runner.run_all(epochs_per_phase=epp,
                   include_multiseed=a.get('include_multiseed', False))
    if a.get('alphas') != [0.3, 0.5, 0.7]:
        runner.run_frm_alpha_sweep(alphas=tuple(a['alphas']), epochs_per_phase=epp)
    logger.info(f"Ablation complete -> {runner.results_path}")


def cmd_evaluate(cfg, args):
    data_yaml = _resolve_data_yaml(cfg)
    if not args.weights:
        sys.exit("evaluate requires --weights <best.pt>")
    out = Path(args.out or (Path(_p(cfg, 'runs_dir')) / 'evaluation'))
    ev = cfg['eval']
    tcfg = ev.get('tiled', {}) or {}

    # Tiled / SAHI evaluation path (eval-only small-object recall boost).
    if args.tiled or tcfg.get('enabled'):
        from meiscf.tiled_eval import tiled_evaluate
        common = dict(conf=tcfg.get('conf', 0.001), nms_iou=tcfg.get('nms_iou', 0.6),
                      max_det=tcfg.get('max_det', 1000),
                      imgsz_full=tcfg.get('imgsz_full', 1280), limit=args.limit)
        tiled = tiled_evaluate(
            args.weights, data_yaml, slice_size=tcfg.get('slice', 640),
            overlap=tcfg.get('overlap', 0.25), full_image=tcfg.get('full_image', True),
            out_json=out / 'tiled_evaluation.json', **common)
        # Full-image pass through the SAME metric -> apples-to-apples comparison
        # (and validates the metric against model.val at this imgsz).
        base = tiled_evaluate(
            args.weights, data_yaml, slice_size=None, full_image=True,
            out_json=out / 'tiled_baseline_same_metric.json', **common)
        d = (tiled['mAP50'] - base['mAP50']) * 100
        logger.info(f"COMPARISON  full-image mAP@50={base['mAP50']*100:.2f}  ->  "
                    f"tiled mAP@50={tiled['mAP50']*100:.2f}  (delta {d:+.2f} pp)")
        return

    from meiscf.evaluate import full_report
    augment = args.tta or ev.get('tta', False)
    full_report(args.weights, data_yaml, out,
                imgszs=tuple(ev['imgszs']),
                per_class_imgsz=ev['per_class_imgsz'],
                fps_imgsz=ev['fps_imgsz'],
                max_det=ev.get('max_det', 300),
                augment=augment,
                batch=ev.get('batch', 4))


def cmd_multiseed(cfg, args):
    from meiscf.multiseed import run_multiseed, aggregate_multiseed
    data_yaml = _resolve_data_yaml(cfg)
    t = cfg['train']
    variants = args.variants or ['meis_p2', 'baseline_p2']
    seeds = [int(s) for s in (args.seeds or [42, 123])]

    if args.summary_only:
        aggregate_multiseed(_p(cfg, 'runs_dir'), variants)
        return

    # P2 variants OOM at batch_scale 1.0 / amp off -- match the settings the
    # seed-0 P2 runs used so seeds are comparable.
    if any(v.endswith('_p2') for v in variants):
        if not t.get('amp', False) or t.get('batch_scale', 1.0) > 0.5:
            logger.warning(
                "P2 variants were trained with amp=true batch_scale=0.5. "
                "Current config differs -- pass "
                "--set train.amp=true train.batch_scale=0.5 for comparable runs.")

    epp = t.get('epochs_per_phase')
    epp = {int(k): v for k, v in epp.items()} if epp else None
    run_multiseed(
        data_yaml, variants, seeds, runs_dir=_p(cfg, 'runs_dir'),
        nc=cfg['model']['nc'], scale=cfg['model']['scale'],
        pretrained=_p(cfg, 'pretrained'), device=t['device'],
        workers=t['workers'], batch_scale=t['batch_scale'],
        amp=t.get('amp', False), optimizer=t.get('optimizer', 'SGD'),
        phases=tuple(t['phases']), epochs_per_phase=epp,
        eval_batch=cfg['eval'].get('batch', 2))


def cmd_significance(cfg, args):
    from meiscf.significance import compare_models
    data_yaml = _resolve_data_yaml(cfg)
    if not (args.weights_a and args.weights_b):
        sys.exit("significance requires --weights-a and --weights-b")
    ev = cfg['eval']
    out = Path(args.out or (Path(_p(cfg, 'runs_dir')) / 'significance'))
    seed_deltas = [float(x) for x in args.seed_deltas] if args.seed_deltas else None
    compare_models(
        args.weights_a, args.weights_b, data_yaml, out,
        imgsz=args.imgsz, label_a=args.label_a, label_b=args.label_b,
        max_det=ev.get('max_det', 600), n_boot=args.n_boot, n_perm=args.n_perm,
        device=cfg['train'].get('device'), seed_deltas=seed_deltas)


def _parse_models(args, command):
    """--models LABEL=WEIGHTS ... -> {label: weights}, with existence checks."""
    if not args.models:
        sys.exit(f"{command} requires --models LABEL=WEIGHTS [LABEL=WEIGHTS ...]")
    models = {}
    for item in args.models:
        if '=' not in item:
            sys.exit(f"--models entries must look like LABEL=WEIGHTS, got '{item}'")
        label, weights = item.split('=', 1)
        if not Path(weights).exists():
            sys.exit(f"weights not found for '{label}': {weights}")
        models[label] = weights
    return models


def _int_list(values, default, flag):
    """Numbers for a list flag, written either "1280 1536" or "1280,1536".

    A flag that lost its dashes in a copied command line arrives here as a word,
    where int() rejects it by name instead of it being taken for a value.
    """
    if not values:
        return tuple(default)
    out = []
    for value in values:
        for part in str(value).replace(',', ' ').split():
            try:
                out.append(int(part))
            except ValueError:
                sys.exit(f"{flag} takes numbers, got '{part}' -- if that is a flag, "
                         f"it is missing its leading dashes")
    return tuple(out)


def cmd_fps(cfg, args):
    """Inference speed at several input sizes on an idle GPU (Table tab:complexity)."""
    from meiscf.evaluate import benchmark_fps_table
    out = Path(args.out or (Path(_p(cfg, 'runs_dir')) / 'fps' / 'fps_benchmark.json'))
    benchmark_fps_table(_parse_models(args, 'fps'),
                        imgszs=_int_list(args.imgszs, [1280, 1536], '--imgszs'),
                        batches=_int_list(args.batches, [1], '--batches'),
                        device=cfg['train'].get('device'), out_json=out)


def cmd_uavdt_prepare(cfg, args):
    """Convert the UAVDT detection split to YOLO format with VisDrone class ids."""
    from meiscf.uavdt import convert_uavdt
    if not args.uavdt_root:
        sys.exit("uavdt-prepare requires --uavdt-root <folder containing "
                 "UAV-benchmark-M/ and UAV-benchmark-MOTD_v1.0/>")
    convert_uavdt(args.uavdt_root, out_root=args.uavdt_yolo, split=args.split,
                  frame_stride=args.frame_stride, link_images=not args.copy_images)


def cmd_uavdt(cfg, args):
    """Zero-shot evaluation of VisDrone-trained checkpoints on UAVDT."""
    from meiscf.uavdt import evaluate_uavdt
    root = Path(args.uavdt_yolo)
    data_yaml = root / 'UAVDT.yaml'
    if not data_yaml.exists():
        sys.exit(f"{data_yaml} not found -- run 'python run.py uavdt-prepare "
                 f"--uavdt-root <raw UAVDT>' first.")
    ignore_json = root / f'ignore_regions_{args.split}.json'
    out = Path(args.out or (Path(_p(cfg, 'runs_dir')) / 'uavdt'))
    evaluate_uavdt(_parse_models(args, 'uavdt'), data_yaml,
                   ignore_json if ignore_json.exists() else None, out,
                   imgszs=_int_list(args.imgszs, [1280], '--imgszs'),
                   max_det=cfg['eval'].get('max_det', 600),
                   device=cfg['train'].get('device'),
                   batch=cfg['eval'].get('batch', 4),
                   merge_van=not args.no_merge_van,
                   capture_stats=args.capture_stats)


def cmd_sizeap(cfg, args):
    """AP by object size from the validator's own matching (Table tab:size)."""
    from meiscf.size_eval import run_size_ap
    data_yaml = _resolve_data_yaml(cfg)
    models = _parse_models(args, 'sizeap')
    out = Path(args.out or (Path(_p(cfg, 'runs_dir')) / 'size_ap'))
    run_size_ap(models, data_yaml, out, imgszs=_int_list(args.imgszs, [1280], '--imgszs'),
                max_det=cfg['eval'].get('max_det', 600),
                device=cfg['train'].get('device'))


def cmd_heatmaps(cfg, args):
    from meiscf.heatmaps import (comparative_heatmaps, heatmaps_for_image,
                                 sample_val_images)
    data_yaml = _resolve_data_yaml(cfg)
    h = cfg['heatmaps']
    out_dir = _p(cfg, 'heatmaps_dir')
    imgs = ([Path(p) for p in args.images] if args.images
            else sample_val_images(data_yaml, n=h['n_images']))
    if not imgs:
        sys.exit("No images found for heatmaps (set --images or check dataset).")
    if not args.weights:
        sys.exit("heatmaps requires --weights <meiscf best.pt>")
    if args.baseline:
        comparative_heatmaps(args.baseline, args.weights, imgs, out_dir,
                             imgsz=h['imgsz'], mode=h['mode'])
    for img in imgs:
        heatmaps_for_image(args.weights, img, out_dir, imgsz=h['imgsz'],
                           mode=h['mode'], tag='meiscf')


def cmd_visualize(cfg, args):
    from meiscf.visualization import generate_all
    runs = Path(_p(cfg, 'runs_dir'))
    abl = Path(_p(cfg, 'ablation_dir'))
    # Find phase CSVs of the most recent run via the paths recorded in its
    # training_summary.json (robust to Ultralytics' save-dir relocation).
    phase_csvs, eval_dir = [], runs / 'evaluation'
    summaries = sorted(runs.glob('*/training_summary.json'),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if summaries:
        summary = json.loads(summaries[0].read_text())
        exp_dir = summaries[0].parent
        for _, d in sorted(summary.get('phase_dirs', {}).items()):
            csv = Path(d) / 'results.csv'
            if csv.exists():
                phase_csvs.append(csv)
        cand_eval = exp_dir / 'evaluation'
        if cand_eval.exists():
            eval_dir = cand_eval
    generate_all(
        runs,
        ablation_json=abl / 'ablation_results.json',
        dataset_stats=Path(_p(cfg, 'yolo_dataset')) / 'dataset_stats.json',
        multi_res_json=eval_dir / 'multi_resolution.json',
        phase_csvs=phase_csvs or None,
        out_dir=_p(cfg, 'figures_dir'))


def cmd_all(cfg, args):
    cmd_prepare(cfg, args)
    best = cmd_train(cfg, args)
    if best:
        args.weights = str(best)
        cmd_heatmaps(cfg, args)
    cmd_visualize(cfg, args)
    logger.info("FULL PIPELINE COMPLETE.")


# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="MEISCF-YOLO unified CLI")
    p.add_argument('command',
                   choices=['smoke', 'prepare', 'train', 'ablation', 'evaluate',
                            'heatmaps', 'visualize', 'multiseed',
                            'significance', 'sizeap', 'uavdt-prepare', 'uavdt',
                            'fps', 'all'])
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--variant', default=None, help="override train.variant")
    p.add_argument('--weights', default=None, help="model checkpoint (evaluate/heatmaps)")
    p.add_argument('--baseline', default=None, help="baseline checkpoint for comparison")
    p.add_argument('--images', nargs='*', default=None, help="explicit image paths")
    p.add_argument('--out', default=None, help="output dir override")
    p.add_argument('--resume-from', default=None,
                   help="checkpoint to start the first (lowest) phase from")
    p.add_argument('--start-phase', type=int, default=None,
                   help="skip phases below this number (use with --resume-from)")
    p.add_argument('--single-phase', action='store_true',
                   help="train: single fixed-resolution control instead of the "
                        "4-phase curriculum (params from train.single_phase)")
    p.add_argument('--weights-a', default=None, help="significance: first checkpoint")
    p.add_argument('--weights-b', default=None, help="significance: second checkpoint")
    p.add_argument('--label-a', default='A', help="significance: name for model A")
    p.add_argument('--label-b', default='B', help="significance: name for model B")
    p.add_argument('--imgsz', type=int, default=1536, help="significance: eval resolution")
    p.add_argument('--n-boot', type=int, default=1000, help="significance: bootstrap resamples")
    p.add_argument('--n-perm', type=int, default=1000, help="significance: permutations")
    p.add_argument('--seed-deltas', nargs='*', default=None,
                   help="significance: per-seed paired deltas in pp for the seed-level test")
    p.add_argument('--models', nargs='*', default=None,
                   help="sizeap: LABEL=WEIGHTS pairs to evaluate")
    p.add_argument('--imgszs', nargs='*', default=None,
                   help="sizeap/uavdt: input sizes (default: 1280); fps: default 1280 1536. "
                        "Space- or comma-separated")
    p.add_argument('--batches', nargs='*', default=None,
                   help="fps: batch sizes to time (default: 1). One image at a time can "
                        "leave a fast GPU idle between kernels; add e.g. 4,8 to compare "
                        "models where the GPU, not the host, is the limit")
    p.add_argument('--uavdt-root', default=None,
                   help="uavdt-prepare: raw UAVDT folder (UAV-benchmark-M + "
                        "UAV-benchmark-MOTD_v1.0)")
    p.add_argument('--uavdt-yolo', default='UAVDT_YOLO',
                   help="converted UAVDT dataset root (default: UAVDT_YOLO)")
    p.add_argument('--split', default='test',
                   help="uavdt: dataset split to convert/evaluate (default: test)")
    p.add_argument('--frame-stride', type=int, default=5,
                   help="uavdt-prepare: keep every n-th video frame (default: 5)")
    p.add_argument('--capture-stats', action='store_true',
                   help="uavdt: also save per-image statistics (batch 1) for "
                        "sequence-level confidence intervals")
    p.add_argument('--no-merge-van', action='store_true',
                   help="uavdt: keep VisDrone 'van' detections separate instead of "
                        "merging them into 'car'")
    p.add_argument('--variants', nargs='*', default=None,
                   help="multiseed: variants to run (default: meis_p2 baseline_p2)")
    p.add_argument('--seeds', nargs='*', default=None,
                   help="multiseed: seeds to train (default: 42 123; seed 0 exists)")
    p.add_argument('--summary-only', action='store_true',
                   help="multiseed: aggregate existing runs without training")
    p.add_argument('--tta', action='store_true',
                   help="evaluate: enable test-time augmentation (multi-scale + flips)")
    p.add_argument('--tiled', action='store_true',
                   help="evaluate: use tiled/SAHI inference (+ same-metric baseline)")
    p.add_argument('--limit', type=int, default=None,
                   help="evaluate --tiled: only process the first N val images (quick test)")
    p.add_argument('--copy-images', action='store_true',
                   help="copy images instead of symlinking during conversion")
    p.add_argument('--set', nargs='*', default=[],
                   help="config overrides as dotted.key=value")
    return p


def parse_overrides(items):
    out = {}
    for it in items:
        if '=' not in it:
            continue
        k, v = it.split('=', 1)
        try:
            v = yaml.safe_load(v)   # parse ints/floats/bools/lists
        except Exception:
            pass
        out[k] = v
    return out


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    {
        'smoke': cmd_smoke, 'prepare': cmd_prepare, 'train': cmd_train,
        'ablation': cmd_ablation, 'evaluate': cmd_evaluate,
        'heatmaps': cmd_heatmaps, 'visualize': cmd_visualize,
        'multiseed': cmd_multiseed, 'significance': cmd_significance,
        'sizeap': cmd_sizeap, 'uavdt-prepare': cmd_uavdt_prepare,
        'uavdt': cmd_uavdt, 'fps': cmd_fps, 'all': cmd_all,
    }[args.command](cfg, args)


if __name__ == '__main__':
    main()
