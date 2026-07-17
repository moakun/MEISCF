# MEISCF-YOLO

Paper-faithful, fully-working reimplementation of **MEISCF: Multi-scale Edge
Information Selection with Cross-scale Fusion for Small Object Detection in UAV
Imagery**, plus a complete research toolkit: VisDrone→YOLO conversion, four-phase
progressive training, automated ablation studies, comparative heatmap analysis,
and publication-ready figures.

This is a ground-up rebuild of the original `trainMEISCF_MULTIPHASE.py`. The
original defined the modules but never converted the dataset, never ran
ablations, produced no heatmaps or charts, and only partially wired evaluation.
Everything here is verified to build and train end-to-end on Ultralytics 8.3.x.

---

## 1. What's in the box

| Module | Purpose |
|---|---|
| `meiscf/modules.py` | MEIS, Cross-scale Sandwich Fusion, FRM (paper Eqs. 1–21) |
| `meiscf/registry.py` | Registers the modules into Ultralytics + patches `parse_model` so they build with correct channels |
| `meiscf/model_builder.py` | Builds the full model **and every ablation variant** from YAML |
| `meiscf/data_prep.py` | **VisDrone/UAVDT → YOLO** annotation conversion + dataset stats |
| `meiscf/trainer.py` | Four-phase progressive trainer (640→800→1024→1280 px) |
| `meiscf/evaluate.py` | Multi-resolution mAP, per-class AP, FPS, params/FLOPs |
| `meiscf/ablation.py` | Automated ablation suites (module ladder, fusion, FRM placement, α-sweep, multi-seed) |
| `meiscf/visualization.py` | Line / bar / pie charts, training curves, SOTA bubble plot |
| `meiscf/heatmaps.py` | Feature-activation + EigenCAM comparative saliency grids |
| `run.py` | One CLI to drive all of the above |
| `config.yaml` | All paths and hyperparameters in one place |

### Architecture variants (for ablation)
`baseline` · `meis` · `meis_sf` · `full` (= MEIS+SF+FRM, the paper model) ·
`frm_prefusion` (placement ablation) · `sf_sequential` (2- vs 3-level fusion).

---

## 2. Setup on the GPU server

```bash
pip install -r requirements.txt          # ultralytics 8.3.x, torch, matplotlib, opencv…
```

Then edit **`config.yaml`** and set three paths:

```yaml
paths:
  pretrained:  /abs/path/to/yolo11s.pt          # the weights you downloaded
  raw_dataset: /abs/path/to/VisDrone            # raw VisDrone download root
  yolo_dataset: ./VisDrone_YOLO                 # where converted data is written
```

`raw_dataset` should contain `VisDrone2019-DET-train/`, `VisDrone2019-DET-val/`
(and optionally `-test-dev/`), each with `images/` and `annotations/`. If you
already have a YOLO-format dataset, point `raw_dataset` at it — conversion is
skipped automatically.

---

## 3. Quick start

```bash
# 0. Prove the architecture builds (no training, ~10s):
python run.py smoke

# 1. Convert VisDrone -> YOLO + write dataset stats:
python run.py prepare

# 2. Full four-phase progressive training of the paper model + evaluation:
python run.py train

# 3. Everything: prepare -> train -> heatmaps -> figures
python run.py all
```

### Reproduce the paper's data tables & figures
```bash
# Full ablation suite (module ladder, fusion, FRM placement, alpha sweep):
python run.py ablation

# Add the 3-seed statistical study (6 extra runs):
python run.py ablation --set ablation.include_multiseed=true

# Regenerate all charts from cached JSON/CSV (no retraining):
python run.py visualize
```

### Comparative heatmaps (baseline vs MEISCF)
```bash
python run.py heatmaps --weights runs/meiscf/<exp>/phase4/weights/best.pt \
                       --baseline runs/ablation/A1_baseline_progressive/phase4/weights/best.pt
```

### Evaluate any checkpoint
```bash
python run.py evaluate --weights path/to/best.pt
# -> multi_resolution.json, per_class_ap.json, fps.json, complexity.json
```

---

## 4. Outputs you get

**Numerical (JSON/CSV):** per-phase `results.csv`, `training_summary.json`,
`multi_resolution.json`, `per_class_ap.json`, `fps.json`, `complexity.json`,
`ablation_results.json`, `ablation_table.{csv,md}`.

**Figures (PNG, 200 dpi):**
- `training_progression.png` — losses, mAP, P/R, LR across all 350 epochs (paper Fig. 5)
- `ablation_bars.png` + `contribution_pie.png` — module ablation (paper Table 2)
- `per_class_comparison.png` — per-class AP baseline vs ours (paper Fig. 6)
- `multi_resolution.png` — mAP vs resolution
- `class_distribution_{pie,bar}.png`, `object_size_hist.png` — dataset analysis
- `alpha_sweep.png` — FRM α ablation
- `sota_comparison.png` — accuracy/speed bubble plot
- `heatmaps/comparative_eigencam.png` — side-by-side saliency

---

## 5. Optimizations applied over the original

- **Robust eager module integration** — `parse_model` is patched once with a
  whitespace-tolerant regex and a sentinel; modules are built with correct
  width-scaled channels so their parameters are actually optimized.
- **Depthwise-separable** edge/fusion convs → the full model is ~10–12M params
  and **24.4 GFLOPs** (paper: 23.4) while keeping real-time speed.
- **AMP** mixed-precision training, **cosine LR**, **AdamW**, paper-matched
  per-phase augmentation decay.
- **Symlinked** dataset images (no disk blow-up), idempotent conversion.
- **`batch_scale`** knob to fit any GPU; **resumable** ablations (cached by id);
  **reduced `epochs_per_phase`** for fast previews on the same code path.
- **Multi-seed** support and matched-resolution baseline evaluation so reported
  gains are apples-to-apples.

---

## 6. Notes on expected numbers

The paper reports 55.1% mAP@50 on VisDrone-val for the full model with the full
350-epoch schedule on the true dataset. Always compare against the baseline at
the **same evaluation resolution** (the per-class figure in the paper uses
1280px for both models). Module-level gains are single-run estimates in the
paper; use `ablation.include_multiseed=true` to get error bars.
