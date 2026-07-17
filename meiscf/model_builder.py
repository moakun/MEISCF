"""Build the full MEISCF model and every ablation variant from YAML templates.

All variants share the stock YOLOv11 backbone (layers 0-10) so the COCO
pretrained backbone transfers via ``model.load(<yolo11s.pt path>)``. Only the
head/neck differs between variants, which is exactly what we want to ablate.

Variants
--------
baseline       : stock YOLOv11s (no custom modules)            -> control
meis           : + MEIS at P3/P4/P5, standard PANet neck        -> MEIS effect
meis_sf        : + Cross-scale Sandwich Fusion neck (3-level)   -> + fusion
full           : + post-fusion FRM   (== MEIS+SF+FRM, the paper model)
frm_prefusion  : FRM placed BEFORE fusion (placement ablation)
sf_sequential  : Sandwich Fusion restricted to 2-level at P4 (3-vs-2 level)

The full model is also parameterizable by FRM ``alpha`` for the residual-blend
sweep (paper: 0.3 / 0.5 / 0.7).
"""

import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

from .registry import register_meiscf_modules

logger = logging.getLogger(__name__)

# VisDrone-DET 10 classes (order matters: matches the annotation->YOLO mapping).
VISDRONE_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                  'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

# Shared backbone for every variant (identical to stock YOLO11 layers 0-10).
_BACKBONE = """
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
"""

# ---- Per-variant head definitions (indices continue from the backbone) ----

_HEAD_BASELINE = """
  - [-1, 1, nn.Upsample, [None, 2, nearest]]   # 11
  - [[-1, 6], 1, Concat, [1]]                  # 12  cat backbone P4
  - [-1, 2, C3k2, [512, False]]                # 13
  - [-1, 1, nn.Upsample, [None, 2, nearest]]   # 14
  - [[-1, 4], 1, Concat, [1]]                  # 15  cat backbone P3
  - [-1, 2, C3k2, [256, False]]                # 16  P3/8-small
  - [-1, 1, Conv, [256, 3, 2]]                 # 17
  - [[-1, 13], 1, Concat, [1]]                 # 18  cat head P4
  - [-1, 2, C3k2, [512, False]]                # 19  P4/16-medium
  - [-1, 1, Conv, [512, 3, 2]]                 # 20
  - [[-1, 10], 1, Concat, [1]]                 # 21  cat head P5
  - [-1, 2, C3k2, [1024, True]]                # 22  P5/32-large
  - [[16, 19, 22], 1, Detect, [nc]]            # 23  Detect(P3, P4, P5)
"""

_HEAD_MEIS = """
  - [4,  1, MEIS, []]                          # 11  MEIS @ P3
  - [6,  1, MEIS, []]                          # 12  MEIS @ P4
  - [10, 1, MEIS, []]                          # 13  MEIS @ P5
  - [13, 1, nn.Upsample, [None, 2, nearest]]   # 14
  - [[14, 12], 1, Concat, [1]]                 # 15
  - [-1, 2, C3k2, [512, False]]                # 16  P4 top-down
  - [-1, 1, nn.Upsample, [None, 2, nearest]]   # 17
  - [[17, 11], 1, Concat, [1]]                 # 18
  - [-1, 2, C3k2, [256, False]]                # 19  P3 out
  - [-1, 1, Conv, [256, 3, 2]]                 # 20
  - [[20, 16], 1, Concat, [1]]                 # 21
  - [-1, 2, C3k2, [512, False]]                # 22  P4 out
  - [-1, 1, Conv, [512, 3, 2]]                 # 23
  - [[23, 13], 1, Concat, [1]]                 # 24
  - [-1, 2, C3k2, [1024, True]]                # 25  P5 out
  - [[19, 22, 25], 1, Detect, [nc]]            # 26  Detect(P3, P4, P5)
"""

_HEAD_MEIS_SF = """
  - [4,  1, MEIS, []]                          # 11  MEIS @ P3
  - [6,  1, MEIS, []]                          # 12  MEIS @ P4
  - [10, 1, MEIS, []]                          # 13  MEIS @ P5
  - [[12, 11, 13], 1, SandwichFusion, []]      # 14  3-level fuse @ P4
  - [-1, 2, C3k2, [512, False]]                # 15  P4 refined
  - [[11, 15], 1, SandwichFusion, []]          # 16  fuse @ P3 (aux P4n)
  - [-1, 2, C3k2, [256, False]]                # 17  P3 refined
  - [[13, 15], 1, SandwichFusion, []]          # 18  fuse @ P5 (aux P4n)
  - [-1, 2, C3k2, [1024, True]]                # 19  P5 refined
  - [[17, 15, 19], 1, Detect, [nc]]            # 20  Detect(P3, P4, P5)
"""

# Full model. {FRM_ARGS} is substituted with [] or [16, alpha] for the alpha sweep.
_HEAD_FULL = """
  - [4,  1, MEIS, []]                          # 11  MEIS @ P3
  - [6,  1, MEIS, []]                          # 12  MEIS @ P4
  - [10, 1, MEIS, []]                          # 13  MEIS @ P5
  - [[12, 11, 13], 1, SandwichFusion, []]      # 14  3-level fuse @ P4
  - [-1, 2, C3k2, [512, False]]                # 15  P4 refined
  - [[11, 15], 1, SandwichFusion, []]          # 16  fuse @ P3 (aux P4n)
  - [-1, 2, C3k2, [256, False]]                # 17  P3 refined
  - [[13, 15], 1, SandwichFusion, []]          # 18  fuse @ P5 (aux P4n)
  - [-1, 2, C3k2, [1024, True]]                # 19  P5 refined
  - [17, 1, FRM, {FRM_ARGS}]                   # 20  FRM post-fusion @ P3
  - [15, 1, FRM, {FRM_ARGS}]                   # 21  FRM post-fusion @ P4
  - [19, 1, FRM, {FRM_ARGS}]                   # 22  FRM post-fusion @ P5
  - [[20, 21, 22], 1, Detect, [nc]]            # 23  Detect(P3, P4, P5)
"""

_HEAD_FRM_PREFUSION = """
  - [4,  1, MEIS, []]                          # 11  MEIS @ P3
  - [6,  1, MEIS, []]                          # 12  MEIS @ P4
  - [10, 1, MEIS, []]                          # 13  MEIS @ P5
  - [11, 1, FRM, []]                           # 14  FRM PRE-fusion @ P3
  - [12, 1, FRM, []]                           # 15  FRM PRE-fusion @ P4
  - [13, 1, FRM, []]                           # 16  FRM PRE-fusion @ P5
  - [[15, 14, 16], 1, SandwichFusion, []]      # 17  3-level fuse @ P4
  - [-1, 2, C3k2, [512, False]]                # 18  P4 refined
  - [[14, 18], 1, SandwichFusion, []]          # 19  fuse @ P3
  - [-1, 2, C3k2, [256, False]]                # 20  P3 refined
  - [[16, 18], 1, SandwichFusion, []]          # 21  fuse @ P5
  - [-1, 2, C3k2, [1024, True]]                # 22  P5 refined
  - [[20, 18, 22], 1, Detect, [nc]]            # 23  Detect(P3, P4, P5)
"""

# Sandwich Fusion restricted to two levels at P4 ([P4, P3] only) to isolate the
# benefit of simultaneous 3-level vs sequential 2-level fusion (paper +1.4%).
_HEAD_SF_SEQUENTIAL = """
  - [4,  1, MEIS, []]                          # 11  MEIS @ P3
  - [6,  1, MEIS, []]                          # 12  MEIS @ P4
  - [10, 1, MEIS, []]                          # 13  MEIS @ P5
  - [[12, 11], 1, SandwichFusion, []]          # 14  2-level fuse @ P4 (no P5)
  - [-1, 2, C3k2, [512, False]]                # 15  P4 refined
  - [[11, 15], 1, SandwichFusion, []]          # 16  fuse @ P3
  - [-1, 2, C3k2, [256, False]]                # 17  P3 refined
  - [[13, 15], 1, SandwichFusion, []]          # 18  fuse @ P5
  - [-1, 2, C3k2, [1024, True]]                # 19  P5 refined
  - [17, 1, FRM, []]                           # 20  FRM post-fusion @ P3
  - [15, 1, FRM, []]                           # 21  FRM post-fusion @ P4
  - [19, 1, FRM, []]                           # 22  FRM post-fusion @ P5
  - [[20, 21, 22], 1, Detect, [nc]]            # 23  Detect(P3, P4, P5)
"""

# ---------------------------------------------------------------------------
# P2-head variants: add a stride-4 (160x160) detection level fed from backbone
# layer 2 (P2/4). 63% of VisDrone objects are <32px; P3 stride-8 is too coarse
# for them, so they are missed. A P2 head gives the detector the resolution to
# find sub-16px objects -- trained into the model (no inference-time scale
# mismatch, unlike tiled inference).
# ---------------------------------------------------------------------------

# full_p2: the full MEISCF (MEIS + Sandwich Fusion + FRM) extended to 4 levels
# (P2/P3/P4/P5). Keeps the paper's module story; adds the P2 head.
_HEAD_FULL_P2 = """
  - [2,  1, MEIS, []]                          # 11  MEIS @ P2 (stride-4)
  - [4,  1, MEIS, []]                          # 12  MEIS @ P3
  - [6,  1, MEIS, []]                          # 13  MEIS @ P4
  - [10, 1, MEIS, []]                          # 14  MEIS @ P5
  - [[13, 12, 14], 1, SandwichFusion, []]      # 15  3-level fuse @ P4 (center P4)
  - [-1, 2, C3k2, [512, False]]                # 16  P4 refined
  - [[12, 11, 16], 1, SandwichFusion, []]      # 17  3-level fuse @ P3 (aux P2, P4n)
  - [-1, 2, C3k2, [256, False]]                # 18  P3 refined
  - [[11, 18], 1, SandwichFusion, []]          # 19  2-level fuse @ P2 (aux P3n)
  - [-1, 2, C3k2, [256, False]]                # 20  P2 refined
  - [[14, 16], 1, SandwichFusion, []]          # 21  2-level fuse @ P5 (aux P4n)
  - [-1, 2, C3k2, [1024, True]]                # 22  P5 refined
  - [20, 1, FRM, {FRM_ARGS}]                   # 23  FRM post-fusion @ P2
  - [18, 1, FRM, {FRM_ARGS}]                   # 24  FRM post-fusion @ P3
  - [16, 1, FRM, {FRM_ARGS}]                   # 25  FRM post-fusion @ P4
  - [22, 1, FRM, {FRM_ARGS}]                   # 26  FRM post-fusion @ P5
  - [[23, 24, 25, 26], 1, Detect, [nc]]        # 27  Detect(P2, P3, P4, P5)
"""

# meis_p2: MEIS + the proven standard PANet neck extended to P2 + FRM on the
# outputs. Drops Sandwich Fusion (which underperformed in ablation). This is the
# pragmatic, high-EV recipe -- the standard "YOLOv11-P2" neck most VisDrone
# papers use, plus the MEIS edge branch and FRM recalibration.
_HEAD_MEIS_P2 = """
  - [2,  1, MEIS, []]                          # 11  MEIS @ P2
  - [4,  1, MEIS, []]                          # 12  MEIS @ P3
  - [6,  1, MEIS, []]                          # 13  MEIS @ P4
  - [10, 1, MEIS, []]                          # 14  MEIS @ P5
  - [-1, 1, nn.Upsample, [None, 2, nearest]]   # 15  (from P5)
  - [[15, 13], 1, Concat, [1]]                 # 16  cat P4
  - [-1, 2, C3k2, [512, False]]                # 17  P4 top-down
  - [-1, 1, nn.Upsample, [None, 2, nearest]]   # 18
  - [[18, 12], 1, Concat, [1]]                 # 19  cat P3
  - [-1, 2, C3k2, [256, False]]                # 20  P3 top-down
  - [-1, 1, nn.Upsample, [None, 2, nearest]]   # 21
  - [[21, 11], 1, Concat, [1]]                 # 22  cat P2
  - [-1, 2, C3k2, [256, False]]                # 23  P2 out (stride-4)
  - [-1, 1, Conv, [256, 3, 2]]                 # 24
  - [[24, 20], 1, Concat, [1]]                 # 25
  - [-1, 2, C3k2, [256, False]]                # 26  P3 out
  - [-1, 1, Conv, [256, 3, 2]]                 # 27
  - [[27, 17], 1, Concat, [1]]                 # 28
  - [-1, 2, C3k2, [512, False]]                # 29  P4 out
  - [-1, 1, Conv, [512, 3, 2]]                 # 30
  - [[30, 14], 1, Concat, [1]]                 # 31
  - [-1, 2, C3k2, [1024, True]]                # 32  P5 out
  - [23, 1, FRM, []]                           # 33  FRM @ P2
  - [26, 1, FRM, []]                           # 34  FRM @ P3
  - [29, 1, FRM, []]                           # 35  FRM @ P4
  - [32, 1, FRM, []]                           # 36  FRM @ P5
  - [[33, 34, 35, 36], 1, Detect, [nc]]        # 37  Detect(P2, P3, P4, P5)
"""

# baseline_p2: the CONTROL for meis_p2 -- the identical standard PANet neck
# extended to P2, with NO MEIS and NO FRM. Isolates the P2-head contribution:
# meis_p2 minus baseline_p2 == the (MEIS + FRM) module contribution.
_HEAD_BASELINE_P2 = """
  - [-1, 1, nn.Upsample, [None, 2, nearest]]   # 11  (from P5)
  - [[-1, 6], 1, Concat, [1]]                  # 12  cat backbone P4
  - [-1, 2, C3k2, [512, False]]                # 13  P4 top-down
  - [-1, 1, nn.Upsample, [None, 2, nearest]]   # 14
  - [[-1, 4], 1, Concat, [1]]                  # 15  cat backbone P3
  - [-1, 2, C3k2, [256, False]]                # 16  P3 top-down
  - [-1, 1, nn.Upsample, [None, 2, nearest]]   # 17
  - [[-1, 2], 1, Concat, [1]]                  # 18  cat backbone P2
  - [-1, 2, C3k2, [256, False]]                # 19  P2 out (stride-4)
  - [-1, 1, Conv, [256, 3, 2]]                 # 20
  - [[-1, 16], 1, Concat, [1]]                 # 21
  - [-1, 2, C3k2, [256, False]]                # 22  P3 out
  - [-1, 1, Conv, [256, 3, 2]]                 # 23
  - [[-1, 13], 1, Concat, [1]]                 # 24
  - [-1, 2, C3k2, [512, False]]                # 25  P4 out
  - [-1, 1, Conv, [512, 3, 2]]                 # 26
  - [[-1, 10], 1, Concat, [1]]                 # 27
  - [-1, 2, C3k2, [1024, True]]                # 28  P5 out
  - [[19, 22, 25, 28], 1, Detect, [nc]]        # 29  Detect(P2, P3, P4, P5)
"""

_HEADS = {
    'baseline':      _HEAD_BASELINE,
    'meis':          _HEAD_MEIS,
    'meis_sf':       _HEAD_MEIS_SF,
    'full':          _HEAD_FULL,
    'meis_sf_frm':   _HEAD_FULL,        # alias
    'frm_prefusion': _HEAD_FRM_PREFUSION,
    'sf_sequential': _HEAD_SF_SEQUENTIAL,
    'full_p2':       _HEAD_FULL_P2,
    'meis_p2':       _HEAD_MEIS_P2,
    'baseline_p2':   _HEAD_BASELINE_P2,
}

VARIANTS = ['baseline', 'meis', 'meis_sf', 'full', 'frm_prefusion',
            'sf_sequential', 'full_p2', 'meis_p2', 'baseline_p2']

# YOLOv11 scale presets [depth, width, max_channels]. We default to 's'.
_SCALES = {
    'n': [0.50, 0.25, 1024],
    's': [0.50, 0.50, 1024],
    'm': [0.50, 1.00, 512],
    'l': [1.00, 1.00, 512],
    'x': [1.00, 1.50, 512],
}


def build_yaml_text(variant='full', nc=10, scale='s', frm_alpha=0.5):
    """Return the full model YAML text for a variant."""
    if variant not in _HEADS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_HEADS)}.")
    head = _HEADS[variant]
    if '{FRM_ARGS}' in head:
        frm_args = '[]' if abs(frm_alpha - 0.5) < 1e-9 else f'[16, {frm_alpha}]'
        head = head.replace('{FRM_ARGS}', frm_args)
    sd, sw, mc = _SCALES[scale]
    return (
        f"# MEISCF-YOLOv11{scale} variant='{variant}'  (auto-generated)\n"
        f"nc: {nc}\n"
        f"scales:\n  {scale}: [{sd}, {sw}, {mc}]\n"
        f"backbone:{_BACKBONE}"
        f"head:{head}"
    )


def write_variant_yaml(variant='full', nc=10, scale='s', frm_alpha=0.5,
                       out_dir='model_configs'):
    """Write a variant YAML to disk; filename carries variant + scale + alpha."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = build_yaml_text(variant, nc=nc, scale=scale, frm_alpha=frm_alpha)
    cfg = yaml.safe_load(text)
    tag = variant if abs(frm_alpha - 0.5) < 1e-9 else f"{variant}_a{frm_alpha}"
    # Filename must contain the scale letter so Ultralytics infers the scale.
    path = out_dir / f"yolo11{scale}-meiscf-{tag}.yaml"
    with open(path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=None)
    return path


def _transfer_head_c3k2_weights(meiscf_model, pretrained_path, variant='full'):
    """Transfer pretrained C3k2 head weights from baseline YOLOv11s to MEISCF.

    The baseline head loads full COCO-pretrained weights, giving it a 2%
    head-start over the MEISCF model whose head is randomly initialized.
    We copy the C3k2 block weights from the baseline head into the MEISCF
    head's C3k2 blocks by matching their channel configurations.

    Because each variant places C3k2 blocks at different layer indices, the
    mapping is variant-aware.  For the `meis` variant the C3k2 blocks still
    use Concat (not SandwichFusion), so input channels match the baseline and
    the transfer is nearly complete.  For SandwichFusion variants only the
    internal bottleneck layers (m, cv2, bn) match; cv1 mismatches because
    SandwichFusion changes the input channel count.
    """
    try:
        from ultralytics import YOLO
        pt = YOLO(str(pretrained_path))
        pt_sd = pt.model.state_dict()
        meiscf_sd = meiscf_model.model.state_dict()

        # Variant-aware mapping: baseline prefix -> MEISCF prefix.
        # Baseline indices (YOLOv11s head): 13(P4), 16(P3), 19(P4), 22(P5).
        # MEIS variants place C3k2 at different indices.
        _VARIANT_C3K2_MAP = {
            'meis': {
                'model.13.': 'model.16.',   # P4 C3k2 [512, False]
                'model.16.': 'model.19.',   # P3 C3k2 [256, False]
                'model.19.': 'model.22.',   # P4 C3k2 [512, False]
                'model.22.': 'model.25.',   # P5 C3k2 [1024, True]
            },
            'meis_sf': {
                'model.13.': 'model.15.',   # P4 C3k2 [512, False]
                'model.16.': 'model.17.',   # P3 C3k2 [256, False]
                'model.22.': 'model.19.',   # P5 C3k2 [1024, True]
            },
            'full': {
                'model.13.': 'model.15.',   # P4 C3k2 [512, False]
                'model.16.': 'model.17.',   # P3 C3k2 [256, False]
                'model.22.': 'model.19.',   # P5 C3k2 [1024, True]
            },
            'frm_prefusion': {
                'model.13.': 'model.18.',   # P4 C3k2 [512, False]
                'model.16.': 'model.20.',   # P3 C3k2 [256, False]
                'model.22.': 'model.22.',   # P5 C3k2 [1024, True] (same index)
            },
            'sf_sequential': {
                'model.13.': 'model.15.',   # P4 C3k2 [512, False]
                'model.16.': 'model.17.',   # P3 C3k2 [256, False]
                'model.22.': 'model.19.',   # P5 C3k2 [1024, True]
            },
            # P2 variants: map the P3/P4/P5 C3k2 blocks; the new P2-level C3k2
            # has no baseline equivalent and stays randomly initialized (the
            # shape-check skips any non-matching tensors safely).
            'full_p2': {
                'model.13.': 'model.16.',   # P4 refined C3k2 [512, False]
                'model.16.': 'model.18.',   # P3 refined C3k2 [256, False]
                'model.22.': 'model.22.',   # P5 refined C3k2 [1024, True]
            },
            'meis_p2': {
                'model.13.': 'model.17.',   # P4 top-down C3k2 [512, False]
                'model.16.': 'model.20.',   # P3 top-down C3k2 [256, False]
                'model.19.': 'model.29.',   # P4 out C3k2 [512, False]
                'model.22.': 'model.32.',   # P5 out C3k2 [1024, True]
            },
            # baseline_p2: layers 11-16 share stock indices/channels, so the
            # natural model.load already transfers them; map the out-blocks.
            'baseline_p2': {
                'model.16.': 'model.22.',   # P3 out C3k2 [256] (cv1 in-ch differs)
                'model.19.': 'model.25.',   # P4 out C3k2 [512, False]
                'model.22.': 'model.28.',   # P5 out C3k2 [1024, True]
            },
        }

        mapping = _VARIANT_C3K2_MAP.get(variant)
        if mapping is None:
            logger.warning(f"No C3k2 head transfer mapping for variant='{variant}'. "
                           f"Head will be randomly initialized.")
            return

        transferred = 0
        skipped = 0
        for pt_prefix, meiscf_prefix in mapping.items():
            for key in pt_sd:
                if not key.startswith(pt_prefix):
                    continue
                suffix = key[len(pt_prefix):]
                meiscf_key = meiscf_prefix + suffix
                if meiscf_key in meiscf_sd:
                    if pt_sd[key].shape == meiscf_sd[meiscf_key].shape:
                        meiscf_sd[meiscf_key].copy_(pt_sd[key])
                        transferred += 1
                    else:
                        logger.warning(
                            f"Shape mismatch transferring {key} -> {meiscf_key}: "
                            f"{pt_sd[key].shape} vs {meiscf_sd[meiscf_key].shape}")
                        skipped += 1
                else:
                    skipped += 1

        meiscf_model.model.load_state_dict(meiscf_sd, strict=False)
        logger.info(f"Transferred {transferred} pretrained C3k2 head tensors "
                    f"({skipped} skipped) for variant='{variant}' from {pretrained_path}")
        del pt
    except Exception as e:
        logger.warning(f"Head weight transfer failed: {e}")


def build_model(variant='full', nc=10, scale='s', frm_alpha=0.5,
                pretrained=None, out_dir='model_configs'):
    """Build a MEISCF variant and optionally transfer pretrained backbone weights.

    Parameters
    ----------
    variant    : one of VARIANTS (or 'meis_sf_frm' alias for 'full')
    nc         : number of classes (VisDrone = 10)
    scale      : YOLOv11 scale letter ('n','s','m','l','x'); paper uses 's'
    frm_alpha  : FRM residual-blend coefficient (full variant only)
    pretrained : path to yolo11s.pt on the GPU server (None to skip transfer)
    """
    register_meiscf_modules()
    yaml_path = write_variant_yaml(variant, nc=nc, scale=scale,
                                   frm_alpha=frm_alpha, out_dir=out_dir)
    model = YOLO(str(yaml_path))
    if pretrained:
        if not Path(pretrained).exists():
            logger.warning(f"Pretrained weights not found at '{pretrained}'. "
                           f"Training from scratch (backbone NOT transferred).")
        else:
            try:
                model.load(pretrained)
                logger.info(f"Transferred pretrained backbone from '{pretrained}'.")
            except Exception as e:
                logger.warning(f"Could not load pretrained '{pretrained}': {e}")
        # Transfer head C3k2 weights for non-baseline variants.
        if variant != 'baseline' and Path(pretrained).exists():
            _transfer_head_c3k2_weights(model, pretrained, variant=variant)
    return model, yaml_path
