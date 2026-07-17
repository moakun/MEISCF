"""MEISCF-YOLO: Multi-scale Edge Information Selection with Cross-scale Fusion.

A paper-faithful, fully-working reimplementation + research-grade tooling for
small object detection in UAV imagery (VisDrone / UAVDT), built on Ultralytics
YOLOv11.

Package layout
--------------
meiscf.modules        - MEIS, SandwichFusion, FRM (paper Eqs. 1-21)
meiscf.registry       - register custom modules + robust parse_model patch
meiscf.model_builder  - build full model + all ablation variants from YAML
meiscf.data_prep      - VisDrone/UAVDT -> YOLO conversion, dataset.yaml, stats
meiscf.trainer        - four-phase progressive trainer (+ single-phase)
meiscf.ablation       - automated ablation experiment runner
meiscf.evaluate       - multi-resolution eval, per-class AP, FPS, params/FLOPs
meiscf.visualization  - line / bar / pie charts and training-curve figures
meiscf.heatmaps       - Grad-CAM + feature-map heatmap comparative analysis
"""

__version__ = "2.0.0"

from .modules import MEIS, SandwichFusion, FRM  # noqa: F401
from .registry import register_meiscf_modules  # noqa: F401
