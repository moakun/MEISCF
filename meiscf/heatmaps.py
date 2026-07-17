"""Comparative heatmap analysis: feature-activation maps + EigenCAM overlays.

Two complementary, dependency-free techniques (no pytorch-grad-cam needed):

  * Feature-activation heatmap - per-layer L2 norm over channels, showing WHERE
    a layer concentrates response. Used to visualise the edge emphasis MEIS adds.
  * EigenCAM - the first principal component of a layer's activation map (no
    gradients required), a robust class-agnostic saliency for detectors.

The headline figure is a side-by-side grid (baseline vs MEISCF) on the same
images at matched layers, which is exactly the comparative heatmap analysis the
reviewers will want.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from .registry import register_meiscf_modules
from .modules import MEIS, FRM, SandwichFusion

logger = logging.getLogger(__name__)


def _load_model(path):
    from ultralytics import YOLO
    register_meiscf_modules()
    return YOLO(str(path))


def _read_image(path, imgsz=1024):
    """Load an image -> (display_rgb HxWx3 uint8, input_tensor 1x3xHxW float)."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # letterbox to square imgsz keeping it simple (resize; detectors are robust).
    disp = cv2.resize(img, (imgsz, imgsz))
    t = torch.from_numpy(disp).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return disp, t


class _ActivationGrabber:
    """Register forward hooks on selected layer indices and grab their outputs."""

    def __init__(self, model, layer_indices):
        self.acts = {}
        self.handles = []
        seq = model.model.model      # nn.Sequential of layers, each has .i index
        for layer in seq:
            idx = getattr(layer, 'i', None)
            if idx in layer_indices:
                self.handles.append(
                    layer.register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, (list, tuple)) else out
            if isinstance(o, torch.Tensor):
                self.acts[idx] = o.detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def _activation_heatmap(act):
    """L2 norm over channels -> normalized HxW saliency in [0,1]."""
    a = act[0]                      # C,H,W
    hm = a.pow(2).sum(0).sqrt()     # H,W
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    return hm.cpu().numpy()


def _eigen_cam(act):
    """First principal component of the activation map -> HxW saliency in [0,1]."""
    a = act[0]                      # C,H,W
    C, H, W = a.shape
    # M: (H*W, C) - each spatial location is a point in channel space.
    M = a.reshape(C, H * W).cpu().numpy().T
    M = M - M.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(M, full_matrices=False)   # vh[0] in C-space
        cam = (M @ vh[0]).reshape(H, W)                    # project onto 1st PC
    except np.linalg.LinAlgError:
        cam = np.linalg.norm(M, axis=1).reshape(H, W)
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def _overlay(disp_rgb, heat, alpha=0.5):
    """Overlay a HxW heatmap (any size) on an RGB image with a jet colormap."""
    import cv2
    H, W = disp_rgb.shape[:2]
    heat = cv2.resize(heat, (W, H))
    cmap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    return (alpha * cmap + (1 - alpha) * disp_rgb).astype(np.uint8)


def _detect_layer_indices(model):
    """Return the layer indices feeding the Detect head (P3/P4/P5 sources)."""
    seq = model.model.model
    detect = seq[-1]
    f = getattr(detect, 'f', None)
    if isinstance(f, (list, tuple)):
        return list(f)
    return [len(seq) - 4, len(seq) - 3, len(seq) - 2]


def heatmaps_for_image(model_path, image_path, out_dir, imgsz=1024,
                       layer_indices=None, mode='both', tag='model'):
    """Generate activation + EigenCAM overlays for one image at chosen layers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model = _load_model(model_path)
    disp, t = _read_image(image_path, imgsz=imgsz)
    if layer_indices is None:
        layer_indices = _detect_layer_indices(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = model.model.to(device).eval()
    grab = _ActivationGrabber(model, set(layer_indices))
    with torch.no_grad():
        net(t.to(device))
    grab.remove()

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    modes = (['activation', 'eigencam'] if mode == 'both' else [mode])
    for li in layer_indices:
        if li not in grab.acts:
            logger.warning(f"No activation captured for layer {li}."); continue
        for mm in modes:
            heat = (_activation_heatmap(grab.acts[li]) if mm == 'activation'
                    else _eigen_cam(grab.acts[li]))
            ov = _overlay(disp, heat)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(ov); ax.axis('off')
            ax.set_title(f'{tag} | layer {li} | {mm}')
            p = out_dir / f'{tag}_L{li}_{mm}_{Path(image_path).stem}.png'
            fig.savefig(p, bbox_inches='tight', dpi=180); plt.close(fig)
            saved.append(p)
    return saved


def comparative_heatmaps(baseline_path, meiscf_path, image_paths, out_dir,
                         imgsz=1024, mode='eigencam'):
    """Side-by-side baseline-vs-MEISCF saliency grid on the same images.

    Each row = one image: [original | baseline saliency | MEISCF saliency].
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = _load_model(baseline_path)
    ours = _load_model(meiscf_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base.model.to(device).eval(); ours.model.to(device).eval()
    base_li = _detect_layer_indices(base)
    ours_li = _detect_layer_indices(ours)

    image_paths = list(image_paths)
    fig, axs = plt.subplots(len(image_paths), 3,
                            figsize=(13, 4.3 * len(image_paths)))
    if len(image_paths) == 1:
        axs = axs.reshape(1, 3)

    def saliency(model, li, t):
        grab = _ActivationGrabber(model, set(li))
        with torch.no_grad():
            model.model(t.to(device))
        grab.remove()
        # Aggregate the P3-level (highest-res, first source) saliency.
        key = li[0] if li[0] in grab.acts else next(iter(grab.acts))
        return (_eigen_cam(grab.acts[key]) if mode == 'eigencam'
                else _activation_heatmap(grab.acts[key]))

    for r, img_path in enumerate(image_paths):
        disp, t = _read_image(img_path, imgsz=imgsz)
        hb = saliency(base, base_li, t)
        ho = saliency(ours, ours_li, t)
        axs[r, 0].imshow(disp); axs[r, 0].set_title('Input'); axs[r, 0].axis('off')
        axs[r, 1].imshow(_overlay(disp, hb)); axs[r, 1].set_title('Baseline')
        axs[r, 1].axis('off')
        axs[r, 2].imshow(_overlay(disp, ho)); axs[r, 2].set_title('MEISCF (ours)')
        axs[r, 2].axis('off')
    fig.suptitle(f'Comparative {mode} Saliency: Baseline vs MEISCF', y=1.005)
    out_path = out_dir / f'comparative_{mode}.png'
    fig.tight_layout(); fig.savefig(out_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    logger.info(f"Saved comparative heatmap grid -> {out_path}")
    return out_path


def sample_val_images(data_yaml, n=4):
    """Pick n validation images from the dataset YAML for qualitative figures."""
    import yaml as _yaml
    cfg = _yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg.get('path', '.'))
    val_dir = root / cfg.get('val', 'images/val')
    if not val_dir.is_dir():
        val_dir = Path(cfg.get('val', 'images/val'))
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    imgs = [p for p in sorted(val_dir.iterdir()) if p.suffix.lower() in exts]
    if not imgs:
        return []
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)
    return [imgs[i] for i in idx]
