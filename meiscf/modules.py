"""Paper-faithful MEISCF modules (MEIS, Cross-scale Sandwich Fusion, FRM).

Each class implements the equations from the paper and is channel-preserving
where the paper requires it, so the COCO-pretrained YOLOv11 backbone transfers
cleanly. See registry.py for how these get wired into the Ultralytics parser.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MEIS - Multi-scale Edge Information Selection (paper Eqs. 1-5)
# ---------------------------------------------------------------------------
class MEIS(nn.Module):
    """Multi-scale Edge Information Selection.

    Dilated depthwise-separable convolutions at dilation rates (1, 2, 4) extract
    edge features at three receptive-field scales (16x16, 32x32, 64x64 px). For
    each scale a per-input channel-wise gate alpha_s in [0,1]^C is computed from
    global context (Eqs. 2-3) and used to weight the scale's edge map (Eq. 4).
    The gated multi-scale sum is added residually to the input (Eq. 5), so the
    module is channel-preserving and preserves the backbone semantics.

    Parameters
    ----------
    c1        : input (== output) channels
    dilations : dilation rates for the three edge branches
    reduction : bottleneck ratio r for the gating MLP (paper r=16)
    """

    def __init__(self, c1, dilations=(1, 2, 4), reduction=16, gate='adaptive'):
        super().__init__()
        self.c1 = c1
        self.gate_mode = gate
        r = max(c1 // reduction, 8)
        self.branches = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.static_gates = nn.ParameterList()
        for d in dilations:
            # Depthwise (dilated) -> pointwise: depthwise-separable edge extractor.
            self.branches.append(nn.Sequential(
                nn.Conv2d(c1, c1, 3, 1, d, dilation=d, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c1, 1, bias=False),
                nn.BatchNorm2d(c1),
                nn.SiLU(),
            ))
            if gate == 'adaptive':
                # alpha_s = sigma(FC2(ReLU(FC1(GAP(E_s)))))  -> per-INPUT gate.
                self.gates.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(c1, r, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(r, c1, 1), nn.Sigmoid(),
                ))
            elif gate == 'static':
                # Ablation: learnable per-channel weight that does NOT depend on
                # the input. Isolates per-input adaptivity from channel weighting.
                # Init 0 -> sigmoid = 0.5, matching the adaptive gate at init.
                self.static_gates.append(nn.Parameter(torch.zeros(1, c1, 1, 1)))
            else:
                raise ValueError(f"gate must be 'adaptive' or 'static', got {gate!r}")

    def __setstate__(self, state):
        # Checkpoints saved before the gate option existed hold no gate_mode;
        # every MEIS block trained then used the adaptive gate.
        super().__setstate__(state)
        if 'gate_mode' not in self.__dict__:
            self.gate_mode = 'adaptive'

    def forward(self, x):
        e = 0
        if self.gate_mode == 'adaptive':
            for branch, gate in zip(self.branches, self.gates):
                es = branch(x)
                e = e + gate(es) * es      # element-wise channel-wise gating (Eq. 4)
        else:
            for branch, w in zip(self.branches, self.static_gates):
                es = branch(x)
                e = e + torch.sigmoid(w) * es
        return x + e                       # residual integration (Eq. 5)


# ---------------------------------------------------------------------------
# Cross-scale Sandwich Fusion (paper Eqs. 6-11)
# ---------------------------------------------------------------------------
class SandwichFusion(nn.Module):
    """Cross-scale Sandwich Fusion with learnable normalized weights.

    Multi-input. The FIRST input is the "current" level: it defines the output
    spatial resolution AND output channel count. Remaining inputs are auxiliary
    pyramid levels, channel-projected to the current level, spatially aligned
    (MaxPool for higher-resolution inputs / Upsample for lower-resolution ones,
    realised here via interpolation to the current size), then combined with a
    ReLU-normalized learnable-weight sum (Eq. 8). Boundary levels (P3, P5) use
    two inputs (Eq. 11); the middle level (P4) uses three.
    """

    def __init__(self, ch_list, eps=1e-4):
        super().__init__()
        self.eps = eps
        c_out = ch_list[0]
        self.c_out = c_out
        self.proj = nn.ModuleList()
        for c in ch_list:
            if c == c_out:
                self.proj.append(nn.Identity())
            else:
                self.proj.append(nn.Sequential(
                    nn.Conv2d(c, c_out, 1, bias=False),
                    nn.BatchNorm2d(c_out),
                ))
        self.weight = nn.Parameter(torch.ones(len(ch_list)))   # w_i, init 1.0
        # Depthwise-separable post-fusion convolution.
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
        w = w / (w.sum() + self.eps)       # normalized weights (Eq. 8)
        out = 0
        for i, x in enumerate(xs):
            x = self.proj[i](x)
            if x.shape[2:] != (H, W):
                # Higher-res aux -> downsample; lower-res aux -> upsample.
                # Use nearest interpolation for both (matches original working code).
                x = F.interpolate(x, size=(H, W), mode='nearest')
            out = out + w[i] * x
        return self.fuse(out)


# ---------------------------------------------------------------------------
# FRM - Feature Recalibration Module (paper Eqs. 12-21; CBAM-identical ops)
# ---------------------------------------------------------------------------
class FRM(nn.Module):
    """Feature Recalibration Module (structurally identical to CBAM).

    Sequential channel attention (avg+max shared MLP, Eqs. 12-15) followed by
    spatial attention (7x7 conv over channel-pooled maps, Eqs. 16-20), combined
    with a residual blend (Eq. 21) using fixed coefficients alpha_ch = alpha_sp.
    Channel-preserving. The paper's contribution is PLACEMENT (post-fusion), not
    the operations; this same class is reused for the pre-fusion ablation.

    Parameters
    ----------
    c1        : input (== output) channels
    reduction : bottleneck ratio r for the channel MLP (paper r=16)
    alpha     : residual blend coefficient (paper alpha_ch = alpha_sp = 0.5)
    """

    def __init__(self, c1, reduction=16, alpha=0.5):
        super().__init__()
        self.c1 = c1
        r = max(c1 // reduction, 8)
        self.alpha_ch = alpha
        self.alpha_sp = alpha
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, r, 1), nn.ReLU(inplace=True), nn.Conv2d(r, c1, 1),
        )
        self.spatial = nn.Conv2d(2, 1, 7, 1, 3, bias=False)

    def forward(self, x):
        # Channel attention (Eqs. 12-15)
        wc = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1))
                           + self.mlp(F.adaptive_max_pool2d(x, 1)))
        fc = x * wc
        # Spatial attention (Eqs. 16-20)
        sa = torch.cat([fc.mean(1, keepdim=True), fc.max(1, keepdim=True)[0]], dim=1)
        ws = torch.sigmoid(self.spatial(sa))
        fs = fc * ws
        # Residual blend (Eq. 21)
        return x + self.alpha_ch * (fc - x) + self.alpha_sp * (fs - fc)


# Convenience: the set of custom modules this package contributes to the parser.
MEISCF_MODULES = (MEIS, SandwichFusion, FRM)
