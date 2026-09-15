"""Ultralytics-8.4 AP definition: no (recall=1, precision=0) end sentinel; zero beyond max recall."""
import numpy as np
from ultralytics.utils import metrics as _m

def compute_ap_v84(recall, precision):
    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([1.0], precision))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre, right=0.0), x)
    return ap, mpre, mrec

_m.compute_ap = compute_ap_v84
ap_per_class = _m.ap_per_class
