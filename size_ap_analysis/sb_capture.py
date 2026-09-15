"""Per-capture bootstrap of size-bin class APs (AP50, Ultralytics 8.4 definition).
All processes use default_rng(0), so resample i selects the same images everywhere."""
import sys, time
import numpy as np
sys.path.insert(0, r"E:\MEISCF")
from meiscf.size_eval import load_capture, _rows
from size_boot import ap_v84

D = r"E:\MEISCF\size_ap"
BINS = [('small', 'native', 0, 32), ('medium', 'native', 32, 96), ('large', 'native', 96, np.inf),
        ('n<8', 'native', 0, 8), ('n8-16', 'native', 8, 16), ('n16-32', 'native', 16, 32),
        ('i4-8', 'input', 4, 8), ('i8-16', 'input', 8, 16), ('i16-32', 'input', 16, 32), ('i>=32', 'input', 32, np.inf)]

def class_sets(cap):
    out = {}
    for name, unit, lo, hi in BINS:
        g = cap[f'gsize_{unit}']
        cnt = np.bincount(cap['gcls'][(g >= lo) & (g < hi)].astype(int), minlength=10)
        out[name] = [c for c in range(10) if cnt[c] >= 100]
    return out

def bin_class_ap(cap, img_idx, sets):
    P = _rows(cap['p_start'], cap['n_pred'], img_idx); G = _rows(cap['g_start'], cap['n_gt'], img_idx)
    order = np.argsort(-cap['pconf'][P])
    pcls_o = cap['pcls'][P][order].astype(int); m_o = cap['matched'][P, 0][order]; has_o = m_o >= 0
    gcls_G = cap['gcls'][G].astype(int)
    out = {}
    for name, unit, lo, hi in BINS:
        gsz = cap[f'gsize_{unit}']; g_in = (gsz >= lo) & (gsz < hi)
        tp_o = np.zeros(len(m_o), dtype=bool); tp_o[has_o] = g_in[m_o[has_o]]
        dsz_o = cap[f'psize_{unit}'][P][order]
        keep_o = ~((has_o & ~tp_o) | (~has_o & ~((dsz_o >= lo) & (dsz_o < hi))))
        ncls = np.bincount(gcls_G[g_in[G]], minlength=10)
        out[name] = np.array([ap_v84(tp_o[keep_o & (pcls_o == c)], ncls[c]) for c in sets[name]])
    return out

if __name__ == '__main__':
    label, z, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cap, meta = load_capture(fr"{D}\boxes_{label}_{z}.npz")
    sets = class_sets(cap); n = len(cap['n_pred'])
    obs = bin_class_ap(cap, np.arange(n), sets)
    boot = {b[0]: np.zeros((B, len(sets[b[0]]))) for b in BINS}
    rng = np.random.default_rng(0); t0 = time.time()
    for i in range(B):
        r = bin_class_ap(cap, rng.integers(0, n, n), sets)
        for k, v in r.items():
            boot[k][i] = v
    np.savez_compressed(f"sb_{label}_{z}.npz", **{f"obs|{k}": v for k, v in obs.items()},
                        **{f"boot|{k}": v for k, v in boot.items()},
                        **{f"set|{k}": np.array(v) for k, v in sets.items()})
    print(f"{label}@{z} done in {time.time()-t0:.0f}s", flush=True)
