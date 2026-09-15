"""Paired bootstrap over images for size-bin AP50 contrasts of the four 2x2 models.

Lean AP (Ultralytics 8.4 definition) computed per class on a global conf ordering;
validated against meiscf.size_eval.size_ap (patched ap_per_class) on the full data.
Class means use a fixed class set per bin: classes with >= 100 ground-truth boxes
in that bin on the full validation set (rare classes make class-mean AP unstable).
"""
import json, sys, time
import numpy as np
sys.path.insert(0, r"E:\MEISCF")
from meiscf.size_eval import load_capture, _rows, size_ap, SCHEMES
from ap_patch import ap_per_class

D = r"E:\MEISCF\size_ap"
M = ['p3_prog', 'p2_prog', 'p3_sp', 'p2_sp']
BINS = [('coco_native', 'small', 'native', 0, 32), ('coco_native', 'medium', 'native', 32, 96),
        ('coco_native', 'large', 'native', 96, np.inf),
        ('input_px', '4-8', 'input', 4, 8), ('input_px', '8-16', 'input', 8, 16),
        ('input_px', '16-32', 'input', 16, 32), ('input_px', '>=32', 'input', 32, np.inf)]
X101 = np.linspace(0, 1, 101)

def ap_v84(tp_sorted, n_l):
    if n_l == 0:
        return np.nan
    if len(tp_sorted) == 0:
        return 0.0
    tpc = np.cumsum(tp_sorted); fpc = np.cumsum(~tp_sorted)
    rec = tpc / (n_l + 1e-16); pre = tpc / (tpc + fpc)
    mrec = np.concatenate(([0.0], rec)); mpre = np.concatenate(([1.0], pre))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return np.trapz(np.interp(X101, mrec, mpre, right=0.0), X101)

def bin_class_ap(cap, img_idx, class_sets):
    """-> {bin_name: array of AP50 over the bin's fixed class set} for one model."""
    P = _rows(cap['p_start'], cap['n_pred'], img_idx); G = _rows(cap['g_start'], cap['n_gt'], img_idx)
    conf = cap['pconf'][P]; pcls = cap['pcls'][P].astype(int); m0 = cap['matched'][P, 0]
    order = np.argsort(-conf)
    pcls_o = pcls[order]; m_o = m0[order]; has_o = m_o >= 0
    gcls_G = cap['gcls'][G].astype(int)
    out = {}
    for scheme, name, unit, lo, hi in BINS:
        gsz = cap[f'gsize_{unit}']
        g_in = (gsz >= lo) & (gsz < hi)
        tp_o = np.zeros(len(m_o), dtype=bool); tp_o[has_o] = g_in[m_o[has_o]]
        dsz_o = cap[f'psize_{unit}'][P][order]
        det_in_o = (dsz_o >= lo) & (dsz_o < hi)
        keep_o = ~((has_o & ~tp_o) | (~has_o & ~det_in_o))
        ncls_bin = np.bincount(gcls_G[g_in[G]], minlength=10)
        vals = []
        for c in class_sets[name]:
            sel = keep_o & (pcls_o == c)
            vals.append(ap_v84(tp_o[sel], ncls_bin[c]))
        out[name] = np.array(vals)
    return out

def main(z, B):
    caps = {m: load_capture(fr"{D}\boxes_{m}_{z}.npz")[0] for m in M}
    names = load_capture(fr"{D}\boxes_p2_sp_{z}.npz")[1]['names']
    ref = caps['p3_prog']; n = len(ref['n_pred']); full = np.arange(n)
    class_sets, all_sets = {}, {}
    for scheme, name, unit, lo, hi in BINS:
        g_in = (ref[f'gsize_{unit}'] >= lo) & (ref[f'gsize_{unit}'] < hi)
        cnt = np.bincount(ref['gcls'][g_in].astype(int), minlength=10)
        class_sets[name] = [c for c in range(10) if cnt[c] >= 100]
        all_sets[name] = [c for c in range(10) if cnt[c] > 0]
    # validation of the lean AP against size_ap (patched ap_per_class), all classes present
    for m in ('p2_sp', 'p3_prog'):
        lean = bin_class_ap(caps[m], full, all_sets)
        cap1 = dict(caps[m]); cap1['matched'] = caps[m]['matched'][:, :1]
        ref_res = size_ap(cap1, {'coco_native': SCHEMES['coco_native'][1:], 'input_px': SCHEMES['input_px'][1:]}, ap_fn=ap_per_class)
        for scheme, name, *_ in BINS:
            a = np.nanmean(lean[name]); b = ref_res[scheme][name]['AP50']
            assert abs(a - b) < 2e-4, (m, name, a, b)
    print(f"[{z}] lean AP validated against size_ap on all bins", flush=True)

    obs = {m: bin_class_ap(caps[m], full, class_sets) for m in M}
    obs_all = {m: bin_class_ap(caps[m], full, all_sets) for m in M}
    rng = np.random.default_rng(0)
    boot = {m: {b[1]: np.zeros((B, len(class_sets[b[1]]))) for b in BINS} for m in M}
    t0 = time.time()
    for i in range(B):
        idx = rng.integers(0, n, n)
        for m in M:
            r = bin_class_ap(caps[m], idx, class_sets)
            for name, v in r.items():
                boot[m][name][i] = v
        if (i + 1) % 100 == 0:
            print(f"  [{z}] {i+1}/{B} {time.time()-t0:.0f}s", flush=True)
    C = {'stride | progressive': {'p2_prog': 1, 'p3_prog': -1}, 'stride | single phase': {'p2_sp': 1, 'p3_sp': -1},
         'schedule | P3-P5': {'p3_sp': 1, 'p3_prog': -1}, 'schedule | P2': {'p2_sp': 1, 'p2_prog': -1},
         'interaction': {'p2_sp': 1, 'p3_sp': -1, 'p2_prog': -1, 'p3_prog': 1}}
    res = {'imgsz': z, 'n_boot': B, 'class_sets': {k: [names[c] for c in v] for k, v in class_sets.items()},
           'means': {}, 'means_all_classes': {}, 'contrasts': {}}
    for scheme, name, *_ in BINS:
        res['means'][name] = {m: float(np.nanmean(obs[m][name]) * 100) for m in M}
        res['means_all_classes'][name] = {m: float(np.nanmean(obs_all[m][name]) * 100) for m in M}
        res['contrasts'][name] = {}
        for cname, w in C.items():
            o = sum(c * np.nanmean(obs[m][name]) for m, c in w.items()) * 100
            d = sum(c * np.nanmean(boot[m][name], axis=1) for m, c in w.items()) * 100
            lo, hi = np.percentile(d, [2.5, 97.5]); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            res['contrasts'][name][cname] = {'obs': float(o), 'ci': [float(lo), float(hi)],
                                             'p': '<0.002' if p == 0 else round(float(min(p, 1)), 3)}
    json.dump(res, open(f"size_boot_{z}.json", "w"), indent=1)
    print(f"[{z}] done", flush=True)

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
