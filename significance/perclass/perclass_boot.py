"""Per-class paired bootstrap from the per-image stats saved by meiscf/significance.py.

Uses the same resampling stream as paired_bootstrap (default_rng(0), integers(0,n,n)
per resample), so the overall-mAP interval must reproduce the server JSON exactly.
"""
import json, sys, time
import numpy as np
from ap_patch import ap_per_class

KEYS = ('tp', 'conf', 'pred_cls', 'target_cls')
NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
         'awning-tricycle', 'bus', 'motor']
SIG = r"E:\MEISCF\significance"

def load(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['_meta']))
    recs = [{k: z[f'{i}_{k}'] for k in KEYS} for i in range(meta['n'])]
    return recs, meta

class Flat:
    """Concatenated per-image arrays + offsets for fast resampling."""
    def __init__(self, recs):
        self.tp = np.concatenate([r['tp'] for r in recs])
        self.conf = np.concatenate([r['conf'] for r in recs])
        self.pcls = np.concatenate([r['pred_cls'] for r in recs])
        self.tcls = np.concatenate([r['target_cls'] for r in recs])
        npred = np.array([len(r['conf']) for r in recs]); ntgt = np.array([len(r['target_cls']) for r in recs])
        self.ps = np.concatenate([[0], np.cumsum(npred)[:-1]]); self.pl = npred
        self.ts = np.concatenate([[0], np.cumsum(ntgt)[:-1]]); self.tl = ntgt

    @staticmethod
    def _rows(starts, lens, idx):
        l = lens[idx]; s = starts[idx]
        tot = l.sum()
        if tot == 0:
            return np.zeros(0, dtype=np.int64)
        # vectorised concatenation of ranges [s_i, s_i + l_i)
        rep_s = np.repeat(s - np.concatenate([[0], np.cumsum(l)[:-1]]), l)
        return np.arange(tot) + rep_s

    def ap50(self, idx):
        pr = self._rows(self.ps, self.pl, idx); tr = self._rows(self.ts, self.tl, idx)
        out = ap_per_class(self.tp[pr], self.conf[pr], self.pcls[pr], self.tcls[tr], plot=False)
        ap, ucls = out[5], out[6]
        per = np.full(10, np.nan); per95 = np.full(10, np.nan)
        per[ucls] = ap[:, 0]; per95[ucls] = ap.mean(1)
        return per, per95

def run(size, n_boot):
    A, ma = load(fr"{SIG}\stats_bp2_sp1280_s0_{size}.npz")     # P2, single phase
    B, mb = load(fr"{SIG}\stats_b_sp1280_s0_{size}.npz")       # P3-P5, single phase
    fa, fb = Flat(A), Flat(B)
    n = len(A); full = np.arange(n)
    a50, a95 = fa.ap50(full); b50, b95 = fb.ap50(full)
    print(f"[{size}] recomputed mAP50  P2 {np.nanmean(a50)*100:.2f}  P3-P5 {np.nanmean(b50)*100:.2f}  "
          f"delta {np.nanmean(a50-b50)*100:+.3f}")
    rng = np.random.default_rng(0)
    D = np.zeros((n_boot, 10)); Dm = np.zeros(n_boot); Dm95 = np.zeros(n_boot)
    t0 = time.time()
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        x50, x95 = fa.ap50(idx); y50, y95 = fb.ap50(idx)
        D[b] = (x50 - y50) * 100
        Dm[b] = (np.nanmean(x50) - np.nanmean(y50)) * 100
        Dm95[b] = (np.nanmean(x95) - np.nanmean(y95)) * 100
    obs = (a50 - b50) * 100
    res = {'size': size, 'n_boot': n_boot, 'seconds': round(time.time() - t0, 1),
           'overall_mAP50': {'obs': float(np.nanmean(a50 - b50) * 100),
                             'ci': [float(x) for x in np.percentile(Dm, [2.5, 97.5])],
                             'p': float(min(1, 2 * min((Dm <= 0).mean(), (Dm >= 0).mean())))},
           'overall_mAP50-95': {'obs': float(np.nanmean(a95 - b95) * 100),
                                'ci': [float(x) for x in np.percentile(Dm95, [2.5, 97.5])]},
           'per_class': {}}
    for c in range(10):
        d = D[:, c]; d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        res['per_class'][NAMES[c]] = {'AP50_P2': float(a50[c] * 100), 'AP50_P3P5': float(b50[c] * 100),
                                      'obs': float(obs[c]), 'ci': [float(lo), float(hi)], 'p': float(p),
                                      'n_tgt': int((fa.tcls == c).sum())}
    # post-hoc grouping: persons + two-wheelers vs. three/four-wheel vehicles
    for g, members in (('persons+two-wheelers', [0, 1, 2, 9]), ('vehicles', [3, 4, 5, 6, 7, 8])):
        d = D[:, members].mean(1)
        res[g] = {'obs': float(obs[members].mean()), 'ci': [float(x) for x in np.percentile(d, [2.5, 97.5])],
                  'p': float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    json.dump(res, open(f"perclass_boot_{size}.json", 'w'), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != 'per_class'}, indent=1))
    for k, v in res['per_class'].items():
        print(f"  {k:16s} n={v['n_tgt']:5d}  P3-P5 {v['AP50_P3P5']:5.1f}  P2 {v['AP50_P2']:5.1f}  "
              f"d {v['obs']:+5.2f}  CI [{v['ci'][0]:+5.2f}, {v['ci'][1]:+5.2f}]  p={v['p']:.3f}")

if __name__ == '__main__':
    run(int(sys.argv[1]), int(sys.argv[2]))
