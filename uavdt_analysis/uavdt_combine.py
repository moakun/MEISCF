import json, sys
import numpy as np
sys.path.insert(0, r"E:\MEISCF")
from uavdt_boot import load, CLASSES

M = ['p3_prog', 'p2_prog', 'p3_sp', 'p2_sp', 'meis_prog']
B = {m: np.load(f"uavdt_boot_{m}.npz", allow_pickle=False) for m in M}
names = list(CLASSES.values())

# how concentrated are the rare classes? (per-sequence ground-truth counts)
d = load('p2_sp')
seq_ids = sorted(set(d['seqs']))
print(f"{'sequence':10s}{'images':>7s}" + "".join(f"{n:>8s}" for n in names))
tot = np.zeros(3, dtype=int)
for s in seq_ids:
    sel = np.where(d['seqs'] == s)[0]
    T = np.concatenate([np.arange(d['ts'][i], d['ts'][i] + d['ntgt'][i]) for i in sel])
    counts = [int((d['tcls'][T] == c).sum()) for c in CLASSES]
    tot += counts
    print(f"{s:10s}{len(sel):7d}" + "".join(f"{c:8d}" for c in counts))
print(f"{'total':10s}{len(d['seqs']):7d}" + "".join(f"{c:8d}" for c in tot))
print("sequences containing each class:",
      {n: int(sum(1 for s in seq_ids
                  for T in [np.concatenate([np.arange(d['ts'][i], d['ts'][i] + d['ntgt'][i])
                                            for i in np.where(d['seqs'] == s)[0]])]
                  if (d['tcls'][T] == c).any())) for c, n in CLASSES.items()})

print(f"\n{'model':11s}" + "".join(f"{n:>9s}" for n in names) + f"{'mean':>9s}")
for m in M:
    o = B[m]['obs'] * 100
    print(f"{m:11s}" + "".join(f"{v:9.2f}" for v in o) + f"{np.nanmean(o):9.2f}")

C = {'schedule | P3-P5': ('p3_sp', 'p3_prog'), 'schedule | P2': ('p2_sp', 'p2_prog'),
     'stride | progressive': ('p2_prog', 'p3_prog'), 'stride | single phase': ('p2_sp', 'p3_sp'),
     'MEIS + FRM | progressive': ('meis_prog', 'p2_prog')}
res = {}
print("\ncontrast (AP@50 points, 95% CI from resampling the 11 sequences)")
for cname, (a, b) in C.items():
    obs = (B[a]['obs'] - B[b]['obs']) * 100
    boot = (B[a]['boot'] - B[b]['boot']) * 100
    row, out = [], {}
    for j, n in enumerate(names + ['mean']):
        o = float(np.nanmean(obs)) if n == 'mean' else float(obs[j])
        dv = np.nanmean(boot, axis=1) if n == 'mean' else boot[:, j]
        dv = dv[~np.isnan(dv)]
        lo, hi = np.percentile(dv, [2.5, 97.5])
        p = 2 * min((dv <= 0).mean(), (dv >= 0).mean())
        out[n] = {'obs': o, 'ci': [float(lo), float(hi)], 'p': '<0.002' if p == 0 else round(float(min(p, 1)), 3)}
        row.append(f"{n} {o:+5.2f} [{lo:+5.1f},{hi:+5.1f}]")
    res[cname] = out
    print(f"  {cname:26s} " + "  ".join(row))
json.dump(res, open("uavdt_contrasts.json", "w"), indent=1)
