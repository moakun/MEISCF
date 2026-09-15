import json
import numpy as np
NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
         'awning-tricycle', 'bus', 'motor']
Z = np.load("joint_boot_1536.npz"); O = json.load(open("joint_obs_1536.json"))
M = ['P3_prog', 'P2_prog', 'P3_sp', 'P2_sp']
boot = {(k, t): Z[f"{k}_{t}"] * 100 for k in M for t in ('50', '95')}
obs = {(k, '50'): np.array(O[k][0]) * 100 for k in M}
obs.update({(k, '95'): np.array(O[k][1]) * 100 for k in M})

CONTRASTS = {
    'schedule | P3-P5   (P3_sp - P3_prog)': {'P3_sp': 1, 'P3_prog': -1},
    'schedule | P2      (P2_sp - P2_prog)': {'P2_sp': 1, 'P2_prog': -1},
    'stride   | single  (P2_sp - P3_sp)': {'P2_sp': 1, 'P3_sp': -1},
    'stride   | progr.  (P2_prog - P3_prog)': {'P2_prog': 1, 'P3_prog': -1},
    'interaction (stride|SP - stride|prog)': {'P2_sp': 1, 'P3_sp': -1, 'P2_prog': -1, 'P3_prog': 1},
}

def lin(src, t, w, per_class=False):
    x = sum(c * src[(k, t)] for k, c in w.items())
    return x if per_class else np.nanmean(x, axis=-1)

def summ(o, d):
    lo, hi = np.percentile(d, [2.5, 97.5]); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    ps = "<0.002" if p == 0 else f"{min(p, 1):.3f}"
    return o, lo, hi, ps, d.std(ddof=1)

res = {}
for name, w in CONTRASTS.items():
    res[name] = {}
    for t, lab in (('50', 'mAP50'), ('95', 'mAP50-95')):
        o, lo, hi, ps, sd = summ(lin(obs, t, w), lin(boot, t, w))
        res[name][lab] = dict(obs=o, ci=[lo, hi], p=ps, se=sd)
        print(f"{name:42s} {lab:8s} {o:+6.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  p={ps:>6s}  boot SE {sd:.2f}")

print("\nPer-class AP50 contrasts @1536 (obs [95% CI], p)")
hdr = ['schedule | P3-P5   (P3_sp - P3_prog)', 'schedule | P2      (P2_sp - P2_prog)',
       'stride   | progr.  (P2_prog - P3_prog)', 'interaction (stride|SP - stride|prog)']
res['per_class'] = {}
for ci, cname in enumerate(NAMES):
    row = []
    res['per_class'][cname] = {}
    for h in hdr:
        w = CONTRASTS[h]
        o = lin(obs, '50', w, True)[ci]; d = lin(boot, '50', w, True)[:, ci]; d = d[~np.isnan(d)]
        _, lo, hi, ps, _ = summ(o, d)
        res['per_class'][cname][h] = dict(obs=float(o), ci=[float(lo), float(hi)], p=ps)
        row.append(f"{o:+5.1f} [{lo:+5.1f},{hi:+5.1f}] p={ps:>6s}")
    print(f"  {cname:16s} | " + " | ".join(row))
print("  columns:", hdr)
json.dump(res, open("joint_contrasts_1536.json", "w"), indent=1, default=float)
