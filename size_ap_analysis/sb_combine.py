import json
import numpy as np
M = ['p3_prog', 'p2_prog', 'p3_sp', 'p2_sp', 'meis_prog', 'p2_prog_s42', 'meis_prog_s42']
NAT = ['small', 'medium', 'large', 'n<8', 'n8-16', 'n16-32']
INP = ['i4-8', 'i8-16', 'i16-32', 'i>=32']
S = {}
for z in (1024, 1280):
    for m in M:
        f = np.load(f"sb_{m}_{z}.npz")
        S[(m, z)] = {k.split('|')[1]: f[k] for k in f.files}
        S[(m, z)]['_keys'] = f.files

def series(m, z, b):
    f = np.load(f"sb_{m}_{z}.npz")
    return float(np.nanmean(f[f"obs|{b}"]) * 100), np.nanmean(f[f"boot|{b}"], axis=1) * 100

def contrast(terms, b):
    """terms: list of (coef, model, imgsz)"""
    o = 0.0; d = 0.0
    for c, m, z in terms:
        ob, bt = series(m, z, b); o += c * ob; d = d + c * bt
    lo, hi = np.percentile(d, [2.5, 97.5]); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return o, lo, hi, ('<0.002' if p == 0 else f"{min(p, 1):.3f}")

def show(title, terms_fn, bins):
    print(f"\n{title}")
    out = {}
    for b in bins:
        o, lo, hi, p = contrast(terms_fn, b)
        out[b] = dict(obs=o, ci=[lo, hi], p=p)
        print(f"   {b:7s} {o:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]  p={p}")
    return out

res = {}
print("=== class-mean AP50 (classes with >=100 boxes per bin) ===")
for z in (1024, 1280):
    print(f"@{z}: " + " | ".join(f"{b}: " + " ".join(f"{series(m, z, b)[0]:5.1f}" for m in M) for b in NAT + INP))
print("   model order:", M)

for z in (1280, 1024):
    res[f'meis_s0_{z}'] = show(f"MEIS - Baseline-P2, progressive, seed 0 @{z}", [(1, 'meis_prog', z), (-1, 'p2_prog', z)], NAT + INP)
    res[f'meis_s42_{z}'] = show(f"MEIS - Baseline-P2, progressive, seed 42 @{z}", [(1, 'meis_prog_s42', z), (-1, 'p2_prog_s42', z)], NAT + INP)
    res[f'meis_mean_{z}'] = show(f"MEIS - Baseline-P2, mean of both seeds @{z}",
                                 [(.5, 'meis_prog', z), (-.5, 'p2_prog', z), (.5, 'meis_prog_s42', z), (-.5, 'p2_prog_s42', z)], NAT + INP)

for m in ['p3_prog', 'p2_prog', 'p3_sp', 'p2_sp']:
    res[f'down_{m}'] = show(f"{m}: AP@1024 - AP@1280 (same objects, native bins)", [(1, m, 1024), (-1, m, 1280)], NAT)
res['down_prog_vs_sp'] = show("mean change 1280->1024: progressive models minus single-phase models (2x2 models)",
                              [(.5, 'p3_prog', 1024), (-.5, 'p3_prog', 1280), (.5, 'p2_prog', 1024), (-.5, 'p2_prog', 1280),
                               (-.5, 'p3_sp', 1024), (.5, 'p3_sp', 1280), (-.5, 'p2_sp', 1024), (.5, 'p2_sp', 1280)], NAT)
for det in ('p3', 'p2'):
    res[f'sched_{det}_1024'] = show(f"schedule gap {det} @1024 (single - progressive)", [(1, f'{det}_sp', 1024), (-1, f'{det}_prog', 1024)], NAT)
    res[f'sched_{det}_change'] = show(f"schedule gap {det}: @1024 minus @1280",
                                      [(1, f'{det}_sp', 1024), (-1, f'{det}_prog', 1024), (-1, f'{det}_sp', 1280), (1, f'{det}_prog', 1280)], NAT)
json.dump(res, open("sb_contrasts.json", "w"), indent=1, default=float)
