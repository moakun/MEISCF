"""Joint paired bootstrap over the 548 val images for the four 2x2 models at 1536 px.

Same resampling stream as meiscf.significance.paired_bootstrap (default_rng(0)), so any
pairwise contrast reproduces the server's test exactly. Per resample we keep per-class AP50
and AP50-95 for every model; contrasts (incl. the stride x schedule interaction) are
computed afterwards from the same resamples.
"""
import json, time
import numpy as np
from ap_patch import ap_per_class

KEYS = ('tp', 'conf', 'pred_cls', 'target_cls')
NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
         'awning-tricycle', 'bus', 'motor']
SIG = r"E:\MEISCF\significance"
MODELS = {'P3_prog': 'b_prog_s0', 'P2_prog': 'baseline_p2_s0',
          'P3_sp': 'b_sp1280_s0', 'P2_sp': 'bp2_sp1280_s0'}

class Flat:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True); m = json.loads(str(z['_meta']))
        r = [{k: z[f'{i}_{k}'] for k in KEYS} for i in range(m['n'])]
        self.n = m['n']; self.reported = m['reported']
        self.tp = np.concatenate([x['tp'] for x in r]); self.conf = np.concatenate([x['conf'] for x in r])
        self.pcls = np.concatenate([x['pred_cls'] for x in r]); self.tcls = np.concatenate([x['target_cls'] for x in r])
        npred = np.array([len(x['conf']) for x in r]); ntgt = np.array([len(x['target_cls']) for x in r])
        self.ps = np.concatenate([[0], np.cumsum(npred)[:-1]]); self.pl = npred
        self.ts = np.concatenate([[0], np.cumsum(ntgt)[:-1]]); self.tl = ntgt
        self.tsorted = [np.sort(x['target_cls']) for x in r]
    @staticmethod
    def rows(s, l, idx):
        L = l[idx]; S = s[idx]; tot = L.sum()
        return np.arange(tot) + np.repeat(S - np.concatenate([[0], np.cumsum(L)[:-1]]), L)
    def ap(self, idx):
        pr = self.rows(self.ps, self.pl, idx); tr = self.rows(self.ts, self.tl, idx)
        out = ap_per_class(self.tp[pr], self.conf[pr], self.pcls[pr], self.tcls[tr], plot=False)
        a50 = np.full(10, np.nan); a95 = np.full(10, np.nan)
        a50[out[6]] = out[5][:, 0]; a95[out[6]] = out[5].mean(1)
        return a50, a95

F = {k: Flat(fr"{SIG}\stats_{v}_1536.npz") for k, v in MODELS.items()}
# pairing check: identical ground truth per image across all four
ref = F['P3_prog']
for k, f in F.items():
    assert f.n == ref.n and all(np.array_equal(a, b) for a, b in zip(f.tsorted, ref.tsorted)), k
n = ref.n; full = np.arange(n)
obs = {k: f.ap(full) for k, f in F.items()}
for k in F:
    print(f"{k:8s} recomputed mAP50 {np.nanmean(obs[k][0])*100:.3f}  mAP50-95 {np.nanmean(obs[k][1])*100:.3f}  "
          f"(validator {F[k].reported['mAP50']*100:.3f}/{F[k].reported['mAP50-95']*100:.3f})")

B = 1000
rng = np.random.default_rng(0)
A50 = {k: np.zeros((B, 10)) for k in F}; A95 = {k: np.zeros((B, 10)) for k in F}
t0 = time.time()
for b in range(B):
    idx = rng.integers(0, n, n)
    for k, f in F.items():
        A50[k][b], A95[k][b] = f.ap(idx)
    if (b + 1) % 200 == 0:
        print(f"  {b+1}/{B}  {time.time()-t0:.0f}s", flush=True)
np.savez_compressed("joint_boot_1536.npz", **{f"{k}_50": A50[k] for k in F}, **{f"{k}_95": A95[k] for k in F})
json.dump({k: [obs[k][0].tolist(), obs[k][1].tolist()] for k in F}, open("joint_obs_1536.json", "w"))
print("done")
