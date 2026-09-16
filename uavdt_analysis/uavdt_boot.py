"""Sequence-level (block) bootstrap for the UAVDT zero-shot comparison.

UAVDT is video: frames within a sequence are nearly identical, so images are not
independent. The resampling unit is therefore the sequence, not the frame.
AP uses the Ultralytics 8.4 definition (see ap_patch / size_boot).
"""
import json, sys, time
import numpy as np
sys.path.insert(0, r"E:\MEISCF")
from meiscf.significance import load_stats
from size_boot import ap_v84

D = r"E:\MEISCF\UAVDT_YOLO\uavdt"
CLASSES = {3: 'car', 5: 'truck', 8: 'bus'}

def load(label, imgsz=1280):
    recs, reported, nc = load_stats(fr"{D}\stats_{label}_{imgsz}.npz")
    order = json.load(open(fr"{D}\image_order_{label}_{imgsz}.json"))
    assert len(recs) == len(order), (len(recs), len(order))
    tp = np.concatenate([r['tp'][:, 0] for r in recs]).astype(bool)
    conf = np.concatenate([r['conf'] for r in recs]).astype(np.float64)
    pcls = np.concatenate([r['pred_cls'] for r in recs]).astype(int)
    tcls = np.concatenate([r['target_cls'] for r in recs]).astype(int)
    npred = np.array([len(r['conf']) for r in recs]); ntgt = np.array([len(r['target_cls']) for r in recs])
    ps = np.concatenate([[0], np.cumsum(npred)[:-1]]); ts = np.concatenate([[0], np.cumsum(ntgt)[:-1]])
    seqs = np.array([s.split('_')[0] for s in order])
    return dict(tp=tp, conf=conf, pcls=pcls, tcls=tcls, ps=ps, npred=npred,
                ts=ts, ntgt=ntgt, seqs=seqs, reported=reported)

def rows(start, length, idx):
    l = length[idx]; tot = int(l.sum())
    if tot == 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(tot) + np.repeat(start[idx] - np.concatenate([[0], np.cumsum(l)[:-1]]), l)

def ap_per_class(d, img_idx):
    P = rows(d['ps'], d['npred'], img_idx); T = rows(d['ts'], d['ntgt'], img_idx)
    conf = d['conf'][P]; pcls = d['pcls'][P]; tp = d['tp'][P]; tcls = d['tcls'][T]
    order = np.argsort(-conf)
    pcls_o, tp_o = pcls[order], tp[order]
    out = {}
    for c in CLASSES:
        n_l = int((tcls == c).sum())
        out[c] = ap_v84(tp_o[pcls_o == c], n_l) if n_l else np.nan
    return out

if __name__ == '__main__':
    label, B = sys.argv[1], int(sys.argv[2])
    d = load(label)
    n_img = len(d['npred'])
    seq_ids = sorted(set(d['seqs']))
    groups = [np.where(d['seqs'] == s)[0] for s in seq_ids]
    obs = ap_per_class(d, np.arange(n_img))
    mean_obs = float(np.nanmean([obs[c] for c in CLASSES]))
    print(f"{label}: recomputed AP50 " +
          ", ".join(f"{CLASSES[c]} {100*obs[c]:.2f}" for c in CLASSES) +
          f" | mean {100*mean_obs:.2f} (validator reported mAP50 {100*d['reported']['mAP50']:.2f})", flush=True)
    rng = np.random.default_rng(0)
    boot = np.zeros((B, len(CLASSES)))
    t0 = time.time()
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        r = ap_per_class(d, idx)
        boot[b] = [r[c] for c in CLASSES]
        if (b + 1) % 200 == 0:
            print(f"  {label} {b+1}/{B} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(f"uavdt_boot_{label}.npz", boot=boot,
                        obs=np.array([obs[c] for c in CLASSES]),
                        classes=np.array(list(CLASSES.values())),
                        seq_ids=np.array(seq_ids))
    print(f"{label} done", flush=True)
