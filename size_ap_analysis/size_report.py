import json
for z in (1280, 1536):
    r = json.load(open(f"size_boot_{z}.json"))
    print(f"\n===== {z} px  (AP50 %, class means over classes with >=100 boxes in the bin; 95% bootstrap CI) =====")
    print(f"{'bin':8s} {'classes':>7s} | {'P3 prog':>7s} {'P2 prog':>7s} {'P3 sp':>7s} {'P2 sp':>7s}")
    for b, v in r['means'].items():
        print(f"{b:8s} {len(r['class_sets'][b]):7d} | " + " ".join(f"{v[m]:7.1f}" for m in ('p3_prog', 'p2_prog', 'p3_sp', 'p2_sp')))
    for cname in ('stride | progressive', 'stride | single phase', 'schedule | P3-P5', 'schedule | P2', 'interaction'):
        print(f"  {cname}")
        for b, c in r['contrasts'].items():
            x = c[cname]
            print(f"    {b:8s} {x['obs']:+6.2f}  [{x['ci'][0]:+6.2f}, {x['ci'][1]:+6.2f}]  p={x['p']}")
