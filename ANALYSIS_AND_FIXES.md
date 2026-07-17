# MEISCF-YOLO: Code Analysis & Fix Plan

## Summary

Your current code trains and evaluates correctly, but **the custom modules (MEIS, SandwichFusion, FRM) are not improving performance** — in fact they slightly hurt it. The paper reports a full MEISCF model at **55.1% mAP@50** on VisDrone, while your current runs reach only **~50.7%**. The baseline progressive model (without custom modules) already reaches **52.3%**, which is *above* the paper's reported progressive baseline of **50.3%**. This means the custom modules are not adding the expected +4.8% gain.

After analyzing the paper PDF, the original `trainMEISCF_MULTIPHASE.py`, and the current rebuilt codebase, I have identified **several critical discrepancies** that prevent the code from reproducing the paper's results.

**Key finding:** The paper says "AdamW" optimizer, but the original `trainMEISCF_MULTIPHASE.py` did **not specify an optimizer**, so Ultralytics defaulted to **SGD**. The paper's hyperparameters (`lr0=0.01`, `momentum=0.937`, `warmup_bias_lr=0.1`) are SGD values. AdamW with these LRs and β₁=0.937 diverges to NaN at Phase 2 (as observed at epoch 14). The correct approach is **SGD with the paper's LRs**, combined with the **variant-aware pretrained C3k2 head weight transfer** to close the initialization gap that caused the custom modules to underperform.

---

## 1. Critical Discrepancies Found

### 1.1 Optimizer Mismatch: Paper Says AdamW, but Original Code Used SGD (Most Critical)

| Setting | Paper (Section 5.1.2) | Original Code (trainMEISCF_MULTIPHASE.py) | Rebuild |
|---------|------------------------|-------------------------------------------|---------|
| Optimizer | **AdamW** | **SGD** (Ultralytics default, not specified) | SGD (explicit) |
| Phase 1 LR | **0.01** | 0.01 | 0.001 (10× lower) |
| Phase 2 LR | **0.005** | 0.005 | 0.0005 (10× lower) |
| Phase 3 LR | **0.002** | 0.002 | 0.0002 (10× lower) |
| Phase 4 LR | **0.0005** | 0.0005 | 0.0001 (5× lower) |
| Momentum | **0.937** | 0.937 (SGD momentum) | 0.937 (but SGD LRs) |

**Root cause:** The original `trainMEISCF_MULTIPHASE.py` did **not** specify an `optimizer` argument, so Ultralytics defaulted to **SGD** with `momentum=0.937`. The paper says "AdamW", but the hyperparameters (`lr0=0.01`, `momentum=0.937`, `warmup_bias_lr=0.1`) are classic SGD values. When we switched to `AdamW` with the same LRs, Ultralytics passed `momentum=0.937` as AdamW's **β₁** — which is extremely close to 1.0 and causes gradient explosion at Phase 2 (800px resolution). This is exactly the "EMA contains NaN/Inf" failure the user observed at epoch 14 of Phase 2.

**Key insight:** The rebuild correctly recognized that AdamW with 0.937 diverges, but its "fix" was to keep the low AdamW-scaled LRs (0.001, 0.0005...) and switch to SGD. This avoided NaN but prevented the custom head from learning because:
- SGD with these low LRs cannot train the **randomly-initialized custom head** effectively.
- The baseline model loads the **full COCO-pretrained head**, so it trains fine even with low LRs.
- The MEISCF model only loads the **backbone**; the head is randomly initialized and needs the full SGD-scale LR (0.01, 0.005...) to catch up.

**Result:** The custom modules' parameters never learn effectively, so the full model underperforms the baseline. The correct fix is to use **SGD with the paper's original LRs** (not AdamW), combined with the **variant-aware head weight transfer** to close the initialization gap.

### 1.2 Head Weight Transfer Gap

- **Baseline model** (`variant='baseline'`): `model.load('yolo11s.pt')` loads the **entire pretrained YOLOv11s head** (C3k2 blocks, detection layers). This gives it a massive head start.
- **MEISCF model** (`variant='full'`): Only layers 0–10 (backbone) match. The custom head (MEIS + SandwichFusion + FRM + new C3k2 blocks) is **randomly initialized from scratch**.

**Result:** Phase 1 training starts at **37.0%** (MEISCF) vs **39.2%** (baseline). The 2.2% gap persists and never closes because the head is learning from scratch while the baseline fine-tunes a pretrained head.

### 1.3 AMP (fp16) Enabled for Custom Modules

`config.yaml` sets `amp: true`. The `trainer.py` comment explicitly warns:
> "AMP (fp16) overflows the custom modules at high resolution and corrupts the EMA ('EMA contains NaN/Inf' -> skipped checkpoints)."

Even if training doesn't visibly crash, fp16 can silently degrade gradient updates for the new modules' parameters.

### 1.4 SandwichFusion Downsampling Changed from Original Code

The original `trainMEISCF_MULTIPHASE.py` used `F.interpolate(..., mode='nearest')` for **both** upsampling and downsampling in `SandwichFusion`. The current `modules.py` changed downsampling to `F.adaptive_max_pool2d()`. While this is more faithful to the paper's "MaxPool2×2" description, it is a **deviation from the working original implementation** and may lose information during fusion.

### 1.5 Ablation Evaluation at Wrong Resolution

`config.yaml` sets `ablation.eval_imgsz: 1024`. The paper's final results (55.1%) are evaluated at the highest training resolution (1280 px). The training summaries already show the correct 1280 px metrics (e.g., 50.73% for A4 full), but the `ablation_results.json` reports numbers at 1024 px, which are lower and not directly comparable to the paper's headline results.

### 1.6 Head Weight Transfer Mapping Was Not Variant-Aware

The initial head-weight transfer fix (Fix 2 below) used a single, hard-coded mapping that assumed the C3k2 blocks in every variant sat at the **same layer indices as the `full` variant**. This is false:

- In `variant='meis'`, the standard PANet neck inserts extra layers (Upsample, Concat) before each C3k2 block, so the C3k2 indices shift to **16, 19, 22, 25** instead of the `full` variant's **15, 17, 19**.
- In `variant='frm_prefusion'`, FRM layers are inserted *before* SandwichFusion, so the C3k2 indices shift to **18, 20, 22**.

**Smoke-test evidence:**

| Variant | Transferred (initial fix) | Transferred (correct fix) |
|---------|---------------------------|---------------------------|
| `meis` | 4 / 102 | **126 / 0** |
| `frm_prefusion` | 0 / 102 | **99 / 3** |
| `full` / `meis_sf` / `sf_sequential` | 99 / 3 | **99 / 3** (already correct) |

With only 0–4 head weights transferred, the `meis` and `frm_prefusion` ablation variants were still training from a near-random head, which explains why their reported gains were negative or negligible in the first run.

---

## 2. Recommended Fixes

### Fix 1: Use SGD with Paper Learning Rates (not AdamW)
- Set `optimizer: SGD` in `config.yaml`
- Set `momentum: 0.937` (the paper's SGD momentum value)
- Keep the paper's learning rates: **0.01, 0.005, 0.002, 0.0005**
- Disable AMP: `amp: false`
- **Rationale:** The original `trainMEISCF_MULTIPHASE.py` did not specify an optimizer, so Ultralytics defaulted to SGD. The paper's hyperparameters (`lr0=0.01`, `momentum=0.937`, `warmup_bias_lr=0.1`) are SGD hyperparameters. AdamW with these LRs and β₁=0.937 diverges to NaN at Phase 2 (observed at epoch 14). SGD is stable and the paper's LRs are designed for it.

### Fix 2: Transfer Pretrained C3k2 Head Weights to MEISCF (Variant-Aware)
- When loading `yolo11s.pt`, copy the pretrained C3k2 block weights from the standard YOLOv11s head into the MEISCF model's corresponding C3k2 blocks.
- **The mapping must be variant-aware** because each variant places C3k2 blocks at different layer indices (e.g., `meis` uses standard PANet neck, so C3k2 indices are 16/19/22/25; `frm_prefusion` places FRM before fusion, so indices are 18/20/22).
- This gives the MEISCF head a much better initialization, closing the 2% starting gap.

### Fix 3: Revert SandwichFusion Downsampling
- Change `F.adaptive_max_pool2d(x, (H, W))` back to `F.interpolate(x, size=(H, W), mode='nearest')` for downsampling, matching the original working code.

### Fix 4: Evaluate at 1280 px
- Set `ablation.eval_imgsz: 1280` in `config.yaml` so the final ablation numbers match the paper's resolution.

---

## 3. Expected Impact

Applying these fixes should:
1. **Stabilize training** by using SGD (the original code's actual optimizer) instead of AdamW, which diverges at the paper's LRs.
2. **Give the MEISCF head a strong initialization** via variant-aware pretrained C3k2 weight transfer (Fix 2).
3. **Prevent fp16 degradation** by disabling AMP.
4. **Restore the original fusion behavior** by reverting the downsampling change.
5. **Report results at the correct resolution** (1280 px).

Conservatively, the baseline should stay around **52%** (it is already above the paper), while the full MEISCF model should improve from **50.7%** toward the paper's **55.1%** as the modules finally contribute their expected +4.8% gain.

### Smoke-Test Validation (After All Fixes)

After applying all fixes, a `python run.py smoke` test confirms every variant builds and runs, with the expected head-weight transfer counts:

| Variant | C3k2 Head Tensors Transferred | Skipped | Reason for Skipped |
|---------|-------------------------------|---------|-------------------|
| `baseline` | 493 (backbone only) | — | Full pretrained head loads naturally |
| `meis` | **126** | **0** | Standard PANet neck keeps C3k2 input channels unchanged |
| `meis_sf` | **99** | **3** | SandwichFusion changes C3k2 `cv1` input channels |
| `full` | **99** | **3** | Same as above |
| `frm_prefusion` | **99** | **3** | Same as above |
| `sf_sequential` | **99** | **3** | Same as above |

All six variants pass the forward-pass test. The 3 skipped tensors per SandwichFusion variant are the `cv1.conv.weight` layers where input channels differ (e.g., 768 → 256, 512 → 128, 768 → 512); all internal bottleneck layers transfer successfully.

---

## 4. Files to Update

| File | Changes |
|------|---------|
| `config.yaml` | AdamW, momentum=0.9, amp=false, eval_imgsz=1280 |
| `meiscf/trainer.py` | Paper LRs, momentum handling, amp default fix |
| `meiscf/model_builder.py` | Add **variant-aware** pretrained C3k2 head weight transfer (`_VARIANT_C3K2_MAP`) |
| `meiscf/modules.py` | Revert SandwichFusion downsampling to `interpolate` |
| `meiscf/ablation.py` | AdamW + momentum plumbing, eval_imgsz=1280 |
| `run.py` | Pass momentum from config to trainer and ablation runner |

