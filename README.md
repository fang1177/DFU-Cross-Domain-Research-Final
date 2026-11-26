# MobileDFU-Net: A Lightweight Domain-Generalized and Uncertainty-Aware Framework for Reliable Cross-Dataset Diabetic Foot Ulcer Segmentation

**Author:** Nagesh Ragolu et al. (SRM University-AP, Vijayawada, India)
**Status:** Completed research project; final IEEE conference manuscript prepared. No clinical validation claimed.

---

## Abstract

Deep-learning segmentation models for diabetic foot ulcers (DFUs) often perform well on the dataset used for training but degrade when images are acquired under different clinical, colour and illumination conditions. This repository contains MobileDFU-Net, a lightweight segmentation framework combining a decoupled chromatic--structural stem with raw-RGB input, a tolerance-aware signed-distance boundary loss, and Monte Carlo (MC) Dropout inference that produces predictive entropy and mutual information. An audited data-reconciliation process reduced several candidate wound-image sources to two independent lineage groups (AZH and WSEG/Mendeley) and froze an approved source-only population of 927 training and 186 internal-validation records, with a native target cohort of 2,187 records reserved for cross-dataset evaluation. Three source-only training runs (seeds 42, 1337, 2024) reached best internal-validation Dice of 0.8370, 0.8322 and 0.8326 (mean 0.8339 +/- 0.0027), but the same checkpoints reached only 0.2485, 0.2203 and 0.2365 macro Dice on the native target, with sensitivity below 0.22. A positive-weight foreground-recovery intervention and two ablations were negative on the target, and entropy/MI failure detection was anti-correlated with failure (AUROC 0.300/0.311). These negative findings are reported explicitly; cross-dataset generalization, foreground suppression and uncertainty reliability remain open problems.

## Research motivation

Automated DFU segmentation can support wound-area quantification and healing tracking. A well-documented limitation of deep segmentation models is that performance on a held-out split of the training dataset does not reliably predict performance on images from another clinical site, camera or lighting setup. The problem is acute in wound photography, where skin tone, ambient light, white balance and acquisition protocol alter colour statistics without changing the underlying pathology.

## Research objectives

1. Audit and reconcile candidate wound-image sources into independent lineage groups and freeze an approved source-only population plus a held-out native target cohort.
2. Design a lightweight DFU segmentation network with a colour-robust input contract and boundary-aware learning.
3. Evaluate cross-dataset generalization honestly, including negative interventions and ablations.
4. Assess MC Dropout uncertainty as a failure-detection signal on the target cohort.
5. Profile model complexity and latency for resource-constrained deployment.

## MobileDFU-Net architecture

MobileDFU-Net (1,116,883 parameters, 517 state-dictionary tensors, 4.47 MB FP32) comprises:

1. **DCS-Stem** -- decoupled chromatic--structural stem. Raw RGB in [0,1] is split into a structural path (BT.601 luminance + Sobel magnitude + 3x3 convolution -> 8 channels) and a chromatic path (ImageNet normalization applied only inside the network, depthwise-separable convolution -> 16 channels). A sigmoid gate derived from the structural features modulates the colour features; the stem outputs `F0 = [Fs, Fc ⊙ g]` (24 channels at 512^2). No dataset-level ImageNet normalization is applied.
2. **Encoder** -- inverted-residual blocks (MobileNetV2 style) with widths (24, 32, 64, 128), block counts (2, 3, 4, 3), stride 2 per stage, producing S1--S4 at 256^2, 128^2, 64^2, 32^2. Dropout 0.1 at four Dropout2d modules.
3. **Bottleneck** -- receptive-field block (RFB) with four parallel dilated branches (rates 1, 3, 5) and 160 channels at 32^2.
4. **Decoder** -- four upsampling stages Up4--Up1 (96, 48, 24, 24 channels at 64^2, 128^2, 256^2, 512^2), each fusing its encoder skip through a channel-attention gate (CAG), plus a native-resolution stem-fusion block D1 (16 channels at 512^2).
5. **Heads** -- auxiliary boundary head (16 -> 8 -> 1, sigmoid) and a boundary-gated segmentation head (`F_seg = D1 + Conv1x1(b_hat) ⊙ D1`).

The model contains no fuzzy sigmoid, no transformer block, and no query--key--value cross-attention. Its gating is limited to the stem's structure-conditioned sigmoid gate, the decoder's CAG skips, and the boundary-conditioned segmentation modulation.

## Dataset provenance and experimental setup

Candidate sources included AZH, WSEG, Mendeley, FUSeg 2021 and DFUC 2022. A lineage audit reduced them to two independent groups: AZH, and WSEG/Mendeley (which share lineage with each other; AZH additionally shares lineage with FUSeg). DFUC 2022 was not used experimentally (institutional authorization and signed licence required). FUSeg was not used experimentally.

- Source training (before exclusions): 945 records; 18 excluded -> 927 approved.
- Internal validation: 193 manifest rows -> 192 unique primary contents -> 191 eligible (one diagnostic row `val_img_0233`, `diagnostic_multiclass`) -> 5 label-uncertain exclusions -> 186 approved.
- Native target cohort: 2,187 records, held out from all training, model selection and threshold tuning.

Preprocessing contract: RGB float32 in [0,1], [3,H,W], no dataset-level ImageNet normalization; masks float32 [1,H,W] in {0,1}; letterbox resize to 512x512 with aspect-ratio preservation and zero padding (bilinear images, nearest masks). Augmentation only on source training; boundary fields recomputed after geometric augmentation; randomness seeded from training seed, record id and epoch.

## Training configuration

Three runs with seeds 42, 1337 and 2024 on the approved source-only cohort (927/186). Configured 100 epochs; batch size 8; 512x512; FP32 (AMP disabled); AdamW (beta1=0.9, beta2=0.999); lr 5e-4; weight decay 1e-4; early-stopping patience 15. Loss weights 1.0/0.5/0.5/0.3 (Dice/BCE/boundary/auxiliary); BCE pos_weight 2.0; boundary tau=2, cap 20. Training GPU: Tesla T4 (Kaggle). Seed 42 and 2024 completed 100 epochs; Seed 1337 stopped early after epoch 96.

Epoch convention: stored `best_epoch` is zero-based; human-readable epoch = stored + 1 (98->99, 80->81, 86->87).

## Cross-dataset evaluation protocol

Best checkpoint per seed evaluated once on the frozen native target (n=2,187) at threshold 0.5. Primary metric: macro Dice. Also reported: macro IoU, sensitivity, specificity, micro Dice. The native target was never used for training, model selection or threshold tuning.

## Results

Internal validation (best Dice): Seed 42 0.8370 (epoch 99), Seed 1337 0.8322 (epoch 81), Seed 2024 0.8326 (epoch 87); mean 0.8339 +/- 0.0027 (sample SD).

Native target (macro, threshold 0.5):

| Seed | Dice | IoU | Sensitivity | Specificity | Micro Dice |
|---|---|---|---|---|---|
| 42 | 0.2485 | 0.1658 | 0.2098 | 0.9914 | 0.2457 |
| 1337 | 0.2203 | 0.1469 | 0.1742 | 0.9951 | 0.2230 |
| 2024 | 0.2365 | 0.1566 | 0.1933 | 0.9915 | 0.2420 |

Gap to internal performance: ~0.59--0.62 Dice; sensitivity below 0.22 with specificity above 0.99 (systematic foreground under-segmentation, not diffuse noise).

Foreground recovery (Seed 2024, pos_weight 2 -> 4): internal Dice 0.8326 -> 0.8225; native Dice 0.2365 -> 0.2436 (+0.0071); sensitivity 0.1933 -> 0.2110 (+0.0177). Both below predefined targets (+0.02 Dice, +0.05 sensitivity); only the internal guardrail (>=0.82) passed.

Ablations (Seed 42, native target): E1 (no DCS-Stem colour path) Dice 0.2398; E2 (boundary logits/probabilities zeroed) Dice 0.2067; baseline 0.2485. Paired mean delta-Dice -0.00869 (E1) and -0.04180 (E2). E2 removes meaningful boundary supervision (boundary loss 0.0, gradients ~7e-5).

Wound-size stratification (Seed 2024 MC outputs, quartiles by ground-truth area): median Dice fell from 0.1873 (Q1) to 0.0764 (Q4); sensitivity 0.1325 -> 0.0401; false-negative fraction 0.8675 -> 0.9599.

## Uncertainty evaluation

MC Dropout with T=15, BN in eval mode, horizontal-flip TTA per pass. Internal: Dice 0.8347, mean entropy 0.00191, MI 0.00055. Native: Dice 0.2135, mean entropy 0.01789, MI 0.00791; Spearman rho with Dice 0.370 (entropy) and 0.344 (MI). Failure = Dice < 0.1 (966 images, 44.17%): AUROC 0.300 (entropy) and 0.311 (MI) -- anti-correlated with failure (failed images had lower mean entropy 0.0111 vs 0.0233 and MI 0.00505 vs 0.01018). MC checkpoint seed 2024 (established by the uncertainty artifact).

## Model complexity and profiling

1,116,883 parameters; 4.47 MB FP32 checkpoint. fvcore 4.209 GFLOPs / 2.1045 GMACs (conv/linear only, lower bound); ptflops 4.18 GMACs; torch.profiler 7.8217 GFLOPs (partial elementwise). Mean latency 75.55 ms (median 75.26, min 74.84, max 77.82), 13.24 images/s, peak 245 MB allocated / 316 MB reserved on RTX 2050, PyTorch 2.11.0, CUDA 12.8, batch 1, synthetic input. Profiling used synthetic input on an RTX 2050, not the training T4.

## Limitations and conclusions

- Only two independent lineage groups (AZH; WSEG/Mendeley); not multi-centre validation.
- Residual label noise after approved exclusions.
- Three seeds on one source cohort; seed dispersion is not domain robustness.
- Seed 1337 stopped early (epoch 96); full per-epoch logs unavailable for Seeds 42 and 1337; only Seed 2024 fully verified; strict load_state_dict() reproduction pending.
- No ECE/Brier calibration metrics; uncertainty characterised only by entropy, MI, rank correlation and AUROC.
- One foreground-recovery intervention tested; both ablations and recovery negative on the target.
- Profiling on synthetic input / RTX 2050; FLOP counts are lower bounds.
- No clinical, prospective or regulatory validation; no clinical-readiness claim.

Conclusion: the domain gap, foreground suppression and uncertainty reliability remain open problems for this framework; negative results are reported rather than presenting a partial success as a solution.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate ; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Install PyTorch separately for your platform:
# https://pytorch.org/get-started/locally/
pip install pytest
```

Python >= 3.10. Tests run with `pytest` (testpaths = tests, pythonpath = src).

## Repository structure

```
src/                  Model, data pipeline, training, evaluation
scripts/              Audit, reconciliation, preprocessing, evaluation scripts
tests/                Unit and integration tests
configs/              Cohort, dataset and preprocessing configuration
data/manifests/       Non-sensitive manifests and audit metadata
manuscript/           Final IEEE manuscript, Overleaf source and figures
docs/                 Project documentation
requirements*.txt     Dependencies
pyproject.toml        Project metadata
```

## Dataset access and licensing

Raw DFU images and masks are not distributed in this repository. Access to AZH, WSEG, Mendeley and other wound datasets is subject to their original licences and terms. Manifests and audit metadata are included where non-sensitive. DFUC 2022 and FUSeg 2021 were not used experimentally.

## Citation

Ragolu, N. et al. (2026). "A Lightweight Domain-Generalized and Uncertainty-Aware Framework for Reliable Cross-Dataset Diabetic Foot Ulcer Segmentation." IEEE conference manuscript. See `manuscript/` for the final PDF and BibTeX.

## Discrepancy note (verified against evidence)

The task brief listed native-target macro Dice as "Seed 1337: 0.2365, Seed 2024: 0.2203". The authoritative evaluation artifacts report Seed 1337 = 0.2203 and Seed 2024 = 0.2365. This README and the manuscript use the authoritative values. See `manuscript/EVIDENCE_TO_MANUSCRIPT_MAP.md`.

## License

Research code provided for academic use. Dataset licences apply to any data obtained separately.