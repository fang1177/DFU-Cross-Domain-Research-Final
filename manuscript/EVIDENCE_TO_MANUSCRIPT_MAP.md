# Evidence-to-Manuscript Map

This document maps the quantitative and methodological claims in
`main.tex` to repository evidence. The authoritative sources are the
manifests, checkpoints, evaluation CSVs/JSONs, code, and profiling reports
listed below. The earlier ChatGPT-generated PDF was used only as a layout
reference; no number was copied from it without re-derivation.

## 1. Source-of-truth policy

- Dataset counts: canonical corrected manifest and reconciliation notes.
- Training results: checkpoint histories and recorded evaluation summaries.
- Native-target results: per-seed evaluation JSON and per-image CSVs.
- Ablations/recovery: experiment summary JSON plus paired per-image CSVs.
- Uncertainty: MC Dropout summary and failure-detection JSON.
- Architecture/losses: `src/models/mobile_dfu_net/*.py`.
- Profiling: `reports/final_execution/computational_profiling_report.md`.

## 2. Dataset and cohort claims

| Manuscript location | Claim | Evidence | Status |
|---|---|---|---|
| Table I; §IV-A | 945 source-training records before exclusions | canonical manifest / reconciliation report | Verified |
| Table I; §IV-A | 191 internal-validation records before exclusions | manifest: 193 rows → 192 unique primary → 191 eligible (1 diagnostic row `val_img_0233`) | Verified |
| Table I; §IV-A | 18 source-training and 5 internal-validation exclusions | approved exclusion records | Verified |
| Table I; §IV-A | 927 training / 186 validation analysis population | source-only builder assertions in each run | Verified |
| Table I; §IV-A | 2,187 native-target records | native target manifest | Verified |
| §II; §IV-A | Two independent lineage groups (AZH; WSEG/Mendeley) | lineage reconciliation report | Verified |
| §IV-A | DFUC 2022 not used experimentally | access/licence notes | Verified |

## 3. Architecture and method claims

| Manuscript location | Claim | Evidence | Status |
|---|---|---|---|
| Fig. 2; §IV-C | Encoder widths 24/32/64/128, blocks 2/3/4/3 | `src/models/mobile_dfu_net/config.py`, `model.py` | Verified |
| Fig. 2; §IV-C | RFB bottleneck 160 channels | `config.py` | Verified |
| Fig. 2; §IV-C | Decoder Up4--Up1 widths 96/48/24/24, plus native stem-fusion block $D_1$ with 16 channels | `feature_fusion.py`, `config.py` | Verified |
| Fig. 2; §IV-C | Boundary-gated segmentation head (boundary probability modulation) and auxiliary boundary head (16->8->1 channels) | `segmentation_head.py`, `boundary_head.py` | Verified |
| Fig. 2; §IV-C | 1,116,883 parameters, 517 tensors | checkpoint state-dict inspection | Verified |
| Fig. 2; §IV-C | No fuzzy sigmoid, transformer block, or attention gate | code audit | Verified |
| Fig. 3; §IV-D | Raw RGB, structural Sobel/luminance branch, internal-only normalization, sigmoid gate | `dcs_stream.py` | Verified |
| Fig. 4; §IV-E | φ_signed = φ_fp − φ_fn; mean(φ_signed · σ(z)) | `losses.py` | Verified |
| Fig. 4; §IV-E | 3×3 erosion, Euclidean distance transform, τ=2, cap 20 | `boundary_fields.py`, `config.py` | Verified |
| Eq. (9); §IV-F | Positive-weighted BCE with `pos_weight=2` | `losses.py`, `config.py` | Verified |
| Eq. (10); §IV-F | Weights 1.0 / 0.5 / 0.5 / 0.3 | `config.py` (`MobileDFULossConfig`) | Verified |
| Fig. 5; §IV-G | T=15, BN eval, flip TTA, entropy, MI, tiny-negative clamp | `uncertainty.py` | Verified |

## 4. Training and internal-validation claims

| Manuscript location | Claim | Evidence | Status |
|---|---|---|---|
| Table III; §V-A | Seed 42: 0.837047, epoch 99 | recovered Seed 42 checkpoint history | Verified |
| Table III; §V-A | Seed 1337: 0.832177, epoch 81 | recovered Seed 1337 checkpoint history | Verified |
| Table III; §V-A | Seed 2024: 0.832556, epoch 87 | Seed 2024 checkpoint history | Verified |
| Table III; §V-A | Mean 0.833927 ± 0.002709 | recomputed from three values (sample SD) | Verified |
| §V-B | Seed 2024 final validation Dice 0.826695 at epoch 100 | checkpoint history | Verified |
| §V-B | Seed 42 final losses 0.227489 / 0.333324 | checkpoint history | Verified |
| §V-B | Seed 1337 stopped after epoch 96 | checkpoint history | Verified |
| §V-D; Table II | Seed 2024 AdamW, lr 5e-4, wd 1e-4, patience 15 | Seed 2024 run record | Verified |

Epoch convention: stored `best_epoch` is zero-based; human-readable epoch =
stored + 1 (98→99, 80→81, 86→87).

## 5. Native-target and ablation claims

| Manuscript location | Claim | Evidence | Status |
|---|---|---|---|
| Table IV | Seed 42: 0.2485 / 0.1658 / 0.2098 / 0.9914 / 0.2457 | `reports/external_evaluation/seed_42/summary_metrics.json` | Verified |
| Table IV | Seed 2024: 0.2365 / 0.1566 / 0.1933 / 0.9915 / 0.2420 | `reports/external_evaluation/seed_2024/summary_metrics.json` | Verified |
| Table IV | Seed 1337: 0.2203 / 0.1469 / 0.1742 / 0.9951 / 0.2230 | `reports/external_evaluation/seed_1337/summary_metrics.json` | Verified |
| Table VII | E1: 0.2398 / 0.1590 / 0.1873 / 0.9933 / 0.2287 | `experiments/final_research/ablation_e1_colour/evaluation/summary_native_target.json` | Verified |
| Table VII | E2: 0.2067 / 0.1355 / 0.1649 / 0.9935 / 0.2025 | `experiments/final_research/ablation_e2_boundary/evaluation/summary_native_target.json` | Verified |
| §V-H | E1 mean ΔDice −0.00869; 957/1081/149 | paired per-image CSVs | Verified |
| §V-H | E2 mean ΔDice −0.04180; 684/1303/200 | paired per-image CSVs | Verified |
| §V-H | E2 boundary loss 0.0; gradients ~7e-5 | gradient audit report | Verified |
| Table VI | Recovery internal 0.8326→0.8225 | `experiments/final_research/fg_recovery_posweight4/evaluation/combined_summary.json` | Verified |
| Table VI | Recovery native Dice 0.2365→0.2436; sensitivity 0.1933→0.2110 | same | Verified |
| §V-G | Threshold 0.75 selected internally (0.8240); native 0.2328 at 0.75 | `threshold_sweep.json` | Verified |

## 6. Uncertainty claims

| Manuscript location | Claim | Evidence | Status |
|---|---|---|---|
| Table V | Internal MC Dice 0.8347, entropy 0.00191, MI 0.00055 | `internal_validation_diagnostic/summary.json` | Verified |
| Table V | Native MC Dice 0.2135, entropy 0.01789, MI 0.00791 | `summary_uncertainty_metrics.json` | Verified |
| §V-F | Spearman ρ 0.370 (entropy), 0.344 (MI) | same | Verified |
| §V-F | AUROC 0.300 / 0.311; 966 failures (44.17%) | `failure_detection_analysis.json` | Verified |
| §V-F | Failed mean entropy 0.0111 vs successful 0.0233 | same | Verified |
| §V-F | Failed mean MI 0.00505 vs successful 0.01018 | same | Verified |

## 7. Profiling claims

| Manuscript location | Claim | Evidence | Status |
|---|---|---|---|
| §V-C | 1,116,883 params; 4.47 MB | profiling report | Verified |
| §V-C | fvcore 4.209 GFLOPs / 2.1045 GMACs | profiling report | Verified |
| §V-C | ptflops 4.18 GMACs | profiling report | Verified |
| §V-C | torch.profiler 7.8217 GFLOPs | profiling report | Verified |
| §V-C | Latency 75.55 ms mean / 75.26 median; 13.24 img/s | profiling report | Verified |
| §V-C | Peak 245 MB allocated / 316 MB reserved | profiling report | Verified |

## 8. Figure provenance

| Figure | Data/description source |
|---|---|
| Fig. 1 | Workflow described in §IV; cohort counts from Table I |
| Fig. 2 | `config.py`, `model.py`, `dcs_stream.py`, `feature_fusion.py` |
| Fig. 3 | `dcs_stream.py` |
| Fig. 4 | `boundary_fields.py`, `losses.py` |
| Fig. 5 | `uncertainty.py` |
| Fig. 6 | `_hist_seed2024.json` exported from the Seed 2024 checkpoint |
| Fig. 7 | three seed summaries at native target; internal best Dice |
| Fig. 8 | `fg_recovery_posweight4` combined summary |
| Fig. 9 | E1/E2 summaries, paired CSVs, and native-target quartile CSV |

Figures are regenerated by `build_figures.py`; all quantitative panels read
the artifacts above. No curve or bar is hand-typed.

## 9. Explicitly not evidenced / pending

- Strict `load_state_dict()` reproduction of recorded scores.
- Complete per-epoch logs for Seeds 42 and 1337.
- Per-pixel ECE/Brier calibration.
- Independent artifact verification for Seeds 42 and 1337 (only Seed 2024
  received full verification).
- Multi-centre or prospective clinical validation.
- Any target-label adaptation or additional training beyond the recorded
  experiments.

These gaps are stated in the manuscript's Limitations and Future Work and are
not presented as completed.
