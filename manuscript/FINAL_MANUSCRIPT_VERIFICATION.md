# Final Manuscript Verification

**Manuscript:** A Lightweight Domain-Generalized and Uncertainty-Aware
Framework for Reliable Cross-Dataset Diabetic Foot Ulcer Segmentation
**Verified build:** `main.pdf` (tectonic 0.15.0, XeTeX backend, IEEEtran conference)
**Verification date:** 2026-11-26

## 1. Output paths

| Artifact | Absolute path |
|---|---|
| Compiled PDF | `E:\DFU_Cross_Domain_Research\FINAL_IEEE_SUBMISSION\DFU_IEEE_Final_Manuscript.pdf` |
| LaTeX source | `E:\DFU_Cross_Domain_Research\FINAL_IEEE_SUBMISSION\main.tex` |
| Bibliography | `E:\DFU_Cross_Domain_Research\FINAL_IEEE_SUBMISSION\references.bib` |
| Figure directory | `E:\DFU_Cross_Domain_Research\FINAL_IEEE_SUBMISSION\figures\` |
| Overleaf archive | `E:\DFU_Cross_Domain_Research\FINAL_IEEE_SUBMISSION\DFU_Overleaf_Source_Final.zip` |

## 2. Counts

| Item | Count | Status |
|---|---:|---|
| PDF pages | 9 | PASS |
| References in `references.bib` | 16 | PASS |
| References cited in text | 16 | PASS |
| Figures included | 9 | PASS |
| Figure captions rendered in PDF | 9 | PASS |
| Tables (I–VII) | 7 | PASS |
| Table captions rendered in PDF | 7 | PASS |
| Numbered equations | 13 | PASS |

## 3. Compilation audit

| Check | Result |
|---|---|
| LaTeX engine | tectonic 0.15.0 (XeTeX) |
| Document class | `IEEEtran` conference |
| PDF produced | Yes, 9 pages, 1,658,585 bytes |
| LaTeX errors | 0 |
| Undefined citations | 0 |
| Unresolved `\ref` / `\eqref` | 0 (`??` count = 0) |
| Missing figure files | 0 |
| Overfull boxes | 0 |
| Underfull boxes | 10 (cosmetic; title/abstract/acknowledgment line breaking) |
| Bibliography | `main.bbl` generated with IEEEtran style, 16 entries |
| Process exit code | 1 |

**Exit-code explanation.** Tectonic returns a non-zero status when the TeX
engine emits warnings. Here the warnings are (i) a Windows `fontconfig`
default-config message and (ii) `TU/ptm` font-shape substitution notices
caused by compiling IEEEtran's Times fonts under XeTeX. Neither is a LaTeX
error, and the PDF is complete. Overleaf's pdfLaTeX path does not emit these
XeTeX-specific messages. A clean extraction of
`DFU_Overleaf_Source_Final.zip` was compiled independently and also produced
a 9-page PDF with zero unresolved references.

**Not performed:** an actual Overleaf cloud compile and a pdfLaTeX/BibTeX
compile on this machine. `pdflatex`, `xelatex`, `lualatex`, and `bibtex` were
not installed; tectonic was used instead. The source is standard IEEEtran and
is expected to compile on Overleaf, but this was not executed here.

## 4. Reference audit

All 16 keys resolve and all 16 are cited. No uncited entries. The bibliography
contains only verified venue/author metadata; no invented DOIs. The added
references beyond the original seven are genuine and appear in the PDF:

- U-Net (MICCAI 2015)
- nnU-Net (Nature Methods 2021)
- Kendall & Gal (NeurIPS 2017)
- Guo et al. (ICML 2017)
- FUSeg (Information 2024)
- Guan & Liu (IEEE TBME 2022)
- Hu et al. (IEEE TMI 2023)
- Liu et al. RFB Net (ECCV 2018)
- Cassidy et al. DFUC 2020 (touchREVIEWS in Endocrinology 2021)

## 5. Figure audit

| Manuscript figure | File | Present | Rendered |
|---|---|---:|---:|
| Fig. 1 Workflow | `figures/fig1_workflow.png` | Yes | Yes |
| Fig. 2 Architecture | `figures/fig2_architecture.png` | Yes | Yes |
| Fig. 3 DCS-Stem | `figures/fig3_dcs_stem.png` | Yes | Yes |
| Fig. 4 Boundary pipeline | `figures/fig4_boundary.png` | Yes | Yes |
| Fig. 5 Uncertainty pipeline | `figures/fig5_uncertainty.png` | Yes | Yes |
| Fig. 6 Seed 2024 training | `figures/fig7_training.png` | Yes | Yes |
| Fig. 7 Three-seed performance | `figures/fig6_performance.png` | Yes | Yes |
| Fig. 8 Foreground recovery | `figures/fig9_recovery.png` | Yes | Yes |
| Fig. 9 Ablation and quartiles | `figures/fig8_ablation.png` | Yes | Yes |

Three-seed performance and ablation figures were checked visually after the
final seed-ordering and quartile-boundary corrections.

## 6. Table V audit (no unexplained N/A or NE)

Table V now has six numeric columns: Dice, Entropy, MI, and two explicit
failure-AUROC columns (Entropy AUROC, MI AUROC). The non-numeric entries
are explained in the table note.

| Row | Dice | Entropy | MI | Ent. AUROC | MI AUROC | Audit |
|---|---:|---:|---:|---:|---:|---|
| Internal, deterministic | 0.8326 | N/A | N/A | N/A | N/A | N/A: deterministic single pass, no MC entropy/MI/AUROC |
| Internal, MC (Seed 2024, T=15) | 0.8347 | 0.00191 | 0.00055 | NE | NE | NE: no internal failure-label set was defined for AUROC |
| Native, deterministic | 0.2365 | N/A | N/A | N/A | N/A | N/A as above |
| Native, MC (Seed 2024, T=15) | 0.2135 | 0.01789 | 0.00791 | 0.300 | 0.311 | AUROC for Dice < 0.1; values below 0.5 are anti-correlated |

The MC rows use the Seed 2024 checkpoint (best epoch 87), established by
`summary_uncertainty_metrics.json` (`"seed": 2024`,
`"checkpoint_best_epoch": 86`).

**Outcome: PASS.** Every `N/A` means *not applicable* and every `NE` means
*not evaluated*; both abbreviations are defined in the table footnote. There
are no bare dashes or unexplained blank cells.

## 7. Numerical reconciliation

Each headline value in the manuscript was recomputed from the authoritative
repository artifact, not copied from the earlier ChatGPT layout PDF.

| Manuscript value | Recomputed value | Source artifact | Result |
|---|---|---|---|
| Internal mean 0.833927 ± 0.002709 | 0.83392645 ± 0.00270888 | three seed summaries | PASS |
| Seed 42 native Dice 0.2485 | 0.24850680 | `reports/external_evaluation/seed_42/summary_metrics.json` | PASS |
| Seed 2024 native Dice 0.2365 | 0.23653303 | `reports/external_evaluation/seed_2024/summary_metrics.json` | PASS |
| Seed 1337 native Dice 0.2203 | 0.22025051 | `reports/external_evaluation/seed_1337/summary_metrics.json` | PASS |
| E1 Dice 0.2398 | 0.23981183 | `experiments/final_research/ablation_e1_colour/evaluation/summary_native_target.json` | PASS |
| E2 Dice 0.2067 | 0.20670944 | `experiments/final_research/ablation_e2_boundary/evaluation/summary_native_target.json` | PASS |
| E1 mean ΔDice −0.00869; 957/1081/149 | −0.00869497; 957/1081/149 | paired per-image CSVs | PASS |
| E2 mean ΔDice −0.04180; 684/1303/200 | −0.04179736; 684/1303/200 | paired per-image CSVs | PASS |
| Recovery Dice 0.2365 → 0.2436 | 0.24356042 | `combined_summary.json` | PASS |
| Recovery sensitivity 0.1933 → 0.2110 | 0.21095987 | `combined_summary.json` | PASS |
| Native MC Dice 0.2135; entropy 0.01789; MI 0.00791 | 0.21349294; 0.01789418; 0.00791091 | `summary_uncertainty_metrics.json` | PASS |
| AUROC 0.300 / 0.311; 966 failures | 0.30003747 / 0.31140259; 966 | `failure_detection_analysis.json` | PASS |
| Quartile medians Q1→Q4 | 0.18732 / 0.17464 / 0.12424 / 0.07641 | Seed 2024 MC per-image CSV + quartile CSV | PASS |
| Table I 191 pre-exclusion internal records | 193 rows → 192 unique primary → 191 eligible → 186 after 5 exclusions | canonical manifest + diagnostic flag | PASS |

**Outcome: PASS.** No numerical mismatch was found between the manuscript and
the verified evaluation artifacts. The earlier Table I 191-vs-192 ambiguity is
resolved by the single diagnostic row `val_img_0233`
(`is_diagnostic=True`, task `diagnostic_multiclass`).

## 8. Consistency checks

| Check | Result |
|---|---|
| Epoch convention (stored `best_epoch` + 1) applied consistently | PASS |
| Q1–Q4 values labelled as Seed 2024 MC Dropout, not Seed 42 | PASS |
| Completed experiments excluded from Future Work | PASS |
| Negative findings retained (domain gap, failed recovery, failed AUROC, negative ablations) | PASS |
| No claim of multi-centre or clinical validation | PASS |
| No state-of-the-art claim | PASS |
| GFLOPs and GMACs conventions distinguished | PASS |
| Math matches implemented losses/inference (weights, signed field, MI clamp) | PASS |
| Table VII compact (5 rows, half-column) and not dominating the page | PASS |

## 9. Remaining limitations stated in the paper (consolidated into four categories)

1. **Dataset lineage and generalizability** -- only two independent lineage
   groups (AZH, which shares lineage with FUSeg; WSEG/Mendeley, which share
   lineage with each other); DFUC 2022 was not used, so this is not
   multi-centre external validation. Residual label noise remains after the
   approved exclusions. Three seeds on one source cohort; seed dispersion is
   not domain robustness.
2. **Training reproducibility and checkpoint verification** -- Seed 1337
   stopped early at epoch 96; full per-epoch logs for Seeds 42 and 1337 were
   unavailable; only Seed 2024 received full artifact verification. Strict
   `load_state_dict()` reproduction remains pending.
3. **Uncertainty calibration and intervention scope** -- no ECE/Brier
   calibration metrics were computed; uncertainty is characterized only by
   entropy, MI, rank correlation, and AUROC. Only one foreground-recovery
   intervention was tested (positive weight 2->4); both ablations and the
   recovery run are negative relative to the baseline on the target, and E2
   removed boundary supervision substantially.
4. **Computational profiling and clinical validation** -- profiling used
   synthetic input on an RTX 2050 rather than the training T4, and FLOP
   counts are lower bounds. No clinical, prospective, or regulatory validation
   was performed.

## 10. Unperformed checks

- Overleaf cloud compilation.
- pdfLaTeX + BibTeX compilation on this machine (not installed).
- Strict checkpoint loading and local inference reproduction.
- Per-pixel calibration and reliability analysis.
- Independent verification of Seeds 42 and 1337 artifacts.
- Any new training, fine-tuning, or target-label adaptation experiment.
