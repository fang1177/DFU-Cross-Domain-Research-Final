# MobileDFU-Net — IEEE Conference Submission Package

This package contains the final, compiled IEEE conference manuscript and a
self-contained LaTeX source tree for the paper:

> **A Lightweight Domain-Generalized and Uncertainty-Aware Framework for
> Reliable Cross-Dataset Diabetic Foot Ulcer Segmentation**

## Package contents

| File | Purpose |
|---|---|
| `DFU_IEEE_Final_Manuscript.pdf` | Compiled manuscript (9 pages) |
| `main.tex` | IEEEtran conference source |
| `references.bib` | 16 verified references |
| `main.bbl` | Pre-built bibliography (fallback for quick compilation) |
| `figures/fig1_workflow.png` … `figures/fig9_recovery.png` | Nine integrated figures |
| `README.md` | This file |
| `FINAL_MANUSCRIPT_VERIFICATION.md` | Audit of counts, compilation, Table V, and numerical reconciliation |
| `EVIDENCE_TO_MANUSCRIPT_MAP.md` | Claim-level mapping from manuscript to repository evidence |

## Compilation

### Overleaf (recommended)

1. Upload `DFU_Overleaf_Source_Final.zip` (or the files in this directory).
2. Set the compiler to **pdfLaTeX**.
3. Set the main document to `main.tex`.
4. Compile. If you edit `references.bib`, delete `main.bbl` and re-run
   **BibTeX** before recompiling.

The class is the standard `IEEEtran` conference class, available on Overleaf.

### Local (tectonic, used for the verified build)

```powershell
tectonic -X compile main.tex --keep-intermediates --keep-logs
```

The verified build used **tectonic 0.15.0** (XeTeX backend). A `main.pdf` is
produced successfully. Tectonic returns exit code 1 because it treats
warnings as failures; the warnings are environmental (`fontconfig` default
config missing under Windows) and cosmetic font-shape substitutions
(`TU/ptm`), not LaTeX errors. There are **zero overfull boxes**, **zero
undefined citations**, and **zero unresolved references**.

## Headline results in the manuscript

- Approved source-only population: **927 training / 186 internal validation**;
  frozen native target: **2,187**.
- Three seeds (42, 1337, 2024), best internal-validation Dice:
  **0.8370 / 0.8322 / 0.8326**; mean **0.8339 ± 0.0027** (sample SD).
- Native-target macro Dice: **0.2485 / 0.2365 / 0.2203** at threshold 0.5.
- Foreground recovery (positive weight 2 → 4): native Dice **0.2365 → 0.2436**;
  sensitivity **0.1933 → 0.2110**; both below predefined targets.
- Uncertainty failure detection (Dice < 0.1): AUROC **0.300** (entropy) and
  **0.311** (MI), i.e. anti-correlated with failure.
- Ablations (Seed 42): E1 colour **0.2398**, E2 boundary **0.2067**, baseline
  **0.2485**.

## Important scope notes

- Only **two independent lineage groups** were available (AZH and
  WSEG/Mendeley). This is **not** multi-centre external validation.
- Seeds 42 and 2024 completed 100 epochs; Seed 1337 stopped early after
  epoch 96.
- Stored `best_epoch` is zero-based; all manuscript epoch numbers are
  human-readable (stored + 1).
- The quartile analysis in Section V-E uses **Seed 2024 MC Dropout** per-image
  outputs, explicitly labelled in the text.
- FLOPs are reported under both conventions: fvcore **4.209 GFLOPs /
  2.1045 GMACs** (conv/linear only, lower bound) and torch.profiler
  **7.8217 GFLOPs** (partial elementwise accounting).
- No clinical, prospective, or regulatory validation was performed. No
  state-of-the-art claim is made.

## Figure files

| Manuscript figure | File | Content |
|---|---|---|
| Fig. 1 | `figures/fig1_workflow.png` | Research workflow |
| Fig. 2 | `figures/fig2_architecture.png` | MobileDFU-Net architecture |
| Fig. 3 | `figures/fig3_dcs_stem.png` | DCS-Stem colour/structure pathway |
| Fig. 4 | `figures/fig4_boundary.png` | Boundary-field construction and loss |
| Fig. 5 | `figures/fig5_uncertainty.png` | MC Dropout uncertainty pipeline |
| Fig. 6 | `figures/fig7_training.png` | Seed 2024 training history |
| Fig. 7 | `figures/fig6_performance.png` | Three-seed internal/native/domain gap |
| Fig. 8 | `figures/fig9_recovery.png` | Foreground-recovery intervention |
| Fig. 9 | `figures/fig8_ablation.png` | Colour/boundary ablations and quartiles |

The filename numbers in `figures/` reflect generation order, not manuscript
figure numbers; the table above is authoritative.
