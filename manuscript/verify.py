"""Final verification for the DFU IEEE submission package.

Checks compilation artifacts, reference resolution, figure inclusion, table and
equation counts, and recomputes headline numbers directly from the authoritative
evaluation files. Writes a JSON report and prints a human-readable summary.
"""
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

tex = (HERE / "main.tex").read_text(encoding="utf-8")
bib = (HERE / "references.bib").read_text(encoding="utf-8")
doc = pymupdf.open(HERE / "main.pdf")
text = "".join(p.get_text() for p in doc)

# ---------------- citations
cites = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    for key in m.group(1).split(","):
        cites.add(key.strip())
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
missing_bib = sorted(cites - bibkeys)
uncited = sorted(bibkeys - cites)

# ---------------- figures / tables / equations
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
missing_figs = [f for f in figs if not (HERE / f).exists()]
n_fig_captions = len(re.findall(r"Fig\. \d+\.", text))
n_table_captions = len(re.findall(r"TABLE [IVX]+", text))
n_equations = tex.count(r"\begin{equation}")
n_tables = tex.count(r"\begin{table}") + tex.count(r"\begin{table*}")

# ---------------- numerical reconciliation
seed42 = json.load(open(ROOT / "reports/external_evaluation/seed_42/summary_metrics.json"))
seed1337 = json.load(open(ROOT / "reports/external_evaluation/seed_1337/summary_metrics.json"))
seed2024 = json.load(open(ROOT / "reports/external_evaluation/seed_2024/summary_metrics.json"))
internal = [0.8370467168028637, 0.8321766840514316, 0.8325559490470475]
mean_internal = float(np.mean(internal))
sd_internal = float(np.std(internal, ddof=1))

e1 = json.load(open(ROOT / "experiments/final_research/ablation_e1_colour/evaluation/summary_native_target.json"))
e2 = json.load(open(ROOT / "experiments/final_research/ablation_e2_boundary/evaluation/summary_native_target.json"))
rec = json.load(open(ROOT / "experiments/final_research/fg_recovery_posweight4/evaluation/combined_summary.json"))
unc = json.load(open(ROOT / "reports/external_evaluation/seed_2024_uncertainty/summary_uncertainty_metrics.json"))
fail = json.load(open(ROOT / "reports/external_evaluation/seed_2024_uncertainty/failure_detection_analysis.json"))

# paired ablation deltas
base = pd.read_csv(ROOT / "reports/external_evaluation/seed_42/per_image_metrics.csv")
base_map = dict(zip(base.record_id, base.dice))
quart = pd.read_csv(ROOT / "experiments/final_research/wound_quartiles_native_target.csv")
quart_map = dict(zip(quart.record_id, quart.quartile))


def paired(path):
    df = pd.read_csv(ROOT / path)
    d = df.dice.values - np.array([base_map[r] for r in df.record_id])
    return float(d.mean()), int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())


e1_paired = paired("experiments/final_research/ablation_e1_colour/evaluation/per_image_native_target.csv")
e2_paired = paired("experiments/final_research/ablation_e2_boundary/evaluation/per_image_native_target.csv")

# quartiles from Seed 2024 MC per-image data (the manuscript values)
mc = pd.read_csv(ROOT / "reports/external_evaluation/seed_2024_uncertainty/per_image_uncertainty_metrics.csv")
mc = mc.merge(quart, on="record_id")
quart_stats = {}
for q, g in mc.groupby("quartile"):
    quart_stats[q] = {
        "n": int(len(g)),
        "median_dice": float(g.dice.median()),
        "median_sensitivity": float(g.sensitivity.median()),
        "median_fn_fraction": float((g.fn / (g.fn + g.tp)).median()),
        "median_area_ratio": float(((g.tp + g.fp) / (g.tp + g.fn)).median()),
    }

report = {
    "pdf_pages": doc.page_count,
    "pdf_bytes": (HERE / "main.pdf").stat().st_size,
    "citations_cited": len(cites),
    "bib_entries": len(bibkeys),
    "missing_bib_entries": missing_bib,
    "uncited_bib_entries": uncited,
    "figures_included": len(figs),
    "figure_captions_in_pdf": n_fig_captions,
    "missing_figure_files": missing_figs,
    "tables": n_tables,
    "table_captions_in_pdf": n_table_captions,
    "equations": n_equations,
    "question_marks": text.count("??"),
    "undefined_references_text": text.lower().count("undefined reference"),
    "table_V_N/A_tokens": text.count("N/A"),
    "table_V_NE_tokens": text.count("NE"),
    "seed_mean": mean_internal,
    "seed_sd": sd_internal,
    "native_seed42": seed42["macro"],
    "native_seed2024": {
        "dice": seed2024["macro_dice"],
        "iou": seed2024["macro_iou"],
        "sensitivity": seed2024["macro_sensitivity"],
        "specificity": seed2024["macro_specificity"],
    },
    "native_seed1337": seed1337["macro"],
    "e1": e1["macro"],
    "e2": e2["macro"],
    "e1_paired": e1_paired,
    "e2_paired": e2_paired,
    "recovery": {
        "internal": rec["internal_val"]["macro"]["dice"],
        "native_dice": rec["native_target"]["macro"]["dice"],
        "native_sensitivity": rec["native_target"]["macro"]["sensitivity"],
        "native_iou": rec["native_target"]["macro"]["iou"],
    },
    "uncertainty_native": {
        "dice": unc["macro_metrics"]["dice"],
        "entropy": unc["macro_metrics"]["mean_predictive_entropy"],
        "mi": unc["macro_metrics"]["mean_mutual_information"],
    },
    "failure_auroc": {
        "entropy": fail["results"]["0.1"]["mean_predictive_entropy_failure_auroc"],
        "mi": fail["results"]["0.1"]["mean_mutual_information_failure_auroc"],
        "failed_images": fail["results"]["0.1"]["failed_images"],
    },
    "quartiles_seed2024_mc": quart_stats,
}

(HERE / "_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
