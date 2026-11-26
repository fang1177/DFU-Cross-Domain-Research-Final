import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

p = Path("reports/external_evaluation/seed_2024_uncertainty/internal_validation_diagnostic/per_image_metrics.csv")
with p.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

dice = np.array([float(r["dice"]) for r in rows])
entropy = np.array([float(r["mean_predictive_entropy"]) for r in rows])
mi = np.array([float(r["mean_mutual_information"]) for r in rows])
tp = np.array([int(r["tp"]) for r in rows])

def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(np.asarray(scores, dtype=float))
    return float((ranks[labels].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg))

rho_e = spearmanr(entropy, dice)
rho_m = spearmanr(mi, dice)

result = {
    "records": len(rows),
    "spearman_entropy_vs_dice": {
        "rho": float(rho_e.statistic), "p_value": float(rho_e.pvalue)
    },
    "spearman_mi_vs_dice": {
        "rho": float(rho_m.statistic), "p_value": float(rho_m.pvalue)
    },
    "failure_detection": {},
    "complete_misses": int((tp == 0).sum()),
    "all_background_predictions": sum(int(r["predicted_pixels"]) == 0 for r in rows)
}

for threshold in (0.10, 0.20, 0.50):
    failure = dice < threshold
    result["failure_detection"][str(threshold)] = {
        "failure_count": int(failure.sum()),
        "entropy_auroc": auc(failure, entropy),
        "low_entropy_reverse_auroc": auc(failure, -entropy),
        "mi_auroc": auc(failure, mi),
        "low_mi_reverse_auroc": auc(failure, -mi)
    }

out = p.parent / "internal_failure_detection_analysis.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print("Saved:", out)
