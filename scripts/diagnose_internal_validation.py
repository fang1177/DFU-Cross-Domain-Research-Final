import csv, json, math, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from src.data.stage_4_dataloaders import (
    DFUCanonicalDataset, DEFAULT_MANIFEST, DEFAULT_CANONICAL_DIR,
    DEFAULT_BOUNDARY_FIELD_DIR, DEFAULT_PROJECT_ROOT
)
from src.data.exclusion_registry import load_exclusion_registry
from src.dfu_cross_domain.training.checkpoint import load_checkpoint
from src.models.mobile_dfu_net import MobileDFUNet, MobileDFUNetConfig

ROOT = Path(DEFAULT_PROJECT_ROOT)
CHECKPOINT = ROOT / "Seed2024_Backup/seed_2024_resume/checkpoints/best.pt"
OUT = ROOT / "reports/external_evaluation/seed_2024_uncertainty/internal_validation_diagnostic"
SEED, T, THRESHOLD, EPS = 20261120, 15, 0.5, 1e-7

if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite: {OUT}")
if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

registry = load_exclusion_registry(
    manifest_path=DEFAULT_MANIFEST,
    project_root=ROOT,
    canonical_dir=DEFAULT_CANONICAL_DIR,
    verify_files=True,
)
print("Registry validated:", registry.registry_version)
print("Registry cohorts:", registry.cohorts)

dataset = DFUCanonicalDataset(
    manifest_path=DEFAULT_MANIFEST,
    cohort="internal_validation",
    split_role="validation",
    project_root=ROOT,
    canonical_dir=DEFAULT_CANONICAL_DIR,
    boundary_field_dir=DEFAULT_BOUNDARY_FIELD_DIR,
    expected_count=186,
    verify_field_hashes=True,
    exclusion_registry=registry,
)
if len(dataset) != 186:
    raise RuntimeError(f"Expected 186 validation records, got {len(dataset)}")

loader = DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=True
)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(device))

model = MobileDFUNet(MobileDFUNetConfig())
payload = load_checkpoint(
    CHECKPOINT, model=model, map_location="cpu",
    restore_rng=False, strict=True
)
model.to(device).eval()

dropouts = []
for name in ("dropout1", "dropout2", "dropout3", "dropout4"):
    module = getattr(model.encoder, name, None)
    if not isinstance(module, torch.nn.Dropout2d):
        raise RuntimeError(f"Invalid encoder.{name}")
    dropouts.append(module)

def entropy(p):
    p = p.clamp(EPS, 1.0-EPS)
    return -(p*torch.log(p) + (1-p)*torch.log(1-p))

rows = []
totals = dict(tp=0, fp=0, fn=0, tn=0)
OUT.mkdir(parents=True)
start = time.perf_counter()

for module in dropouts:
    module.training = True

try:
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            paired = torch.cat((x, x.flip(dims=[3])), dim=0)
            sum_p = sum_h = None

            for _ in range(T):
                logits = model(paired, return_auxiliary=False)
                b = x.shape[0]
                logits_tta = (logits[:b] + logits[b:].flip(dims=[3])) / 2
                p = torch.sigmoid(logits_tta)
                h = entropy(p)
                if sum_p is None:
                    sum_p, sum_h = p.clone(), h.clone()
                else:
                    sum_p.add_(p)
                    sum_h.add_(h)

            mean_p = sum_p / T
            pe = entropy(mean_p)
            mi = pe - sum_h / T
            mi = torch.where((mi < 0) & (mi >= -1e-7), torch.zeros_like(mi), mi)

            pred = (mean_p >= THRESHOLD).cpu().numpy().astype(bool)
            truth = (y >= 0.5).cpu().numpy().astype(bool)

            for j in range(x.shape[0]):
                a, bmask = pred[j, 0], truth[j, 0]
                tp = int((a & bmask).sum())
                fp = int((a & ~bmask).sum())
                fn = int((~a & bmask).sum())
                tn = int((~a & ~bmask).sum())
                dice = 1.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn)
                row = {
                    "record_id": str(batch["record_id"][j]),
                    "dice": dice, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "gt_pixels": int(bmask.sum()),
                    "predicted_pixels": int(a.sum()),
                    "mean_probability": float(mean_p[j].mean().item()),
                    "max_probability": float(mean_p[j].max().item()),
                    "mean_predictive_entropy": float(pe[j].mean().item()),
                    "mean_mutual_information": float(mi[j].mean().item()),
                }
                rows.append(row)
                for key in totals:
                    totals[key] += row[key]

            if step % 25 == 0 or step == len(loader):
                print(f"Processed {step}/{len(loader)}; elapsed {time.perf_counter()-start:.1f}s")
finally:
    model.eval()

if len(rows) != 186:
    raise RuntimeError(f"Evaluated {len(rows)}, expected 186")

def safe_ratio(a, b):
    return float(a/b) if b else None

dice_values = np.asarray([r["dice"] for r in rows], dtype=float)
summary = {
    "experiment": "AZH_source_to_internal_validation",
    "experiment_type": "MC_Dropout_with_horizontal_flip_TTA",
    "checkpoint": str(CHECKPOINT),
    "checkpoint_epoch": int(payload.get("epoch", -1)),
    "checkpoint_best_epoch": 86,
    "inference_seed": SEED,
    "mc_passes": T,
    "threshold": THRESHOLD,
    "tta": "horizontal_flip_logits_averaged_within_each_mc_pass",
    "registry_version": registry.registry_version,
    "records_evaluated": len(rows),
    "macro_dice": float(np.mean(dice_values)),
    "macro_iou": float(np.mean([
        r["tp"]/(r["tp"]+r["fp"]+r["fn"])
        if r["tp"]+r["fp"]+r["fn"] else 1.0 for r in rows
    ])),
    "macro_sensitivity": float(np.nanmean([
        safe_ratio(r["tp"], r["tp"]+r["fn"]) for r in rows
        if r["tp"]+r["fn"] > 0
    ])),
    "macro_specificity": float(np.mean([
        safe_ratio(r["tn"], r["tn"]+r["fp"]) for r in rows
    ])),
    "all_background_predictions": sum(r["predicted_pixels"] == 0 for r in rows),
    "zero_dice_cases": int((dice_values == 0).sum()),
    "mean_predictive_entropy": float(np.mean([r["mean_predictive_entropy"] for r in rows])),
    "mean_mutual_information": float(np.mean([r["mean_mutual_information"] for r in rows])),
    "confusion_totals": totals,
    "elapsed_seconds": time.perf_counter()-start,
}
for filename, content in (
    ("per_image_metrics.csv", rows),
    ("summary.json", summary),
):
    path = OUT / filename
    if filename.endswith(".csv"):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text(json.dumps(content, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("Saved:", OUT)
