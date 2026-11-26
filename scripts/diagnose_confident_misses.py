import csv, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from src.data.stage_4_dataloaders import build_stage_4_dataloaders
from src.dfu_cross_domain.training.checkpoint import load_checkpoint
from src.models.mobile_dfu_net import MobileDFUNet, MobileDFUNetConfig

ROOT = Path.cwd()
CHECKPOINT = ROOT / "Seed2024_Backup/seed_2024_resume/checkpoints/best.pt"
OUT = ROOT / "reports/external_evaluation/seed_2024_uncertainty/diagnostics/targeted_prediction_seed20261119"
SEED, T, THRESHOLD = 20261119, 15, 0.5

wanted = {
    "native_target_58375566", "native_target_50690890",
    "native_target_75560211", "native_target_60407937",
    "native_target_75529165", "native_target_54806340",
    "native_target_75539422", "native_target_75528249",
    "native_target_60965856", "native_target_61835201",
}

if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite: {OUT}")
if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

OUT.mkdir(parents=True)
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

dataset = build_stage_4_dataloaders(
    batch_size=1, num_workers=0, pin_memory=False
)["target_native_dataset"]

indices = {
    r["record_id"]: i for i, r in enumerate(dataset.records)
    if r["record_id"] in wanted
}
if set(indices) != wanted:
    raise RuntimeError(f"Missing records: {wanted - set(indices)}")

for module in dropouts:
    module.training = True

rows = []
sheet = Image.new("RGB", (1400, 350 * len(wanted)), "white")
draw = ImageDraw.Draw(sheet)

try:
    with torch.inference_mode():
        for row_index, rid in enumerate(sorted(wanted)):
            item = dataset[indices[rid]]
            x = item["image"].unsqueeze(0).to(device)
            y = item["mask"].squeeze().cpu().numpy() >= 0.5

            paired = torch.cat((x, x.flip(dims=[3])), dim=0)
            sum_p = None

            for _ in range(T):
                logits = model(paired, return_auxiliary=False)
                b = x.shape[0]
                logits_tta = (
                    logits[:b] + logits[b:].flip(dims=[3])
                ) / 2
                p = torch.sigmoid(logits_tta)
                sum_p = p.clone() if sum_p is None else sum_p.add_(p)

            mean_p = sum_p / T
            pred = (mean_p[0, 0] >= THRESHOLD).cpu().numpy()
            prob = mean_p[0, 0].cpu().numpy()

            tp = int((pred & y).sum())
            fp = int((pred & ~y).sum())
            fn = int((~pred & y).sum())
            tn = int((~pred & ~y).sum())
            dice = 1.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn)

            image = Image.open(dataset.records[indices[rid]]["image_path"]).convert("RGB")
            rgb = np.asarray(image).copy()

            gt_overlay = rgb.copy()
            gt_overlay[y] = (0.55*gt_overlay[y] + 0.45*np.array([255,0,0])).astype(np.uint8)

            pred_overlay = rgb.copy()
            pred_overlay[pred] = (0.55*pred_overlay[pred] + 0.45*np.array([0,255,0])).astype(np.uint8)

            combined = rgb.copy()
            combined[y & ~pred] = [255,0,0]       # missed GT
            combined[pred & ~y] = [0,255,0]       # false positive
            combined[pred & y] = [255,255,0]      # overlap

            Image.fromarray(pred.astype(np.uint8)*255).save(OUT / f"{rid}_prediction.png")
            Image.fromarray(prob).save(OUT / f"{rid}_mean_probability.tiff")
            Image.fromarray(combined).save(OUT / f"{rid}_comparison.png")

            panels = [
                Image.fromarray(rgb), Image.fromarray(gt_overlay),
                Image.fromarray(pred_overlay), Image.fromarray(combined)
            ]
            labels = ["Original", "GT red overlay", "Prediction green overlay",
                      "Red=miss | Green=FP | Yellow=overlap"]

            yy = row_index * 350
            draw.text((8, yy+5), rid, fill="black")
            draw.text(
                (8, yy+23),
                f"Dice={dice:.5f} TP={tp} FP={fp} FN={fn} | Pred pixels={int(pred.sum())}",
                fill="black"
            )
            for j, (panel, label) in enumerate(zip(panels, labels)):
                panel.thumbnail((330, 290))
                xx = 8 + j*345
                draw.text((xx, yy+48), label, fill="black")
                sheet.paste(panel, (xx, yy+68))

            rows.append({
                "record_id": rid, "dice": dice, "tp": tp, "fp": fp,
                "fn": fn, "tn": tn, "gt_pixels": int(y.sum()),
                "predicted_pixels": int(pred.sum()),
                "mean_probability": float(prob.mean()),
                "mc_passes": T, "threshold": THRESHOLD,
                "inference_seed": SEED, "tta": "horizontal_flip"
            })
            print(f"{rid}: Dice={dice:.5f}, TP={tp}, FP={fp}, FN={fn}")

finally:
    model.eval()

sheet.save(OUT / "targeted_prediction_contact_sheet.png")
with (OUT / "targeted_prediction_metrics.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "experiment": "targeted_10_confident_complete_misses",
    "checkpoint": str(CHECKPOINT),
    "checkpoint_best_epoch": 86,
    "inference_seed": SEED,
    "mc_passes": T,
    "threshold": THRESHOLD,
    "tta": "horizontal_flip_logits_averaged_within_each_mc_pass",
    "records_evaluated": len(rows),
    "results": rows,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Contact sheet:", OUT / "targeted_prediction_contact_sheet.png")
print("Metrics:", OUT / "targeted_prediction_metrics.csv")
print("Completed:", len(rows), "records")
