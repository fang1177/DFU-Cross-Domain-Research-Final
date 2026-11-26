from __future__ import annotations
import csv, json, math, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from src.data.stage_4_dataloaders import build_stage_4_dataloaders
from src.dfu_cross_domain.training.checkpoint import load_checkpoint
from src.models.mobile_dfu_net import MobileDFUNet, MobileDFUNetConfig

ROOT = Path(r"E:\DFU_Cross_Domain_Research")
CHECKPOINT = ROOT / "Seed2024_Backup/seed_2024_resume/checkpoints/best.pt"
OUT = ROOT / "reports/external_evaluation/seed_2024_uncertainty"
T, THRESHOLD, EPS = 15, 0.5, 1e-7

def entropy(p):
    p = p.clamp(EPS, 1.0-EPS)
    return -(p*torch.log(p) + (1-p)*torch.log(1-p))

def ratio(a, b):
    return float(a/b) if b else float("nan")

def metrics(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    dd, jd = 2*tp+fp+fn, tp+fp+fn
    return {"dice": 1.0 if dd == 0 else 2*tp/dd,
            "iou": 1.0 if jd == 0 else tp/jd,
            "sensitivity": ratio(tp,tp+fn), "specificity": ratio(tn,tn+fp),
            "tp":tp,"fp":fp,"fn":fn,"tn":tn}

def main():
    if not CHECKPOINT.is_file(): raise FileNotFoundError(CHECKPOINT)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {OUT}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20241119)
    torch.cuda.manual_seed_all(20241119)
    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(device))

    model = MobileDFUNet(MobileDFUNetConfig())
    payload = load_checkpoint(CHECKPOINT, model=model, map_location="cpu",
                              restore_rng=False, strict=True)
    model.to(device).eval()
    dropout_names = ("dropout1","dropout2","dropout3","dropout4")
    dropout_modules = []
    for name in dropout_names:
        module = getattr(model.encoder, name, None)
        if not isinstance(module, torch.nn.Dropout2d):
            raise RuntimeError(f"Missing/invalid encoder.{name} Dropout2d")
        dropout_modules.append(module)

    loaders = build_stage_4_dataloaders(batch_size=1, num_workers=0, pin_memory=True)
    loader = loaders["target_native_loader"]
    if len(loaders["target_native_dataset"]) != 2187:
        raise RuntimeError("Native target count is not 2,187")

    rows=[]; totals={"tp":0,"fp":0,"fn":0,"tn":0}; start=time.perf_counter()
    for module in dropout_modules: module.training = True
    try:
        with torch.inference_mode():
            for step, batch in enumerate(loader, 1):
                x=batch["image"].to(device, non_blocking=True)
                y=batch["mask"].to(device, non_blocking=True)
                # Batch contains original and flipped views. Each view receives
                # its own dropout mask; flipped logits are realigned per pass.
                paired=torch.cat((x,x.flip(dims=[3])),dim=0)
                sum_p=sum_h=None
                for _ in range(T):
                    logits=model(paired,return_auxiliary=False)
                    b=x.shape[0]
                    logits_tta=(logits[:b]+logits[b:].flip(dims=[3]))/2
                    p=torch.sigmoid(logits_tta); h=entropy(p)
                    if sum_p is None: sum_p=p; sum_h=h
                    else: sum_p.add_(p); sum_h.add_(h)
                mean_p=sum_p/T
                pe=entropy(mean_p)
                mi=pe-(sum_h/T)
                mi=torch.where((mi<0)&(mi>=-1e-7),torch.zeros_like(mi),mi)
                pred=(mean_p>=THRESHOLD).to(torch.uint8).cpu().numpy()
                truth=(y>=0.5).to(torch.uint8).cpu().numpy()
                pe_img=pe.mean(dim=(1,2,3)).cpu().numpy()
                mi_img=mi.mean(dim=(1,2,3)).cpu().numpy()
                for j in range(x.shape[0]):
                    m=metrics(pred[j,0],truth[j,0])
                    for k in totals: totals[k]+=m[k]
                    row={"record_id":str(batch["record_id"][j]),
                         "pair_id":str(batch["pair_id"][j]),
                         "orig_img_sha256":str(batch["orig_img_sha256"][j]),
                         **m,"mean_predictive_entropy":float(pe_img[j]),
                         "mean_mutual_information":float(mi_img[j]),
                         "mc_passes":T,"inference_seed":20241119,"tta":"horizontal_flip","threshold":THRESHOLD}
                    rows.append(row)
                if step%100==0 or step==len(loader):
                    print(f"Processed {len(rows)}/2187; elapsed {time.perf_counter()-start:.1f}s")
    finally:
        # Return all modules to evaluation state after inference.
        model.eval()

    if len(rows)!=2187: raise RuntimeError(f"Evaluated {len(rows)}, expected 2187")
    csv_path=OUT/"per_image_uncertainty_metrics.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    macro={}
    for key in ("dice","iou","sensitivity","specificity",
                "mean_predictive_entropy","mean_mutual_information"):
        vals=np.asarray([r[key] for r in rows],dtype=float)
        macro[key]=float(np.nanmean(vals)) if np.isfinite(vals).any() else None
    dice=np.asarray([r["dice"] for r in rows],float)
    def corr(key):
        vals=np.asarray([r[key] for r in rows],float)
        z=spearmanr(vals,dice,nan_policy="omit")
        return {"rho":float(z.statistic) if math.isfinite(float(z.statistic)) else None,
                "p_value":float(z.pvalue) if math.isfinite(float(z.pvalue)) else None}
    tp,fp,fn,tn=(totals[k] for k in ("tp","fp","fn","tn"))
    summary={
      "experiment":"AZH_source_to_MENDELEY_native_target",
      "experiment_type":"MC_Dropout_with_horizontal_flip_TTA",
      "seed":2024,"checkpoint":str(CHECKPOINT),
      "checkpoint_best_epoch":86,"checkpoint_epoch":int(payload.get("epoch",-1)),
      "target_images_evaluated":len(rows),"mc_passes":T,"inference_seed":20241119,
      "tta":"horizontal_flip_logits_averaged_within_each_mc_pass",
      "threshold":THRESHOLD,"device":torch.cuda.get_device_name(device),
      "batch_size":1,"macro_metrics":macro,
      "micro_dice":1.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn),
      "micro_iou":1.0 if tp+fp+fn==0 else tp/(tp+fp+fn),
      "micro_sensitivity":ratio(tp,tp+fn),"micro_specificity":ratio(tn,tn+fp),
      "total_confusion_counts":totals,
      "spearman_uncertainty_vs_dice":{"predictive_entropy":corr("mean_predictive_entropy"),
                                      "mutual_information":corr("mean_mutual_information")},
      "elapsed_seconds":round(time.perf_counter()-start,2),
      "notes":["Native target cohort: 2,187 primary images; no target augmentation.",
       "Each MC pass evaluates original and horizontal-flip views in one paired batch.",
       "Flipped logits are realigned and averaged before sigmoid.",
       "Uncertainty is calculated across 15 pass-level TTA probabilities.",
       "Tiny negative MI in [-1e-7,0) is clamped; larger negatives are retained.",
       "Empty-empty Dice/IoU=1; undefined sensitivity/specificity are NaN.",
       "No training, optimizer steps, or checkpoint modifications."]}
    with (OUT/"summary_uncertainty_metrics.json").open("w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2,allow_nan=False)
    print("\nCOMPLETE")
    print("Macro Dice:",macro["dice"])
    print("Mean predictive entropy:",macro["mean_predictive_entropy"])
    print("Mean mutual information:",macro["mean_mutual_information"])
    print("Output:",OUT)

if __name__=="__main__": main()

