import torch
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.stage_4_dataloaders import build_stage_4_dataloaders
from src.models.mobile_dfu_net.model import MobileDFUNet
from src.models.mobile_dfu_net.losses import MobileDFUCompoundLoss

loaders = build_stage_4_dataloaders()
train_ds = loaders["train_dataset"]
val_ds = loaders["val_dataset"]

model = MobileDFUNet()
loss_fn = MobileDFUCompoundLoss()

print("Loaded Train Dataset:", len(train_ds))
print("Loaded Val Dataset:", len(val_ds))

sample = train_ds[0]
print("Sample 0 info:", sample["pair_id"], sample["cohort"], sample["proc_mask_filename"])

img_path = Path("data/processed/stage_3_2B_canonical/images") / f"{sample['cohort']}_{sample['pair_id']}_bilinear.png"
mask_path = Path("data/processed/stage_3_2B_canonical/masks") / sample["proc_mask_filename"]

print("Image exists:", img_path.exists())
print("Mask exists:", mask_path.exists())

img_arr = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)

mask_arr = (np.array(Image.open(mask_path).convert("L")) == 255).astype(np.float32)
mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)

phi_fp = torch.zeros(1, 1, 512, 512)
phi_fn = torch.zeros(1, 1, 512, 512)

out = model(img_tensor)
loss_dict = loss_fn(out, mask_tensor, phi_fp, phi_fn)

print("Forward output shape:", out["seg_logits"].shape)
print("Total loss on real batch:", loss_dict["total_loss"].item())
loss_dict["total_loss"].backward()
print("Backward pass on real data sample successful!")
