# python statistics_generator.py

import os
import cv2
import numpy as np
import torch
import lpips
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# Paths to folders CHANGE AS NEEDED
gt_folder = "/Users/ceabyfernandez/bachelorsthesis/statistics_colour/groundtruth"
lq_folder = "/Users/ceabyfernandez/bachelorsthesis/statistics_colour/oapt_restored/qf10"

# Initialize LPIPS model (AlexNet variant)
loss_fn = lpips.LPIPS(net='alex')
loss_fn.eval()

# Get sorted file lists
gt_images = sorted([f for f in os.listdir(gt_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
lq_images = sorted([f for f in os.listdir(lq_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Make sure they match
assert len(gt_images) == len(lq_images), "Mismatch in number of images"

records = []

for gt_name, lq_name in zip(gt_images, lq_images):
    # Read images
    gt_path = os.path.join(gt_folder, gt_name)
    lq_path = os.path.join(lq_folder, lq_name)

    img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

    # Convert to float32 in range [0,1]
    img_gt_f = img_gt.astype(np.float32) / 255.0
    img_lq_f = img_lq.astype(np.float32) / 255.0

    # PSNR
    psnr = compare_psnr(img_gt_f, img_lq_f, data_range=1.0)

    # SSIM
    ssim = compare_ssim(img_gt_f, img_lq_f, channel_axis=-1, data_range=1.0)

    # LPIPS (requires tensor: NCHW, [-1,1] range)
    gt_tensor = torch.from_numpy(img_gt_f).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    lq_tensor = torch.from_numpy(img_lq_f).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        lpips_val = loss_fn(gt_tensor, lq_tensor).item()

    # Save record
    records.append({
        "Image": gt_name,
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lpips_val
    })

# Convert to DataFrame
df = pd.DataFrame(records)

# Compute averages
avg_row = {
    "Image": "AVERAGE",
    "PSNR": np.mean(df["PSNR"]),
    "SSIM": np.mean(df["SSIM"]),
    "LPIPS": np.mean(df["LPIPS"])
}
df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

# Output CSV in lq_folder
#change filename to appropriate
output_path = os.path.join(lq_folder, "qf90_statistics.csv")
df.to_csv(output_path, index=False)

print(f"✅ Statistics saved to: {output_path}")
print(df.to_string)