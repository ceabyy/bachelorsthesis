# figure comparison generator (grayscale version)

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- paths to your images ---
gt_path = "/Users/ceabyfernandez/bachelorsthesis/statistics_gray/groundtruth/0027.png"
restored_paths = [
    "/Users/ceabyfernandez/bachelorsthesis/statistics_gray/fbcnn_restored/qf50/0027.png",
    "/Users/ceabyfernandez/bachelorsthesis/statistics_gray/qcn_restored/qf50/0027.png",
    "/Users/ceabyfernandez/bachelorsthesis/statistics_gray/oapt_restored/qf50/0027.png"
]

titles = ["Ground Truth", "FBCNN QF=50", "QGCN QF=50", "OAPT QF=50"]

# --- load grayscale images ---
images = [cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)] + [
    cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in restored_paths
]

# --- define zoom-in region (x, y, w, h) ---
x, y, w, h = 600, 1300, 100, 100   # adjust to the interesting region

# --- crop zoom patches ---
zooms = [img[y:y+h, x:x+w] for img in images]

# optional: magnify zoom patches
scale = 3
zooms = [
    np.array(Image.fromarray(z).resize((w*scale, h*scale), Image.NEAREST)) for z in zooms
]

# --- plot setup ---
fig, axes = plt.subplots(2, len(images), figsize=(16, 7))

for i, (img, zoom, title) in enumerate(zip(images, zooms, titles)):
    # main image
    axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, i].set_title(title, fontsize=12)
    axes[0, i].axis("off")

    # draw rectangle for zoom region
    rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
    axes[0, i].add_patch(rect)

    # zoomed-in patch
    axes[1, i].imshow(zoom, cmap='gray', vmin=0, vmax=255)
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()