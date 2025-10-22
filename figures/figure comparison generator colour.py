#figure comparison generator

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- paths to your images ---
gt_path = "/Users/ceabyfernandez/bachelorsthesis/statistics_colour/groundtruth/0026.png"
restored_paths = [
    "/Users/ceabyfernandez/bachelorsthesis/statistics_colour/fbcnn_restored/qf10/0026.png",
    "/Users/ceabyfernandez/bachelorsthesis/statistics_colour/qcn_restored/qf10/0026.png",
    "/Users/ceabyfernandez/bachelorsthesis/statistics_colour/oapt_restored/qf10/0026.png"
]

titles = ["Ground Truth", "FBCNN QF=10", "QCN QF=10", "OAPT QF=10"]

# --- load images (convert BGR→RGB for matplotlib) ---
images = [cv2.imread(gt_path)] + [cv2.imread(p) for p in restored_paths]
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

# --- define zoom-in region (x, y, w, h) ---
x, y, w, h = 1800, 1160, 100, 100   # adjust to the interesting region

# --- crop zoom patches ---
zooms = [img[y:y+h, x:x+w] for img in images]

# optional: magnify zoom patches
scale = 3
zooms = [np.array(Image.fromarray(z).resize((w*scale, h*scale), Image.NEAREST)) for z in zooms]

# --- plot setup ---
fig, axes = plt.subplots(2, len(images), figsize=(16, 7))

for i, (img, zoom, title) in enumerate(zip(images, zooms, titles)):
    # main image
    axes[0, i].imshow(img)
    axes[0, i].set_title(title, fontsize=12)
    axes[0, i].axis("off")

    # draw rectangle for zoom region
    rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
    axes[0, i].add_patch(rect)

    # zoomed-in patch
    axes[1, i].imshow(zoom)
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()