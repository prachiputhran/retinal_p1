import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

files = sorted(glob.glob("outputs/gradcam_images/dr_grade*_*.png"))[:5]

fig, axes = plt.subplots(1, len(files), figsize=(20, 4))

for ax, f in zip(axes, files):
    ax.imshow(mpimg.imread(f))
    ax.set_title(os.path.basename(f).split("_")[1])
    ax.axis("off")

plt.tight_layout()
plt.savefig("assets/gradcam_grid.png", dpi=150)

print("Saved assets/gradcam_grid.png")