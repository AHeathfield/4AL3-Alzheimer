"""
Run ONCE (or whenever you change image size / HOG settings).
Reads all images, extracts HOG features, and saves them to features.npz.

Expected data layout: DATA_DIR contains one subfolder per class, each full of .jpg images.
"""
import os
import numpy as np
from skimage import io, color, transform
from skimage.feature import hog

DATA_DIR = "../normal_dataset"            # <-- parent folder that contains the 4 class folders
OUT_FILE = "../output/features.npz"
IMG_SIZE = (128, 128)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
IMG_EXTS = (".jpg", ".jpeg", ".png")


def main():
    class_names = sorted(
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    if not class_names:
        raise SystemExit(f"No class folders found inside {DATA_DIR!r}. Check DATA_DIR.")
    print("Classes found:", class_names)

    features, labels, paths = [], [], []
    for label, cls in enumerate(class_names):
        folder = os.path.join(DATA_DIR, cls)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))
        print(f"  {cls}: {len(files)} images")
        for i, fname in enumerate(files):
            img = io.imread(os.path.join(folder, fname))
            if img.ndim == 3:
                img = color.rgb2gray(img[..., :3])
            img = transform.resize(img, IMG_SIZE, anti_aliasing=True)
            features.append(hog(img, **HOG_PARAMS))
            labels.append(label)
            paths.append(os.path.join(cls, fname))
            if (i + 1) % 500 == 0:
                print(f"    processed {i + 1}/{len(files)}")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    np.savez_compressed(
        OUT_FILE, X=X, y=y, class_names=np.array(class_names), paths=np.array(paths)
    )
    print(f"Saved {OUT_FILE}: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
