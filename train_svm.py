"""
Baseline: HOG features + linear SVM trained with SGD, for 4-class Alzheimer's MRI classification.

Expected folder layout (one subfolder per class, containing that class's images):

    data/
      Non-Demented/        *.jpg / *.png
      Very Mild Demented/  ...
      Mild Demented/       ...
      Moderate Demented/   ...

Change DATA_DIR below to wherever you unzipped the ORIGINAL (not synthesized) scans.
Libraries: numpy, scikit-image, scikit-learn, matplotlib, seaborn, joblib
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, color, transform
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = "data"          # <-- change this
IMG_SIZE = (128, 128)      # every image is resized to this
SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_dataset(data_dir):
    """Read every image, convert to grayscale, resize, extract HOG features."""
    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    print("Classes found:", class_names)

    features, labels = [], []
    for label, cls in enumerate(class_names):
        folder = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
        print(f"  {cls}: {len(files)} images")
        for fname in files:
            img = io.imread(os.path.join(folder, fname))
            if img.ndim == 3:                      # RGB/RGBA -> grayscale
                img = color.rgb2gray(img[..., :3])
            img = transform.resize(img, IMG_SIZE, anti_aliasing=True)
            feat = hog(
                img,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
            )
            features.append(feat)
            labels.append(label)
    return np.array(features), np.array(labels), class_names


def main():
    X, y, class_names = load_dataset(DATA_DIR)
    print("Feature matrix:", X.shape)

    # 75/25 split, keeping class proportions equal in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    # Scale features (fit on train only, so no information leaks from test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Linear SVM trained with stochastic gradient descent (hinge loss = SVM)
    clf = SGDClassifier(
        loss="hinge",
        alpha=1e-4,        # regularization strength: try 1e-5 ... 1e-2
        max_iter=1000,
        tol=1e-3,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    pred = clf.predict(X_test)
    print(f"\nTrain accuracy: {accuracy_score(y_train, clf.predict(X_train)):.4f}")
    print(f"Test accuracy:  {accuracy_score(y_test, pred):.4f}   (baseline to beat: 0.7421)\n")
    print(classification_report(y_test, pred, target_names=class_names))

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()
