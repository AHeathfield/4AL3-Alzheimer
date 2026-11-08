"""
Loads features.npz (made by preprocess.py)
and trains/evaluates the linear SVM (hinge loss, SGD).
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES_FILE = "../output/features.npz"
SEED = 42

data = np.load(FEATURES_FILE)
X, y = data["X"], data["y"]
class_names = list(data["class_names"])
print("Loaded", X.shape, "classes:", class_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=SEED)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(f"Train accuracy: {accuracy_score(y_train, clf.predict(X_train)):.4f}")
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
