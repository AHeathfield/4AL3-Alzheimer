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
OUTPUT_DIR = "../output"

SEED = 42   # hard-coded seed for reproducible result
TEST_SPLIT = 0.25   # 25% test, 75% train
ALPHA = 1e-4   # regularization strength for SGDClassifier
MAX_ITER = 1000   # max num iterations for SGDClassifier
TOL = 1e-3   # stopping criterion for SGDClassifier

data = np.load(FEATURES_FILE)
X, y = data["X"], data["y"]
class_names = list(data["class_names"])
print("Loaded", X.shape, "classes:", class_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, stratify=y, random_state=SEED
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train linear svm (hinge loss) using SGD
classifier = SGDClassifier(loss="hinge", alpha=ALPHA, max_iter=MAX_ITER, tol=TOL, random_state=SEED)
classifier.fit(X_train, y_train)

# evaluate on train and test set
train_pred = classifier.predict(X_train)
test_pred = classifier.predict(X_test)
print(f"Train accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(f"Test accuracy:  {accuracy_score(y_test, test_pred):.4f}   (baseline to beat: 0.7421)\n")
print(classification_report(y_test, test_pred, target_names=class_names))

# genereate confusion matrix and save to file
CONFUSION_MATRIX_OUTPUT = f"{OUTPUT_DIR}/confusion_matrix.png"
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_OUTPUT, dpi=200)
print("Saved confusion_matrix.png to", CONFUSION_MATRIX_OUTPUT)
