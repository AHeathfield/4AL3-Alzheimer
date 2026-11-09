"""
Hyperparameter tuning for the Alzheimer's disease classification model.
"""
import csv
import itertools
import os
import datetime
import ast
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage import io, color, transform
from skimage.feature import hog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "normal_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "tuning_results.csv")
RECOVERY_FILE = os.path.join(OUTPUT_DIR, "tuning_results_recovery.csv")
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Add values to any list to test them. 
# The total number of runs is the product of the list lengths
PARAM_GRID = {
    "test_split": [0.2],
    # hyperparameters
    "seed": [42, 100, 999],
    "alpha": [0.001, 0.01, 0.1],
    "max_iter": [1000, 3000],
    "tol": [1e-2, 1e-3, 1e-4],
    # preprocessing parameters
    "img_size": [(128, 128)],
    "orientations": [9],
    "pixels_per_cell": [(8, 8)],
    "cells_per_block": [(2, 2)],
}

def main():
    # load class names from DATA_DIR
    class_names = sorted(
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    if not class_names:
        raise SystemExit(f"No class folders found inside {DATA_DIR!r}. Check DATA_DIR.")
    print("Classes found:", class_names)

    # load parameter combinations and filter out those already completed
    parameter_names = list(PARAM_GRID)
    combinations = list(itertools.product(*(PARAM_GRID[name] for name in parameter_names)))

    # check for valid test_split values (can save a lot of time by catching these errors lol)
    for split in PARAM_GRID["test_split"]:
        if not 0 < split < 1:
            raise ValueError(f"test_split must be between 0 and 1, got {split}")

    existing_results = load_results()
    known_keys = {row["combination_key"] for row in existing_results}
    pending_combinations = [
        (model_id, dict(zip(parameter_names, values)))
        for model_id, values in enumerate(combinations, start=1)
        if combination_key(dict(zip(parameter_names, values))) not in known_keys
    ]
    print(
        f"Found {len(existing_results)} saved results; "
        f"{len(pending_combinations)} combinations remain"
    )

    start_time = datetime.datetime.now()

    # Preprocessing depends only on the image and HOG settings, so do not
    # repeat the expensive image pass for models sharing those settings.
    feature_cache = {}
    results = existing_results
    for run_number, (model_id, parameters) in enumerate(pending_combinations, start=1):
        key = combination_key(parameters)
        preprocessing_key = (
            parameters["img_size"],
            parameters["orientations"],
            parameters["pixels_per_cell"],
            parameters["cells_per_block"],
        )
        if preprocessing_key not in feature_cache:
            feature_cache[preprocessing_key] = preprocess(
                class_names,
                img_size=parameters["img_size"],
                hog_params={
                    "orientations": parameters["orientations"],
                    "pixels_per_cell": parameters["pixels_per_cell"],
                    "cells_per_block": parameters["cells_per_block"],
                },
            )
        X, y = feature_cache[preprocessing_key]
        X, y = feature_cache[preprocessing_key]
        train_accuracy, test_accuracy = train(
            X,
            y,
            class_names,
            seed=parameters["seed"],
            test_split=parameters["test_split"],
            alpha=parameters["alpha"],
            max_iter=parameters["max_iter"],
            tol=parameters["tol"],
        )
        result = {
            "model_id": model_id,
            "combination_key": key,
            **parameters,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
        results.append(result)
        save_results(results)
        print(
            f"[{run_number}/{len(pending_combinations)}] "
            f"test_accuracy={test_accuracy:.4f} parameters={parameters}"
            f"\ntime_elapsed={datetime.datetime.now() - start_time}"
        )

    time_elapsed = datetime.datetime.now() - start_time
    print(f"\nTuning completed in {time_elapsed}.")

    # get accuracy statistics for each unique combination of parameters (ignoring seed)
    save_results(results)

    print(f"\nSaved sorted results to {RESULTS_FILE}")
    print("\nBest combinations:")
    for result in results[:10]:
        print(
            f"model_id={result['model_id']}, "
            f"mean={float(result['test_accuracy_mean']):.4f}, "
            f"std={float(result['test_accuracy_std']):.4f}, "
            f"runs={result['runs_in_group']}"
        )


def combination_key(parameters):
    """Create a stable key that can be used to resume a tuning run."""
    return repr(sorted(parameters.items()))


def save_results(results):
    """Save results and checkpoint after every completed model."""
    if not results:
        return

    parameter_names = list(PARAM_GRID)
    accuracy_groups = defaultdict(list)
    for result in results:
        group_key = combination_key(
            {name: result[name] for name in parameter_names if name != "seed"}
        )
        accuracy_groups[group_key].append(float(result["test_accuracy"]))

    for result in results:
        group_key = combination_key(
            {name: result[name] for name in parameter_names if name != "seed"}
        )
        accuracies = accuracy_groups[group_key]
        result["test_accuracy_mean"] = np.mean(accuracies)
        result["test_accuracy_std"] = np.std(accuracies)
        result["runs_in_group"] = len(accuracies)

    results.sort(key=lambda result: result["test_accuracy_mean"], reverse=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = [
        "model_id",
        "combination_key",
        *parameter_names,
        "train_accuracy",
        "test_accuracy",
        "test_accuracy_mean",
        "test_accuracy_std",
        "runs_in_group",
    ]
    try:
        write_results_file(RESULTS_FILE, results, fieldnames)
        output_file = RESULTS_FILE
    except PermissionError:
        write_results_file(RECOVERY_FILE, results, fieldnames)
        output_file = RECOVERY_FILE
        print(
            f"Could not write {RESULTS_FILE}. "
            f"Saved results to {RECOVERY_FILE} instead. "
            "Close the CSV in Excel and copy the recovery file over it later."
        )
    print(f"\nSaved checkpoint to {output_file}")


def write_results_file(path, results, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def load_results():
    """Load prior results so combinations are not ran twice."""
    if not os.path.exists(RESULTS_FILE):
        return []

    with open(RESULTS_FILE, newline="", encoding="utf-8") as results_file:
        rows = list(csv.DictReader(results_file))

    for row in rows:
        for name in PARAM_GRID:
            row[name] = parse_csv_parameter(name, row[name])
        if "combination_key" not in row:
            row["combination_key"] = combination_key(
                {
                    name: row[name]
                    for name in PARAM_GRID
                }
            )
        for name in ("model_id", "runs_in_group"):
            if name in row and row[name]:
                row[name] = int(float(row[name]))
        for name in ("train_accuracy", "test_accuracy"):
            if name in row and row[name]:
                row[name] = float(row[name])
    return rows


def parse_csv_parameter(name, value):
    if name in {"seed", "max_iter", "orientations"}:
        return int(value)
    if name in {"test_split", "alpha", "tol"}:
        return float(value)
    return ast.literal_eval(value)


def preprocess(
    class_names,
    img_size=(128, 128),
    hog_params=None,
):
    """
    Preprocess images: read, convert to grayscale, resize, extract HOG features, and return features and labels.
    """
    if hog_params is None:
        hog_params = {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
        }

    features, labels, paths = [], [], []
    for label, cls in enumerate(class_names):   # loop over each class folder, label=0,1,..., cls=folder name (class name)
        folder = os.path.join(DATA_DIR, cls)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))   # all files in the folder that end with acceptable extensions
        for i, fname in enumerate(files):   # loop over each file in the class folder, i=index, fname=filename
            img = io.imread(os.path.join(folder, fname))
            if img.ndim == 3:   # if image has 3 dimensions (color image), convert to grayscale
                img = color.rgb2gray(img[..., :3])
            img = transform.resize(img, img_size, anti_aliasing=True)
            features.append(hog(img, **hog_params))
            labels.append(label)
            paths.append(os.path.join(cls, fname))

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    return X, y

def train(
    X,
    y,
    class_names,
    seed,
    test_split,
    alpha,
    max_iter,
    tol
):
    """
    Train and evaluate the linear SVM (hinge loss, SGD) with given hyperparameters.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train linear svm (hinge loss) using SGD
    classifier = SGDClassifier(loss="hinge", alpha=alpha, max_iter=max_iter, tol=tol, random_state=seed)
    classifier.fit(X_train, y_train)

    # evaluate train and test set
    train_pred = classifier.predict(X_train)
    test_pred = classifier.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred) 
    # cr = classification_report(y_test, test_pred, target_names=class_names)
    # cm = confusion_matrix(y_test, test_pred)

    return train_accuracy, test_accuracy #, cr, cm

if __name__ == "__main__":
    main()