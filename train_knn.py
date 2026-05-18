# Final version used for submission
# Dataset size: 152 samples
# k = 7
# Accuracy: 86.67%




import csv
from pathlib import Path

import numpy as np

DATA_CSV = Path("data/gestures_dataset.csv")
MODEL_OUT = Path("models/knn_model.npz")


def load_dataset(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # label + 63 features
        rows = list(reader)

    labels = [row[0] for row in rows]
    X = np.array([[float(v) for v in row[1:]] for row in rows], dtype=np.float32)
    return labels, X


def split_train_test(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    n_test = int(len(X) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for x in X_test:
        d2 = np.sum((X_train - x) ** 2, axis=1)
        nn = np.argsort(d2)[:k]
        votes = y_train[nn]
        values, counts = np.unique(votes, return_counts=True)
        preds.append(values[np.argmax(counts)])
    return np.array(preds, dtype=np.int64)


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    if not DATA_CSV.exists():
        print("Missing gestures_dataset.csv. Run collect_dataset.py first.")
        return

    labels, X = load_dataset(DATA_CSV)

    unique_labels = sorted(set(labels))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}

    y = np.array([label_to_id[l] for l in labels], dtype=np.int64)

    X_train, y_train, X_test, y_test = split_train_test(X, y, test_ratio=0.2, seed=42)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough samples for train/test split.")
        return

    k = 7
    if len(X_train) < k:
        k = max(1, len(X_train))

    y_pred = knn_predict(X_train, y_train, X_test, k=k)

    acc = float(np.mean(y_pred == y_test))
    print("Total samples:", len(X))
    print("Train:", len(X_train), "Test:", len(X_test), "k:", k)
    print("Accuracy: {:.2f}%".format(acc * 100.0))

    cm = confusion_matrix(y_test, y_pred, n_classes=len(unique_labels))
    print("\nLabels (id order):", unique_labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save model (store ALL data so runtime can classify using kNN)
    np.savez(
        MODEL_OUT,
        X_train=X,
        y_train=y,
        labels=np.array(unique_labels, dtype=object),
    )
    print("\nSaved model to:", MODEL_OUT.resolve())


if __name__ == "__main__":
    main()

