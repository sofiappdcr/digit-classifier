import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42


def load_data(use_mnist: bool):
    if use_mnist:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        y = np.asarray(y).astype(int)
        X = np.asarray(X) / 255.0
        name = "MNIST (28x28, 70,000 samples)"
    else:
        X, y = load_digits(return_X_y=True)
        X = np.asarray(X) / 16.0  # pixels range from 0 to 16
        name = "sklearn load_digits (8x8, 1,797 samples)"
    return X, y, name


def train(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    acc_tr = accuracy_score(y_tr, model.predict(X_tr))
    pred = model.predict(X_te)
    acc_te = accuracy_score(y_te, pred)
    print(f"{name:<24} train={acc_tr:.4f}  test={acc_te:.4f}  time={elapsed:.1f}s")
    return {"acc_train": acc_tr, "acc_test": acc_te, "time_s": elapsed}, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", action="store_true")
    args = ap.parse_args()

    X, y, dataset_name = load_data(args.mnist)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    print(f"Dataset: {dataset_name}")
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}\n")

    res = {"dataset": dataset_name, "n_train": len(X_tr), "n_test": len(X_te)}

    # 1) Baseline: logistic regression
    logreg = LogisticRegression(max_iter=1000, random_state=SEED)
    res["logistic_regression"], _ = train(
        "Logistic regression", logreg, X_tr_s, y_tr, X_te_s, y_te
    )

    # 2) Small neural network: 1 hidden layer with 64 neurons
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        max_iter=300,
        early_stopping=False,
        random_state=SEED,
    )
    res["neural_network"], pred_mlp = train(
        "Neural network (64)", mlp, X_tr_s, y_tr, X_te_s, y_te
    )
    res["neural_network"]["architecture"] = "input -> 64 (ReLU) -> 10 (softmax)"
    res["neural_network"]["epochs"] = int(mlp.n_iter_)

    # Repeat with 5 different train/test splits to check the gap is stable
    accs = {"logistic_regression": [], "neural_network": []}
    for s in (1, 2, 3, 4, 5):
        a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
        sc = StandardScaler().fit(a)
        a, b = sc.transform(a), sc.transform(b)
        accs["logistic_regression"].append(
            accuracy_score(d, LogisticRegression(max_iter=1000, random_state=s).fit(a, c).predict(b))
        )
        accs["neural_network"].append(
            accuracy_score(
                d,
                MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=False, random_state=s)
                .fit(a, c)
                .predict(b),
            )
        )
    print("\nAccuracy over 5 different splits (mean ± std):")
    for k, v in accs.items():
        print(f"  {k:<22} {np.mean(v):.4f} ± {np.std(v):.4f}")
        res[k]["mean_5_splits"] = float(np.mean(v))
        res[k]["std_5_splits"] = float(np.std(v))

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # Comparison chart
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Logistic\nregression", "Neural network\n(64)"]
    vals = [res["logistic_regression"]["acc_test"], res["neural_network"]["acc_test"]]
    bars = ax.bar(labels, vals, color=["#7a8aa0", "#2f6fdb"])
    ax.set_ylim(min(vals) - 0.05, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Sprint 2: model comparison")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=150)

    # Confusion matrix of the neural network
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_te, pred_mlp, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix - neural network")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
