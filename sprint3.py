import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # saves figures to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42 # so that the results will be the same.
RESULTS_DIR = Path("results")


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


def split_data(X, y):
    """60% train / 20% validation / 20% test.

    Train: teach the model.
    Validation: compare and choose the best model (network).
    Test: honest final result.
    """
    X_temp, X_te, y_temp, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    # 25% of the remaining 80% = 20% of the total
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
    )
    # Fit the scaler on train only, then apply it to the other two sets
    scaler = StandardScaler().fit(X_tr)
    return (
        scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te),
        y_tr, y_val, y_te,
    )


def train_model(data, hidden_layers=(64,), lr=0.001, alpha=0.0001, max_iter=300):
    """Train one MLP with the given configuration. Returns (model, seconds)."""
    X_tr, _, _, y_tr, _, _ = data
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=False,
        random_state=SEED,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr)
    return model, time.time() - t0


def val_accuracy(model, data):
    _, X_val, _, _, y_val, _ = data
    return accuracy_score(y_val, model.predict(X_val))


def train_with_history(data, hidden_layers=(64,), lr=0.001, alpha=0.0001, epochs=100):
    """Train epoch by epoch with partial_fit, recording metrics after each epoch.

    Returns (model, history) where history has one value per epoch for:
    train_loss, train_acc, val_acc.
    """
    X_tr, X_val, _, y_tr, y_val, _ = data
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=lr,
        alpha=alpha,
        random_state=SEED,
    )
    # train_loss: the "error" the network is trying to reduce.
    # train_acc: % correct on the training set.
    # val_acc: % correct on the validation set.
    classes = np.unique(y_tr)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        model.partial_fit(X_tr, y_tr, classes=classes)  # one pass over the train set
        history["train_loss"].append(model.loss_)
        history["train_acc"].append(accuracy_score(y_tr, model.predict(X_tr)))
        history["val_acc"].append(accuracy_score(y_val, model.predict(X_val)))
    # at the end, the function returns the model and the history, with 100 values in each list, one per epoch.
    return model, history

# draws two charts side by side-left: the loss per epoch. right: training and validation accuracy together, for comparison.
def plot_training_curves(history, path, title="Training curves"):
    """Left: train loss per epoch. Right: train vs validation accuracy per epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"])
    ax1.set_title("Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    best = int(np.argmax(history["val_acc"]))
    ax2.axvline(best + 1, color="gray", linestyle="--", linewidth=1)
    ax2.set_title(f"Accuracy (best validation at epoch {best + 1})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    X, y, dataset_name = load_data(args.mnist)
    data = split_data(X, y)
    X_tr, X_val, X_te, *_ = data
    print(f"Dataset: {dataset_name}")
    print(f"Train: {len(X_tr)} | Validation: {len(X_val)} | Test: {len(X_te)}\n")

    # the baseline (same config as Sprint 2) on the validation set
    model, seconds = train_model(data)
    print(f"Baseline (64) -> val accuracy={val_accuracy(model, data):.4f}  time={seconds:.1f}s")

    # training curves for the baseline
    _, history = train_with_history(data, hidden_layers=(64,), epochs=100)
    plot_training_curves(history, RESULTS_DIR / "training_curve.png", "Baseline (64)")
    best = int(np.argmax(history["val_acc"]))
    print(
        f"\nTask 1 -> final train acc={history['train_acc'][-1]:.4f}  "
        f"final val acc={history['val_acc'][-1]:.4f}  "
        f"best val acc={history['val_acc'][best]:.4f} (epoch {best + 1})"
    )
    print("Saved results/training_curve.png")


if __name__ == "__main__":
    main()