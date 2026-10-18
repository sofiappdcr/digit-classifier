import argparse
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # saves figures to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
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
    data = (
        scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te),
        y_tr, y_val, y_te,
    )
    return data, scaler  # the scaler is returned so we can undo it to draw images


def train_model(data, hidden_layers=(64,), lr=0.001, alpha=0.0001, max_iter=300, seed=SEED):
    """Train one MLP with the given configuration. Returns (model, seconds)."""
    X_tr, _, _, y_tr, _, _ = data
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=False,
        random_state=seed,
    )
    t0 = time.time()
    with warnings.catch_warnings():
        # a run that hits max_iter without converging is a result, not an error
        warnings.simplefilter("ignore", ConvergenceWarning)
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
    classes = np.unique(y_tr)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        model.partial_fit(X_tr, y_tr, classes=classes)  # one pass over the train set
        history["train_loss"].append(model.loss_)
        history["train_acc"].append(accuracy_score(y_tr, model.predict(X_tr)))
        history["val_acc"].append(accuracy_score(y_val, model.predict(X_val)))
    return model, history


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


def plot_confusion_matrix(y_true, y_pred, path, title="Confusion matrix"):
    """Rows = real digit, columns = predicted digit. Off-diagonal cells are errors."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(y_true, y_pred, k=3):
    """Return the k most frequent mistakes as (real, predicted, count)."""
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # ignore correct predictions
    order = np.argsort(cm, axis=None)[::-1][:k]  # largest cells first
    result = []
    for idx in order:
        real, pred = divmod(int(idx), cm.shape[1])
        if cm[real, pred] > 0:
            result.append((real, pred, int(cm[real, pred])))
    return result


def plot_errors(X, y_true, y_pred, path, n=12):
    """Save a grid with images the model got wrong (real digit / predicted digit)."""
    wrong = np.flatnonzero(y_true != y_pred)[:n]
    if len(wrong) == 0:
        return
    side = int(np.sqrt(X.shape[1]))  # 8 for load_digits, 28 for MNIST
    fig, axes = plt.subplots(2, 6, figsize=(12, 6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(X[i].reshape(side, side), cmap="gray_r")
        ax.set_title(f"real {y_true[i]} / pred {y_pred[i]}", fontsize=10)
    fig.suptitle("Validation images the model got wrong")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Task 3: each experiment changes ONE thing compared to the baseline.
BASELINE_CONFIG = {"hidden_layers": (64,), "lr": 0.001, "alpha": 0.0001}
EXPERIMENTS = [
    ("Baseline (64)", {}),
    ("Fewer neurons (32)", {"hidden_layers": (32,)}),
    ("More neurons (128)", {"hidden_layers": (128,)}),
    ("More neurons (256)", {"hidden_layers": (256,)}),
    ("Two layers (64, 64)", {"hidden_layers": (64, 64)}),
    ("Two layers (128, 64)", {"hidden_layers": (128, 64)}),
    ("Lower learning rate (0.0001)", {"lr": 0.0001}),
    ("Higher learning rate (0.01)", {"lr": 0.01}),
    ("Regularization alpha=0.01", {"alpha": 0.01}),
    ("Regularization alpha=0.1", {"alpha": 0.1}),
]
EXPERIMENT_SEEDS = (42, 1, 2)  # same data split, 3 different random initializations


def run_experiment(data, name, changes):
    """Train one configuration with several seeds and average the results.

    With only 360 validation images, one image is ~0.28 points, so a single run
    can differ from another just by luck. Averaging 3 seeds makes it fairer.
    """
    config = {**BASELINE_CONFIG, **changes}
    X_tr, _, _, y_tr, _, _ = data
    vals, trains, times, iters = [], [], [], []
    for seed in EXPERIMENT_SEEDS:
        model, seconds = train_model(data, seed=seed, **config)
        vals.append(val_accuracy(model, data))
        trains.append(accuracy_score(y_tr, model.predict(X_tr)))
        times.append(seconds)
        iters.append(model.n_iter_)
    return {
        "name": name,
        "config": config,
        "val_mean": float(np.mean(vals)),
        "val_std": float(np.std(vals)),
        "train_mean": float(np.mean(trains)),
        "time": float(np.mean(times)),
        "epochs": int(round(np.mean(iters))),
    }


def write_experiments_md(rows, best, path):
    lines = [
        "# Sprint 3 - Experiments (validation set)",
        "",
        f"Each row changes ONE thing from the baseline. Results are the mean of "
        f"{len(EXPERIMENT_SEEDS)} runs with different random initializations "
        f"(seeds {', '.join(map(str, EXPERIMENT_SEEDS))}).",
        "",
        "| # | Configuration | Val acc (mean ± std) | Train acc | Epochs | Time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, start=1):
        mark = " **(best)**" if r is best else ""
        lines.append(
            f"| {i} | {r['name']}{mark} | {r['val_mean']:.2%} ± {r['val_std']:.2%} | "
            f"{r['train_mean']:.2%} | {r['epochs']} | {r['time']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", action="store_true")
    ap.add_argument(
        "--final", action="store_true",
        help="also evaluate the chosen model on the TEST set (do this once, at the very end)",
    )
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    X, y, dataset_name = load_data(args.mnist)
    data, scaler = split_data(X, y)
    X_tr, X_val, X_te, *_ = data
    print(f"Dataset: {dataset_name}")
    print(f"Train: {len(X_tr)} | Validation: {len(X_val)} | Test: {len(X_te)}\n")

    # Step 0 check: the baseline (same config as Sprint 2) on the validation set
    model, seconds = train_model(data)
    print(f"Baseline (64) -> val accuracy={val_accuracy(model, data):.4f}  time={seconds:.1f}s")

    # Task 1: training curves for the baseline
    _, history = train_with_history(data, hidden_layers=(64,), epochs=100)
    plot_training_curves(history, RESULTS_DIR / "training_curve.png", "Baseline (64)")
    best = int(np.argmax(history["val_acc"]))
    print(
        f"\nTask 1 -> final train acc={history['train_acc'][-1]:.4f}  "
        f"final val acc={history['val_acc'][-1]:.4f}  "
        f"best val acc={history['val_acc'][best]:.4f} (epoch {best + 1})"
    )
    print("Saved results/training_curve.png")

    # Task 2: confusion matrix of the baseline on the VALIDATION set (test stays locked)
    y_val = data[4]
    y_pred = model.predict(X_val)
    plot_confusion_matrix(y_val, y_pred, RESULTS_DIR / "confusion_matrix.png", "Baseline (64) - validation")
    X_val_raw = scaler.inverse_transform(X_val)  # back to the original pixel values
    plot_errors(X_val_raw, y_val, y_pred, RESULTS_DIR / "wrong_examples.png")
    n_wrong = int((y_val != y_pred).sum())
    print(f"\nTask 2 -> {n_wrong} wrong out of {len(y_val)} validation images")
    print("Most confused pairs (real -> predicted : count):")
    for real, pred, count in top_confusions(y_val, y_pred):
        print(f"  {real} -> {pred} : {count}")
    print("Saved results/confusion_matrix.png and results/wrong_examples.png")

    # Task 3: experiments, compared on the VALIDATION set
    print("\nTask 3 -> experiments (mean of 3 seeds, validation set)")
    rows = []
    for name, changes in EXPERIMENTS:
        r = run_experiment(data, name, changes)
        rows.append(r)
        print(
            f"  {name:<30} val={r['val_mean']:.4f} ± {r['val_std']:.4f}  "
            f"train={r['train_mean']:.4f}  epochs={r['epochs']:<3}  time={r['time']:.1f}s"
        )
    best = max(rows, key=lambda r: r["val_mean"])  # chosen by validation, never by test
    write_experiments_md(rows, best, RESULTS_DIR / "experiments.md")
    print(f"\nBest by validation: {best['name']} (val={best['val_mean']:.4f})")
    print("Saved results/experiments.md")

    # Final model: the test set is touched only here, once, and only with --final
    if args.final:
        cfg = best["config"]
        final_model, _ = train_model(data, **cfg)
        X_te, y_te = data[2], data[5]
        test_pred = final_model.predict(X_te)
        test_acc = accuracy_score(y_te, test_pred)
        print(f"\nFINAL (test set, used once) -> {best['name']}: test accuracy = {test_acc:.4f}")
        _, final_hist = train_with_history(data, epochs=max(best["epochs"], 50), **cfg)
        plot_training_curves(final_hist, RESULTS_DIR / "final_training_curve.png", best["name"])
        plot_confusion_matrix(
            y_te, test_pred, RESULTS_DIR / "final_confusion_matrix.png",
            f"{best['name']} - test",
        )
        print("Saved results/final_training_curve.png and results/final_confusion_matrix.png")
    else:
        print("\n(Test set not used. Run with --final once you are done choosing.)")


if __name__ == "__main__":
    main()