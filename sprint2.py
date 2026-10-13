"""Sprint 2 - Modelo simples vs rede neural pequena.

Treina uma regressão logística (baseline) e uma rede neural pequena (MLP)
no mesmo conjunto de dígitos manuscritos, mede a acurácia no teste e salva
os resultados em resultados.json, matriz_confusao.png e comparacao.png.

Uso:
    python sprint2.py            # dataset leve (8x8, embutido no scikit-learn)
    python sprint2.py --mnist    # MNIST completo (28x28, baixa do OpenML)
"""
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


def carregar_dados(usar_mnist: bool):
    if usar_mnist:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        y = np.asarray(y).astype(int)
        X = np.asarray(X) / 255.0
        nome = "MNIST (28x28, 70.000 amostras)"
    else:
        X, y = load_digits(return_X_y=True)
        X = np.asarray(X) / 16.0  # pixels vão de 0 a 16
        nome = "sklearn load_digits (8x8, 1.797 amostras)"
    return X, y, nome


def treinar(nome, modelo, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    modelo.fit(X_tr, y_tr)
    tempo = time.time() - t0
    acc_tr = accuracy_score(y_tr, modelo.predict(X_tr))
    pred = modelo.predict(X_te)
    acc_te = accuracy_score(y_te, pred)
    print(f"{nome:<24} treino={acc_tr:.4f}  teste={acc_te:.4f}  tempo={tempo:.1f}s")
    return {"acc_treino": acc_tr, "acc_teste": acc_te, "tempo_s": tempo}, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", action="store_true")
    args = ap.parse_args()

    X, y, nome_dataset = carregar_dados(args.mnist)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    print(f"Dataset: {nome_dataset}")
    print(f"Treino: {len(X_tr)} | Teste: {len(X_te)}\n")

    res = {"dataset": nome_dataset, "n_treino": len(X_tr), "n_teste": len(X_te)}

    # 1) Baseline: regressão logística
    logreg = LogisticRegression(max_iter=1000, random_state=SEED)
    res["regressao_logistica"], _ = treinar(
        "Regressão logística", logreg, X_tr_s, y_tr, X_te_s, y_te
    )

    # 2) Rede neural pequena: 1 camada oculta de 64 neurônios
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        max_iter=300,
        early_stopping=False,  # com True parava em ~43 épocas e perdia ~2 p.p.
        random_state=SEED,
    )
    res["rede_neural"], pred_mlp = treinar(
        "Rede neural (64)", mlp, X_tr_s, y_tr, X_te_s, y_te
    )
    res["rede_neural"]["arquitetura"] = "entrada -> 64 (ReLU) -> 10 (softmax)"
    res["rede_neural"]["epocas"] = int(mlp.n_iter_)

    # Repete com 5 divisões treino/teste para ver se a diferença é estável
    accs = {"regressao_logistica": [], "rede_neural": []}
    for s in (1, 2, 3, 4, 5):
        a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
        sc = StandardScaler().fit(a)
        a, b = sc.transform(a), sc.transform(b)
        accs["regressao_logistica"].append(
            accuracy_score(d, LogisticRegression(max_iter=1000, random_state=s).fit(a, c).predict(b))
        )
        accs["rede_neural"].append(
            accuracy_score(
                d,
                MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=False, random_state=s)
                .fit(a, c)
                .predict(b),
            )
        )
    print("\nAcurácia em 5 divisões diferentes (média ± desvio):")
    for k, v in accs.items():
        print(f"  {k:<22} {np.mean(v):.4f} ± {np.std(v):.4f}")
        res[k]["media_5_divisoes"] = float(np.mean(v))
        res[k]["desvio_5_divisoes"] = float(np.std(v))

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # Gráfico de comparação
    fig, ax = plt.subplots(figsize=(5, 4))
    nomes = ["Regressão\nlogística", "Rede neural\n(64)"]
    vals = [res["regressao_logistica"]["acc_teste"], res["rede_neural"]["acc_teste"]]
    barras = ax.bar(nomes, vals, color=["#7a8aa0", "#2f6fdb"])
    ax.set_ylim(min(vals) - 0.05, 1.0)
    ax.set_ylabel("Acurácia no teste")
    ax.set_title("Sprint 2: comparação de modelos")
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig("comparacao.png", dpi=150)

    # Matriz de confusão da rede neural
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_te, pred_mlp, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriz de confusão - rede neural")
    fig.tight_layout()
    fig.savefig("matriz_confusao.png", dpi=150)


if __name__ == "__main__":
    main()
