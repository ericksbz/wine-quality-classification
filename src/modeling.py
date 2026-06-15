"""
modeling.py
Funções de treinamento e avaliação dos modelos de classificação.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)


def train_models(X_train, y_train, random_state: int = 42) -> dict:
    """
    Treina três modelos de classificação:
    - Regressão Logística
    - Random Forest
    - Gradient Boosting
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=random_state,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=random_state
        ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✓ {name} treinado.")
    return models


def evaluate_model(model, X_test, y_test, model_name: str = "") -> dict:
    """Calcula métricas de avaliação para um modelo."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }
    return metrics


def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """Gera DataFrame comparativo de métricas para todos os modelos."""
    results = []
    for name, model in models.items():
        m = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(m)
    df = pd.DataFrame(results).set_index("model")
    return df


def plot_confusion_matrices(models: dict, X_test, y_test, save_path: str = None):
    """Plota matrizes de confusão lado a lado."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Baixa/Média", "Alta"])
        ax.set_yticklabels(["Baixa/Média", "Alta"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.suptitle("Matrizes de Confusão", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(models: dict, X_test, y_test, save_path: str = None):
    """Plota curvas ROC para todos os modelos."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e63946", "#2a9d8f", "#e9c46a"]
    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name, color=color)
    ax.plot([0, 1], [0, 1], "k--", label="Aleatório (AUC = 0.50)")
    ax.set_title("Curvas ROC — Comparação dos Modelos", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names: list, model_name: str, save_path: str = None):
    """Plota importância das features para modelos baseados em árvore."""
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} não possui feature_importances_.")
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features_sorted = [feature_names[i] for i in indices]
    values_sorted = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2a9d8f" if v >= np.percentile(values_sorted, 75) else "#457b9d" for v in values_sorted]
    bars = ax.barh(features_sorted[::-1], values_sorted[::-1], color=colors[::-1])
    ax.set_title(f"Importância das Features — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importância Relativa")
    ax.axvline(x=np.mean(values_sorted), color="red", linestyle="--", alpha=0.5, label="Média")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Gráfico de barras comparando métricas entre modelos."""
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Acurácia", "Precisão", "Recall", "F1-Score", "ROC-AUC"]

    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#e63946", "#2a9d8f", "#e9c46a"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_name, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=model_name, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Métricas entre os Modelos", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
