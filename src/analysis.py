"""
analysis.py
Script principal — executa a pipeline completa:
EDA + Pré-processamento + Modelos + Avaliação + Resultados
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data, create_binary_target, feature_engineering, preprocess
from modeling import (
    train_models, compare_models,
    plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_metrics_comparison
)

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "WineQT.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
SEED = 42

# ─────────────────────────────────────────────
# 1. CARREGAMENTO E COMPREENSÃO DO PROBLEMA
# ─────────────────────────────────────────────
print("=" * 60)
print("1. CARREGAMENTO DOS DADOS")
print("=" * 60)

df_raw = load_data(DATA_PATH)
print(f"Shape: {df_raw.shape}")
print(df_raw.head())
print(df_raw.describe().round(3))

# ─────────────────────────────────────────────
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. EDA")
print("=" * 60)

# Distribuição da qualidade original
fig, ax = plt.subplots(figsize=(8, 5))
quality_counts = df_raw["quality"].value_counts().sort_index()
bars = ax.bar(quality_counts.index, quality_counts.values,
              color=["#e63946" if v < 7 else "#2a9d8f" for v in quality_counts.index],
              edgecolor="white", linewidth=1.2)
ax.set_title("Distribuição das Notas de Qualidade do Vinho", fontsize=13, fontweight="bold")
ax.set_xlabel("Nota de Qualidade")
ax.set_ylabel("Contagem")
for bar, val in zip(bars, quality_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha="center", va="bottom", fontsize=10)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#e63946", label="Baixa/Média Qualidade (< 7)"),
                   Patch(facecolor="#2a9d8f", label="Alta Qualidade (≥ 7)")]
ax.legend(handles=legend_elements)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "01_quality_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Gráfico 01 salvo: distribuição da qualidade")

# Balanceamento binário
df_bin = create_binary_target(df_raw.copy())
class_counts = df_bin["high_quality"].value_counts()
fig, ax = plt.subplots(figsize=(6, 5))
ax.pie(class_counts.values, labels=["Baixa/Média\n(< 7)", "Alta\n(≥ 7)"],
       colors=["#e63946", "#2a9d8f"], autopct="%1.1f%%", startangle=90,
       textprops={"fontsize": 12})
ax.set_title("Balanceamento das Classes\n(Classificação Binária)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "02_class_balance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Gráfico 02 salvo: balanceamento de classes")
print(f"     Alta qualidade: {class_counts[1]} ({class_counts[1]/len(df_bin)*100:.1f}%)")
print(f"     Baixa/Média:    {class_counts[0]} ({class_counts[0]/len(df_bin)*100:.1f}%)")

# Histogramas de todas as features
features = [c for c in df_raw.columns if c != "quality"]
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.flatten()
for i, feat in enumerate(features):
    axes[i].hist(df_raw[feat], bins=30, color="#457b9d", edgecolor="white", alpha=0.85)
    axes[i].set_title(feat, fontsize=10, fontweight="bold")
    axes[i].set_ylabel("Frequência")
    axes[i].grid(alpha=0.3)
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])
plt.suptitle("Distribuição das Variáveis Físico-Químicas", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "03_feature_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Gráfico 03 salvo: distribuições das features")

# Matriz de correlação
fig, ax = plt.subplots(figsize=(12, 9))
corr = df_raw.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Matriz de Correlação — Variáveis Físico-Químicas", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "04_correlation_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Gráfico 04 salvo: matriz de correlação")

# Boxplots: features vs target binário
df_eda = df_raw.copy()
df_eda["Qualidade"] = df_eda["quality"].apply(lambda x: "Alta (≥7)" if x >= 7 else "Baixa/Média (<7)")
key_features = ["alcohol", "volatile acidity", "sulphates", "citric acid",
                "density", "total sulfur dioxide"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
palette = {"Alta (≥7)": "#2a9d8f", "Baixa/Média (<7)": "#e63946"}
for i, feat in enumerate(key_features):
    sns.boxplot(data=df_eda, x="Qualidade", y=feat, palette=palette, ax=axes[i])
    axes[i].set_title(feat, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].grid(axis="y", alpha=0.3)
plt.suptitle("Distribuição das Principais Features por Qualidade", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "05_features_vs_quality.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Gráfico 05 salvo: boxplots features vs qualidade")

# Detecção de outliers via IQR
print("\n  Outliers detectados por feature (método IQR):")
for feat in features:
    Q1, Q3 = df_raw[feat].quantile(0.25), df_raw[feat].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df_raw[feat] < Q1 - 1.5*IQR) | (df_raw[feat] > Q3 + 1.5*IQR)).sum()
    print(f"    {feat:30s}: {n_out} outliers ({n_out/len(df_raw)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PRÉ-PROCESSAMENTO")
print("=" * 60)

X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df_raw)
print(f"  Treino: {X_train.shape} | Teste: {X_test.shape}")
print(f"  Features após engineering: {feature_names}")
print(f"  Dados faltantes: {df_raw.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 4. TREINAMENTO DOS MODELOS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TREINAMENTO DOS MODELOS")
print("=" * 60)

models = train_models(X_train, y_train, random_state=SEED)

# ─────────────────────────────────────────────
# 5. AVALIAÇÃO DOS MODELOS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. AVALIAÇÃO DOS MODELOS")
print("=" * 60)

results_df = compare_models(models, X_test, y_test)
print("\n  Métricas:")
print(results_df.round(4).to_string())

# Salvar métricas em CSV
results_df.round(4).to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))
print("\n  ✓ Métricas salvas: metrics_comparison.csv")

# Plots de avaliação
plot_confusion_matrices(models, X_test, y_test,
                        save_path=os.path.join(RESULTS_DIR, "06_confusion_matrices.png"))
print("  ✓ Gráfico 06 salvo: matrizes de confusão")

plot_roc_curves(models, X_test, y_test,
                save_path=os.path.join(RESULTS_DIR, "07_roc_curves.png"))
print("  ✓ Gráfico 07 salvo: curvas ROC")

plot_metrics_comparison(results_df,
                        save_path=os.path.join(RESULTS_DIR, "08_metrics_comparison.png"))
print("  ✓ Gráfico 08 salvo: comparação de métricas")

# ─────────────────────────────────────────────
# 6. INTERPRETAÇÃO — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. INTERPRETAÇÃO DOS RESULTADOS")
print("=" * 60)

for model_name in ["Random Forest", "Gradient Boosting"]:
    plot_feature_importance(
        models[model_name], feature_names, model_name,
        save_path=os.path.join(RESULTS_DIR, f"09_feature_importance_{model_name.replace(' ', '_')}.png")
    )
    print(f"  ✓ Feature importance salva: {model_name}")

# Ranking final
best_model = results_df["f1_score"].idxmax()
print(f"\n  🏆 Melhor modelo (F1-Score): {best_model}")
print(f"     F1: {results_df.loc[best_model, 'f1_score']:.4f}")
print(f"     AUC: {results_df.loc[best_model, 'roc_auc']:.4f}")

print("\n" + "=" * 60)
print("Pipeline concluída! Resultados salvos em /results/")
print("=" * 60)
