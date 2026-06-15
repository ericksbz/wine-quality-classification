# 🍷 Wine Quality Classification — Machine Learning

> **POSTECH | Pós-Graduação em Data Analytics & Tecnologia**  
> Tech Challenge — Fase 2 | Classificação com Machine Learning

---

## 📋 Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Dataset](#dataset)
- [Pipeline de Análise](#pipeline-de-análise)
- [Modelos Treinados](#modelos-treinados)
- [Resultados](#resultados)
- [Como Executar](#como-executar)
- [Principais Conclusões](#principais-conclusões)
- [Tecnologias](#tecnologias)

---

## Sobre o Projeto

A avaliação da qualidade de vinhos é, historicamente, um processo subjetivo realizado por especialistas por meio de análise sensorial. Este projeto propõe uma solução baseada em **Machine Learning** para prever automaticamente se um vinho é de **alta qualidade** com base em suas características físico-químicas — tornando o processo mais objetivo, escalável e econômico.

**Problema:** Classificação binária da qualidade do vinho  
**Target:** `high_quality` — 1 (Alta Qualidade: nota ≥ 7) | 0 (Baixa/Média: nota < 7)

---

## Estrutura do Repositório

```
wine-quality-classification/
│
├── data/
│   └── WineQT.csv                  # Dataset original
│
├── notebooks/
│   └── wine_quality_classification.ipynb  # Análise completa (EDA + Modelos)
│
├── src/
│   ├── preprocessing.py            # Pipeline de pré-processamento
│   ├── modeling.py                 # Treinamento, avaliação e visualizações
│   └── analysis.py                 # Script principal (execução completa)
│
├── results/
│   ├── metrics_comparison.csv      # Métricas comparativas dos modelos
│   ├── 01_quality_distribution.png
│   ├── 02_class_balance.png
│   ├── 03_feature_distributions.png
│   ├── 04_correlation_matrix.png
│   ├── 05_features_vs_quality.png
│   ├── 06_confusion_matrices.png
│   ├── 07_roc_curves.png
│   ├── 08_metrics_comparison.png
│   ├── 09_feature_importance_Random_Forest.png
│   └── 09_feature_importance_Gradient_Boosting.png
│
├── requirements.txt                # Dependências do projeto
└── README.md                       # Este arquivo
```

---

## Dataset

**Fonte:** [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

| Variável | Descrição |
|---|---|
| `fixed acidity` | Acidez fixa (tartárica) do vinho |
| `volatile acidity` | Acidez volátil (acética) — excesso resulta em gosto de vinagre |
| `citric acid` | Ácido cítrico — confere frescor ao vinho |
| `residual sugar` | Açúcar residual após fermentação |
| `chlorides` | Concentração de cloretos (sal) |
| `free sulfur dioxide` | SO₂ livre — age como conservante e antimicrobiano |
| `total sulfur dioxide` | Total de SO₂ (livre + ligado) |
| `density` | Densidade do vinho |
| `pH` | Acidez/basicidade do vinho (escala 0-14) |
| `sulphates` | Aditivo que aumenta SO₂ — atua como antioxidante |
| `alcohol` | Teor alcoólico (% volume) |
| `quality` | **Nota de qualidade** (0–10) atribuída por especialistas |

**Dimensões:** 1.143 amostras × 12 variáveis | Sem valores faltantes

---

## Pipeline de Análise

### 1. 🎯 Compreensão do Problema
- Contexto da indústria vitivinícola
- Definição da variável alvo binária (threshold: nota ≥ 7)

### 2. 📊 Análise Exploratória de Dados (EDA)
- Distribuição das notas de qualidade originais
- Análise do desbalanceamento de classes (86.1% vs 13.9%)
- Histogramas de todas as features
- Matriz de correlação e análise de relacionamentos
- Boxplots por classe de qualidade
- Detecção de outliers via método IQR

### 3. ⚙️ Pré-processamento
- Verificação e tratamento de dados faltantes (dataset limpo)
- Transformação da variável target em classificação binária
- **Feature Engineering** — 3 novas features criadas:
  - `acidity_ratio`: relação acidez fixa/volátil
  - `sulfur_ratio`: eficiência do conservante (SO₂ livre/total)
  - `alcohol_density_ratio`: leveza alcoólica do vinho
- Normalização com `StandardScaler`
- Split estratificado treino/teste (80/20)

### 4. 🤖 Desenvolvimento dos Modelos
Três modelos foram treinados e comparados:
- **Regressão Logística** (baseline linear)
- **Random Forest** (ensemble por bagging)
- **Gradient Boosting** (ensemble por boosting)

### 5. 📈 Avaliação
Métricas: Acurácia, Precisão, Recall, **F1-Score**, **ROC-AUC**  
Visualizações: Matrizes de confusão, Curvas ROC, Comparativo de métricas

### 6. 💡 Interpretação
Feature importance dos modelos baseados em árvore e implicações para a produção vinícola.

---

## Modelos Treinados

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Regressão Logística | 0.7904 | 0.3621 | 0.6562 | 0.4667 | 0.8637 |
| Random Forest | 0.8996 | 0.6667 | 0.5625 | 0.6102 | 0.9036 |
| **Gradient Boosting** ⭐ | **0.9127** | **0.7000** | **0.6562** | **0.6774** | 0.8831 |

> **Métrica principal: F1-Score** — escolhida por ser mais adequada ao cenário de classes desbalanceadas.

---

## Resultados

### 🏆 Melhor Modelo: Gradient Boosting
- **F1-Score: 0.677** — melhor equilíbrio entre precisão e recall
- **Acurácia: 91.3%** — alta taxa de acerto geral
- **ROC-AUC: 0.883** — excelente capacidade discriminativa

### 📌 Fatores Mais Relevantes para a Qualidade

| Feature | Importância | Interpretação |
|---|---|---|
| **alcohol** | ⭐⭐⭐⭐⭐ | Maior teor alcoólico → maior qualidade |
| **alcohol_density_ratio** | ⭐⭐⭐⭐ | Feature engineered — captura leveza do vinho |
| **volatile acidity** | ⭐⭐⭐⭐ | Alta acidez volátil → menor qualidade |
| **sulphates** | ⭐⭐⭐ | Bom conservante → melhora qualidade |
| **citric acid** | ⭐⭐⭐ | Frescor → contribui positivamente |
| **total sulfur dioxide** | ⭐⭐ | Excesso → penaliza qualidade |

---

## Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/wine-quality-classification.git
cd wine-quality-classification
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute o script principal
```bash
python src/analysis.py
```
> Gera todos os gráficos e métricas na pasta `results/`

### 4. Ou abra o Jupyter Notebook
```bash
jupyter notebook notebooks/wine_quality_classification.ipynb
```

---

## Principais Conclusões

1. **O teor alcoólico é o principal preditor da qualidade** — vinhos com maior graduação alcóolica tendem a receber notas mais altas pelos especialistas.

2. **A acidez volátil é o maior fator negativo** — altos níveis indicam degradação bacteriana que produz ácido acético (vinagre), comprometendo o sabor.

3. **Dataset fortemente desbalanceado** (86.1% × 13.9%) exigiu uso de `class_weight="balanced"` e métricas orientadas a F1/AUC.

4. **Gradient Boosting superou os demais modelos** em F1-Score e acurácia, sendo indicado para implantação em ambiente produtivo.

5. **Feature Engineering gerou valor real**: `alcohol_density_ratio` figurou entre as features mais importantes nos modelos baseados em árvore.

### 💡 Implicações para a Produção

- **Monitoramento contínuo** de `alcohol` e `volatile acidity` durante a fermentação pode antecipar a qualidade final do lote.
- **Redução de custos**: diminui dependência de avaliações sensoriais manuais e agiliza o controle de qualidade.
- **Integração ao processo**: o modelo pode emitir alertas automáticos para lotes com alta probabilidade de baixa qualidade.
- **Trabalhos futuros**: aplicar SMOTE para oversampling da classe minoritária e testar modelos como XGBoost e LightGBM.

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/numpy-1.26-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.8-11557C)
![Seaborn](https://img.shields.io/badge/seaborn-0.13-4C72B0)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

---

*Projeto desenvolvido para o Tech Challenge — Fase 2 | POSTECH DTAT*
