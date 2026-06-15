"""
preprocessing.py
Funções de pré-processamento para o dataset Wine Quality.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset e remove a coluna de ID desnecessária."""
    df = pd.read_csv(path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df


def create_binary_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """
    Transforma a variável 'quality' em classificação binária.
    Alta Qualidade (1): nota >= threshold
    Baixa/Média Qualidade (0): nota < threshold
    """
    df = df.copy()
    df["high_quality"] = (df["quality"] >= threshold).astype(int)
    df = df.drop(columns=["quality"])
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features a partir das variáveis existentes.
    - acidity_ratio: relação entre acidez fixa e volátil
    - sulfur_ratio: relação entre SO2 livre e total
    - alcohol_density_ratio: interação entre álcool e densidade
    """
    df = df.copy()
    df["acidity_ratio"] = df["fixed acidity"] / (df["volatile acidity"] + 1e-9)
    df["sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-9)
    df["alcohol_density_ratio"] = df["alcohol"] / (df["density"] + 1e-9)
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline completo de pré-processamento:
    1. Criação do target binário
    2. Feature engineering
    3. Split treino/teste
    4. Normalização (StandardScaler)

    Retorna: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = create_binary_target(df)
    df = feature_engineering(df)

    X = df.drop(columns=["high_quality"])
    y = df["high_quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    X_train_sc = pd.DataFrame(X_train_sc, columns=X.columns)
    X_test_sc = pd.DataFrame(X_test_sc, columns=X.columns)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, list(X.columns)
