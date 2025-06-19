"""
Módulo de utilidades para procesamiento de datos
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data_for_training(
    data: pd.DataFrame,
    target_col: str = "target",
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Prepara los datos para entrenamiento, incluyendo división y escalado

    Args:
        data: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        feature_cols: Lista de columnas a usar como features (None = todas excepto target)
        test_size: Proporción de datos para test
        val_size: Proporción de datos para validación (del conjunto de train)
        random_state: Semilla aleatoria
        scale_features: Si se deben escalar las features

    Returns:
        Tuple con (X_train, X_val, y_train, y_val, scaler)
    """
    # Determinar columnas de features
    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != target_col]

    # Extraer features y target
    X = data[feature_cols].values
    y = data[target_col].values

    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Dividir train en train y validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size_adjusted, random_state=random_state
    )

    # Escalar features si se requiere
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return X_train, X_val, y_train, y_val, scaler


def create_sequences(
    data: Union[pd.DataFrame, np.ndarray],
    sequence_length: int,
    target_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea secuencias de datos para modelos de series temporales

    Args:
        data: DataFrame o array con los datos
        sequence_length: Longitud de las secuencias
        target_col: Columna objetivo (solo para DataFrame)
        feature_cols: Columnas de features (solo para DataFrame)
        step: Paso entre secuencias

    Returns:
        Tuple con (X, y) donde X son las secuencias y y son los targets
    """
    if isinstance(data, pd.DataFrame):
        if target_col is None:
            raise ValueError("target_col debe especificarse para DataFrame")
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]

        X_data = data[feature_cols].values
        y_data = data[target_col].values
    else:
        X_data = data
        y_data = None

    n_samples = len(X_data) - sequence_length + 1
    n_features = X_data.shape[1]

    X = np.zeros((n_samples, sequence_length, n_features))
    y = np.zeros(n_samples) if y_data is not None else None

    for i in range(0, n_samples, step):
        X[i] = X_data[i : i + sequence_length]
        if y_data is not None:
            y[i] = y_data[i + sequence_length - 1]

    if y_data is None:
        return X
    return X, y


__all__ = ["prepare_data_for_training", "create_sequences"]
