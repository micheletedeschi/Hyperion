import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
pytest.importorskip("pandas")
import torch
import numpy as np
import pandas as pd
from hyperion3.models.model_factory import ModelFactory
from hyperion3.data.preprocessor import DataPreprocessor
from hyperion3.config.base_config import (
    HyperionV2Config,
    ModelType,
    DataConfig,
    TransformerConfig,
    RLConfig,
    TradingConfig,
    MLOpsConfig,
)

# Configuración base para los modelos
base_config = HyperionV2Config(
    data=DataConfig(
        lookback_window=60,
        prediction_horizon=24,
        feature_columns=[
            "log_return",
            "volatility",
            "rsi",
            "sma_20",
            "std_20",
            "upper_band",
            "lower_band",
            "macd",
            "macd_signal",
            "volume_norm",
            "momentum",
            "atr",
        ],
    ),
    transformer=TransformerConfig(
        learning_rate=0.001,
        weight_decay=0.0001,
        dropout=0.1,
        d_model=128,
        n_heads=8,
        n_layers=2,
    ),
    rl=RLConfig(
        learning_rate=0.001,
        batch_size=32,
        buffer_size=10000,
        gamma=0.99,
        tau=0.005,
        sac_alpha=0.2,
        sac_automatic_entropy_tuning=True,
        td3_policy_noise=0.2,
        td3_noise_clip=0.5,
        td3_policy_freq=2,
        rainbow_n_steps=3,
        rainbow_num_atoms=51,
        rainbow_v_min=-10,
        rainbow_v_max=10,
    ),
    trading=TradingConfig(
        max_position_size=1.0, initial_balance=10000, commission=0.001
    ),
    mlops=MLOpsConfig(use_gpu=False),
)


@pytest.fixture
def sample_data():
    # Crear datos de prueba sintéticos
    dates = pd.date_range(
        start="2023-01-01", periods=100, freq="h"
    )  # Cambiado de 'H' a 'h'
    data = pd.DataFrame(
        {
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 102,
            "low": np.random.randn(100).cumsum() + 98,
            "close": np.random.randn(100).cumsum() + 101,
            "volume": np.random.randint(1000, 10000, 100),
            "log_return": np.random.randn(100) * 0.01,
            "volatility": np.random.randn(100) * 0.02,
            "rsi": np.random.uniform(0, 100, 100),
            "sma_20": np.random.randn(100).cumsum() + 100,
            "std_20": np.random.randn(100) * 0.5,
            "upper_band": np.random.randn(100).cumsum() + 102,
            "lower_band": np.random.randn(100).cumsum() + 98,
            "macd": np.random.randn(100),
            "macd_signal": np.random.randn(100),
            "volume_norm": np.random.randn(100),
            "momentum": np.random.randn(100),
            "atr": np.random.randn(100) * 0.5,
        },
        index=dates,
    )
    return data


@pytest.fixture
def preprocessed_data(sample_data):
    config = {
        "feature_columns": base_config.data.feature_columns,
        "target_column": "close",
        "scaler": "standard",
        "fillna_method": "ffill",
        "add_indicators": True,
    }
    preprocessor = DataPreprocessor(config)
    return preprocessor.fit_transform(sample_data)


def test_sac_model(preprocessed_data):
    config = HyperionV2Config.from_dict(base_config.to_dict())
    config.model_type = ModelType.SAC
    factory = ModelFactory(config)
    model = factory.create_model("BTC/USDT")

    # Verificar inicialización
    assert model is not None
    assert hasattr(model, "sac")
    assert hasattr(model.sac, "select_action")

    # Preparar datos de prueba
    num_features = len(config.data.feature_columns)
    lookback_window = config.data.lookback_window
    state_dim = num_features * lookback_window + 4  # +4 for portfolio state

    # Crear estado con la forma correcta
    market_data = np.random.randn(1, num_features * lookback_window)
    portfolio_state = np.array(
        [[1.0, 0.0, 0.0, 1.0]]
    )  # [balance, position, position_value, total_value]
    state = np.concatenate([market_data, portfolio_state], axis=1)

    # Probar selección de acción
    action = model.sac.select_action(state)
    assert isinstance(action, np.ndarray)
    if action.shape == (1, 1):
        action = action.flatten()
    assert action.shape == (1,)

    # No se puede probar train_step directamente, así que solo verificamos la interfaz principal


def test_td3_model(preprocessed_data):
    config = HyperionV2Config.from_dict(base_config.to_dict())
    config.model_type = ModelType.TD3
    factory = ModelFactory(config)
    model = factory.create_model("BTC/USDT")

    # Verificar inicialización
    assert model is not None
    assert hasattr(model, "td3")
    assert hasattr(model.td3, "select_action")

    # Preparar datos de prueba
    num_features = len(config.data.feature_columns)
    lookback_window = config.data.lookback_window
    state_dim = num_features * lookback_window + 4  # +4 for portfolio state

    # Crear estado con la forma correcta
    market_data = np.random.randn(1, num_features * lookback_window)
    portfolio_state = np.array(
        [[1.0, 0.0, 0.0, 1.0]]
    )  # [balance, position, position_value, total_value]
    state = np.concatenate([market_data, portfolio_state], axis=1)

    # Probar selección de acción
    action = model.td3.select_action(state)
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)

    # No se puede probar train_step directamente, así que solo verificamos la interfaz principal


def test_rainbow_model(preprocessed_data):
    config = HyperionV2Config.from_dict(base_config.to_dict())
    config.model_type = ModelType.RAINBOW_DQN
    factory = ModelFactory(config)
    model = factory.create_model("BTC/USDT")

    # Verificar inicialización
    assert model is not None
    assert hasattr(model, "rainbow")
    assert hasattr(model.rainbow, "select_action")

    # Preparar datos de prueba
    num_features = len(config.data.feature_columns)
    lookback_window = config.data.lookback_window
    state_dim = num_features * lookback_window + 4  # +4 for portfolio state

    # Crear estado con la forma correcta
    market_data = np.random.randn(1, num_features * lookback_window)
    portfolio_state = np.array(
        [[1.0, 0.0, 0.0, 1.0]]
    )  # [balance, position, position_value, total_value]
    state = np.concatenate([market_data, portfolio_state], axis=1)

    # Probar selección de acción
    action = model.rainbow.select_action(state)
    assert isinstance(action, int)
    assert 0 <= action < 8  # Rainbow tiene 8 acciones discretas


def test_model_save_load(preprocessed_data):
    # Probar guardado y carga para cada modelo
    model_types = [ModelType.SAC, ModelType.TD3, ModelType.RAINBOW_DQN]

    for model_type in model_types:
        config = HyperionV2Config.from_dict(base_config.to_dict())
        config.model_type = model_type
        factory = ModelFactory(config)
        model = factory.create_model("BTC/USDT")

        # Guardar modelo
        save_path = f"test_{model_type.value}_model.pth"
        model.save(save_path)

        # Cargar modelo
        loaded_model = factory.create_model("BTC/USDT")
        loaded_model.load(save_path)

        # Verificar que el modelo cargado tiene el atributo correcto
        if model_type == ModelType.SAC:
            assert hasattr(loaded_model, "sac")
        elif model_type == ModelType.TD3:
            assert hasattr(loaded_model, "td3")
        elif model_type == ModelType.RAINBOW_DQN:
            assert hasattr(loaded_model, "rainbow")
