"""
Script de diagnóstico para verificar el funcionamiento de los modelos de Hyperion3
"""

import os
import sys
import logging
import asyncio
import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("torch")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperion3.config import get_config
from hyperion3.data import DataPreprocessor
from hyperion3.models import ModelFactory
from hyperion3.utils.mlops import setup_mlops

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_test.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ModelTester:
    def __init__(self):
        self.config = get_config()
        self.data_preprocessor = DataPreprocessor(self.config)
        self.model_factory = ModelFactory(self.config)
        self.test_results: Dict[str, Dict] = {}

    async def prepare_test_data(self, symbol: str = "BTC/USDT") -> pd.DataFrame:
        """Prepara datos de prueba para los modelos"""
        try:
            # Usar un período corto para pruebas
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            data = await self.data_preprocessor.load_and_preprocess(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            if data is None or data.empty:
                raise ValueError(f"No se pudieron cargar datos para {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error preparando datos de prueba: {str(e)}")
            raise

    async def test_model(self, model_name: str, data: pd.DataFrame) -> Dict:
        """Prueba un modelo individual"""
        try:
            logger.info(f"Probando modelo: {model_name}")

            # Crear y entrenar el modelo
            model = self.model_factory.create_model(model_name)

            # Dividir datos para prueba rápida
            train_size = int(len(data) * 0.7)
            train_data = data[:train_size]
            test_data = data[train_size:]

            # Entrenamiento rápido
            model.fit(train_data)

            # Evaluación básica
            predictions = model.predict(test_data)
            metrics = {
                "mse": np.mean((test_data["close"].values - predictions) ** 2),
                "mae": np.mean(np.abs(test_data["close"].values - predictions)),
            }

            logger.info(f"Modelo {model_name} probado exitosamente")
            return {"status": "success", "metrics": metrics, "model": model}

        except Exception as e:
            logger.error(f"Error probando modelo {model_name}: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def run_parallel_tests(self, models: List[str], data: pd.DataFrame):
        """Ejecuta pruebas en paralelo para múltiples modelos"""
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {
                executor.submit(self.test_model, model, data): model for model in models
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    self.test_results[model_name] = result
                except Exception as e:
                    logger.error(
                        f"Error en prueba paralela para {model_name}: {str(e)}"
                    )
                    self.test_results[model_name] = {"status": "error", "error": str(e)}

    def print_results(self):
        """Imprime los resultados de las pruebas"""
        print("\n=== Resultados de Pruebas de Modelos ===")
        for model_name, result in self.test_results.items():
            print(f"\nModelo: {model_name}")
            if result["status"] == "success":
                print(f"Estado: ✅ Exitoso")
                print("Métricas:")
                for metric, value in result["metrics"].items():
                    print(f"  - {metric}: {value:.4f}")
            else:
                print(f"Estado: ❌ Error")
                print(f"Error: {result['error']}")


async def main():
    """Función principal de prueba"""
    try:
        # Inicializar tester
        tester = ModelTester()

        # Preparar datos de prueba
        print("Preparando datos de prueba...")
        test_data = await tester.prepare_test_data()

        # Lista de modelos a probar
        models_to_test = ["patchtst", "lightgbm", "xgboost", "catboost", "sac", "td3"]

        # Ejecutar pruebas en paralelo
        print("\nIniciando pruebas de modelos en paralelo...")
        await tester.run_parallel_tests(models_to_test, test_data)

        # Mostrar resultados
        tester.print_results()

    except Exception as e:
        logger.error(f"Error en la ejecución de pruebas: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
