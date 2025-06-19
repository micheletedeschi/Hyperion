"""
Script para entrenamiento paralelo de modelos en Hyperion3
"""

import os
import sys
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperion3.config import get_config
from hyperion3.data import DataPreprocessor
from hyperion3.models import ModelFactory
from hyperion3.utils.mlops import setup_mlops
from hyperion3.utils.model_utils import save_model

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("parallel_train.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ParallelTrainer:
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.data_preprocessor = DataPreprocessor(self.config)
        self.model_factory = ModelFactory(self.config)
        self.training_results: Dict[str, Dict] = {}
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def prepare_training_data(self, symbol: str) -> pd.DataFrame:
        """Prepara datos para entrenamiento"""
        try:
            data = await self.data_preprocessor.load_and_preprocess(
                symbol=symbol,
                start_date=self.config.data.start_date,
                end_date=self.config.data.end_date,
            )

            if data is None or data.empty:
                raise ValueError(f"No se pudieron cargar datos para {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error preparando datos de entrenamiento: {str(e)}")
            raise

    async def train_model(
        self, model_name: str, data: pd.DataFrame, symbol: str
    ) -> Dict:
        """Entrena un modelo individual"""
        try:
            logger.info(f"Iniciando entrenamiento de {model_name} para {symbol}")

            # Crear modelo
            model = self.model_factory.create_model(model_name)

            # Dividir datos
            train_size = int(len(data) * self.config.data.train_split)
            val_size = int(len(data) * self.config.data.val_split)

            train_data = data[:train_size]
            val_data = data[train_size : train_size + val_size]
            test_data = data[train_size + val_size :]

            # Entrenamiento
            model.fit(train_data=train_data, val_data=val_data, test_data=test_data)

            # Evaluación
            metrics = model.evaluate(test_data)

            # Guardar modelo
            model_path = (
                self.checkpoint_dir
                / f"{symbol}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
            save_model(model, str(model_path))

            logger.info(f"Modelo {model_name} entrenado exitosamente para {symbol}")
            return {
                "status": "success",
                "metrics": metrics,
                "model_path": str(model_path),
            }

        except Exception as e:
            logger.error(
                f"Error entrenando modelo {model_name} para {symbol}: {str(e)}"
            )
            return {"status": "error", "error": str(e)}

    async def train_all_models(self, symbols: List[str], models: List[str]):
        """Entrena todos los modelos en paralelo para todos los símbolos"""
        for symbol in symbols:
            logger.info(f"\nProcesando símbolo: {symbol}")

            # Preparar datos para este símbolo
            data = await self.prepare_training_data(symbol)

            # Entrenar modelos en paralelo
            with ThreadPoolExecutor(max_workers=len(models)) as executor:
                futures = {
                    executor.submit(self.train_model, model, data, symbol): (
                        model,
                        symbol,
                    )
                    for model in models
                }

                for future in as_completed(futures):
                    model_name, symbol = futures[future]
                    try:
                        result = future.result()
                        key = f"{symbol}_{model_name}"
                        self.training_results[key] = result

                        if result["status"] == "success":
                            logger.info(f"✅ {key}: Entrenamiento completado")
                            logger.info(f"Métricas: {result['metrics']}")
                        else:
                            logger.error(f"❌ {key}: Error en entrenamiento")
                            logger.error(f"Error: {result['error']}")

                    except Exception as e:
                        logger.error(
                            f"Error en entrenamiento paralelo para {symbol}_{model_name}: {str(e)}"
                        )
                        self.training_results[f"{symbol}_{model_name}"] = {
                            "status": "error",
                            "error": str(e),
                        }

    def print_summary(self):
        """Imprime un resumen del entrenamiento"""
        print("\n=== Resumen de Entrenamiento ===")

        # Agrupar por símbolo
        symbols = set(key.split("_")[0] for key in self.training_results.keys())

        for symbol in symbols:
            print(f"\nSímbolo: {symbol}")
            symbol_results = {
                k: v for k, v in self.training_results.items() if k.startswith(symbol)
            }

            successful = sum(
                1 for r in symbol_results.values() if r["status"] == "success"
            )
            failed = sum(1 for r in symbol_results.values() if r["status"] == "error")

            print(f"Modelos exitosos: {successful}")
            print(f"Modelos fallidos: {failed}")

            if successful > 0:
                print("\nMétricas de modelos exitosos:")
                for model_key, result in symbol_results.items():
                    if result["status"] == "success":
                        model_name = model_key.split("_")[1]
                        print(f"\n{model_name}:")
                        for metric, value in result["metrics"].items():
                            print(f"  - {metric}: {value:.4f}")


async def main():
    """Función principal de entrenamiento paralelo"""
    try:
        # Inicializar trainer
        trainer = ParallelTrainer()

        # Obtener símbolos y modelos del config
        symbols = trainer.config.data.symbols
        models = trainer.config.ensemble_models

        print(f"Símbolos a procesar: {symbols}")
        print(f"Modelos a entrenar: {models}")

        # Entrenar todos los modelos
        await trainer.train_all_models(symbols, models)

        # Mostrar resumen
        trainer.print_summary()

    except Exception as e:
        logger.error(f"Error en el entrenamiento paralelo: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
