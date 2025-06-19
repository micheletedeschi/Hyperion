import unittest
import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("torch")
import numpy as np
import pandas as pd
import torch
from hyperion3.models.transformers.patchtst import PatchTST
from hyperion3.training.trainer import Trainer
import time
import psutil
import os


class TestOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todos los tests de optimización"""
        # Configuración del modelo
        cls.config = {
            "model": {
                "seq_len": 96,
                "pred_len": 24,
                "d_model": 128,
                "n_heads": 8,
                "d_ff": 256,
                "dropout": 0.1,
                "patch_len": 16,
                "stride": 8,
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-3,
                "max_epochs": 100,
                "early_stopping_patience": 15,
            },
        }

        # Crear datos sintéticos más grandes para tests de rendimiento
        np.random.seed(42)
        n_samples = 10000  # Más datos para tests de rendimiento
        n_features = 10
        cls.data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        cls.data["target"] = np.random.randn(n_samples)

        # Dividir en train y validation
        train_size = int(0.8 * n_samples)
        cls.train_data = cls.data.iloc[:train_size]
        cls.val_data = cls.data.iloc[train_size:]

    def setUp(self):
        """Configuración para cada test individual"""
        self.model = PatchTST(
            seq_len=self.config["model"]["seq_len"],
            pred_len=self.config["model"]["pred_len"],
            d_model=self.config["model"]["d_model"],
            n_heads=self.config["model"]["n_heads"],
            d_ff=self.config["model"]["d_ff"],
            dropout=self.config["model"]["dropout"],
            patch_len=self.config["model"]["patch_len"],
            stride=self.config["model"]["stride"],
        )

    def get_memory_usage(self):
        """Obtener uso de memoria actual del proceso"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def test_batch_processing_efficiency(self):
        """Test de eficiencia en el procesamiento por lotes"""
        batch_sizes = [16, 32, 64, 128]
        times = []
        memory_usage = []

        for batch_size in batch_sizes:
            # Medir tiempo y memoria
            start_time = time.time()
            start_memory = self.get_memory_usage()

            # Procesar datos
            feature_cols = [col for col in self.train_data.columns if col != "target"]
            self.model.fit(
                self.train_data,
                self.val_data,
                feature_cols=feature_cols,
                target_col="target",
                batch_size=batch_size,
                max_epochs=2,  # Pocas épocas para el test
            )

            end_time = time.time()
            end_memory = self.get_memory_usage()

            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)

        # Verificar que los tiempos son razonables
        for t in times:
            self.assertLess(t, 60)  # No más de 60 segundos por batch size

        # Verificar que el uso de memoria es proporcional al batch size
        for i in range(1, len(memory_usage)):
            self.assertLess(memory_usage[i], memory_usage[i - 1] * 2)

    def test_gpu_memory_efficiency(self):
        """Test de eficiencia en el uso de memoria GPU"""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS no está disponible")

        # Medir uso de memoria antes
        torch.mps.empty_cache()
        start_memory = torch.mps.current_allocated_memory() / 1024 / 1024  # MB

        # Entrenar modelo
        feature_cols = [col for col in self.train_data.columns if col != "target"]
        self.model.fit(
            self.train_data,
            self.val_data,
            feature_cols=feature_cols,
            target_col="target",
            max_epochs=2,
        )

        # Medir uso de memoria después
        end_memory = torch.mps.current_allocated_memory() / 1024 / 1024  # MB
        memory_used = end_memory - start_memory

        # Verificar que el uso de memoria es razonable
        self.assertLess(memory_used, 1000)  # No más de 1GB de memoria GPU

    def test_data_loading_efficiency(self):
        """Test de eficiencia en la carga de datos"""
        # Medir tiempo de carga de datos
        start_time = time.time()

        # Crear dataloader
        feature_cols = [col for col in self.train_data.columns if col != "target"]
        train_loader = self.model._create_dataloader(
            self.train_data,
            feature_cols=feature_cols,
            target_col="target",
            batch_size=self.config["training"]["batch_size"],
        )

        # Iterar sobre el dataloader
        for _ in train_loader:
            pass

        end_time = time.time()
        load_time = end_time - start_time

        # Verificar que el tiempo de carga es razonable
        self.assertLess(
            load_time, 5
        )  # No más de 5 segundos para cargar todos los datos

    def test_model_inference_speed(self):
        """Test de velocidad de inferencia"""
        # Preparar datos
        feature_cols = [col for col in self.train_data.columns if col != "target"]
        self.model.fit(
            self.train_data,
            self.val_data,
            feature_cols=feature_cols,
            target_col="target",
            max_epochs=1,
        )

        # Medir tiempo de inferencia
        batch_size = 32
        n_batches = 10
        total_time = 0

        for _ in range(n_batches):
            x = torch.randn(
                batch_size, self.config["model"]["seq_len"], len(feature_cols)
            )

            start_time = time.time()
            with torch.no_grad():
                _ = self.model(x)
            end_time = time.time()

            total_time += end_time - start_time

        avg_inference_time = total_time / n_batches

        # Verificar que el tiempo de inferencia es razonable
        self.assertLess(avg_inference_time, 0.1)  # No más de 100ms por batch

    def test_memory_leaks(self):
        """Test de fugas de memoria"""
        initial_memory = self.get_memory_usage()

        # Realizar múltiples entrenamientos
        for _ in range(3):
            feature_cols = [col for col in self.train_data.columns if col != "target"]
            self.model.fit(
                self.train_data,
                self.val_data,
                feature_cols=feature_cols,
                target_col="target",
                max_epochs=1,
            )

            # Forzar limpieza de memoria
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        # Verificar que no hay fugas de memoria significativas
        self.assertLess(memory_increase, 100)  # No más de 100MB de aumento


if __name__ == "__main__":
    unittest.main()
