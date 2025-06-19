"""
PatchTST Implementation for Hyperion V2
State-of-the-art transformer for time series forecasting
21% better accuracy than traditional methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any, List, Callable, Union
import math
from einops import rearrange, repeat
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
from torch.serialization import add_safe_globals
import os
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from collections import defaultdict

# Añadir numpy._core.multiarray._reconstruct a los globals seguros
add_safe_globals(["numpy._core.multiarray._reconstruct"])

# Configurar logger
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]


class PatchEmbedding(nn.Module):
    """Embedding de patches para PatchTST"""

    def __init__(self, d_model: int, patch_len: int, n_vars: int):
        """
        Inicializar embedding de patches

        Args:
            d_model: Dimensión del modelo
            patch_len: Longitud de cada patch
            n_vars: Número de variables de entrada
        """
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.n_vars = n_vars

        # Embedding lineal
        self.embedding = nn.Linear(patch_len * n_vars, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Tensor de forma (batch_size, seq_len, n_vars)

        Returns:
            Tensor de forma (batch_size, n_patches, d_model)
        """
        # Verificar y limpiar datos
        if torch.isnan(x).any():
            logger.warning("Datos de entrada contienen NaN. Reemplazando con 0...")
            x = torch.nan_to_num(x, nan=0.0)

        batch_size, seq_len, n_vars = x.shape

        # Verificar dimensiones
        if n_vars != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} input variables, got {n_vars}")

        # Calcular número de patches
        n_patches = seq_len // self.patch_len

        # Reshape para procesar patches
        x = x.reshape(batch_size, n_patches, self.patch_len * n_vars)

        # Embedding
        x = self.embedding(x)  # (batch_size, n_patches, d_model)

        return x


class AttentionLayer(nn.Module):
    """Multi-head attention layer with normalization"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        d_ff = d_ff or 4 * d_model

        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed forward
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Self attention
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Feed forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class PatchTSTEncoder(nn.Module):
    """PatchTST Encoder"""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_vars: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars

        # Patch embedding
        # Correct parameter order: d_model, patch_len, n_vars
        # seq_len was mistakenly used as d_model, leading to shape errors
        self.patch_embedding = PatchEmbedding(d_model, patch_len, n_vars)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                AttentionLayer(d_model, n_heads, d_ff, dropout, activation)
                for _ in range(e_layers)
            ]
        )

        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Forward pass through the encoder."""
        # Patch embedding
        x = self.patch_embedding(x)
        n_vars = self.patch_embedding.n_vars

        # Positional encoding
        x = self.positional_encoding(x)

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        return x, n_vars


class FlattenHead(nn.Module):
    """Flatten output head for prediction"""

    def __init__(
        self,
        n_vars: int,
        seq_len: int,
        pred_len: int,
        d_model: int,
        head_dropout: float = 0,
    ):
        super().__init__()

        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size * n_vars, patch_num, d_model]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)

        # Reshape back
        x = rearrange(x, "(b n) l -> b l n", n=self.n_vars)

        return x


class PatchTST(nn.Module):
    """
    PatchTST: Patch Time Series Transformer

    Key innovations:
    - Patching: Segments time series into subseries-level patches
    - Channel-independence: Each channel processed independently
    - Efficient attention with reduced complexity
    """

    def __init__(self, config: Dict):
        """
        Inicializa el modelo PatchTST.

        Args:
            config: Diccionario con la configuración del modelo
        """
        super().__init__()

        # Parámetros del modelo
        feature_cols = config.get("feature_columns", [])

        # Determinar número de variables de entrada de forma robusta
        if "n_vars" in config:
            self.n_vars = int(config["n_vars"])
            if feature_cols and len(feature_cols) != self.n_vars:
                logger.warning(
                    "'n_vars' no coincide con 'feature_columns'. "
                    "Usando longitud de 'feature_columns'"
                )
                self.n_vars = len(feature_cols)
        else:
            self.n_vars = len(feature_cols) if feature_cols else 1

        self.d_model = int(config.get("d_model", 128))
        self.n_heads = int(config.get("n_heads", 8))
        self.n_layers = int(config.get("n_layers", 3))
        self.dropout = float(config.get("dropout", 0.1))
        self.patch_size = int(config.get("patch_size", 16))
        self.lookback_window = int(config.get("lookback_window", 96))
        self.pred_len = int(config.get("pred_len", 24))

        self.d_model = int(config.get("d_model", 128))
        self.n_heads = int(config.get("n_heads", 8))
        self.n_layers = int(config.get("n_layers", 3))
        self.dropout = float(config.get("dropout", 0.1))
        self.patch_size = int(config.get("patch_size", 16))
        self.lookback_window = int(config.get("lookback_window", 96))
        self.pred_len = int(config.get("pred_len", 24))

        # Calcular dimensiones
        # Calculate number of patches. If lookback_window is not divisible by
        # patch_size we will pad the input later in ``forward``. Emit a warning
        # instead of raising an error so training can proceed.
        self.n_patches = math.ceil(self.lookback_window / self.patch_size)
        if self.lookback_window % self.patch_size != 0:
            padded = self.n_patches * self.patch_size
            logger.warning(
                "Ventana de lookback (%s) no divisible por tamaño de patch (%s). Se aplicará padding hasta %s.",
                self.lookback_window,
                self.patch_size,
                padded,
            )

        # Capas del modelo

        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model, patch_len=self.patch_size, n_vars=self.n_vars
        )

        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model, patch_len=self.patch_size, n_vars=self.n_vars
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        # Proyección final
        self.final_projection = nn.Linear(self.d_model, self.pred_len)

        # Función de pérdida
        self.criterion = nn.MSELoss()

        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Guardar configuración
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.

        Args:
            x: Tensor de entrada (batch_size, seq_len, n_vars)

        Returns:
            Tensor de salida (batch_size, pred_len)
        """
        # Asegurar que x es 3D
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # Añadir dimensión de variables si es necesario

        batch_size, seq_len, n_vars = x.shape

        # Validar dimensiones
        if n_vars != self.n_vars:

            raise ValueError(
                f"Número de variables incorrecto. Esperado: {self.n_vars}, Recibido: {n_vars}"
            )

            raise ValueError(
                f"Número de variables incorrecto. Esperado: {self.n_vars}, Recibido: {n_vars}"
            )

        # Asegurar que seq_len es divisible por patch_size
        if seq_len % self.patch_size != 0:
            # Padding con el último valor para hacer seq_len divisible por patch_size
            pad_size = self.patch_size - (seq_len % self.patch_size)
            x = torch.cat([x, x[:, -1:, :].repeat(1, pad_size, 1)], dim=1)
            seq_len = x.shape[1]

        # Aplicar patch embedding
        x = self.patch_embedding(x)  # (batch_size, n_patches, d_model)

        # Añadir posicional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, n_patches, d_model)

        # Proyección final (usar solo el último patch)
        x = self.final_projection(x[:, -1, :])  # (batch_size, pred_len)

        return x

    def fit(
        self,
        train_data: Union[pd.DataFrame, Dict],
        val_data: Optional[Union[pd.DataFrame, Dict]] = None,
        **kwargs,
    ) -> Dict:
        """
        Entrena el modelo PatchTST.

        Args:
            train_data: DataFrame o diccionario con los datos de entrenamiento
            val_data: DataFrame o diccionario opcional con los datos de validación
            **kwargs: Argumentos adicionales para el entrenamiento

        Returns:
            Dict con métricas de entrenamiento
        """
        try:
            # Preparar datos de entrenamiento
            if isinstance(train_data, pd.DataFrame):

                feature_cols = self.config.get(
                    "feature_columns", train_data.columns.tolist()
                )

                feature_cols = self.config.get(
                    "feature_columns", train_data.columns.tolist()
                )

                X_train = train_data[feature_cols].values
            elif isinstance(train_data, dict):
                if "market_data" in train_data and "feature_columns" in train_data:
                    feature_cols = train_data["feature_columns"]
                    if isinstance(train_data["market_data"], pd.DataFrame):
                        X_train = train_data["market_data"][feature_cols].values
                    else:

                        raise ValueError(
                            "market_data debe ser un DataFrame para PatchTST"
                        )

                        raise ValueError(
                            "market_data debe ser un DataFrame para PatchTST"
                        )

                else:
                    raise ValueError(
                        "train_data debe contener 'market_data' y 'feature_columns' cuando es un diccionario"
                    )
            else:
                raise ValueError("train_data debe ser DataFrame o dict")

            # Validar columnas
            if len(feature_cols) != self.n_vars:
                raise ValueError(
                    f"Número de variables incorrecto. Esperado: {self.n_vars}, Recibido: {len(feature_cols)}"
                )

            # Convertir a tensor
            X_train = torch.FloatTensor(X_train).to(self.device)
            # Reshape a (batch, lookback_window, n_vars)
            seq_len = self.lookback_window
            n_vars = self.n_vars
            if X_train.shape[1] != n_vars:
                raise ValueError(
                    f"PatchTST espera {n_vars} columnas, pero el array tiene {X_train.shape[1]}"
                )
            if X_train.shape[0] % seq_len != 0:
                X_train = X_train[: X_train.shape[0] - (X_train.shape[0] % seq_len)]
            X_train = X_train.reshape(-1, seq_len, n_vars)
            # Preparar datos de validación si existen
            if val_data is not None:
                if isinstance(val_data, pd.DataFrame):
                    X_val = val_data[feature_cols].values
                elif isinstance(val_data, dict):
                    if "market_data" in val_data and "feature_columns" in val_data:
                        if isinstance(val_data["market_data"], pd.DataFrame):
                            X_val = val_data["market_data"][feature_cols].values
                        else:

                            raise ValueError(
                                "market_data debe ser un DataFrame para PatchTST"
                            )

                            raise ValueError(
                                "market_data debe ser un DataFrame para PatchTST"
                            )

                    else:
                        raise ValueError(
                            "val_data debe contener 'market_data' y 'feature_columns' cuando es un diccionario"
                        )
                else:
                    raise ValueError("val_data debe ser DataFrame o dict")
                X_val = torch.FloatTensor(X_val).to(self.device)
                if X_val.shape[1] != n_vars:
                    raise ValueError(
                        f"PatchTST espera {n_vars} columnas, pero el array tiene {X_val.shape[1]}"
                    )
                if X_val.shape[0] % seq_len != 0:
                    X_val = X_val[: X_val.shape[0] - (X_val.shape[0] % seq_len)]
                X_val = X_val.reshape(-1, seq_len, n_vars)
            else:
                X_val = None

            # Optimizador

            optimizer = optim.Adam(
                self.parameters(), lr=self.config.get("learning_rate", 0.001)
            )

            optimizer = optim.Adam(
                self.parameters(), lr=self.config.get("learning_rate", 0.001)
            )

            # Entrenamiento
            self.train()
            best_val_loss = float("inf")
            metrics = defaultdict(list)

            for epoch in range(kwargs.get("epochs", 100)):
                # Forward pass
                y_pred = self(X_train)

                # Calcular pérdida
                loss = self.criterion(y_pred, X_train[:, -self.pred_len :, 0])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Guardar métricas
                metrics["train_loss"].append(loss.item())

                # Validación
                if X_val is not None:
                    self.eval()
                    with torch.no_grad():
                        y_val_pred = self(X_val)

                        val_loss = self.criterion(
                            y_val_pred, X_val[:, -self.pred_len :, 0]
                        )

                        val_loss = self.criterion(
                            y_val_pred, X_val[:, -self.pred_len :, 0]
                        )

                        metrics["val_loss"].append(val_loss.item())

                        # Guardar mejor modelo
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.best_state = self.state_dict()

                    self.train()

            # Restaurar mejor modelo
            if hasattr(self, "best_state"):
                self.load_state_dict(self.best_state)

            return dict(metrics)

        except Exception as e:
            logger.error(f"Error en entrenamiento de PatchTST: {str(e)}")
            raise

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            data: Input data of shape (n_samples, lookback_window, n_features)

        Returns:
            Predictions of shape (n_samples, prediction_horizon)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(data).to(self.device)
            predictions = self(x)
            return predictions.cpu().numpy()


class PatchTSTPredictor:
    """
    PatchTST wrapper for cryptocurrency prediction
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        self.config = config
        self.device = device

        # Verificar y derivar parámetros requeridos
        if "n_vars" not in config:
            feature_cols = config.get("feature_columns", [])
            if not feature_cols:
                raise ValueError("Missing 'n_vars' and 'feature_columns'")
            config["n_vars"] = len(feature_cols)

        if "seq_len" not in config or "pred_len" not in config:
            raise ValueError("Missing required sequence parameters")

        # Initialize model
        self.model = PatchTST(config).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.get("T_0", 10), T_mult=config.get("T_mult", 2)
        )

        # Training state
        self.best_val_loss = float("inf")
        self.patience = config.get("patience", 10)
        self.patience_counter = 0

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "learning_rates": [],
        }

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch in train_loader:
                step_metrics = self.train_step(batch)
                train_losses.append(step_metrics["total_loss"])

            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation phase
            val_metrics = self.evaluate(val_loader)
            history["val_loss"].append(val_metrics["val_mse"])
            history["val_rmse"].append(val_metrics["val_rmse"])
            history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping check
            if val_metrics["val_mse"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_mse"]
                self.patience_counter = 0
                # Save best model
                self.save("best_model.pt")
            else:
                self.patience_counter += 1

            # Log progress
            if progress_callback:
                progress = int((epoch + 1) / epochs * 100)
                progress_callback(progress)

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return history

    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()

        total_mse = 0
        total_mae = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)

                predictions = self.model(inputs)

                mse = F.mse_loss(predictions, targets, reduction="sum")
                mae = F.l1_loss(predictions, targets, reduction="sum")

                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += targets.numel()

        return {
            "val_mse": total_mse / total_samples,
            "val_mae": total_mae / total_samples,
            "val_rmse": np.sqrt(total_mse / total_samples),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Get data
        inputs = batch["inputs"].to(self.device)  # [batch_size, seq_len, n_vars]
        targets = batch["targets"].to(self.device)  # [batch_size, pred_len, 1]

        # Verificar dimensiones
        batch_size, seq_len, n_vars = inputs.shape
        if n_vars != self.config["n_vars"]:

            raise ValueError(
                f"Expected {self.config['n_vars']} input variables, got {n_vars}"
            )

            raise ValueError(
                f"Expected {self.config['n_vars']} input variables, got {n_vars}"
            )

        # Forward pass
        predictions = self.model(inputs)  # [batch_size, pred_len, n_vars]

        # Calcular pérdida solo en la columna de precio de cierre

        close_idx = self.config.get(
            "close_idx", 0
        )  # Índice de la columna de precio de cierre

        close_idx = self.config.get(
            "close_idx", 0
        )  # Índice de la columna de precio de cierre

        predictions_close = predictions[:, :, close_idx : close_idx + 1]

        # Calculate losses
        mse_loss = F.mse_loss(predictions_close, targets)
        mae_loss = F.l1_loss(predictions_close, targets)

        # Custom loss for financial data
        direction_weight = torch.where(
            targets > inputs[:, -1:, close_idx : close_idx + 1],
            torch.ones_like(targets) * 1.5,
            torch.ones_like(targets),
        )

        weighted_loss = (direction_weight * (predictions_close - targets) ** 2).mean()

        # Total loss
        total_loss = mse_loss + 0.1 * mae_loss + 0.1 * weighted_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "mse_loss": mse_loss.item(),
            "mae_loss": mae_loss.item(),
            "weighted_loss": weighted_loss.item(),
        }

    def save(self, path: str):
        """Save model"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Obtener un valor del diccionario de configuración de forma segura"""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _build_model(self):
        """Build the PatchTST model architecture"""
        self.patch_len = self._get_config_value("patch_len", 16)
        self.stride = self._get_config_value("stride", 8)

        # Transformer architecture
        self.d_model = self._get_config_value("d_model", 128)
        self.n_heads = self._get_config_value("n_heads", 8)
        self.e_layers = self._get_config_value("e_layers", 3)
        self.d_ff = self._get_config_value("d_ff", 512)
        self.dropout = self._get_config_value("dropout", 0.1)

    def _setup_optimizer(self):
        """Configure optimizer and learning rate scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._get_config_value("learning_rate", 1e-3),
            weight_decay=self._get_config_value("weight_decay", 1e-4),
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self._get_config_value("T_0", 10),
            T_mult=self._get_config_value("T_mult", 2),
        )


class PatchTSTTrainer:
    """Trainer for PatchTST model"""

    def __init__(self, model: PatchTSTPredictor, config: Dict, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device

    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.model.eval()

        total_mse = 0
        total_mae = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)

                predictions = self.model.model(inputs)

                mse = F.mse_loss(predictions, targets, reduction="sum")
                mae = F.l1_loss(predictions, targets, reduction="sum")

                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += targets.numel()

        return {
            "val_mse": total_mse / total_samples,
            "val_mae": total_mae / total_samples,
            "val_rmse": np.sqrt(total_mse / total_samples),
        }
