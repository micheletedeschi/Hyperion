"""
Temporal Fusion Transformer (TFT) Implementation for Hyperion V2
Multi-horizon time series forecasting with interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


class TimeDistributed(nn.Module):
    """Applies a module over temporal dimension"""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        # x shape: [batch_size, time_steps, features]
        batch_size, time_steps, features = x.size()

        # Reshape to [batch_size * time_steps, features]
        x = x.contiguous().view(-1, features)

        # Apply module
        x = self.module(x)

        # Reshape back
        x = x.view(batch_size, time_steps, -1)

        return x


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for feature selection"""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        mult = self.fc2(x)
        return self.dropout(sig * mult)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for non-linear processing"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
        residual: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.residual = residual

        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu1 = nn.ELU()

        # Layer 2
        input_to_layer2 = hidden_size
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size)
            input_to_layer2 += hidden_size

        self.fc2 = nn.Linear(input_to_layer2, hidden_size)
        self.elu2 = nn.ELU()

        # Output
        self.dropout = nn.Dropout(dropout)
        self.gate_fc = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Skip connection
        if residual and input_size != output_size:
            self.skip_fc = nn.Linear(input_size, output_size)
        else:
            self.skip_fc = None

        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, context=None):
        # Layer 1
        hidden = self.fc1(x)
        hidden = self.elu1(hidden)

        # Layer 2 with optional context
        if self.context_size is not None and context is not None:
            context_projection = self.context_fc(context)
            if len(context_projection.shape) == 2:
                context_projection = context_projection.unsqueeze(1)
            hidden = torch.cat([hidden, context_projection], dim=-1)

        hidden = self.fc2(hidden)
        hidden = self.elu2(hidden)

        # Gated output
        gate = torch.sigmoid(self.gate_fc(hidden))
        output = gate * self.fc3(hidden)
        output = self.dropout(output)

        # Residual connection
        if self.residual:
            if self.skip_fc is not None:
                output = output + self.skip_fc(x)
            else:
                output = output + x

        return self.layer_norm(output)


class VariableSelectionNetwork(nn.Module):
    """Variable selection network for identifying important features"""

    def __init__(
        self,
        input_size: int,
        num_inputs: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.hidden_size = hidden_size

        # Variable selection weights
        self.flattened_grn = GatedResidualNetwork(
            input_size=num_inputs * input_size,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout=dropout,
            context_size=context_size,
            residual=False,
        )

        # Variable processing
        self.single_variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                    residual=False,
                )
                for _ in range(num_inputs)
            ]
        )

        # Variable combination
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, context=None):
        # inputs shape: [batch_size, time_steps, num_inputs, input_size]
        batch_size, time_steps, _, _ = inputs.shape

        # Flatten for variable selection
        flattened = inputs.view(batch_size, time_steps, -1)

        # Get variable selection weights
        if context is not None:
            weights = self.flattened_grn(flattened, context)
        else:
            weights = self.flattened_grn(flattened)

        weights = self.softmax(weights).unsqueeze(-1)

        # Process each variable
        processed_inputs = []
        for i in range(self.num_inputs):
            processed = self.single_variable_grns[i](inputs[:, :, i, :])
            processed_inputs.append(processed)

        # Stack processed variables
        processed_inputs = torch.stack(processed_inputs, dim=2)

        # Apply variable selection weights
        outputs = (processed_inputs * weights).sum(dim=2)

        return outputs, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention for temporal relationships"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.shape[:2]

        # Linear projections in batch from d_model => h x d_k
        Q = self.query_projection(query).view(
            batch_size, seq_len, self.n_heads, self.d_k
        )
        K = self.key_projection(key).view(batch_size, -1, self.n_heads, self.d_k)
        V = self.value_projection(value).view(batch_size, -1, self.n_heads, self.d_k)

        # Transpose for attention dot product
        Q = Q.transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.out_projection(context)

        return output, attention_weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon time series forecasting

    Key features:
    - Variable selection networks for feature importance
    - Gated residual networks for non-linear processing
    - Interpretable multi-head attention
    - Quantile outputs for uncertainty estimation
    """

    def __init__(self, config):
        self.config = config
        if torch.backends.mps.is_available():
            self.device = torch.device(
                "mps"
            )  # Metal Performance Shaders para Apple Silicon
        else:
            self.device = torch.device("cpu")

        # Initialize nn.Module before assigning submodules
        super().__init__()

        # Build model layers after device is set
        self._build_model()

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Obtener un valor del diccionario de configuración de forma segura"""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _build_model(self):
        """Build the Temporal Fusion Transformer model architecture"""
        # Model hyperparameters
        self.hidden_size = self._get_config_value("hidden_size", 160)
        self.num_attention_heads = self._get_config_value("attention_heads", 4)
        self.dropout = self._get_config_value("dropout", 0.1)
        self.num_lstm_layers = self._get_config_value("lstm_layers", 2)

        # Feature dimensions
        self.num_static_features = self._get_config_value("num_static_features", 0)
        self.num_time_features = self._get_config_value("num_time_features", 4)
        self.num_inputs = self._get_config_value("num_inputs", 7)  # OHLCV + indicators
        self.num_known_features = self._get_config_value(
            "num_known_features", 2
        )  # Time features

        # Output
        self.prediction_length = self._get_config_value("prediction_length", 24)
        self.quantiles = self._get_config_value("quantiles", [0.1, 0.5, 0.9])
        self.num_quantiles = len(self.quantiles)

        # Static variable selection (if we have static features)
        if self.num_static_features > 0:
            self.static_selection = VariableSelectionNetwork(
                input_size=1,
                num_inputs=self.num_static_features,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
            )

        # Historical variable selection
        self.hist_selection = VariableSelectionNetwork(
            input_size=1,
            num_inputs=self.num_inputs,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size if self.num_static_features > 0 else None,
        )

        # Future variable selection
        self.future_selection = VariableSelectionNetwork(
            input_size=1,
            num_inputs=self.num_known_features,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size if self.num_static_features > 0 else None,
        )

        # LSTM encoder-decoder
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        # Gate for combining static and temporal
        self.static_enrichment_gate = GatedLinearUnit(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

        # Temporal self-attention
        self.temporal_attention = InterpretableMultiHeadAttention(
            d_model=self.hidden_size,
            n_heads=self.num_attention_heads,
            dropout=self.dropout,
        )

        # Position-wise feed-forward
        self.position_wise_ff = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size * 4,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # Output layers
        self.output_layer = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(self.num_quantiles)]
        )

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            past_values: Historical values [batch_size, past_length, num_features]
            past_time_features: Historical time features [batch_size, past_length, num_time_features]
            future_time_features: Future time features [batch_size, pred_length, num_time_features]
            static_features: Static features [batch_size, num_static_features]

        Returns:
            Dictionary with predictions and attention weights
        """
        batch_size = past_values.shape[0]

        # Process static features
        static_context = None
        if static_features is not None and self.num_static_features > 0:
            static_features = static_features.unsqueeze(1).unsqueeze(-1)
            static_encoded, static_weights = self.static_selection(static_features)
            static_context = static_encoded.squeeze(1)

        # Prepare historical inputs
        hist_inputs = past_values.unsqueeze(-1)  # Add feature dimension

        # Historical variable selection
        hist_encoded, hist_weights = self.hist_selection(hist_inputs, static_context)

        # Prepare future inputs (known features)
        future_inputs = future_time_features[:, :, : self.num_known_features].unsqueeze(
            -1
        )

        # Future variable selection
        future_encoded, future_weights = self.future_selection(
            future_inputs, static_context
        )

        # LSTM encoding
        lstm_input = hist_encoded
        if static_context is not None:
            # Enrich with static information
            static_enrichment = self.static_enrichment_gate(static_context).unsqueeze(1)
            lstm_input = lstm_input + static_enrichment

        encoder_output, (hidden, cell) = self.lstm_encoder(lstm_input)

        # LSTM decoding with future features
        decoder_input = future_encoded
        if static_context is not None:
            decoder_input = decoder_input + static_enrichment

        decoder_output, _ = self.lstm_decoder(decoder_input, (hidden, cell))

        # Temporal attention
        attention_output, attention_weights = self.temporal_attention(
            decoder_output, encoder_output, encoder_output
        )

        # Residual connection and layer norm
        temporal_features = decoder_output + attention_output

        # Position-wise feed-forward
        output = self.position_wise_ff(temporal_features)

        # Generate quantile predictions
        quantile_outputs = []
        for i, quantile_layer in enumerate(self.output_layer):
            quantile_pred = quantile_layer(output).squeeze(-1)
            quantile_outputs.append(quantile_pred)

        # Stack quantile predictions
        predictions = torch.stack(quantile_outputs, dim=-1)

        return {
            "predictions": predictions,  # [batch_size, pred_length, num_quantiles]
            "attention_weights": attention_weights,
            "variable_weights": {
                "historical": hist_weights,
                "future": future_weights,
                "static": static_weights if static_features is not None else None,
            },
        }

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Quantile loss function

        Args:
            predictions: Model predictions [batch_size, pred_length, num_quantiles]
            targets: True values [batch_size, pred_length]

        Returns:
            Quantile loss
        """
        targets = targets.unsqueeze(-1).expand_as(predictions)

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))

        loss = torch.cat(losses, dim=-1)
        return loss.mean()

    def predict(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with the model

        Returns dictionary with:
        - predictions: Point predictions (median)
        - prediction_intervals: Confidence intervals
        - attention_weights: Attention weights for interpretability
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(
                past_values, past_time_features, future_time_features, static_features
            )

        predictions = outputs["predictions"].cpu().numpy()

        # Extract median prediction and confidence intervals
        median_idx = self.quantiles.index(0.5)
        point_predictions = predictions[..., median_idx]

        # Get prediction intervals
        lower_idx = 0  # 0.1 quantile
        upper_idx = -1  # 0.9 quantile
        prediction_intervals = np.stack(
            [predictions[..., lower_idx], predictions[..., upper_idx]], axis=-1
        )

        return {
            "predictions": point_predictions,
            "prediction_intervals": prediction_intervals,
            "attention_weights": outputs["attention_weights"].cpu().numpy(),
            "variable_importance": {
                k: v.cpu().numpy() if v is not None else None
                for k, v in outputs["variable_weights"].items()
            },
        }


class TFTCryptoPredictor(TemporalFusionTransformer):
    """
    TFT specifically configured for cryptocurrency trading
    """

    def __init__(self, config: Dict, device: str = "cpu"):
        # Add crypto-specific configuration
        crypto_config = {
            "num_inputs": 10,  # OHLCV + technical indicators
            "num_time_features": 4,  # Hour, day, week, month encoding
            "num_static_features": 3,  # Exchange, symbol category, market cap tier
            "hidden_size": 256,
            "attention_heads": 8,
            "lstm_layers": 2,
            "dropout": 0.2,
            "prediction_length": 24,  # 24 hours ahead
            "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            **config,
        }

        super().__init__(crypto_config)

        # Additional crypto-specific layers
        self.regime_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),  # Bull, Bear, Sideways
        )

        self.volatility_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Ensure positive volatility
        )

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with crypto-specific outputs"""

        # Get base TFT outputs
        base_outputs = super().forward(
            past_values, past_time_features, future_time_features, static_features
        )

        # We need access to the hidden features from the decoder, but they're not exposed
        # As a workaround, let's re-compute the encoder to get hidden features
        batch_size = past_values.shape[0]

        # Process static features (same as parent)
        static_context = None
        if static_features is not None and self.num_static_features > 0:
            static_features_expanded = static_features.unsqueeze(1).unsqueeze(-1)
            static_encoded, _ = self.static_selection(static_features_expanded)
            static_context = static_encoded.squeeze(1)

        # Get historical encoded features 
        hist_inputs = past_values.unsqueeze(-1)
        hist_encoded, _ = self.hist_selection(hist_inputs, static_context)

        # Apply static enrichment if available
        lstm_input = hist_encoded
        if static_context is not None:
            static_enrichment = self.static_enrichment_gate(static_context).unsqueeze(1)
            lstm_input = lstm_input + static_enrichment

        # Get LSTM encoder output for pooling
        encoder_output, _ = self.lstm_encoder(lstm_input)
        
        # Pool temporal features (mean over sequence length)
        pooled_features = encoder_output.mean(dim=1)  # [batch_size, hidden_size]
        
        # Ensure correct batch dimension
        if pooled_features.dim() == 1:
            pooled_features = pooled_features.unsqueeze(0)

        # Predict market regime
        regime_logits = self.regime_classifier(pooled_features)
        regime_probs = F.softmax(regime_logits, dim=-1)

        # Predict volatility
        volatility = self.volatility_predictor(pooled_features)

        # Combine outputs
        outputs = {
            **base_outputs,
            "regime_probs": regime_probs,
            "predicted_volatility": volatility,
        }

        return outputs
