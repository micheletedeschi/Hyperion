#!/usr/bin/env python3
"""
🤖 MODELOS PYTORCH PARA HYPERION3
Definición centralizada de todas las arquitecturas de redes neuronales
"""

from typing import Optional, Union, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    PYTORCH_AVAILABLE = False

class SimpleMLP(nn.Module):
    """MLP simple optimizado para velocidad"""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class DeepMLP(nn.Module):
    """MLP profunda con múltiples capas"""
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class LSTMRegressor(nn.Module):
    """LSTM para regresión con protección contra errores MPS"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, force_cpu: bool = False):
        super().__init__()
        self.force_cpu = force_cpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Crear LSTM con validación de dimensiones
        if num_layers < 2:
            num_layers = 2
            print(f"⚠️ LSTM: num_layers ajustado a {num_layers}")
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Validar dimensiones de entrada más estrictamente
        if x.ndim == 2:
            # Convertir (batch, features) a (batch, seq_len=1, features)
            x = x.unsqueeze(1)
        elif x.ndim == 3:
            # Ya está en formato correcto (batch, seq_len, features)
            pass
        else:
            raise ValueError(f"Input debe ser 2D o 3D, recibido: {x.shape}")
        
        # LSTM forward con manejo de errores para MPS
        try:
            lstm_out, _ = self.lstm(x)
            # Tomar la última salida de la secuencia
            last_output = lstm_out[:, -1, :]
            return self.linear(last_output)
        except RuntimeError as e:
            if "MPS" in str(e) and not self.force_cpu:
                print(f"⚠️ Error MPS en LSTM: {e}")
                print("🔄 Reintentando en CPU...")
                # Mover a CPU y procesar
                x_cpu = x.cpu()
                self.cpu()
                lstm_out, _ = self.lstm(x_cpu)
                last_output = lstm_out[:, -1, :]
                result = self.linear(last_output)
                return result
            else:
                raise e

class SimpleTransformer(nn.Module):
    """Transformer simple para series temporales"""
    
    def __init__(self, input_size: int, nhead: int = 4, 
                 num_layers: int = 2, dim_feedforward: int = 512):
        super().__init__()
        
        # Ajustar input_size para que sea divisible por nhead
        while input_size % nhead != 0:
            input_size += 1
        
        self.model_type = 'Transformer'
        self.input_projection = nn.Linear(input_size, input_size)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
        self.decoder = nn.Linear(input_size, 1)
    
    def forward(self, src):
        if src.ndim == 2:
            # Añadir dimensión de secuencia
            src = src.unsqueeze(1)
        
        src = self.input_projection(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Global average pooling
        return self.decoder(output)

class CNNRegressor(nn.Module):
    """CNN 1D para series temporales"""
    
    def __init__(self, input_size: int, num_filters: int = 64, 
                 kernel_size: int = 3, num_layers: int = 3):
        super().__init__()
        
        layers = []
        in_channels = 1
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters),
                nn.Dropout(0.2)
            ])
            in_channels = num_filters
            if i < num_layers - 1:  # No max pooling en la última capa
                layers.append(nn.MaxPool1d(2))
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_filters, 1)
    
    def forward(self, x):
        # Reshape para CNN: (batch, channels, sequence)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Añadir dimensión de canal
        elif x.ndim == 3 and x.size(1) != 1:
            x = x.transpose(1, 2)  # (batch, features, seq) -> (batch, seq, features)
            x = x.unsqueeze(1)  # Añadir dimensión de canal
        
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Eliminar dimensión de secuencia
        return self.classifier(x)

def create_pytorch_models(input_size: int) -> dict:
    """
    Crear todos los modelos PyTorch disponibles
    
    Args:
        input_size: Número de features de entrada
        
    Returns:
        Dict con modelos inicializados
    """
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch no disponible - no se pueden crear modelos")
        return {}
    
    models = {}
    
    try:
        # 1. MLP Simple (OPTIMIZADO PARA VELOCIDAD)
        models['SimpleMLP'] = SimpleMLP(input_size)
        
        # 2. MLP Profunda (OPTIMIZADA)
        models['DeepMLP'] = DeepMLP(input_size)
        
        # 3. LSTM (CON PROTECCIÓN CONTRA ERRORES MPS)
        # Verificar si debemos forzar CPU para LSTM
        force_cpu_for_lstm = False
        if torch.backends.mps.is_available():
            try:
                # Test rápido para ver si MPS funciona con LSTM
                test_lstm = nn.LSTM(input_size, 32, batch_first=True)
                test_input = torch.randn(1, 5, input_size)
                if torch.backends.mps.is_available():
                    test_lstm.to('mps')
                    test_input = test_input.to('mps')
                    _ = test_lstm(test_input)
            except Exception:
                force_cpu_for_lstm = True
                print("⚠️ MPS no compatible con LSTM, usando CPU")
        
        models['LSTM'] = LSTMRegressor(input_size, force_cpu=force_cpu_for_lstm)
        
        # 4. Transformer Simple
        models['SimpleTransformer'] = SimpleTransformer(input_size)
        
        # 5. CNN 1D
        models['CNN1D'] = CNNRegressor(input_size)
        
        print(f"✅ Creados {len(models)} modelos PyTorch exitosamente")
        
    except Exception as e:
        print(f"❌ Error creando modelos PyTorch: {e}")
    
    return models

def train_pytorch_model(model, X_train, y_train, X_val, y_val, 
                       model_name: str, epochs: int = 30, 
                       optimized_params: Optional[dict] = None,
                       device: Optional[torch.device] = None) -> dict:
    """
    Entrenar modelo PyTorch con configuración optimizada
    
    Args:
        model: Modelo PyTorch a entrenar
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación  
        model_name: Nombre del modelo para logging
        epochs: Número de epochs
        optimized_params: Parámetros optimizados (opcional)
        device: Device a usar (opcional)
        
    Returns:
        Dict con modelo entrenado y métricas
    """
    if not PYTORCH_AVAILABLE:
        return {'error': 'PyTorch no disponible'}
    
    try:
        # Configurar parámetros
        lr = optimized_params.get('lr', 0.005) if optimized_params else 0.005
        batch_size = optimized_params.get('batch_size', 256) if optimized_params else 256
        weight_decay = optimized_params.get('weight_decay', 0.01) if optimized_params else 0.01
        
        # Configurar device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        # Manejar LSTM con MPS que puede fallar
        if "LSTM" in model_name and device.type == "mps":
            try:
                model.to(device)
                test_input = torch.randn(1, 1, X_train.shape[1]).to(device)
                _ = model(test_input)
            except Exception:
                print(f"⚠️ {model_name}: MPS falló, usando CPU")
                device = torch.device("cpu")
                model.to(device)
        else:
            model.to(device)
        
        # Convertir datos a tensores
        X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train).reshape(-1, 1).to(device)
        X_val_tensor = torch.FloatTensor(X_val.values if hasattr(X_val, 'values') else X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val).reshape(-1, 1).to(device)
        
        # Crear DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Configurar optimizador y loss
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Entrenamiento
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            # Validación
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"  Early stopping en epoch {epoch+1}")
                    break
            
            model.train()
        
        # Evaluación final
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).cpu().numpy().flatten()
            val_pred = model(X_val_tensor).cpu().numpy().flatten()
        
        # Calcular métricas
        from sklearn.metrics import r2_score, mean_squared_error
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        return {
            'model': model,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'val_mse': val_mse,
            'predictions': val_pred,
            'type': 'pytorch',
            'device': str(device)
        }
        
    except Exception as e:
        return {'error': f'Error entrenando {model_name}: {str(e)}'}

def create_sequences(X, y, seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crear secuencias para modelos que requieren datos secuenciales
    
    Args:
        X: Features
        y: Target
        seq_len: Longitud de la secuencia
        
    Returns:
        Tuple con (X_sequences, y_sequences)
    """
    if len(X) <= seq_len:
        raise ValueError(f"Datos insuficientes: {len(X)} <= {seq_len}")
    
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:(i + seq_len)])
        y_seq.append(y[i + seq_len])
    
    return np.array(X_seq), np.array(y_seq)

def create_ensemble_models(device="cpu", n_jobs=8):
    """
    Crear colección de modelos de ensemble para entrenamiento
    
    Args:
        device: Dispositivo para PyTorch (cpu, cuda, mps)
        n_jobs: Número de trabajos paralelos
        
    Returns:
        Diccionario con modelos configurados
    """
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor, 
        ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
    )
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.neural_network import MLPRegressor
    
    import xgboost as xgb
    import lightgbm as lgb
    
    try:
        import catboost as cb
        CATBOOST_AVAILABLE = True
    except ImportError:
        cb = None
        CATBOOST_AVAILABLE = False
    
    models = {}
    
    # 🌲 MODELOS SKLEARN BÁSICOS
    models['random_forest'] = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        n_jobs=n_jobs,
        random_state=42
    )
    
    models['gradient_boosting'] = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    models['extra_trees'] = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=10,
        n_jobs=n_jobs,
        random_state=42
    )
    
    # 📈 MODELOS LINEALES
    models['ridge'] = Ridge(alpha=1.0, random_state=42)
    models['lasso'] = Lasso(alpha=1.0, random_state=42)
    models['elastic_net'] = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    
    # 🧠 MLP CLÁSICO
    models['mlp_sklearn'] = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    # 🚀 MODELOS ENSEMBLE AVANZADOS
    models['xgboost'] = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',
        n_jobs=n_jobs,
        random_state=42
    )
    
    models['lightgbm'] = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        n_jobs=n_jobs,
        random_state=42,
        verbose=-1
    )
    
    if CATBOOST_AVAILABLE:
        models['catboost'] = cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            thread_count=n_jobs,
            random_seed=42,
            verbose=False
        )
    
    # 🤖 MODELOS PYTORCH (si está disponible y device != cpu)
    if PYTORCH_AVAILABLE and device != "cpu":
        try:
            # Estos se crearán dinámicamente cuando se necesiten
            # porque requieren conocer input_size
            models['mlp_pytorch'] = 'pytorch_mlp'
            models['lstm'] = 'pytorch_lstm'
            models['transformer'] = 'pytorch_transformer'
            models['cnn_1d'] = 'pytorch_cnn'
        except Exception as e:
            print(f"⚠️ Error configurando modelos PyTorch: {e}")
    
    # 🎯 ENSEMBLE VOTING
    if len(models) >= 3:
        # Seleccionar mejores modelos para voting
        base_models = [
            ('rf', models['random_forest']),
            ('xgb', models['xgboost']),
            ('lgb', models['lightgbm'])
        ]
        
        if CATBOOST_AVAILABLE:
            base_models.append(('cb', models['catboost']))
        
        models['voting_ensemble'] = VotingRegressor(
            estimators=base_models,
            n_jobs=n_jobs
        )
    
    print(f"✅ Creados {len(models)} modelos:")
    for category, model_list in [
        ('Sklearn', ['random_forest', 'gradient_boosting', 'extra_trees']),
        ('Linear', ['ridge', 'lasso', 'elastic_net']),
        ('Ensemble', ['xgboost', 'lightgbm', 'catboost']),
        ('Neural', ['mlp_sklearn', 'mlp_pytorch', 'lstm', 'transformer']),
        ('Meta', ['voting_ensemble'])
    ]:
        available = [m for m in model_list if m in models]
        if available:
            print(f"   {category}: {', '.join(available)}")
    
    return models

def create_model_factory(device="cpu", n_jobs=8):
    """
    Factory function para crear modelos
    Alias para create_ensemble_models para compatibilidad
    """
    return create_ensemble_models(device=device, n_jobs=n_jobs)
