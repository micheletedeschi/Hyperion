#!/usr/bin/env python3
"""
🤖 ENTRENADOR SIMPLIFICADO PARA MODELOS PYTORCH
Módulo optimizado para entrenar modelos PyTorch con datos de criptomonedas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

class PytorchTrainer:
    """Entrenador simplificado para modelos PyTorch"""
    
    def __init__(self, device='mps'):
        self.device = device if torch.backends.mps.is_available() else 'cpu'
        self.scaler = StandardScaler()
        print(f"🔧 PytorchTrainer inicializado con device: {self.device}")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """Preparar datos para entrenamiento PyTorch"""
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1)).to(self.device)
        
        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    def train_pytorch_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray, 
                           epochs: int = 100, batch_size: int = 32, lr: float = 0.001) -> Dict:
        """
        Entrenar modelo PyTorch
        
        Args:
            model: Modelo PyTorch
            X: Features
            y: Target
            epochs: Número de épocas
            batch_size: Tamaño del batch
            lr: Learning rate
            
        Returns:
            Dict con métricas y modelo entrenado
        """
        
        try:
            # Preparar datos
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)
            
            # Mover modelo a device
            model = model.to(self.device)
            
            # Optimizer y loss
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Dataset y DataLoader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            model.train()
            losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                losses.append(avg_loss)
                
                if epoch % 20 == 0:
                    print(f"  Época {epoch}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train).cpu().numpy()
                test_pred = model(X_test).cpu().numpy()
                
                # Métricas
                train_r2 = r2_score(y_train.cpu().numpy(), train_pred)
                test_r2 = r2_score(y_test.cpu().numpy(), test_pred)
                train_mse = mean_squared_error(y_train.cpu().numpy(), train_pred)
                test_mse = mean_squared_error(y_test.cpu().numpy(), test_pred)
            
            result = {
                'success': True,
                'model': model,
                'scaler': self.scaler,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'final_loss': losses[-1] if losses else None,
                'epochs_trained': epochs,
                'device': self.device
            }
            
            test_r2_str = f"{test_r2:.4f}" if isinstance(test_r2, (int, float)) and not (np.isnan(test_r2) or np.isinf(test_r2)) else "N/A"
            print(f"  ✅ Entrenamiento completado - Test R²: {test_r2_str}")
            return result
            
        except Exception as e:
            print(f"  ❌ Error en entrenamiento PyTorch: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': None,
                'device': self.device
            }

def create_and_train_pytorch_model(model_type: str, X: np.ndarray, y: np.ndarray, 
                                  device: str = 'mps') -> Dict:
    """
    Crear y entrenar un modelo PyTorch específico
    
    Args:
        model_type: Tipo de modelo ('mlp', 'lstm', 'transformer', 'cnn')
        X: Features
        y: Target
        device: Device para entrenamiento
        
    Returns:
        Dict con resultado del entrenamiento
    """
    
    from utils.models import SimpleMLP, LSTMRegressor, SimpleTransformer, CNNRegressor
    
    trainer = PytorchTrainer(device=device)
    input_size = X.shape[1]
    
    print(f"🤖 Creando modelo {model_type} con input_size={input_size}")
    
    try:
        # Crear modelo según tipo
        if model_type == 'mlp' or model_type == 'mlp_pytorch':
            model = SimpleMLP(input_size=input_size, hidden_size=128)
            
        elif model_type == 'lstm':
            model = LSTMRegressor(input_size=input_size, hidden_size=64, num_layers=2)
            
        elif model_type == 'transformer':
            # Ajustar input_size para que sea divisible por num_heads
            adjusted_size = input_size
            while adjusted_size % 4 != 0:
                adjusted_size += 1
            model = SimpleTransformer(input_size=adjusted_size, nhead=4, num_layers=2)
            
            # Ajustar X si es necesario
            if adjusted_size != input_size:
                padding = np.zeros((X.shape[0], adjusted_size - input_size))
                X = np.concatenate([X, padding], axis=1)
                print(f"  🔧 Input ajustado de {input_size} a {adjusted_size}")
            
        elif model_type == 'cnn' or model_type == 'cnn_1d':
            model = CNNRegressor(input_size=input_size, num_filters=64, num_layers=3)
            
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        print(f"  📝 Modelo {model_type} creado: {sum(p.numel() for p in model.parameters())} parámetros")
        
        # Entrenar modelo
        result = trainer.train_pytorch_model(model, X, y, epochs=50, batch_size=32)
        result['model_type'] = model_type
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error creando modelo {model_type}: {e}")
        return {
            'success': False,
            'error': str(e),
            'model_type': model_type
        }

def test_pytorch_models():
    """Test rápido de modelos PyTorch"""
    print("🧪 TESTING MODELOS PYTORCH")
    
    # Datos sintéticos
    np.random.seed(42)
    X = np.random.randn(1000, 20)  # 20 features
    y = np.sum(X[:, :5], axis=1) + np.random.randn(1000) * 0.1  # Target con ruido
    
    models_to_test = ['mlp_pytorch', 'lstm', 'transformer', 'cnn_1d']
    results = {}
    
    for model_type in models_to_test:
        print(f"\n🔄 Probando {model_type}...")
        result = create_and_train_pytorch_model(model_type, X, y)
        results[model_type] = result
        
        if result['success']:
            test_r2 = result['test_r2']
            test_r2_str = f"{test_r2:.4f}" if isinstance(test_r2, (int, float)) and not (np.isnan(test_r2) or np.isinf(test_r2)) else "N/A"
            print(f"  ✅ {model_type} - R²: {test_r2_str}")
        else:
            print(f"  ❌ {model_type} - Error: {result['error']}")
    
    return results

if __name__ == "__main__":
    results = test_pytorch_models()
    print(f"\n🎯 Test completado - {sum(1 for r in results.values() if r['success'])}/{len(results)} modelos exitosos")
