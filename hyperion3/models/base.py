"""Base model class for Hyperion models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Base class for all Hyperion models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        pass
        
    @abstractmethod
    def predict(self, *args, **kwargs):
        """Make predictions with the model."""
        pass
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.copy()
        
    def set_config(self, config: Dict[str, Any]):
        """Update model configuration."""
        self.config.update(config)


class BaseTradingModel(BaseModel):
    """Base class for trading models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    @abstractmethod
    def generate_signals(self, data: torch.Tensor) -> torch.Tensor:
        """Generate trading signals."""
        pass
        
    @abstractmethod
    def calculate_returns(self, signals: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        """Calculate returns from signals and prices."""
        pass
