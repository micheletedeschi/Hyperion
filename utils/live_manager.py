#!/usr/bin/env python3
"""
🎯 LIVE DISPLAY MANAGER MEJORADO
Manager centralizado para prevenir conflictos de Rich Live displays
"""

from rich.live import Live as RichLive
from rich.console import Console
import threading
import time
from typing import Optional, Any

class LiveDisplayManager:
    """Manager centralizado para Rich Live displays"""
    
    _active_live: Optional[RichLive] = None
    _lock = threading.Lock()
    _console = Console()
    
    @classmethod
    def get_live(cls, *args, **kwargs) -> RichLive:
        """Obtener instancia Live única"""
        with cls._lock:
            if cls._active_live is not None:
                # Si hay un Live activo, esperar a que termine
                try:
                    cls._active_live.stop()
                except:
                    pass
                cls._active_live = None
            
            cls._active_live = RichLive(*args, **kwargs)
            return cls._active_live
    
    @classmethod
    def clear_live(cls):
        """Limpiar Live activo"""
        with cls._lock:
            if cls._active_live is not None:
                try:
                    cls._active_live.stop()
                except:
                    pass
                cls._active_live = None
    
    @classmethod
    def is_active(cls) -> bool:
        """Verificar si hay un Live activo"""
        return cls._active_live is not None

class SafeLive:
    """Wrapper seguro para Rich Live"""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._live = None
    
    def __enter__(self):
        self._live = LiveDisplayManager.get_live(*self.args, **self.kwargs)
        return self._live.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._live:
                result = self._live.__exit__(exc_type, exc_val, exc_tb)
            return result
        finally:
            LiveDisplayManager.clear_live()

# Alias para compatibilidad
Live = SafeLive
