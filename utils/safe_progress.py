#!/usr/bin/env python3
"""
🛡️ WRAPPER SEGURO PARA RICH PROGRESS
Previene conflictos de múltiples displays activos
"""

from rich.progress import Progress as RichProgress
from rich.console import Console
import threading
import time

class SafeProgress:
    """Progress wrapper que previene múltiples displays activos y forwarda métodos"""
    
    _active_progress = None
    _lock = threading.Lock()
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._progress = None
        self._is_active = False
        self._entered_progress = None
    
    def __enter__(self):
        with self._lock:
            # Esperar a que cualquier progress activo termine
            while SafeProgress._active_progress is not None:
                time.sleep(0.1)
            
            self._progress = RichProgress(*self.args, **self.kwargs)
            SafeProgress._active_progress = self
            self._is_active = True
            
            # Retornar el objeto progress que tiene todos los métodos
            self._entered_progress = self._progress.__enter__()
            return self._entered_progress
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        result = None
        try:
            if self._progress:
                result = self._progress.__exit__(exc_type, exc_val, exc_tb)
        finally:
            with self._lock:
                if SafeProgress._active_progress == self:
                    SafeProgress._active_progress = None
                self._is_active = False
                self._entered_progress = None
        
        return result

# Reemplazar Progress con SafeProgress
Progress = SafeProgress
