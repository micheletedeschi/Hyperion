#!/usr/bin/env python3
"""
🚀 HYPERION3 - PUNTO DE ENTRADA PRINCIPAL
Launcher principal que redirige al sistema profesional
Versión 3.0 - Arquitectura Modular Profesional
"""

import sys
import os
from pathlib import Path

def main():
    """Punto de entrada principal que redirige al sistema profesional"""
    print("🚀 Iniciando Hyperion3...")
    print("📍 Redirigiendo al sistema profesional...")
    
    # Importar y ejecutar el sistema profesional
    try:
        from main_professional import main as professional_main
        professional_main()
    except ImportError as e:
        print(f"❌ Error al importar sistema profesional: {e}")
        print("🔧 Asegúrate de que todas las dependencias estén instaladas:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
