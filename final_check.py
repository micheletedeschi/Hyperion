#!/usr/bin/env python3
"""
✅ HYPERION3 - VERIFICACIÓN FINAL PRE-PUBLICACIÓN
Script para verificar que todo está listo para GitHub
"""

import os
import sys
from pathlib import Path

def main():
    print("🔍 HYPERION3 - VERIFICACIÓN FINAL")
    print("=" * 50)
    
    # Archivos críticos
    critical_files = {
        "README.md": "Documentación principal",
        "main.py": "Punto de entrada principal", 
        "example_minimal.py": "Ejemplo funcionando",
        "setup.py": "Configuración de instalación",
        "requirements.txt": "Dependencias completas",
        "requirements-minimal.txt": "Dependencias mínimas",
        "LICENSE": "Licencia Apache 2.0",
        "config.json": "Configuración de ejemplo",
        "CHANGELOG.md": "Historial de cambios",
        "QUICKSTART.md": "Guía de inicio rápido",
        "PRE_LAUNCH_CHECKLIST.md": "Lista de verificación"
    }
    
    print("📁 Verificando archivos críticos:")
    missing_files = []
    for file, desc in critical_files.items():
        if Path(file).exists():
            print(f"✅ {file} - {desc}")
        else:
            print(f"❌ {file} - {desc} - FALTANTE")
            missing_files.append(file)
    
    # Directorios importantes
    important_dirs = [
        "hyperion3/",
        "tests/", 
        "docs/",
        "scripts/",
        "data/",
        "utils/"
    ]
    
    print("\n📂 Verificando directorios:")
    for dir in important_dirs:
        if Path(dir).exists():
            print(f"✅ {dir}")
        else:
            print(f"⚠️  {dir} - Opcional")
    
    # Verificar contenido del README
    print("\n📖 Verificando README:")
    try:
        with open("README.md", "r") as f:
            readme_content = f.read()
            
        checks = [
            ("Advertencia de riesgo al inicio", "🚨 ADVERTENCIA CRÍTICA" in readme_content),
            ("Historia personal", "Hola, soy el creador" in readme_content),
            ("Quick start", "Quick Start" in readme_content),
            ("Ejemplo mínimo", "example_minimal.py" in readme_content),
            ("Licencia Apache", "Apache 2.0" in readme_content),
        ]
        
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"{status} {check_name}")
            
    except Exception as e:
        print(f"❌ Error leyendo README: {e}")
    
    # Resumen final
    print("\n" + "=" * 50)
    print("🎯 RESUMEN FINAL")
    print("=" * 50)
    
    if not missing_files:
        print("🎉 ¡PROYECTO LISTO PARA PUBLICACIÓN!")
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. 🔧 git init (si no lo has hecho)")
        print("2. 📝 git add .")
        print("3. 💬 git commit -m 'Initial commit: Hyperion3 v3.0.0'")
        print("4. 🌐 Crear repositorio en GitHub")
        print("5. 🚀 git push origin main")
        print("6. 📢 ¡Anunciar tu proyecto!")
        
        print("\n💡 TIPS DE MARKETING:")
        print("- Comparte en r/MachineLearning")
        print("- Tweet con hashtags #trading #AI #opensource")
        print("- Post en LinkedIn con tu historia")
        print("- Considera hacer un demo en YouTube")
        
        return True
    else:
        print(f"⚠️  FALTAN {len(missing_files)} archivos críticos:")
        for file in missing_files:
            print(f"   - {file}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 ¡Hyperion está listo para conquistar GitHub!")
    else:
        print("\n🔧 Completa los elementos faltantes y vuelve a ejecutar")
    
    sys.exit(0 if success else 1)
