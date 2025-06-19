# 🚀 Guía de Inicio Rápido - Hyperion3

## ⚡ Instalación Express

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/hyperion3.git
cd hyperion3

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. ¡Lanzar Hyperion!
python main.py
```

## 🍎 Instalación para Apple Silicon (M1/M2/M3)

```bash
# Usar el script de instalación optimizado
chmod +x install_mac.sh
./install_mac.sh
```

## 🎯 Primer Experimento (5 minutos)

### 1. Configuración Básica
Edita `config.json` para tu primer experimento:

```json
{
  "data": {
    "symbols": ["BTC/USDT"],
    "start_date": "2024-01-01",
    "end_date": "2024-06-01"
  },
  "models": {
    "enabled_models": ["lightgbm", "xgboost"]
  }
}
```

### 2. Ejecutar Pipeline Completo
```bash
python main.py
```

Selecciona del menú:
- `1` → 🤖 MODELOS → `e1` (LightGBM) → Enter
- `4` → 📊 ANÁLISIS → Ver resultados

### 3. Entrenar Todos los Modelos
```bash
python main.py
```
- `1` → 🤖 MODELOS → `all` → Enter (¡Toma un café! ☕)

## 🎮 Comandos Rápidos del Menú

### Entrenar Modelos Específicos
- `s1, s2, s3...` → Modelos sklearn (RandomForest, etc.)
- `e1, e2, e3...` → Modelos ensemble (XGBoost, LightGBM, CatBoost)
- `p1, p2, p3...` → Modelos PyTorch (MLP, LSTM, Transformer)
- `a1, a2...` → Modelos AutoML (FLAML, Optuna)

### Entrenar por Categoría
- `sklearn` → Todos los modelos sklearn
- `ensemble` → Todos los modelos ensemble
- `pytorch` → Todos los modelos PyTorch
- `automl` → Todos los modelos AutoML
- `all` → ¡Todos los modelos! 🚀

### Crear Ensembles
```bash
python main.py
# Menú → 3 → 🎭 ENSEMBLES
# Seleccionar método: voting, weighted, stacking, bagging
```

## 📊 Estructura de Resultados

Después de entrenar, encontrarás:

```
results/
├── models/           # Modelos entrenados
├── predictions/      # Predicciones
├── backtests/       # Resultados de backtesting
├── plots/           # Gráficos generados
└── reports/         # Reportes en formato PDF/HTML

mlops/
├── experiments/     # Experimentos MLflow
├── artifacts/       # Artefactos del modelo
└── metrics/         # Métricas detalladas
```

## 🎯 Casos de Uso Rápidos

### 💡 Caso 1: Comparar Modelos Rápidamente
```bash
# Configurar modelos ligeros en config.json
{
  "models": { "enabled_models": ["lightgbm", "xgboost", "catboost"] },
  "optimization": { "time_budget": 300 }  # 5 minutos
}

# Ejecutar y comparar
python main.py → 1 → all
```

### 🧠 Caso 2: Probar Reinforcement Learning
```bash
# En config.json
{
  "models": { "enabled_models": ["sac", "td3"] },
  "training": { "epochs": 50 }  # Entrenamientos cortos
}

# Ejecutar agentes RL
python main.py → 1 → p4, p5
```

### 🔮 Caso 3: Usar Transformers de Última Generación
```bash
# En config.json
{
  "models": { "enabled_models": ["patchtst", "tft"] },
  "data": { "lookback_window": 168 }  # 1 semana de datos
}

# Entrenar Transformers
python main.py → 1 → p1, p2
```

## ❓ Solución de Problemas Rápidos

### Error de Memoria
```bash
# Reducir batch_size en config.json
{
  "training": { "batch_size": 16 }
}
```

### Error de GPU
```bash
# Forzar CPU en caso de problemas con GPU
export CUDA_VISIBLE_DEVICES=""
python main.py
```

### Dependencias Faltantes
```bash
# Reinstalar dependencias
pip install -r requirements.txt --upgrade
```

## 🚨 Recordatorio Importante

> ⚠️ **Hyperion es una herramienta de INVESTIGACIÓN y APRENDIZAJE**
> 
> - Los backtests NO garantizan resultados futuros
> - NUNCA inviertas dinero que no puedas permitirte perder
> - Siempre practica primero en testnet/paper trading

## 🎉 ¡Listo para Despegar!

¡Ya tienes todo lo necesario para empezar tu viaje con Hyperion! 

**Próximos pasos sugeridos:**
1. Experimenta con diferentes modelos
2. Prueba distintas configuraciones
3. Analiza los resultados en detalle
4. Lee la documentación completa en `docs/`
5. ¡Comparte tus resultados con la comunidad!

---

**¿Necesitas ayuda?** 
- 📖 Lee `README.md` para la historia completa
- 🔍 Revisa `docs/` para guías detalladas
- 🐛 Reporta issues en GitHub
- 💬 Únete a la discusión de la comunidad

*¡Que el código esté contigo!* 🚀
