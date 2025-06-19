# Hyperion: Un Framework de Trading Algorítmico Forjado en la Obsesión

![Version](https://img.shields.io/badge/version-3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![Trading](https://img.shields.io/badge/trading-algorithmic-gold.svg)
![AI](https://img.shields.io/badge/AI-enabled-purple.svg)
![MLOps](https://img.shields.io/badge/MLOps-integrated-orange.svg)

> "Hyperion no es solo código. Es la respuesta a un desafío que me cautivó por completo. Es la herramienta que desearía haber tenido cuando empecé."

## 🚨 ADVERTENCIA CRÍTICA - LEE ESTO PRIMERO

<table>
<tr>
<td>
<h3>⚠️ RIESGO FINANCIERO EXTREMO</h3>

**Hyperion es una herramienta de INVESTIGACIÓN y APRENDIZAJE, NO una máquina de hacer dinero.**

- 🔮 **Los backtests NO garantizan rendimientos futuros**
- 🌪️ **El mercado real es caótico e impredecible**
- 💰 **NUNCA inviertas dinero que no puedas permitirte perder**
- ⚖️ **Úsalo bajo tu COMPLETA responsabilidad**
- 🧪 **Practica SIEMPRE en paper trading primero**

**El trading algorítmico requiere conocimiento profundo, gestión de riesgos y comprensión de que las pérdidas son parte del juego.**
</td>
</tr>
</table>

## 💫 La Historia Detrás de Hyperion

Hola, soy el creador de Hyperion. Permíteme contarte una historia.

Siempre soñé con crear un bot de trading que operara de forma autónoma, una máquina inteligente capaz de navegar los turbulentos mares del mercado. Cuando empecé, mi ingenuidad me hizo pensar que sería una tarea fácil. Creí que en un par de fines de semana tendría algo funcionando.

**Qué equivocado estaba.**

Pronto me di de bruces con la realidad: construir un bot de vanguardia, partiendo desde cero y en solitario, es un desafío colosal. Me sumergí en un océano de papers de investigación, arquitecturas de modelos y técnicas de preprocesamiento, y cada nueva capa de conocimiento revelaba diez más que desconocía.

Al buscar ayuda, me di cuenta de que el panorama de los bots de trading públicos y gratuitos era desolador. La mayoría eran demasiado simples, cajas negras sin flexibilidad, o directamente inútiles. Sentí una frustración inmensa. ¿Cómo podía alguien, con más ganas que experiencia, empezar en este mundo?

Fue en ese momento de frustración y desafío que **nació Hyperion**. Decidí que si la herramienta que necesitaba no existía, la construiría yo mismo.

Este proyecto es el resultado de incontables horas de trabajo, de prueba y error, de pequeños fracasos y grandes victorias. Mi esperanza más sincera es que Hyperion te ahorre parte de ese difícil camino y te dé el poder para que tú también puedas transformar tus ideas en estrategias reales.

## 🚀 Quick Start

### ⚡ Ejemplo Súper Rápido (2 minutos)
```bash
# 1. Ejemplo mínimo sin configuración compleja
pip install numpy pandas matplotlib
python example_minimal.py
```

### 🎯 Inicio Completo (5 minutos)
```bash
# 1. Inicia el sistema profesional (RECOMENDADO)
pip install -r requirements.txt
python main.py

# 2. O accede directamente a la interfaz profesional
python main_professional.py

# 3. Valida la estructura modular
python test_modular_structure.py
```

## 🏗️ El Pipeline de Hyperion: Anatomía de una Idea

Hyperion está diseñado como un **pipeline modular y automatizado** que transforma datos de mercado en bruto en estrategias de trading robustas y validadas. Todo el proceso se controla desde un único archivo de configuración (`config.json`), simplificando un flujo de trabajo que de otro modo sería extremadamente complejo.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Config    │ -> │    Data     │ -> │Preprocessor │ -> │   Model     │
│ (config.json)│    │ Downloader  │    │& Features   │    │  Trainer    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                  │
                                                                  v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    MLOps    │ <- │ Backtester  │ <- │  Ensemble   │ <- │ Hyperopt    │
│  Tracking   │    │& Validation │    │ Creation    │    │ (FLAML)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 🧠 Etapa 1: Configuración (config.json)
**El Cerebro de la Operación**: Aquí defines todo: el par de criptomonedas (ej. BTC/USDT), el intervalo de tiempo (1h, 4h, 1d), las fechas de inicio y fin, y lo más importante: la lista de modelos que quieres poner a prueba y la configuración del optimizador.

### 📊 Etapa 2: Adquisición de Datos (hyperion3/data/downloader.py)
Hyperion se conecta a las fuentes de datos y descarga el historial de precios OHLCV (Open, High, Low, Close, Volume) para el activo que especificaste. Los datos se guardan localmente para un acceso rápido y reutilización.

### ⚙️ Etapa 3: Preprocesamiento e Ingeniería de Características
**Limpieza y Preparación**: Los datos crudos se limpian de valores faltantes y se preparan para el análisis.

**Creación de Inteligencia**: ¡Aquí ocurre la magia! Hyperion no solo usa el precio. Genera un arsenal de **más de 100 características (features)** para darle a los modelos un contexto profundo del mercado:

- **📈 Indicadores de Momentum**: RSI, Estocástico, MACD, Williams %R
- **📊 Indicadores de Tendencia**: Medias Móviles (SMA, EMA), Bandas de Bollinger, ADX, Ichimoku Cloud, Vortex
- **� Indicadores de Volatilidad**: ATR (Average True Range), Keltner Channels
- **📊 Análisis de Volumen**: On-Balance Volume (OBV)
- **🕯️ Patrones de Velas Japonesas**: Doji, Engulfing, Hammer, etc.
- **🎭 Aumentación de Datos**: Para evitar sobreajuste, se crean variaciones sintéticas de los datos

### 🔬 Etapa 4: Optimización de Hiperparámetros y Entrenamiento
**La Búsqueda de la Perfección (AutoML)**: Un modelo sin los hiperparámetros correctos es ineficiente. Hyperion integra **FLAML**, una potente y ligera librería de AutoML de Microsoft. En lugar de una búsqueda a ciegas, FLAML utiliza algoritmos de búsqueda inteligentes para explorar eficientemente el espacio de posibles configuraciones.

### 🤝 Etapa 5: Creación de Ensambles
**La Sabiduría de la Multitud**: En lugar de confiar en un solo "genio", Hyperion puede combinar las predicciones de tus mejores modelos en un ensamble. Esto a menudo conduce a decisiones más estables y robustas.

### 🧪 Etapa 6: Backtesting Riguroso
**La Prueba de Fuego**: El Backtester simula cómo habría funcionado tu estrategia en el pasado, operación por operación. Te proporciona métricas críticas como el Retorno Total, el Sharpe Ratio, el Máximo Drawdown y la tasa de aciertos.

### 📈 Etapa 7: Análisis y MLOps
**Reproducibilidad y Transparencia**: Cada detalle de tu experimento se registra automáticamente con MLflow. Esto te permite comparar diferentes enfoques y volver a cualquier punto de tu investigación sin perderte.

## 🤖 El Arsenal de Modelos: Un Espectro Completo de Inteligencia

Hyperion integra una biblioteca de modelos excepcionalmente diversa, permitiéndote abordar el problema desde múltiples ángulos. Todos los modelos son instanciados a través de `hyperion3/models/model_factory.py`.

### 📊 1. Modelos Clásicos y Estadísticos
- **Prophet**: Desarrollado por Facebook, excelente para capturar estacionalidades y tendencias de forma robusta

### 🌳 2. Modelos de Machine Learning (Basados en Árboles)
La columna vertebral de la ciencia de datos moderna. Son rápidos, interpretables y muy potentes:

- **🚀 LightGBM**: La opción más rápida. Utiliza crecimiento leaf-wise extremadamente eficiente
- **🏆 XGBoost**: El estándar de oro. Famoso por su rendimiento y regularización anti-overfitting
- **🎯 CatBoost**: Especialmente diseñado para manejar datos de forma eficiente, muy robusto
- **🌲 RandomForest y ExtraTrees**: Ensambles de múltiples árboles para mejorar robustez

### 🧠 3. Modelos de Deep Learning para Series Temporales
Diseñados específicamente para capturar dependencias temporales complejas:

- **📈 N-BEATS**: Descompone la serie temporal en componentes interpretables
- **⚡ N-HITS**: Evolución de N-BEATS con mejor eficiencia y espectro de frecuencias
- **🔥 TFT (Temporal Fusion Transformer)**: Fusiona diferentes tipos de datos con mecanismos de atención
- **💎 PatchTST (Transformer)**: ¡La joya de la corona! Basado en la arquitectura Transformer de Google, procesa la serie temporal en "parches" para capturar relaciones a corto y largo plazo

### 🎮 4. Aprendizaje por Refuerzo (Reinforcement Learning)
**El cambio de paradigma más radical**. En lugar de predecir el futuro, los agentes aprenden a actuar para maximizar recompensas:

- **🎭 SAC (Soft Actor-Critic)**: Algoritmo moderno, eficiente y muy estable
- **🎯 TD3 (Twin Delayed DDPG)**: Robusto, diseñado para mitigar sobreestimación de valores
- **🌈 Rainbow DQN**: Mejora del clásico DQN que combina múltiples técnicas

**¿Cómo funciona el RL?** El agente es el "trader". Observa el mercado y decide acciones (comprar/vender/mantener). Si gana, recibe recompensa positiva. Después de miles de simulaciones, aprende una política para maximizar ganancias. Es lo más cercano a enseñar a una IA a "pensar" como un trader.

## ✨ Interfaz Profesional

Hyperion3 cuenta con una **interfaz profesional completa**:

### 🎯 **Características del Menú Principal**
- **🤖 MODELS**: Entrena modelos individuales por categoría (sklearn, ensemble, pytorch, automl)
- **🎯 HYPERPARAMETERS**: Optimización automática y manual de hiperparámetros
- **🎭 ENSEMBLES**: Crea y gestiona ensambles (voting, weighted, stacking, bagging)
- **📊 ANALYSIS**: Análisis completo de resultados y métricas de rendimiento
- **⚙️ CONFIGURATION**: Gestión de configuración del sistema
- **📈 MONITORING**: Monitoreo del sistema en tiempo real

### 🔧 **Opciones de Entrenamiento Modular**
```bash
# Entrena modelos específicos
s1, s2, s3...    # modelos sklearn (Random Forest, Gradient Boosting, etc.)
e1, e2, e3...    # modelos ensemble (XGBoost, LightGBM, CatBoost)
p1, p2, p3...    # modelos pytorch (MLP, LSTM, Transformer)
a1, a2...        # modelos automl (FLAML, Optuna)

# Entrena por categoría
sklearn, ensemble, pytorch, automl

# Entrena todos los modelos
all
```

## 🚀 Instalación

### Instalación Rápida
```bash
pip install -r requirements.txt
```

### 🍎 Para usuarios de Apple Silicon
```bash
./install_mac.sh
```

### 📋 Requisitos
- **Python 3.8+**
- **SO Unix** (Linux o macOS recomendado)
- **Opcional**: GPU con CUDA para modelos de deep learning

Ver [docs/INSTALLATION.md](docs/INSTALLATION.md) para instrucciones detalladas.

## 🏗️ Arquitectura del Proyecto

El código está organizado en paquetes modulares:

- **`hyperion3/models/`** – transformers y agentes de RL
- **`hyperion3/training/`** – loops de entrenamiento y callbacks
- **`hyperion3/evaluations/`** – backtester y métricas financieras
- **`hyperion3/optimization/`** – utilidades AutoML con FLAML
- **`hyperion3/data/`** – descargadores, preprocessing e ingeniería de features
- **`scripts/deployment/`** – motor de trading en vivo y monitoreo
- **`scripts/`** – comandos auxiliares para entrenamiento y testing
- **`docs/`** – documentación adicional

### 🎨 Características Profesionales
- **🎨 Rich UI**: Interfaz de consola hermosa con la librería Rich
- **🔧 Diseño Modular**: Separación limpia en módulos utils/
- **⚡ Rendimiento**: Optimizado para Apple Silicon (MPS) y CUDA
- **💾 Auto-Save**: Guardado automático de modelos, resultados y configuraciones
- **📊 Analytics**: Herramientas de análisis y comparación integradas

### 📈 Características Avanzadas
- **📊 Datos en tiempo real** con API de Binance
- **🧪 Backtesting avanzado** con múltiples estrategias
- **⚠️ Gestión de riesgos** y optimización de portfolio
- **🔬 Integración MLOps** con seguimiento de experimentos
- **⏰ Análisis multi-timeframe** y predicción

## 📊 Gestión de Datasets

Los datasets en bruto residen en `data/`. Utiliza los scripts de preprocesamiento proporcionados para generar características y aumentaciones. La clase `DataConfig` controla símbolos, ventanas de lookback y fuentes de datos adicionales como sentiment, orderbook o métricas on-chain.

Ver [`docs/DATA_MANAGEMENT.md`](docs/DATA_MANAGEMENT.md) para un tutorial completo.

## 📚 Documentación

Guías adicionales disponibles en el directorio `docs/`:

- [`BACKTESTER.md`](docs/BACKTESTER.md) – motor de backtesting avanzado
- [`EXPERIMENTS.md`](docs/EXPERIMENTS.md) – ejecutar experimentos configurables
- [`VALIDATORS.md`](docs/VALIDATORS.md) – helpers de validación cruzada
- [`INSTALLATION.md`](docs/INSTALLATION.md) – instrucciones de instalación detalladas
- [`DEVELOPMENT_GUIDE.md`](docs/DEVELOPMENT_GUIDE.md) – guía de desarrollo

## 🤝 Únete al Viaje

Hyperion es un **proyecto vivo y en constante evolución**. Si te apasiona este mundo, tu ayuda es bienvenida. Puedes contribuir:

- 🐛 **Reportando errores** via Issues
- 💡 **Sugiriendo nuevas características** 
- 🔧 **Añadiendo tu propio código** via Pull Requests
- 📖 **Mejorando la documentación**
- 🧪 **Compartiendo resultados de experimentos**

La estructura del proyecto es modular, lo que facilita la adición de nuevos modelos, métricas o procesadores de datos.

### 🛠️ Cómo Contribuir
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

Asegúrate de que todas las pruebas pasen antes de enviar.

## 📜 Licencia

Este proyecto se publica bajo los términos de la Licencia Apache 2.0. Ver [`LICENSE`](LICENSE) para más detalles.

---

## 🌟 Agradecimientos

Gracias a la comunidad open source y a todos los investigadores cuyo trabajo ha hecho posible Hyperion. Especial reconocimiento a:

- **Microsoft FLAML** por AutoML
- **Binance API** por datos de mercado
- **PyTorch** y **TensorFlow** ecosystems
- **Rich** por la hermosa interfaz de consola
- **MLflow** por el tracking de experimentos

---

**✨ ¿Listo para transformar tus ideas en estrategias reales? ¡Comienza tu viaje con Hyperion hoy!**

```bash
git clone https://github.com/tu-usuario/hyperion.git
cd hyperion
pip install -r requirements.txt
python main.py
```

*Que el código esté contigo* 🚀

