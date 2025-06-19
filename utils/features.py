#!/usr/bin/env python3
"""
📊 GENERACIÓN DE FEATURES PARA HYPERION3
Creación de indicadores técnicos y features avanzadas sin data leakage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def create_clean_features(data: pd.DataFrame, console=None) -> pd.DataFrame:
    """
    Crear features avanzadas con indicadores técnicos - VERSIÓN MEJORADA
    
    Args:
        data: DataFrame con datos OHLCV
        console: Console para progress tracking (opcional)
        
    Returns:
        DataFrame con features limpias y target
    """
    
    # Mostrar progreso si hay console disponible
    if console:
        from utils.safe_progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]🔧 Creando features avanzadas...", total=12)
            
            # Asegurar orden temporal
            data = data.sort_values('timestamp').reset_index(drop=True)
            features = {}
            
            # 1. Returns logarítmicos (mejores que diferencias simples)
            progress.update(task, description="[cyan]🔧 Returns logarítmicos...")
            features['log_return_1'] = np.log(data['close'] / data['close'].shift(1))
            features['log_return_5'] = np.log(data['close'] / data['close'].shift(5))
            features['log_return_15'] = np.log(data['close'] / data['close'].shift(15))
            features['log_return_30'] = np.log(data['close'] / data['close'].shift(30))
            progress.advance(task)
            
            # 2. Volatilidad rolling mejorada
            progress.update(task, description="[cyan]🔧 Volatilidad avanzada...")
            for window in [10, 20, 50]:
                log_ret = np.log(data['close'] / data['close'].shift(1))
                features[f'volatility_{window}'] = log_ret.rolling(window).std()
                features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features[f'volatility_{window}'].rolling(window*2).mean()
            progress.advance(task)
            
            # 3. Momentum indicators (RSI mejorado)
            progress.update(task, description="[cyan]🔧 Momentum indicators...")
            try:
                import ta
                features['rsi_7'] = ta.momentum.RSIIndicator(data['close'], window=7).rsi()
                features['rsi_14'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
                features['macd'] = ta.trend.MACD(data['close']).macd()
                features['macd_signal'] = ta.trend.MACD(data['close']).macd_signal()
                features['macd_histogram'] = features['macd'] - features['macd_signal']
            except ImportError:
                # Implementación manual del RSI
                for period in [7, 14]:
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                
                # MACD manual
                ema_12 = data['close'].ewm(span=12).mean()
                ema_26 = data['close'].ewm(span=26).mean()
                features['macd'] = ema_12 - ema_26
                features['macd_signal'] = features['macd'].ewm(span=9).mean()
                features['macd_histogram'] = features['macd'] - features['macd_signal']
            progress.advance(task)
            
            # 4. Trend indicators mejorados
            progress.update(task, description="[cyan]🔧 Trend indicators...")
            for window in [10, 20, 50]:
                sma = data['close'].rolling(window).mean()
                features[f'sma_{window}'] = sma
                features[f'price_vs_sma_{window}'] = (data['close'] - sma) / sma
                features[f'price_position_{window}'] = (data['close'] - data['close'].rolling(window).min()) / (data['close'].rolling(window).max() - data['close'].rolling(window).min())
            progress.advance(task)
            
            # 5. Bollinger Bands
            progress.update(task, description="[cyan]🔧 Bollinger Bands...")
            try:
                import ta
                bb = ta.volatility.BollingerBands(data['close'], window=20)
                features['bb_upper'] = bb.bollinger_hband()
                features['bb_lower'] = bb.bollinger_lband()
                features['bb_middle'] = bb.bollinger_mavg()
                features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
                features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            except ImportError:
                # Implementación manual
                sma_20 = data['close'].rolling(20).mean()
                std_20 = data['close'].rolling(20).std()
                features['bb_upper'] = sma_20 + (std_20 * 2)
                features['bb_lower'] = sma_20 - (std_20 * 2)
                features['bb_middle'] = sma_20
                features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
                features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            progress.advance(task)
            
            # 6. Volume features avanzadas
            progress.update(task, description="[cyan]🔧 Volume features...")
            features['volume_sma_20'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma_20']
            
            # VWAP (Volume Weighted Average Price)
            features['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
            features['price_vs_vwap'] = (data['close'] - features['vwap']) / features['vwap']
            
            # Volume momentum
            features['volume_momentum'] = data['volume'].rolling(10).mean() / data['volume'].rolling(30).mean()
            progress.advance(task)
            
            # 7. Price momentum and acceleration
            progress.update(task, description="[cyan]🔧 Price momentum...")
            for period in [5, 10, 20]:
                price_change = data['close'].pct_change(period)
                features[f'momentum_{period}'] = price_change
                features[f'acceleration_{period}'] = price_change - data['close'].pct_change(period).shift(period)
            progress.advance(task)
            
            # 8. Time-based features
            progress.update(task, description="[cyan]🔧 Time features...")
            if 'timestamp' in data.columns:
                dt = pd.to_datetime(data['timestamp'])
                features['hour'] = dt.dt.hour
                features['day_of_week'] = dt.dt.dayofweek
                features['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
            else:
                # Features temporales básicas basadas en posición
                features['hour'] = (data.index % 24)
                features['day_of_week'] = ((data.index // 24) % 7)
                features['is_weekend'] = ((features['day_of_week'] >= 5)).astype(int)
            progress.advance(task)
            
            # 9. Support and Resistance levels
            progress.update(task, description="[cyan]🔧 Support/Resistance...")
            for window in [20, 50]:
                rolling_max = data['high'].rolling(window).max()
                rolling_min = data['low'].rolling(window).min()
                features[f'resistance_{window}'] = rolling_max
                features[f'support_{window}'] = rolling_min
                features[f'price_vs_resistance_{window}'] = (data['close'] - rolling_max) / rolling_max
                features[f'price_vs_support_{window}'] = (data['close'] - rolling_min) / rolling_min
            progress.advance(task)
            
            # 10. Advanced volatility features
            progress.update(task, description="[cyan]🔧 Advanced volatility...")
            # True Range and ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features['atr_14'] = true_range.rolling(14).mean()
            features['atr_ratio'] = true_range / features['atr_14']
            
            # Volatility regime
            features['vol_regime'] = features['volatility_20'] / features['volatility_50']
            progress.advance(task)
            
            # 11. Cross-asset momentum (if available)
            progress.update(task, description="[cyan]🔧 Cross momentum...")
            # Use existing indicators as proxies for market momentum
            features['market_momentum'] = (features['rsi_14'] + features['macd']) / 2
            features['trend_strength'] = np.abs(features['price_position_20'])
            features['momentum_convergence'] = features['rsi_7'] - features['rsi_14']
            progress.advance(task)
            
            # 12. Crear DataFrame y agregar lag features
            progress.update(task, description="[cyan]🔧 Finalizando y lag features...")
            feature_df = pd.DataFrame(features)
            
            # AGREGAR LAG FEATURES para capturar dependencias temporales
            console.print("📊 Agregando lag features...")
            
            # Lag features de las características más importantes
            important_features = ['log_return_1', 'rsi_14', 'volatility_20', 'volume_ratio', 'bb_position']
            
            for feature in important_features:
                if feature in feature_df.columns:
                    for lag in [1, 2, 3, 5]:
                        feature_df[f'{feature}_lag_{lag}'] = feature_df[feature].shift(lag)
            
            # Rolling averages de lag features
            for feature in ['log_return_1', 'rsi_14']:
                if feature in feature_df.columns:
                    feature_df[f'{feature}_ma_3'] = feature_df[feature].rolling(3).mean()
                    feature_df[f'{feature}_ma_5'] = feature_df[feature].rolling(5).mean()
            
            # TARGET MEJORADO: 3-day forward returns (balance entre predictabilidad y horizonte)
            valid_length = len(data) - 24  # Necesitamos 24 puntos futuros (24h)
            future_returns = []
            
            for i in range(valid_length):
                current_price = data['close'].iloc[i]
                future_price = data['close'].iloc[i + 24]  # 24 horas adelante
                future_return = (future_price - current_price) / current_price
                future_returns.append(future_return)
            
            # Truncar y agregar target
            feature_df = feature_df.iloc[:valid_length].copy()
            feature_df['target'] = future_returns
            feature_df['timestamp'] = data['timestamp'].iloc[:valid_length]
            progress.advance(task)
    
    else:
        # Versión sin progress bar
        feature_df = _create_features_without_progress(data)
    
    # LIMPIEZA AVANZADA
    if console:
        console.print("🧹 Limpieza avanzada de features...")
    
    # Eliminar NaN e Inf
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    initial_rows = len(feature_df)
    feature_df = feature_df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(feature_df)
    
    if dropped_rows > 0 and console:
        console.print(f"🧹 Eliminadas {dropped_rows} filas con NaN/Inf")
    
    # Clip features usando percentiles
    numeric_cols = [col for col in feature_df.columns if col not in ['target', 'timestamp']]
    
    for col in numeric_cols:
        q01 = feature_df[col].quantile(0.01)
        q99 = feature_df[col].quantile(0.99)
        feature_df[col] = feature_df[col].clip(lower=q01, upper=q99)
    
    # Target final - mantener consistencia con el cálculo arriba
    return feature_df

def _create_features_without_progress(data: pd.DataFrame) -> pd.DataFrame:
    """Versión simplificada sin progress bar para uso interno"""
    
    # Asegurar orden temporal
    data = data.sort_values('timestamp').reset_index(drop=True)
    features = {}
    
    # 1. Returns logarítmicos
    features['log_return_1'] = np.log(data['close'] / data['close'].shift(1))
    features['log_return_5'] = np.log(data['close'] / data['close'].shift(5))
    features['log_return_15'] = np.log(data['close'] / data['close'].shift(15))
    
    # 2. Volatilidad básica
    for window in [10, 20]:
        log_ret = np.log(data['close'] / data['close'].shift(1))
        features[f'volatility_{window}'] = log_ret.rolling(window).std()
    
    # 3. RSI básico
    for period in [7, 14]:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # 4. Moving averages
    for window in [10, 20]:
        sma = data['close'].rolling(window).mean()
        features[f'sma_{window}'] = sma
        features[f'price_vs_sma_{window}'] = (data['close'] - sma) / sma
    
    # 5. Volume features básicas
    features['volume_sma_20'] = data['volume'].rolling(20).mean()
    features['volume_ratio'] = data['volume'] / features['volume_sma_20']
    
    # 6. Target
    valid_length = len(data) - 24
    future_returns = []
    for i in range(valid_length):
        current_price = data['close'].iloc[i]
        future_price = data['close'].iloc[i + 24]
        future_return = (future_price - current_price) / current_price
        future_returns.append(future_return)
    
    # Crear DataFrame
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.iloc[:valid_length].copy()
    feature_df['target'] = future_returns
    feature_df['timestamp'] = data['timestamp'].iloc[:valid_length]
    
    return feature_df

def verify_no_leakage(feature_df: pd.DataFrame, console=None) -> bool:
    """
    Verificar que no hay data leakage en las features
    
    Args:
        feature_df: DataFrame con features y target
        console: Console para logging (opcional)
        
    Returns:
        True si no hay leakage, False si se detecta leakage
    """
    
    feature_cols = [col for col in feature_df.columns if col not in ['target', 'timestamp']]
    
    leakage_found = False
    max_correlation = 0
    suspicious_features = []
    
    for feat in feature_cols:
        try:
            corr = abs(feature_df[feat].corr(feature_df['target']))
            if corr > max_correlation:
                max_correlation = corr
            
            if corr > 0.95:  # Umbral muy alto indica posible leakage
                leakage_found = True
                suspicious_features.append((feat, corr))
        except:
            continue
    
    if console:
        console.print(f"📊 Máxima correlación encontrada: {max_correlation:.6f}")
    
    if leakage_found:
        if console:
            console.print(f"🚨 POSIBLE DATA LEAKAGE DETECTADO!")
            for feat, corr in suspicious_features:
                console.print(f"   {feat}: correlación = {corr:.6f}")
        return False
    elif max_correlation > 0.8:
        if console:
            console.print(f"⚠️ Correlación alta detectada: {max_correlation:.6f}")
            console.print("   Revisar features sospechosas")
        return True
    else:
        if console:
            console.print("✅ No se detectó data leakage")
        return True

def select_top_features(feature_df: pd.DataFrame, n_features: int = 30, 
                       method: str = 'correlation') -> List[str]:
    """
    Seleccionar las mejores features usando diferentes métodos
    
    Args:
        feature_df: DataFrame con features y target
        n_features: Número de features a seleccionar
        method: Método de selección ('correlation', 'mutual_info', 'variance')
        
    Returns:
        Lista con nombres de las mejores features
    """
    
    feature_cols = [col for col in feature_df.columns if col not in ['target', 'timestamp']]
    
    if method == 'correlation':
        # Selección por correlación absoluta con target
        correlations = {}
        for col in feature_cols:
            try:
                corr = abs(feature_df[col].corr(feature_df['target']))
                if not np.isnan(corr):
                    correlations[col] = corr
            except:
                continue
        
        # Ordenar por correlación y tomar top N
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, _ in sorted_features[:n_features]]
        
    elif method == 'mutual_info':
        # Selección por información mutua
        try:
            from sklearn.feature_selection import mutual_info_regression
            X = feature_df[feature_cols].fillna(0)
            y = feature_df['target']
            
            mi_scores = mutual_info_regression(X, y, random_state=42)
            feature_scores = list(zip(feature_cols, mi_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, _ in feature_scores[:n_features]]
        except ImportError:
            # Fallback a correlación si sklearn no está disponible
            return select_top_features(feature_df, n_features, 'correlation')
    
    elif method == 'variance':
        # Selección por varianza (eliminar features con baja varianza)
        try:
            from sklearn.feature_selection import VarianceThreshold
            X = feature_df[feature_cols].fillna(0)
            
            # Primero eliminar features con varianza muy baja
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            selected_cols = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
            
            # Luego seleccionar por correlación
            temp_df = feature_df[selected_cols + ['target']].copy()
            selected_features = select_top_features(temp_df, n_features, 'correlation')
        except ImportError:
            return select_top_features(feature_df, n_features, 'correlation')
    
    else:
        raise ValueError(f"Método no reconocido: {method}")
    
    return selected_features[:n_features]

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Añadir indicadores técnicos adicionales usando TA-Lib si está disponible
    
    Args:
        data: DataFrame con datos OHLCV
        
    Returns:
        DataFrame con indicadores técnicos añadidos
    """
    
    try:
        import talib
        
        # Indicadores de momento
        data['RSI'] = talib.RSI(data['close'].values)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['close'].values)
        data['CCI'] = talib.CCI(data['high'].values, data['low'].values, data['close'].values)
        data['MOM'] = talib.MOM(data['close'].values)
        
        # Indicadores de tendencia
        data['SMA_20'] = talib.SMA(data['close'].values, timeperiod=20)
        data['EMA_12'] = talib.EMA(data['close'].values, timeperiod=12)
        data['EMA_26'] = talib.EMA(data['close'].values, timeperiod=26)
        data['ADX'] = talib.ADX(data['high'].values, data['low'].values, data['close'].values)
        
        # Indicadores de volatilidad
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['close'].values)
        data['ATR'] = talib.ATR(data['high'].values, data['low'].values, data['close'].values)
        
        # Indicadores de volumen
        data['OBV'] = talib.OBV(data['close'].values, data['volume'].values)
        data['AD'] = talib.AD(data['high'].values, data['low'].values, data['close'].values, data['volume'].values)
        
        print("✅ Indicadores TA-Lib añadidos exitosamente")
        
    except ImportError:
        print("⚠️ TA-Lib no disponible, usando implementaciones manuales")
        
        # Implementaciones manuales básicas
        data['RSI'] = _calculate_rsi(data['close'])
        data['SMA_20'] = data['close'].rolling(20).mean()
        data['EMA_12'] = data['close'].ewm(span=12).mean()
        data['BB_middle'] = data['close'].rolling(20).mean()
        data['BB_std'] = data['close'].rolling(20).std()
        data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
        data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
    
    return data

def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calcular RSI manualmente"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class EnhancedFeatureEngineer:
    """
    Clase principal para ingeniería de features avanzada
    Envuelve las funciones de feature engineering en una interfaz orientada a objetos
    """
    
    def __init__(self, console=None):
        """
        Args:
            console: Rich console para mostrar progreso (opcional)
        """
        self.console = console
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features avanzadas para el DataFrame de entrada
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con features y target
        """
        return create_clean_features(data, self.console)
    
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features con indicadores avanzados
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con features avanzadas
        """
        # Primero crear features básicas
        features_df = create_clean_features(data, self.console)
        # Luego agregar indicadores técnicos
        features_df = add_technical_indicators(features_df)
        return features_df

# Función wrapper para compatibilidad
def engineer_features(data: pd.DataFrame, console=None) -> pd.DataFrame:
    """
    Función wrapper para crear features usando la clase EnhancedFeatureEngineer
    """
    engineer = EnhancedFeatureEngineer(console=console)
    return engineer.create_features(data)
