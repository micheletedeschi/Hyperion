#!/usr/bin/env python3
"""
🚀 PREPROCESADOR AVANZADO PARA HYPERION3
Sistema de preprocesamiento de datos con múltiples timeframes y features avanzadas
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Rich imports para UI
try:
    from rich.console import Console
    from utils.safe_progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Technical Analysis imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AdvancedDataPreprocessor:
    """Preprocesador avanzado con múltiples timeframes y features técnicas"""
    
    def __init__(self):
        """Inicializar preprocesador"""
        self.console = Console() if RICH_AVAILABLE else None
        self.data_dir = Path("data")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de features
        self.feature_config = {
            'basic_features': True,        # OHLCV básicas
            'price_features': True,        # Retornos, volatilidad
            'technical_indicators': True,  # RSI, MACD, Bollinger
            'volume_features': True,       # OBV, VWAP, Volume Profile
            'time_features': True,         # Hora, día de semana, etc.
            'multi_timeframe': True,       # Features de múltiples timeframes
            'lag_features': True,          # Features con lag
            'rolling_features': True,      # Medias móviles, rolling stats
            'target_engineering': True     # Múltiples targets
        }
        
        # Parámetros de ventanas
        self.windows = {
            'short': [5, 10, 15],
            'medium': [20, 30, 50],
            'long': [100, 200]
        }
        
        # Timeframes para multi-timeframe analysis
        self.timeframe_hierarchy = {
            '1m': ['5m', '15m', '1h'],
            '5m': ['15m', '1h', '4h'],
            '15m': ['1h', '4h', '1d'],
            '1h': ['4h', '1d', '1w'],
            '4h': ['1d', '1w', '1M'],
            '1d': ['1w', '1M']
        }
    
    def show_preprocessing_menu(self):
        """Mostrar menú de preprocesamiento"""
        if not self.console:
            return self._simple_preprocessing_menu()
        
        self.console.clear()
        
        # Header
        header_panel = Panel.fit(
            "[bold green]🔧 PREPROCESADOR AVANZADO DE DATOS[/bold green]\n"
            "[dim]Sistema de preprocesamiento con múltiples timeframes y features técnicas[/dim]",
            border_style="green"
        )
        self.console.print(header_panel)
        
        # Mostrar archivos disponibles
        self._show_available_data()
        
        # Opciones de preprocesamiento
        options_table = Table(title="🔧 Opciones de Preprocesamiento", show_header=True, header_style="bold cyan")
        options_table.add_column("ID", style="cyan", width=6)
        options_table.add_column("Opción", style="white", width=25)
        options_table.add_column("Descripción", style="dim", width=35)
        
        preprocessing_options = [
            ("1", "Auto-Procesar Descargados", "Procesar automáticamente datos del descargador"),
            ("2", "Preprocesar Archivo", "Preprocesar un archivo específico"),
            ("3", "Preprocesar Todo", "Preprocesar todos los archivos disponibles"),
            ("4", "Multi-Timeframe", "Crear dataset con múltiples timeframes"),
            ("5", "Features Personalizadas", "Configurar features específicas"),
            ("6", "Validar Datos", "Validar calidad de datos"),
            ("7", "Ver Procesados", "Explorar datos procesados"),
            ("8", "Configurar Features", "Configurar tipos de features")
        ]
        
        for opt_id, option, description in preprocessing_options:
            options_table.add_row(opt_id, option, description)
        
        self.console.print(options_table)
        
        choice = Prompt.ask("🎯 Selecciona opción", choices=["1", "2", "3", "4", "5", "6", "7", "8", "back"], default="1")
        
        if choice == "back":
            return
        elif choice == "1":
            return self._auto_process_downloaded()
        elif choice == "2":
            return self._preprocess_single_file()
        elif choice == "3":
            return self._preprocess_all_files()
        elif choice == "4":
            return self._create_multi_timeframe_dataset()
        elif choice == "5":
            return self._custom_features()
        elif choice == "6":
            return self._validate_data()
        elif choice == "7":
            return self._explore_processed_data()
        elif choice == "8":
            return self._configure_features()
    
    def _simple_preprocessing_menu(self):
        """Menú simple de preprocesamiento"""
        print("\n🔧 PREPROCESADOR DE DATOS")
        print("=" * 40)
        print("1. Preprocesar archivo")
        print("2. Preprocesar todo")
        print("3. Multi-timeframe")
        print("4. back - Volver")
        
        choice = input("\nSelecciona: ")
        if choice == "1":
            self._preprocess_single_file()
        elif choice == "2":
            self._preprocess_all_files()
        elif choice == "3":
            self._create_multi_timeframe_dataset()
    
    def _show_available_data(self):
        """Mostrar datos disponibles para procesar"""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            if self.console:
                self.console.print("[yellow]⚠️ No hay datos disponibles. Descarga datos primero.[/yellow]")
            return
        
        if self.console:
            files_table = Table(title="📂 Datos Disponibles", show_header=True, header_style="bold blue")
            files_table.add_column("ID", style="cyan", width=6)
            files_table.add_column("Archivo", style="white", width=30)
            files_table.add_column("Tamaño", style="green", width=10)
            files_table.add_column("Estado", style="yellow", width=12)
            
            for i, file in enumerate(csv_files, 1):
                size_mb = file.stat().st_size / (1024 * 1024)
                
                # Verificar si ya está procesado (enhanced_ o processed_)
                enhanced_file = self.processed_dir / f"enhanced_{file.name}"
                processed_file = self.processed_dir / f"processed_{file.name}"
                
                if enhanced_file.exists() or processed_file.exists():
                    status = "✅ Procesado"
                else:
                    status = "⚙️ Pendiente"
                
                files_table.add_row(str(i), file.name, f"{size_mb:.1f}MB", status)
            
            self.console.print(files_table)
    
    def _preprocess_single_file(self):
        """Preprocesar un archivo específico"""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            if self.console:
                self.console.print("[yellow]⚠️ No hay archivos para procesar[/yellow]")
            else:
                print("⚠️ No hay archivos para procesar")
            return
        
        if self.console:
            # Seleccionar archivo
            file_choices = [f.name for f in csv_files]
            filename = Prompt.ask("Selecciona archivo", choices=file_choices, default=file_choices[0])
            
            selected_file = next(f for f in csv_files if f.name == filename)
            
            with self.console.status(f"[bold green]Procesando {filename}..."):
                result = self._process_file(selected_file)
                self._show_processing_result(result)
        else:
            # Versión simple
            print("Archivos disponibles:")
            for i, file in enumerate(csv_files, 1):
                print(f"  {i}. {file.name}")
            
            try:
                choice_idx = int(input("Selecciona archivo (número): ")) - 1
                if 0 <= choice_idx < len(csv_files):
                    selected_file = csv_files[choice_idx]
                    print(f"Procesando {selected_file.name}...")
                    result = self._process_file(selected_file)
                    self._show_processing_result(result)
            except (ValueError, IndexError):
                print("❌ Selección inválida")
    
    def _preprocess_all_files(self):
        """Preprocesar todos los archivos disponibles"""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            if self.console:
                self.console.print("[yellow]⚠️ No hay archivos para procesar[/yellow]")
            else:
                print("⚠️ No hay archivos para procesar")
            return
        
        if self.console:
            self.console.print(f"[green]🔧 Procesando {len(csv_files)} archivos...[/green]")
            
            results = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Procesando archivos...", total=len(csv_files))
                
                for file in csv_files:
                    progress.update(task, description=f"Procesando {file.name}")
                    result = self._process_file(file)
                    results.append(result)
                    progress.advance(task)
            
            self._show_multi_processing_results(results)
        else:
            print(f"Procesando {len(csv_files)} archivos...")
            results = []
            for file in csv_files:
                print(f"  Procesando {file.name}...")
                result = self._process_file(file)
                results.append(result)
            
            self._show_multi_processing_results(results)
    
    def _process_file(self, file_path: Path) -> Dict:
        """Procesar un archivo individual"""
        try:
            # Cargar datos
            df = pd.read_csv(file_path)
            original_rows = len(df)
            
            # Validar columnas básicas
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return {
                    'error': f'Columnas requeridas faltantes: {required_cols}',
                    'file': file_path.name
                }
            
            # Convertir timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Limpiar datos básicos
            df = self._basic_cleaning(df)
            
            # Crear features según configuración
            if self.feature_config['basic_features']:
                df = self._add_basic_features(df)
            
            if self.feature_config['price_features']:
                df = self._add_price_features(df)
            
            if self.feature_config['technical_indicators']:
                df = self._add_technical_indicators(df)
            
            if self.feature_config['volume_features']:
                df = self._add_volume_features(df)
            
            if self.feature_config['time_features']:
                df = self._add_time_features(df)
            
            if self.feature_config['rolling_features']:
                df = self._add_rolling_features(df)
            
            if self.feature_config['lag_features']:
                df = self._add_lag_features(df)
            
            if self.feature_config['target_engineering']:
                df = self._add_target_features(df)
            
            # Limpiar NaN finales
            df = df.dropna()
            
            # Guardar archivo procesado
            processed_filename = f"processed_{file_path.name}"
            processed_path = self.processed_dir / processed_filename
            df.to_csv(processed_path, index=False)
            
            # Guardar metadatos
            metadata = {
                'original_file': file_path.name,
                'processed_file': processed_filename,
                'original_rows': original_rows,
                'processed_rows': len(df),
                'features_count': len(df.columns),
                'processing_date': datetime.now().isoformat(),
                'feature_config': self.feature_config.copy(),
                'columns': list(df.columns)
            }
            
            metadata_path = self.processed_dir / f"metadata_{file_path.stem}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'success': True,
                'file': file_path.name,
                'original_rows': original_rows,
                'processed_rows': len(df),
                'features_count': len(df.columns),
                'processed_file': str(processed_path),
                'size_mb': processed_path.stat().st_size / (1024 * 1024),
                'data_quality': self._assess_data_quality(df)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'file': file_path.name
            }
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza básica de datos"""
        # Eliminar duplicados
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Validar precios
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['close'] > 0) &
            (df['volume'] >= 0)
        ]
        
        return df.reset_index(drop=True)
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir features básicas"""
        # Spreads y ratios
        df['hl_spread'] = df['high'] - df['low']
        df['oc_spread'] = abs(df['open'] - df['close'])
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        # Posición del cierre dentro del rango
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Body y shadow de velas
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir features de precio"""
        # Retornos
        df['return_1'] = df['close'].pct_change()
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatilidad rolling
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['return_1'].rolling(window).std()
            df[f'volatility_log_{window}'] = df['log_return'].rolling(window).std()
        
        # Price momentum
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir indicadores técnicos"""
        # Medias móviles
        for window in [5, 10, 20, 50, 100]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # RSI
        if TALIB_AVAILABLE:
            df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            df['rsi_21'] = talib.RSI(df['close'].values, timeperiod=21)
        else:
            # RSI manual
            for period in [14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
        else:
            # MACD manual
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for window in [20, 50]:
            rolling_mean = df['close'].rolling(window).mean()
            rolling_std = df['close'].rolling(window).std()
            df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            df[f'bb_width_{window}'] = df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / df[f'bb_width_{window}']
        
        # Stochastic
        for window in [14, 21]:
            lowest_low = df['low'].rolling(window).min()
            highest_high = df['high'].rolling(window).max()
            df[f'stoch_k_{window}'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            df[f'stoch_d_{window}'] = df[f'stoch_k_{window}'].rolling(3).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir features de volumen"""
        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        for window in [10, 20, 50]:
            vwap_num = (typical_price * df['volume']).rolling(window).sum()
            vwap_den = df['volume'].rolling(window).sum()
            df[f'vwap_{window}'] = vwap_num / vwap_den
        
        # OBV (On Balance Volume)
        df['price_change'] = df['close'].diff()
        df['obv'] = (df['volume'] * np.sign(df['price_change'])).cumsum()
        
        # Volume ratios
        for window in [5, 10, 20]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # Volume-Price Trend
        df['vpt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir features temporales"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Features cíclicas
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Es fin de semana?
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir features rolling estadísticas"""
        price_cols = ['close', 'high', 'low', 'volume']
        
        for col in price_cols:
            for window in [5, 10, 20, 50]:
                # Estadísticas básicas
                df[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_max_{window}'] = df[col].rolling(window).max()
                df[f'{col}_median_{window}'] = df[col].rolling(window).median()
                
                # Posición relativa
                df[f'{col}_position_{window}'] = (df[col] - df[f'{col}_min_{window}']) / (df[f'{col}_max_{window}'] - df[f'{col}_min_{window}'])
                
                # Z-score
                df[f'{col}_zscore_{window}'] = (df[col] - df[f'{col}_mean_{window}']) / df[f'{col}_std_{window}']
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir features con lag"""
        key_features = ['close', 'volume', 'return_1', 'rsi_14', 'macd']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5, 10]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _add_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir múltiples targets"""
        # Target principal: retorno futuro
        for horizon in [1, 3, 5, 10]:
            df[f'target_return_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Target de clasificación: dirección
        for horizon in [1, 3, 5]:
            df[f'target_direction_{horizon}'] = (df[f'target_return_{horizon}'] > 0).astype(int)
        
        # Target de volatilidad
        for horizon in [5, 10]:
            df[f'target_volatility_{horizon}'] = df['return_1'].shift(-horizon).rolling(horizon).std()
        
        # Target por defecto
        df['target'] = df['target_return_1']
        
        return df
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Evaluar calidad de los datos"""
        quality_metrics = {
            'completeness': 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
            'duplicate_rows': df.duplicated().sum(),
            'zero_volume_ratio': (df['volume'] == 0).mean() if 'volume' in df.columns else 0,
            'price_gaps': 0,  # Calcular gaps de precios si es necesario
            'outlier_ratio': 0  # Calcular outliers si es necesario
        }
        
        return quality_metrics
    
    def _show_processing_result(self, result: Dict):
        """Mostrar resultado de procesamiento individual"""
        if 'error' in result:
            if self.console:
                self.console.print(f"[red]❌ Error procesando {result['file']}: {result['error']}[/red]")
            else:
                print(f"❌ Error: {result['error']}")
        else:
            if self.console:
                content = f"[green]✅ Procesamiento completado[/green]\n"
                content += f"[cyan]Archivo:[/cyan] {result['file']}\n"
                content += f"[cyan]Filas originales:[/cyan] {result['original_rows']:,}\n"
                content += f"[cyan]Filas procesadas:[/cyan] {result['processed_rows']:,}\n"
                content += f"[cyan]Features creadas:[/cyan] {result['features_count']}\n"
                content += f"[cyan]Archivo procesado:[/cyan] {result['processed_file']}\n"
                content += f"[cyan]Tamaño:[/cyan] {result['size_mb']:.2f} MB\n"
                
                quality = result['data_quality']
                content += f"[cyan]Calidad:[/cyan] {quality['completeness']:.2%} completo"
                
                self.console.print(Panel(content, title="🔧 Resultado", border_style="green"))
            else:
                print(f"✅ Procesado: {result['file']}")
                print(f"   {result['original_rows']:,} → {result['processed_rows']:,} filas")
                print(f"   {result['features_count']} features creadas")
    
    def _show_multi_processing_results(self, results: List[Dict]):
        """Mostrar resultados de múltiples procesamientos"""
        successful = [r for r in results if 'success' in r]
        failed = [r for r in results if 'error' in r]
        
        if self.console:
            # Tabla de resumen
            summary_table = Table(title="🔧 Resumen de Procesamiento", show_header=True, header_style="bold green")
            summary_table.add_column("Archivo", style="cyan", width=25)
            summary_table.add_column("Filas", style="white", width=12)
            summary_table.add_column("Features", style="green", width=10)
            summary_table.add_column("Tamaño", style="yellow", width=10)
            summary_table.add_column("Estado", style="blue", width=10)
            
            for result in results:
                if 'success' in result:
                    summary_table.add_row(
                        result['file'][:20] + "..." if len(result['file']) > 20 else result['file'],
                        f"{result['processed_rows']:,}",
                        str(result['features_count']),
                        f"{result['size_mb']:.1f}MB",
                        "✅ OK"
                    )
                else:
                    summary_table.add_row(
                        result['file'][:20] + "..." if len(result['file']) > 20 else result['file'],
                        "N/A",
                        "N/A",
                        "N/A",
                        "❌ Error"
                    )
            
            self.console.print(summary_table)
            
            # Estadísticas
            if successful:
                total_rows = sum(r['processed_rows'] for r in successful)
                total_features = sum(r['features_count'] for r in successful)
                avg_features = total_features / len(successful)
                
                stats_text = f"[green]📊 Estadísticas:[/green]\n"
                stats_text += f"• Archivos exitosos: {len(successful)}\n"
                stats_text += f"• Archivos fallidos: {len(failed)}\n"
                stats_text += f"• Total filas procesadas: {total_rows:,}\n"
                stats_text += f"• Promedio features: {avg_features:.0f}"
                
                self.console.print(Panel(stats_text, title="📊 Resumen", border_style="green"))
        else:
            print(f"📊 RESUMEN: {len(successful)} exitosos, {len(failed)} fallidos")
            for result in successful:
                print(f"  ✅ {result['file']}: {result['processed_rows']:,} filas, {result['features_count']} features")
    
    def _create_multi_timeframe_dataset(self):
        """Crear dataset con múltiples timeframes"""
        if self.console:
            self.console.print("[cyan]🕐 Creación de dataset multi-timeframe en desarrollo[/cyan]")
        else:
            print("🕐 Dataset multi-timeframe en desarrollo")
    
    def _custom_features(self):
        """Configurar features personalizadas"""
        if self.console:
            self.console.print("[cyan]⚙️ Features personalizadas en desarrollo[/cyan]")
        else:
            print("⚙️ Features personalizadas en desarrollo")
    
    def _validate_data(self):
        """Validar calidad de datos"""
        if self.console:
            self.console.print("[cyan]✅ Validación de datos en desarrollo[/cyan]")
        else:
            print("✅ Validación de datos en desarrollo")
    
    def _explore_processed_data(self):
        """Explorar datos procesados"""
        processed_files = list(self.processed_dir.glob("processed_*.csv"))
        
        if not processed_files:
            if self.console:
                self.console.print("[yellow]⚠️ No hay datos procesados[/yellow]")
            else:
                print("⚠️ No hay datos procesados")
            return
        
        if self.console:
            files_table = Table(title="📂 Datos Procesados", show_header=True, header_style="bold green")
            files_table.add_column("Archivo", style="cyan", width=30)
            files_table.add_column("Tamaño", style="white", width=10)
            files_table.add_column("Modificado", style="dim", width=20)
            
            for file in processed_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                files_table.add_row(file.name, f"{size_mb:.1f}MB", mod_time)
            
            self.console.print(files_table)
        else:
            print("📂 DATOS PROCESADOS:")
            for file in processed_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.name} ({size_mb:.1f}MB)")
    
    def _configure_features(self):
        """Configurar tipos de features"""
        if self.console:
            config_table = Table(title="⚙️ Configuración de Features", show_header=True, header_style="bold cyan")
            config_table.add_column("Feature Type", style="cyan", width=20)
            config_table.add_column("Estado", style="green", width=10)
            config_table.add_column("Descripción", style="dim", width=30)
            
            for feature_type, enabled in self.feature_config.items():
                status = "✅ Activo" if enabled else "❌ Inactivo"
                descriptions = {
                    'basic_features': 'OHLCV básicas y spreads',
                    'price_features': 'Retornos y volatilidad',
                    'technical_indicators': 'RSI, MACD, Bollinger',
                    'volume_features': 'OBV, VWAP, ratios',
                    'time_features': 'Hora, día, estacionalidad',
                    'multi_timeframe': 'Features de múltiples TF',
                    'lag_features': 'Features con retardo',
                    'rolling_features': 'Estadísticas móviles',
                    'target_engineering': 'Múltiples targets'
                }
                desc = descriptions.get(feature_type, 'Personalizado')
                
                config_table.add_row(feature_type, status, desc)
            
            self.console.print(config_table)
            
            # Permitir modificar configuración
            if Confirm.ask("¿Modificar configuración de features?", default=False):
                for feature_type in self.feature_config.keys():
                    current = self.feature_config[feature_type]
                    new_value = Confirm.ask(f"Activar {feature_type}", default=current)
                    self.feature_config[feature_type] = new_value
                
                self.console.print("[green]✅ Configuración actualizada[/green]")
        else:
            print("⚙️ CONFIGURACIÓN DE FEATURES:")
            for feature_type, enabled in self.feature_config.items():
                status = "✅" if enabled else "❌"
                print(f"  {status} {feature_type}")
    
    def auto_preprocess_downloaded_data(self, filename: str = None) -> Dict:
        """Preprocesar automáticamente datos descargados del sistema"""
        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Detectando datos descargados...", total=100)
                
                # Buscar archivos recientes del descargador
                if filename:
                    target_files = [self.data_dir / filename]
                else:
                    # Buscar archivos CSV recientes (últimas 24h)
                    target_files = []
                    for file in self.data_dir.glob("*.csv"):
                        # Detectar formato del descargador (symbol_timeframe_date.csv)
                        if '_' in file.stem and not file.name.startswith('processed_'):
                            target_files.append(file)
                
                if not target_files:
                    self.console.print("[yellow]⚠️ No se encontraron datos del descargador[/yellow]")
                    return {'error': 'No data found'}
                
                progress.update(task, advance=20, description="Analizando archivos...")
                
                results = []
                for file in target_files:
                    try:
                        # Cargar datos
                        df = pd.read_csv(file)
                        
                        # Detectar formato y configurar preprocesamiento
                        if self._detect_downloader_format(df):
                            progress.update(task, advance=30, description=f"Procesando {file.name}...")
                            
                            # Preprocesar con configuración optimizada para trading
                            processed_df = self._enhance_trading_data(df)
                            
                            # Guardar datos procesados
                            output_file = self.processed_dir / f"enhanced_{file.name}"
                            processed_df.to_csv(output_file, index=False)
                            
                            results.append({
                                'file': str(file),
                                'output': str(output_file),
                                'rows': len(processed_df),
                                'features': len(processed_df.columns)
                            })
                            
                            progress.update(task, advance=20, description=f"Completado {file.name}")
                    
                    except Exception as e:
                        self.console.print(f"[red]❌ Error procesando {file.name}: {e}[/red]")
                
                progress.update(task, advance=30, description="¡Procesamiento completado!")
                
                return {'success': True, 'results': results}
        else:
            print("🔧 Preprocesando datos descargados...")
            # Version simple
            csv_files = list(self.data_dir.glob("*.csv"))
            results = []
            
            for file in csv_files:
                if not file.name.startswith('processed_'):
                    try:
                        df = pd.read_csv(file)
                        if self._detect_downloader_format(df):
                            processed_df = self._enhance_trading_data(df)
                            output_file = self.processed_dir / f"enhanced_{file.name}"
                            processed_df.to_csv(output_file, index=False)
                            results.append({'file': str(file), 'output': str(output_file)})
                    except Exception as e:
                        print(f"❌ Error: {e}")
            
            return {'success': True, 'results': results}
    
    def _detect_downloader_format(self, df: pd.DataFrame) -> bool:
        """Detectar si los datos vienen del descargador de Hyperion3"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required_columns)
    
    def _enhance_trading_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mejorar datos de trading con features avanzadas optimizadas"""
        try:
            # Crear copia para procesamiento
            enhanced_df = df.copy()
            
            # Asegurar que timestamp es datetime
            if 'timestamp' in enhanced_df.columns:
                enhanced_df['timestamp'] = pd.to_datetime(enhanced_df['timestamp'])
                enhanced_df = enhanced_df.sort_values('timestamp').reset_index(drop=True)
            
            # Features básicas de precio
            enhanced_df['price_change'] = enhanced_df['close'].pct_change()
            enhanced_df['price_range'] = (enhanced_df['high'] - enhanced_df['low']) / enhanced_df['close']
            enhanced_df['body_size'] = abs(enhanced_df['close'] - enhanced_df['open']) / enhanced_df['close']
            
            # Features de volatilidad
            enhanced_df['volatility_5'] = enhanced_df['price_change'].rolling(5).std()
            enhanced_df['volatility_20'] = enhanced_df['price_change'].rolling(20).std()
            
            # Features de volumen (si disponible)
            if 'volume' in enhanced_df.columns:
                enhanced_df['volume_sma'] = enhanced_df['volume'].rolling(20).mean()
                enhanced_df['volume_ratio'] = enhanced_df['volume'] / enhanced_df['volume_sma']
                enhanced_df['price_volume'] = enhanced_df['close'] * enhanced_df['volume']
            
            # Indicadores técnicos básicos
            if TALIB_AVAILABLE:
                # RSI
                enhanced_df['rsi'] = talib.RSI(enhanced_df['close'].values)
                
                # MACD
                macd, signal, histogram = talib.MACD(enhanced_df['close'].values)
                enhanced_df['macd'] = macd
                enhanced_df['macd_signal'] = signal
                enhanced_df['macd_histogram'] = histogram
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(enhanced_df['close'].values)
                enhanced_df['bb_upper'] = bb_upper
                enhanced_df['bb_middle'] = bb_middle
                enhanced_df['bb_lower'] = bb_lower
                enhanced_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
                enhanced_df['bb_position'] = (enhanced_df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Medias móviles simples
            for window in [5, 10, 20, 50]:
                enhanced_df[f'sma_{window}'] = enhanced_df['close'].rolling(window).mean()
                enhanced_df[f'price_vs_sma_{window}'] = enhanced_df['close'] / enhanced_df[f'sma_{window}'] - 1
            
            # Features de tiempo (si hay timestamp)
            if 'timestamp' in enhanced_df.columns:
                enhanced_df['hour'] = enhanced_df['timestamp'].dt.hour
                enhanced_df['day_of_week'] = enhanced_df['timestamp'].dt.dayofweek
                enhanced_df['month'] = enhanced_df['timestamp'].dt.month
                
                # Features cíclicas para tiempo
                enhanced_df['hour_sin'] = np.sin(2 * np.pi * enhanced_df['hour'] / 24)
                enhanced_df['hour_cos'] = np.cos(2 * np.pi * enhanced_df['hour'] / 24)
                enhanced_df['dow_sin'] = np.sin(2 * np.pi * enhanced_df['day_of_week'] / 7)
                enhanced_df['dow_cos'] = np.cos(2 * np.pi * enhanced_df['day_of_week'] / 7)
            
            # Target para ML (predicción de precio futuro)
            enhanced_df['target_1h'] = enhanced_df['close'].shift(-1) / enhanced_df['close'] - 1
            enhanced_df['target_4h'] = enhanced_df['close'].shift(-4) / enhanced_df['close'] - 1
            enhanced_df['target_1d'] = enhanced_df['close'].shift(-24) / enhanced_df['close'] - 1
            
            # Clasificación de dirección
            enhanced_df['direction_1h'] = (enhanced_df['target_1h'] > 0).astype(int)
            enhanced_df['direction_4h'] = (enhanced_df['target_4h'] > 0).astype(int)
            enhanced_df['direction_1d'] = (enhanced_df['target_1d'] > 0).astype(int)
            
            # Limpiar valores infinitos y NaN
            enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
            
            # Eliminar filas con demasiados NaN
            enhanced_df = enhanced_df.dropna(thresh=len(enhanced_df.columns) * 0.7)
            
            return enhanced_df
            
        except Exception as e:
            print(f"❌ Error en enhance_trading_data: {e}")
            return df

def main():
    """Función principal para pruebas"""
    preprocessor = AdvancedDataPreprocessor()
    preprocessor.show_preprocessing_menu()


if __name__ == "__main__":
    main()
