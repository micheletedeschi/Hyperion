#!/usr/bin/env python3
"""
🚀 ENTRENADOR MULTI-TIMEFRAME PARA HYPERION3
Sistema para entrenar modelos con múltiples timeframes simultáneamente
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
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ML imports
try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Imports de nuestros módulos
try:
    from utils.models import create_ensemble_models
    from utils.features import EnhancedFeatureEngineer
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


class MultiTimeframeTrainer:
    """Entrenador de modelos con múltiples timeframes"""
    
    def __init__(self):
        """Inicializar entrenador multi-timeframe"""
        self.console = Console() if RICH_AVAILABLE else None
        self.data_dir = Path("data")
        self.processed_dir = Path("data/processed")
        self.results_dir = Path("results")
        self.multi_tf_dir = Path("results/multi_timeframe")
        
        # Crear directorios
        for directory in [self.results_dir, self.multi_tf_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Configuración de timeframes
        self.timeframe_configs = {
            'scalping': {
                'primary': '1m',
                'secondary': ['5m', '15m'],
                'features': ['price_action', 'volume', 'momentum'],
                'target_horizon': 1
            },
            'intraday': {
                'primary': '15m',
                'secondary': ['1h', '4h'],
                'features': ['technical', 'volume', 'trend'],
                'target_horizon': 3
            },
            'swing': {
                'primary': '1h',
                'secondary': ['4h', '1d'],
                'features': ['technical', 'trend', 'macro'],
                'target_horizon': 5
            },
            'position': {
                'primary': '4h',
                'secondary': ['1d', '1w'],
                'features': ['trend', 'macro', 'sentiment'],
                'target_horizon': 10
            }
        }
        
        # Estrategias de combinación
        self.combination_strategies = {
            'hierarchical': 'Modelos jerárquicos por timeframe',
            'ensemble': 'Ensemble de todos los timeframes',
            'weighted': 'Promedio ponderado por timeframe',
            'meta_learning': 'Meta-modelo que combina predicciones',
            'attention': 'Mecanismo de atención entre timeframes'
        }
    
    def show_multi_timeframe_menu(self):
        """Mostrar menú de entrenamiento multi-timeframe"""
        if not self.console:
            return self._simple_multi_tf_menu()
        
        self.console.clear()
        
        # Header
        header_panel = Panel.fit(
            "[bold magenta]🕐 ENTRENAMIENTO MULTI-TIMEFRAME[/bold magenta]\n"
            "[dim]Entrena modelos usando múltiples timeframes para mejorar predicciones[/dim]",
            border_style="magenta"
        )
        self.console.print(header_panel)
        
        # Mostrar configuraciones disponibles
        self._show_timeframe_configs()
        
        # Opciones principales
        options_table = Table(title="🕐 Opciones Multi-Timeframe", show_header=True, header_style="bold cyan")
        options_table.add_column("ID", style="cyan", width=6)
        options_table.add_column("Opción", style="white", width=25)
        options_table.add_column("Descripción", style="dim", width=35)
        
        options = [
            ("1", "Estrategia Predefinida", "Usar configuración de trading predefinida"),
            ("2", "Configuración Personalizada", "Seleccionar timeframes manualmente"),
            ("3", "Entrenar Todo", "Entrenar todos los timeframes disponibles"),
            ("4", "Comparar Estrategias", "Comparar diferentes enfoques"),
            ("5", "Análisis de Resultados", "Analizar modelos multi-TF existentes"),
            ("6", "Configurar Estrategias", "Modificar configuraciones")
        ]
        
        for opt_id, option, description in options:
            options_table.add_row(opt_id, option, description)
        
        self.console.print(options_table)
        
        choice = Prompt.ask("🎯 Selecciona opción", choices=["1", "2", "3", "4", "5", "6", "back"], default="1")
        
        if choice == "back":
            return
        elif choice == "1":
            return self._predefined_strategy()
        elif choice == "2":
            return self._custom_configuration()
        elif choice == "3":
            return self._train_all_timeframes()
        elif choice == "4":
            return self._compare_strategies()
        elif choice == "5":
            return self._analyze_results()
        elif choice == "6":
            return self._configure_strategies()
    
    def _simple_multi_tf_menu(self):
        """Menú simple multi-timeframe"""
        print("\n🕐 ENTRENAMIENTO MULTI-TIMEFRAME")
        print("=" * 40)
        print("1. Estrategia Swing (1h + 4h + 1d)")
        print("2. Estrategia Intraday (15m + 1h + 4h)")
        print("3. Entrenar todo")
        print("4. back - Volver")
        
        choice = input("\nSelecciona: ")
        if choice == "1":
            self._train_strategy("swing")
        elif choice == "2":
            self._train_strategy("intraday")
        elif choice == "3":
            self._train_all_timeframes()
    
    def _show_timeframe_configs(self):
        """Mostrar configuraciones de timeframe disponibles"""
        if self.console:
            config_table = Table(title="⚙️ Estrategias de Timeframe", show_header=True, header_style="bold yellow")
            config_table.add_column("Estrategia", style="cyan", width=12)
            config_table.add_column("TF Principal", style="white", width=12)
            config_table.add_column("TF Secundarios", style="green", width=15)
            config_table.add_column("Features", style="yellow", width=20)
            config_table.add_column("Horizonte", style="blue", width=10)
            
            for strategy, config in self.timeframe_configs.items():
                config_table.add_row(
                    strategy.title(),
                    config['primary'],
                    ", ".join(config['secondary']),
                    ", ".join(config['features']),
                    f"{config['target_horizon']} períodos"
                )
            
            self.console.print(config_table)
    
    def _predefined_strategy(self):
        """Usar estrategia predefinida"""
        if self.console:
            strategy_choices = list(self.timeframe_configs.keys())
            strategy = Prompt.ask("Selecciona estrategia", choices=strategy_choices, default="swing")
            
            symbol = Prompt.ask("Símbolo a entrenar", default="BTC/USDT")
            
            with self.console.status(f"[bold green]Entrenando estrategia {strategy}..."):
                result = self._train_strategy(strategy, symbol)
                self._show_training_result(result)
        else:
            strategy = input("Estrategia [swing]: ") or "swing"
            symbol = input("Símbolo [BTC/USDT]: ") or "BTC/USDT"
            
            print(f"Entrenando estrategia {strategy}...")
            result = self._train_strategy(strategy, symbol)
            self._show_training_result(result)
    
    def _train_strategy(self, strategy: str, symbol: str = "BTC/USDT") -> Dict:
        """Entrenar una estrategia específica"""
        return self.train_multi_timeframe_model(symbol, strategy)
    
    def _custom_configuration(self):
        """Configuración personalizada de timeframes"""
        if self.console:
            self.console.print("\n[cyan]🔧 CONFIGURACIÓN PERSONALIZADA[/cyan]")
            
            symbol = Prompt.ask("Símbolo", default="BTC/USDT")
            
            # Seleccionar timeframes
            available_tfs = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
            self.console.print("\n[cyan]Timeframes disponibles:[/cyan]")
            for tf in available_tfs:
                self.console.print(f"  • {tf}")
            
            tf_input = Prompt.ask("Timeframes (separados por coma)", default="1h,4h,1d")
            timeframes = [tf.strip() for tf in tf_input.split(",")]
            
            # Validar timeframes
            valid_tfs = [tf for tf in timeframes if tf in available_tfs]
            
            if len(valid_tfs) < 2:
                self.console.print("[red]❌ Se necesitan al menos 2 timeframes válidos[/red]")
                return
            
            # Crear configuración temporal
            custom_config = {
                'primary': valid_tfs[0],
                'secondary': valid_tfs[1:],
                'features': ['technical', 'volume', 'trend'],
                'target_horizon': 3
            }
            
            # Temporalmente agregar a configuraciones
            self.timeframe_configs['custom'] = custom_config
            
            # Entrenar
            result = self.train_multi_timeframe_model(symbol, 'custom')
            self._show_training_result(result)
            
            # Remover configuración temporal
            del self.timeframe_configs['custom']
        
        else:
            symbol = input("Símbolo [BTC/USDT]: ") or "BTC/USDT"
            tf_input = input("Timeframes (ej: 1h,4h,1d): ") or "1h,4h,1d"
            timeframes = [tf.strip() for tf in tf_input.split(",")]
            
            # Crear configuración simple
            if len(timeframes) >= 2:
                custom_config = {
                    'primary': timeframes[0],
                    'secondary': timeframes[1:],
                    'features': ['technical', 'volume', 'trend'],
                    'target_horizon': 3
                }
                self.timeframe_configs['custom'] = custom_config
                result = self.train_multi_timeframe_model(symbol, 'custom')
                self._show_training_result(result)
                del self.timeframe_configs['custom']
    
    def _train_all_timeframes(self):
        """Entrenar todos los timeframes disponibles"""
        if self.console:
            symbol = Prompt.ask("Símbolo", default="BTC/USDT")
            
            self.console.print("\n[cyan]🚀 ENTRENANDO TODAS LAS ESTRATEGIAS[/cyan]")
            
            all_results = {}
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Entrenando estrategias...", total=len(self.timeframe_configs))
                
                for strategy in self.timeframe_configs.keys():
                    progress.update(task, advance=1, description=f"Entrenando {strategy}...")
                    result = self.train_multi_timeframe_model(symbol, strategy)
                    all_results[strategy] = result
            
            # Mostrar resumen de resultados
            self._show_comparison_results(all_results)
        
        else:
            symbol = input("Símbolo [BTC/USDT]: ") or "BTC/USDT"
            print("Entrenando todas las estrategias...")
            
            all_results = {}
            for strategy in self.timeframe_configs.keys():
                print(f"  - {strategy}...")
                result = self.train_multi_timeframe_model(symbol, strategy)
                all_results[strategy] = result
            
            self._show_comparison_results(all_results)
    
    def _compare_strategies(self):
        """Comparar diferentes estrategias"""
        if self.console:
            self.console.print("\n[cyan]📊 COMPARACIÓN DE ESTRATEGIAS[/cyan]")
            
            # Buscar resultados existentes
            result_files = list(self.multi_tf_dir.glob("multi_tf_*.json"))
            
            if not result_files:
                self.console.print("[yellow]⚠️ No hay resultados guardados para comparar[/yellow]")
                return
            
            # Agrupar por símbolo
            symbol_results = {}
            for file in result_files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    symbol = data.get('symbol', 'Unknown')
                    if symbol not in symbol_results:
                        symbol_results[symbol] = []
                    symbol_results[symbol].append(data)
                
                except Exception as e:
                    self.console.print(f"[red]Error leyendo {file.name}: {e}[/red]")
            
            # Mostrar comparación
            for symbol, results in symbol_results.items():
                self._show_symbol_comparison(symbol, results)
        
        else:
            print("📊 Comparando estrategias...")
            # Implementación simple
            result_files = list(self.multi_tf_dir.glob("multi_tf_*.json"))
            if result_files:
                for file in result_files[:5]:  # Mostrar los 5 más recientes
                    try:
                        with open(file, 'r') as f:
                            data = json.load(f)
                        print(f"  {data.get('symbol')} - {data.get('strategy')}: R² = {data.get('results', {}).get('random_forest', {}).get('r2', 'N/A')}")
                    except:
                        pass
    
    def _analyze_results(self):
        """Analizar resultados de modelos multi-timeframe"""
        if self.console:
            self.console.print("\n[cyan]📈 ANÁLISIS DE RESULTADOS[/cyan]")
            
            # Buscar archivos de resultados
            result_files = list(self.multi_tf_dir.glob("multi_tf_*.json"))
            
            if not result_files:
                self.console.print("[yellow]⚠️ No hay resultados para analizar[/yellow]")
                return
            
            # Cargar y analizar resultados
            all_results = []
            for file in result_files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    all_results.append(data)
                except Exception as e:
                    self.console.print(f"[red]Error leyendo {file.name}: {e}[/red]")
            
            if all_results:
                self._show_detailed_analysis(all_results)
        
        else:
            print("📈 Analizando resultados...")
            result_files = list(self.multi_tf_dir.glob("multi_tf_*.json"))
            print(f"Encontrados {len(result_files)} archivos de resultados")
    
    def _configure_strategies(self):
        """Configurar estrategias de timeframe"""
        if self.console:
            self.console.print("\n[cyan]⚙️ CONFIGURACIÓN DE ESTRATEGIAS[/cyan]")
            
            # Mostrar configuraciones actuales
            self._show_timeframe_configs()
            
            modify = Confirm.ask("¿Modificar alguna estrategia?", default=False)
            
            if modify:
                strategy_choices = list(self.timeframe_configs.keys())
                strategy = Prompt.ask("Estrategia a modificar", choices=strategy_choices)
                
                config = self.timeframe_configs[strategy]
                
                self.console.print(f"\n[cyan]Configuración actual de {strategy}:[/cyan]")
                self.console.print(f"  TF Principal: {config['primary']}")
                self.console.print(f"  TF Secundarios: {', '.join(config['secondary'])}")
                self.console.print(f"  Features: {', '.join(config['features'])}")
                self.console.print(f"  Horizonte: {config['target_horizon']}")
                
                # Permitir modificaciones básicas
                new_horizon = IntPrompt.ask("Nuevo horizonte de predicción", default=config['target_horizon'])
                config['target_horizon'] = new_horizon
                
                self.console.print("[green]✅ Configuración actualizada[/green]")
        
        else:
            print("⚙️ Configurando estrategias...")
            # Implementación simple
            for strategy, config in self.timeframe_configs.items():
                print(f"{strategy}: {config['primary']} + {', '.join(config['secondary'])}")
    
    def _show_training_result(self, result: Dict):
        """Mostrar resultado del entrenamiento"""
        if result.get('error'):
            if self.console:
                self.console.print(f"[red]❌ {result['error']}[/red]")
            else:
                print(f"❌ {result['error']}")
            return
        
        if self.console and result.get('success'):
            save_data = result.get('save_data', {})
            results = result.get('results', {})
            
            self.console.print("\n[green]✅ ENTRENAMIENTO COMPLETADO[/green]")
            
            # Información básica
            info_table = Table(title="📊 Información del Entrenamiento", show_header=True)
            info_table.add_column("Métrica", style="cyan")
            info_table.add_column("Valor", style="white")
            
            info_table.add_row("Símbolo", save_data.get('symbol', 'N/A'))
            info_table.add_row("Estrategia", save_data.get('strategy', 'N/A'))
            info_table.add_row("Timeframes", ', '.join(save_data.get('timeframes', [])))
            info_table.add_row("Tamaño dataset", str(save_data.get('dataset_size', 'N/A')))
            info_table.add_row("Features", str(save_data.get('features_count', 'N/A')))
            
            self.console.print(info_table)
            
            # Resultados de modelos
            if results:
                models_table = Table(title="🤖 Resultados de Modelos", show_header=True, header_style="bold green")
                models_table.add_column("Modelo", style="cyan")
                models_table.add_column("R² Score", style="green")
                models_table.add_column("MSE", style="yellow")
                models_table.add_column("MAE", style="blue")
                
                for model_name, model_result in results.items():
                    if isinstance(model_result, dict):
                        r2 = f"{model_result.get('r2', 0):.4f}"
                        mse = f"{model_result.get('mse', 0):.6f}"
                        mae = f"{model_result.get('mae', 0):.6f}"
                        models_table.add_row(model_name, r2, mse, mae)
                
                self.console.print(models_table)
        
        elif result.get('success'):
            save_data = result.get('save_data', {})
            print(f"✅ Entrenamiento completado para {save_data.get('symbol', 'N/A')}")
            print(f"   Estrategia: {save_data.get('strategy', 'N/A')}")
            print(f"   Dataset: {save_data.get('dataset_size', 'N/A')} registros")
    
    def _show_comparison_results(self, all_results: Dict):
        """Mostrar comparación de resultados"""
        if self.console:
            comparison_table = Table(title="📊 Comparación de Estrategias", show_header=True, header_style="bold magenta")
            comparison_table.add_column("Estrategia", style="cyan")
            comparison_table.add_column("Estado", style="white")
            comparison_table.add_column("Mejor R²", style="green")
            comparison_table.add_column("Timeframes", style="yellow")
            
            for strategy, result in all_results.items():
                if result.get('success'):
                    results = result.get('results', {})
                    best_r2 = 0
                    for model_result in results.values():
                        if isinstance(model_result, dict):
                            r2 = model_result.get('r2', 0)
                            best_r2 = max(best_r2, r2)
                    
                    save_data = result.get('save_data', {})
                    timeframes = ', '.join(save_data.get('timeframes', []))
                    
                    comparison_table.add_row(
                        strategy,
                        "✅ Exitoso",
                        f"{best_r2:.4f}",
                        timeframes
                    )
                else:
                    comparison_table.add_row(
                        strategy,
                        "❌ Error",
                        "N/A",
                        "N/A"
                    )
            
            self.console.print(comparison_table)
        else:
            print("\n📊 Comparación de estrategias:")
            for strategy, result in all_results.items():
                status = "✅" if result.get('success') else "❌"
                print(f"  {status} {strategy}")
    
    def _show_symbol_comparison(self, symbol: str, results: List[Dict]):
        """Mostrar comparación para un símbolo específico"""
        if self.console:
            symbol_table = Table(title=f"📊 Resultados para {symbol}", show_header=True, header_style="bold cyan")
            symbol_table.add_column("Estrategia", style="cyan")
            symbol_table.add_column("Fecha", style="dim")
            symbol_table.add_column("R² (RF)", style="green")
            symbol_table.add_column("R² (GB)", style="yellow")
            symbol_table.add_column("Features", style="blue")
            
            for result in sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True):
                strategy = result.get('strategy', 'Unknown')
                timestamp = result.get('timestamp', 'Unknown')[:8]  # Solo fecha
                rf_r2 = result.get('results', {}).get('random_forest', {}).get('r2', 0)
                gb_r2 = result.get('results', {}).get('gradient_boosting', {}).get('r2', 0)
                features = result.get('features_count', 0)
                
                symbol_table.add_row(
                    strategy,
                    timestamp,
                    f"{rf_r2:.4f}",
                    f"{gb_r2:.4f}",
                    str(features)
                )
            
            self.console.print(symbol_table)
    
    def _show_detailed_analysis(self, all_results: List[Dict]):
        """Mostrar análisis detallado de resultados"""
        if self.console:
            # Estadísticas generales
            total_models = len(all_results)
            successful_models = sum(1 for r in all_results if r.get('results'))
            
            # Análisis de rendimiento
            all_r2_scores = []
            for result in all_results:
                results = result.get('results', {})
                for model_result in results.values():
                    if isinstance(model_result, dict):
                        r2 = model_result.get('r2', 0)
                        all_r2_scores.append(r2)
            
            if all_r2_scores:
                avg_r2 = sum(all_r2_scores) / len(all_r2_scores)
                max_r2 = max(all_r2_scores)
                min_r2 = min(all_r2_scores)
                
                stats_table = Table(title="📈 Estadísticas Generales", show_header=True)
                stats_table.add_column("Métrica", style="cyan")
                stats_table.add_column("Valor", style="white")
                
                stats_table.add_row("Total de entrenamientos", str(total_models))
                stats_table.add_row("Entrenamientos exitosos", str(successful_models))
                stats_table.add_row("R² promedio", f"{avg_r2:.4f}")
                stats_table.add_row("Mejor R²", f"{max_r2:.4f}")
                stats_table.add_row("Peor R²", f"{min_r2:.4f}")
                
                self.console.print(stats_table)
            
            # Top 5 mejores modelos
            best_models = []
            for result in all_results:
                symbol = result.get('symbol', 'Unknown')
                strategy = result.get('strategy', 'Unknown')
                results = result.get('results', {})
                
                for model_name, model_result in results.items():
                    if isinstance(model_result, dict):
                        r2 = model_result.get('r2', 0)
                        best_models.append({
                            'symbol': symbol,
                            'strategy': strategy,
                            'model': model_name,
                            'r2': r2
                        })
            
            if best_models:
                best_models.sort(key=lambda x: x['r2'], reverse=True)
                
                top_table = Table(title="🏆 Top 5 Mejores Modelos", show_header=True, header_style="bold gold1")
                top_table.add_column("Ranking", style="gold1")
                top_table.add_column("Símbolo", style="cyan")
                top_table.add_column("Estrategia", style="green")
                top_table.add_column("Modelo", style="blue")
                top_table.add_column("R² Score", style="bold green")
                
                for i, model in enumerate(best_models[:5], 1):
                    top_table.add_row(
                        f"#{i}",
                        model['symbol'],
                        model['strategy'],
                        model['model'],
                        f"{model['r2']:.4f}"
                    )
                
                self.console.print(top_table)
    
    def _create_timeframe_features(self, df: pd.DataFrame, timeframe: str, feature_types: List[str]) -> pd.DataFrame:
        """Crear features específicas para un timeframe"""
        # Añadir prefijo del timeframe a las columnas
        tf_prefix = f"tf_{timeframe}_"
        
        # Features básicas de precio
        if 'price_action' in feature_types:
            df[f'{tf_prefix}return_1'] = df['close'].pct_change()
            df[f'{tf_prefix}return_5'] = df['close'].pct_change(5)
            df[f'{tf_prefix}volatility'] = df[f'{tf_prefix}return_1'].rolling(20).std()
            df[f'{tf_prefix}hl_ratio'] = df['high'] / df['low']
        
        # Features técnicas
        if 'technical' in feature_types:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df[f'{tf_prefix}rsi'] = 100 - (100 / (1 + rs))
            
            # Moving averages
            df[f'{tf_prefix}sma_20'] = df['close'].rolling(20).mean()
            df[f'{tf_prefix}sma_50'] = df['close'].rolling(50).mean()
            df[f'{tf_prefix}price_sma20_ratio'] = df['close'] / df[f'{tf_prefix}sma_20']
        
        # Features de volumen
        if 'volume' in feature_types:
            df[f'{tf_prefix}volume_sma'] = df['volume'].rolling(20).mean()
            df[f'{tf_prefix}volume_ratio'] = df['volume'] / df[f'{tf_prefix}volume_sma']
            
            # OBV simplificado
            price_change = df['close'].diff()
            df[f'{tf_prefix}obv'] = (df['volume'] * np.sign(price_change)).cumsum()
        
        # Features de tendencia
        if 'trend' in feature_types:
            # Trend strength
            df[f'{tf_prefix}trend_20'] = (df['close'] > df['close'].shift(20)).astype(int)
            df[f'{tf_prefix}trend_50'] = (df['close'] > df['close'].shift(50)).astype(int)
        
        # Features de momentum
        if 'momentum' in feature_types:
            df[f'{tf_prefix}momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df[f'{tf_prefix}momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        return df
    
    def _combine_timeframe_data(self, timeframe_data: Dict[str, pd.DataFrame], config: Dict) -> Optional[pd.DataFrame]:
        """Combinar datos de múltiples timeframes"""
        primary_tf = config['primary']
        primary_data = timeframe_data[primary_tf].copy()
        
        # Usar el timeframe principal como base
        result_df = primary_data.copy()
        
        # Añadir features de timeframes secundarios
        for tf in config['secondary']:
            if tf not in timeframe_data:
                continue
            
            secondary_data = timeframe_data[tf].copy()
            
            # Hacer merge por timestamp más cercano
            # Esto es una simplificación - en producción necesitaríamos interpolación más sofisticada
            merged = pd.merge_asof(
                result_df.sort_values('timestamp'),
                secondary_data.sort_values('timestamp'),
                on='timestamp',
                direction='backward',
                suffixes=('', f'_{tf}')
            )
            
            result_df = merged
        
        # Crear target basado en el horizonte configurado
        horizon = config['target_horizon']
        result_df['target'] = result_df['close'].shift(-horizon) / result_df['close'] - 1
        
        # Limpiar datos
        result_df = result_df.dropna()
        
        return result_df if len(result_df) > 0 else None
    
    def _train_multi_timeframe_models(self, data: pd.DataFrame, config: Dict, strategy: str, symbol: str) -> Dict:
        """Entrenar modelos con datos multi-timeframe"""
        try:
            # Preparar features y target
            feature_cols = [col for col in data.columns if col not in ['timestamp', 'target']]
            X = data[feature_cols].fillna(0)
            y = data['target'].fillna(0)
            
            # Split temporal
            split_idx = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Escalar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entrenar múltiples modelos
            models = {}
            results = {}
            
            # Random Forest
            if SKLEARN_AVAILABLE:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train_scaled, y_train)
                rf_pred = rf_model.predict(X_test_scaled)
                
                models['random_forest'] = rf_model
                results['random_forest'] = {
                    'r2_score': r2_score(y_test, rf_pred),
                    'mse': mean_squared_error(y_test, rf_pred),
                    'mae': mean_absolute_error(y_test, rf_pred)
                }
                
                # Gradient Boosting
                gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                gb_model.fit(X_train_scaled, y_train)
                gb_pred = gb_model.predict(X_test_scaled)
                
                models['gradient_boosting'] = gb_model
                results['gradient_boosting'] = {
                    'r2_score': r2_score(y_test, gb_pred),
                    'mse': mean_squared_error(y_test, gb_pred),
                    'mae': mean_absolute_error(y_test, gb_pred)
                }
            
            # Ensemble de timeframes (weighted average)
            if len(results) > 1:
                # Crear ensemble simple
                ensemble_pred = np.mean([
                    results['random_forest']['r2_score'] * rf_pred,
                    results['gradient_boosting']['r2_score'] * gb_pred
                ], axis=0) / np.mean([
                    results['random_forest']['r2_score'],
                    results['gradient_boosting']['r2_score']
                ])
                
                results['ensemble'] = {
                    'r2_score': r2_score(y_test, ensemble_pred),
                    'mse': mean_squared_error(y_test, ensemble_pred),
                    'mae': mean_absolute_error(y_test, ensemble_pred)
                }
            
            # Información adicional
            best_model = max(results.keys(), key=lambda k: results[k]['r2_score'])
            
            return {
                'success': True,
                'strategy': strategy,
                'symbol': symbol,
                'timeframes': [config['primary']] + config['secondary'],
                'features_count': len(feature_cols),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'models': list(results.keys()),
                'results': results,
                'best_model': best_model,
                'best_score': results[best_model]['r2_score'],
                'scaler': scaler,
                'feature_columns': feature_cols
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'strategy': strategy,
                'symbol': symbol
            }
    
    def _save_multi_tf_results(self, results: Dict, strategy: str, symbol: str):
        """Guardar resultados multi-timeframe"""
        if 'success' not in results:
            return
        
        # Crear filename único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_tf_{strategy}_{symbol.replace('/', '_')}_{timestamp}.json"
        filepath = self.multi_tf_dir / filename
        
        # Preparar datos para JSON (remover objetos no serializables)
        save_data = results.copy()
        if 'scaler' in save_data:
            del save_data['scaler']
        
        # Guardar
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
    
    def _show_training_result(self, result: Dict):
        """Mostrar resultado de entrenamiento"""
        if 'error' in result:
            if self.console:
                self.console.print(f"[red]❌ Error: {result['error']}[/red]")
            else:
                print(f"❌ Error: {result['error']}")
        else:
            if self.console:
                content = f"[green]✅ Entrenamiento Multi-TF completado[/green]\n"
                content += f"[cyan]Estrategia:[/cyan] {result['strategy']}\n"
                content += f"[cyan]Símbolo:[/cyan] {result['symbol']}\n"
                content += f"[cyan]Timeframes:[/cyan] {', '.join(result['timeframes'])}\n"
                content += f"[cyan]Features:[/cyan] {result['features_count']}\n"
                content += f"[cyan]Mejor modelo:[/cyan] {result['best_model']}\n"
                best_score = result['best_score']
                score_str = f"{best_score:.4f}" if isinstance(best_score, (int, float)) and not (np.isnan(best_score) or np.isinf(best_score)) else "N/A"
                content += f"[cyan]Best R² Score:[/cyan] {score_str}\n\n"
                
                content += "[yellow]Resultados por modelo:[/yellow]\n"
                for model, metrics in result['results'].items():
                    r2 = metrics['r2_score']
                    mse = metrics['mse']
                    r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) and not (np.isnan(r2) or np.isinf(r2)) else "N/A"
                    mse_str = f"{mse:.6f}" if isinstance(mse, (int, float)) and not (np.isnan(mse) or np.isinf(mse)) else "N/A"
                    content += f"• {model}: R²={r2_str}, MSE={mse_str}\n"
                
                self.console.print(Panel(content, title="🕐 Multi-Timeframe Results", border_style="magenta"))
            else:
                print(f"✅ Entrenamiento completado: {result['strategy']}")
                print(f"   Símbolo: {result['symbol']}")
                print(f"   Timeframes: {', '.join(result['timeframes'])}")
                print(f"   Mejor modelo: {result['best_model']} (R²: {result['best_score']:.4f})")
    
    def _custom_configuration(self):
        """Configuración personalizada de timeframes"""
        if self.console:
            self.console.print("[cyan]⚙️ Configuración personalizada en desarrollo[/cyan]")
        else:
            print("⚙️ Configuración personalizada en desarrollo")
    
    def _train_all_timeframes(self):
        """Entrenar todos los timeframes disponibles"""
        if self.console:
            self.console.print("[cyan]🚀 Entrenamiento completo en desarrollo[/cyan]")
        else:
            print("🚀 Entrenamiento completo en desarrollo")
    
    def _compare_strategies(self):
        """Comparar diferentes estrategias"""
        if self.console:
            self.console.print("[cyan]📊 Comparación de estrategias en desarrollo[/cyan]")
        else:
            print("📊 Comparación de estrategias en desarrollo")
    
    def _analyze_results(self):
        """Analizar resultados de modelos multi-TF"""
        result_files = list(self.multi_tf_dir.glob("multi_tf_*.json"))
        
        if not result_files:
            if self.console:
                self.console.print("[yellow]⚠️ No hay resultados multi-timeframe[/yellow]")
            else:
                print("⚠️ No hay resultados multi-timeframe")
            return
        
        if self.console:
            results_table = Table(title="📊 Resultados Multi-Timeframe", show_header=True, header_style="bold blue")
            results_table.add_column("Estrategia", style="cyan", width=12)
            results_table.add_column("Símbolo", style="white", width=12)
            results_table.add_column("Mejor Modelo", style="green", width=15)
            results_table.add_column("R² Score", style="yellow", width=10)
            results_table.add_column("Fecha", style="dim", width=16)
            
            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    if 'success' in data:
                        results_table.add_row(
                            data.get('strategy', 'N/A'),
                            data.get('symbol', 'N/A'),
                            data.get('best_model', 'N/A'),
                            f"{data.get('best_score', 0):.4f}",
                            result_file.stem.split('_')[-1][:8] if '_' in result_file.stem else 'N/A'
                        )
                except Exception:
                    continue
            
            self.console.print(results_table)
        else:
            print("📊 RESULTADOS MULTI-TIMEFRAME:")
            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    if 'success' in data:
                        print(f"  {data.get('strategy', 'N/A')} - {data.get('symbol', 'N/A')} - R²: {data.get('best_score', 0):.4f}")
                except Exception:
                    continue
    
    def _configure_strategies(self):
        """Configurar estrategias de timeframe"""
        if self.console:
            self.console.print("[cyan]⚙️ Configuración de estrategias en desarrollo[/cyan]")
        else:
            print("⚙️ Configuración de estrategias en desarrollo")
    
    def show_training_menu(self):
        """Alias para show_multi_timeframe_menu"""
        return self.show_multi_timeframe_menu()
    
    def create_multi_timeframe_dataset(self, symbol: str, timeframes: List[str], days: int = 30) -> pd.DataFrame:
        """Crear dataset combinando múltiples timeframes"""
        try:
            if self.console:
                self.console.print(f"[cyan]📊 Combinando datos de {symbol} en timeframes: {', '.join(timeframes)}[/cyan]")
            
            # Cargar datos de cada timeframe
            timeframe_data = {}
            
            for tf in timeframes:
                # Buscar archivo correspondiente
                symbol_clean = symbol.replace('/', '_')
                pattern = f"{symbol_clean}_{tf}_*.csv"
                files = list(self.data_dir.glob(pattern))
                
                if files:
                    # Usar el archivo más reciente
                    latest_file = max(files, key=lambda f: f.stat().st_mtime)
                    df = pd.read_csv(latest_file)
                    
                    # Asegurar formato correcto
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        timeframe_data[tf] = df
                    else:
                        if self.console:
                            self.console.print(f"[yellow]⚠️ Archivo {latest_file.name} sin columna timestamp[/yellow]")
                else:
                    if self.console:
                        self.console.print(f"[yellow]⚠️ No se encontraron datos para {tf}[/yellow]")
            
            if not timeframe_data:
                raise ValueError("No se encontraron datos para ningún timeframe")
            
            # Usar el timeframe con más datos como base
            base_tf = max(timeframe_data.keys(), key=lambda tf: len(timeframe_data[tf]))
            base_df = timeframe_data[base_tf].copy()
            
            if self.console:
                self.console.print(f"[green]Usando {base_tf} como timeframe base ({len(base_df)} registros)[/green]")
            
            # Agregar features de otros timeframes
            for tf, df in timeframe_data.items():
                if tf == base_tf:
                    continue
                
                # Agregar prefijo al nombre de columnas
                feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                # Resamplear datos al timeframe base
                resampled_features = self._resample_to_base_timeframe(
                    df, base_df['timestamp'], tf, base_tf, feature_columns
                )
                
                # Agregar al dataset base
                for col, values in resampled_features.items():
                    base_df[f"{tf}_{col}"] = values
            
            # Agregar features de interacción entre timeframes
            base_df = self._add_multi_timeframe_features(base_df, timeframes)
            
            # Limpiar datos
            base_df = base_df.dropna()
            
            if self.console:
                self.console.print(f"[green]✅ Dataset multi-timeframe creado: {len(base_df)} registros, {len(base_df.columns)} features[/green]")
            
            return base_df
            
        except Exception as e:
            if self.console:
                self.console.print(f"[red]❌ Error creando dataset multi-timeframe: {e}[/red]")
            else:
                print(f"❌ Error: {e}")
            return pd.DataFrame()
    
    def _resample_to_base_timeframe(self, source_df: pd.DataFrame, base_timestamps: pd.Series, 
                                  source_tf: str, base_tf: str, feature_columns: List[str]) -> Dict:
        """Resamplear features de un timeframe a otro"""
        resampled_features = {}
        
        # Configurar el resampling según los timeframes
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        source_minutes = timeframe_minutes.get(source_tf, 60)
        base_minutes = timeframe_minutes.get(base_tf, 60)
        
        # Si el timeframe fuente es mayor que el base, interpolar
        if source_minutes > base_minutes:
            for col in feature_columns:
                if col in source_df.columns:
                    # Interpolación simple
                    interp_values = np.interp(
                        base_timestamps.astype(int),
                        source_df['timestamp'].astype(int),
                        source_df[col].fillna(method='ffill')
                    )
                    resampled_features[col] = interp_values
        else:
            # Si el timeframe fuente es menor, agregar
            for col in feature_columns:
                if col in source_df.columns:
                    # Forward fill para timeframes menores
                    resampled_values = []
                    for timestamp in base_timestamps:
                        # Buscar el valor más reciente
                        mask = source_df['timestamp'] <= timestamp
                        if mask.any():
                            latest_value = source_df.loc[mask, col].iloc[-1]
                            resampled_values.append(latest_value)
                        else:
                            resampled_values.append(np.nan)
                    
                    resampled_features[col] = resampled_values
        
        return resampled_features
    
    def _add_multi_timeframe_features(self, df: pd.DataFrame, timeframes: List[str]) -> pd.DataFrame:
        """Agregar features de interacción entre timeframes"""
        try:
            # Features de correlación entre timeframes
            if len(timeframes) > 1:
                # Comparar RSI entre timeframes
                rsi_columns = [col for col in df.columns if 'rsi' in col.lower()]
                if len(rsi_columns) > 1:
                    for i in range(len(rsi_columns)):
                        for j in range(i+1, len(rsi_columns)):
                            col_diff = f"rsi_diff_{rsi_columns[i]}_{rsi_columns[j]}"
                            df[col_diff] = df[rsi_columns[i]] - df[rsi_columns[j]]
                
                # Convergencia/divergencia de precios entre timeframes
                close_columns = [col for col in df.columns if 'close' in col.lower() and 'target' not in col.lower()]
                if len(close_columns) > 1:
                    for i in range(len(close_columns)):
                        for j in range(i+1, len(close_columns)):
                            if close_columns[i] != close_columns[j]:
                                conv_col = f"convergence_{close_columns[i]}_{close_columns[j]}"
                                df[conv_col] = (df[close_columns[i]] / df[close_columns[j]]) - 1
                
                # Tendencia dominante entre timeframes
                trend_columns = [col for col in df.columns if 'sma' in col.lower() or 'ema' in col.lower()]
                if trend_columns:
                    # Calcular dirección de tendencia para cada timeframe
                    trend_directions = []
                    for col in trend_columns:
                        if col in df.columns:
                            trend_direction = (df[col] > df[col].shift(1)).astype(int)
                            trend_directions.append(trend_direction)
                    
                    if trend_directions:
                        # Consenso de tendencia
                        df['trend_consensus'] = sum(trend_directions) / len(trend_directions)
            
            return df
        
        except Exception as e:
            print(f"❌ Error agregando features multi-timeframe: {e}")
            return df
    
    def train_multi_timeframe_model(self, symbol: str, strategy: str = "swing") -> Dict:
        """Entrenar modelo con múltiples timeframes"""
        try:
            if strategy not in self.timeframe_configs:
                raise ValueError(f"Estrategia {strategy} no válida")
            
            config = self.timeframe_configs[strategy]
            all_timeframes = [config['primary']] + config['secondary']
            
            if self.console:
                self.console.print(f"[cyan]🕐 Entrenando modelo multi-timeframe para {symbol}[/cyan]")
                self.console.print(f"[dim]Estrategia: {strategy} | Timeframes: {', '.join(all_timeframes)}[/dim]")
            
            # Crear dataset combinado
            combined_df = self.create_multi_timeframe_dataset(symbol, all_timeframes)
            
            if combined_df.empty:
                return {'error': 'No se pudo crear dataset combinado'}
            
            # Preparar features y targets
            feature_columns = [col for col in combined_df.columns 
                             if col not in ['timestamp', 'target_1h', 'target_4h', 'target_1d',
                                          'direction_1h', 'direction_4h', 'direction_1d']]
            
            X = combined_df[feature_columns].fillna(0)
            
            # Seleccionar target según horizonte de estrategia
            horizon = config['target_horizon']
            if horizon <= 1:
                y = combined_df['target_1h'].fillna(0)
            elif horizon <= 5:
                y = combined_df['target_4h'].fillna(0)
            else:
                y = combined_df['target_1d'].fillna(0)
            
            # Dividir datos
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Entrenar modelos de ensemble
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            results = {}
            
            for model_name, model in models.items():
                if self.console:
                    with self.console.status(f"[bold green]Entrenando {model_name}..."):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Métricas
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        results[model_name] = {
                            'model': model,
                            'r2': r2,
                            'mse': mse,
                            'mae': mae,
                            'predictions': y_pred
                        }
                
                else:
                    print(f"Entrenando {model_name}...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    results[model_name] = {
                        'model': model,
                        'r2': r2,
                        'mse': mse,
                        'mae': mae,
                        'predictions': y_pred
                    }
            
            # Guardar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.multi_tf_dir / f"multi_tf_{symbol.replace('/', '_')}_{strategy}_{timestamp}.json"
            
            # Preparar datos para guardar (sin objetos model)
            save_results = {}
            for model_name, result in results.items():
                save_results[model_name] = {
                    'r2': result['r2'],
                    'mse': result['mse'],
                    'mae': result['mae']
                }
            
            save_data = {
                'symbol': symbol,
                'strategy': strategy,
                'timeframes': all_timeframes,
                'dataset_size': len(combined_df),
                'features_count': len(feature_columns),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'results': save_results,
                'timestamp': timestamp
            }
            
            with open(results_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            if self.console:
                self.console.print(f"[green]✅ Entrenamiento completado. Resultados guardados en {results_file.name}[/green]")
            
            return {
                'success': True,
                'results': results,
                'save_data': save_data,
                'file': str(results_file)
            }
        
        except Exception as e:
            error_msg = f"Error entrenando modelo multi-timeframe: {e}"
            if self.console:
                self.console.print(f"[red]❌ {error_msg}[/red]")
            else:
                print(f"❌ {error_msg}")
            return {'error': error_msg}


def main():
    """Función principal para pruebas"""
    trainer = MultiTimeframeTrainer()
    trainer.show_multi_timeframe_menu()


if __name__ == "__main__":
    main()
