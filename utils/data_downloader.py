#!/usr/bin/env python3
"""
🚀 DESCARGADOR DE DATOS PARA HYPERION3
Sistema avanzado para descargar datos de criptomonedas con múltiples timeframes
"""

import os
import sys
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from pathlib import Path

# Rich imports para UI
try:
    from rich.console import Console
    from utils.safe_progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class CryptoDataDownloader:
    """Descargador avanzado de datos de criptomonedas"""
    
    def __init__(self):
        """Inicializar descargador"""
        self.console = Console() if RICH_AVAILABLE else None
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Exchanges disponibles
        self.exchanges = {
            'binance': ccxt.binance(),
            'coinbase': ccxt.coinbase(),
            'kraken': ccxt.kraken(),
            'bybit': ccxt.bybit(),
            'okx': ccxt.okx()
        }
        
        # Timeframes disponibles
        self.timeframes = {
            '1m': '1 minuto',
            '3m': '3 minutos',
            '5m': '5 minutos',
            '15m': '15 minutos',
            '30m': '30 minutos',
            '1h': '1 hora',
            '2h': '2 horas',
            '4h': '4 horas',
            '6h': '6 horas',
            '8h': '8 horas',
            '12h': '12 horas',
            '1d': '1 día',
            '3d': '3 días',
            '1w': '1 semana',
            '1M': '1 mes'
        }
        
        # Símbolos populares por categorías
        self.popular_symbols = {
            'Major': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT'],
            'DeFi': ['UNI/USDT', 'LINK/USDT', 'AAVE/USDT', 'COMP/USDT', 'MKR/USDT'],
            'Layer1': ['SOL/USDT', 'AVAX/USDT', 'DOT/USDT', 'ATOM/USDT', 'NEAR/USDT'],
            'Meme': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT'],
            'AI': ['FET/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'TAO/USDT', 'RNDR/USDT']
        }
        
        # Configuración por defecto
        self.default_config = {
            'exchange': 'binance',
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'days': 30,
            'include_volume': True,
            'clean_data': True
        }
    
    def show_download_menu(self):
        """Mostrar menú de descarga de datos"""
        if not self.console:
            return self._simple_download_menu()
        
        self.console.clear()
        
        # Header
        header_panel = Panel.fit(
            "[bold cyan]📊 DESCARGA DE DATOS DE CRIPTOMONEDAS[/bold cyan]\n"
            "[dim]Sistema avanzado para descargar datos históricos con múltiples timeframes[/dim]",
            border_style="cyan"
        )
        self.console.print(header_panel)
        
        # Opciones principales
        options_table = Table(title="📊 Opciones de Descarga", show_header=True, header_style="bold magenta")
        options_table.add_column("ID", style="cyan", width=6)
        options_table.add_column("Opción", style="white", width=25)
        options_table.add_column("Descripción", style="dim", width=35)
        
        download_options = [
            ("1", "Descarga Rápida", "Descarga con configuración por defecto"),
            ("2", "Descarga Personalizada", "Seleccionar exchange, símbolo, timeframe"),
            ("3", "Múltiples Timeframes", "Descargar varios timeframes del mismo símbolo"),
            ("4", "Múltiples Símbolos", "Descargar múltiples criptomonedas"),
            ("5", "Dataset Completo", "Crear dataset completo para entrenamiento"),
            ("6", "Ver Datos Descargados", "Explorar datos existentes"),
            ("7", "Configuración", "Modificar configuración por defecto")
        ]
        
        for opt_id, option, description in download_options:
            options_table.add_row(opt_id, option, description)
        
        self.console.print(options_table)
        
        choice = Prompt.ask("🎯 Selecciona opción", choices=["1", "2", "3", "4", "5", "6", "7", "back"], default="2")
        
        if choice == "back":
            return
        elif choice == "1":
            return self._quick_download()
        elif choice == "2":
            return self._custom_download()
        elif choice == "3":
            return self._multi_timeframe_download()
        elif choice == "4":
            return self._multi_symbol_download()
        elif choice == "5":
            return self._complete_dataset_download()
        elif choice == "6":
            return self._explore_downloaded_data()
        elif choice == "7":
            return self._configure_defaults()
    
    def _simple_download_menu(self):
        """Menú simple de descarga"""
        print("\n📊 DESCARGA DE DATOS")
        print("=" * 40)
        print("1. Descarga rápida (BTC/USDT, 1h, 30 días)")
        print("2. Descarga personalizada")
        print("3. Múltiples timeframes")
        print("4. back - Volver")
        
        choice = input("\nSelecciona: ")
        if choice == "1":
            self._quick_download()
        elif choice == "2":
            self._custom_download()
        elif choice == "3":
            self._multi_timeframe_download()
    
    def _quick_download(self):
        """Descarga rápida con configuración por defecto"""
        config = self.default_config.copy()
        
        if self.console:
            with self.console.status("[bold green]Descargando datos...") as status:
                result = self._download_data(config)
                self._show_download_result(result)
        else:
            print("Descargando datos...")
            result = self._download_data(config)
            self._show_download_result(result)
    
    def _custom_download(self):
        """Descarga personalizada con configuración completa"""
        config = {}
        
        if self.console:
            # Exchange selection
            self.console.print("\n[cyan]📈 Exchanges disponibles:[/cyan]")
            for exchange in self.exchanges.keys():
                self.console.print(f"  • {exchange}")
            
            config['exchange'] = Prompt.ask("Exchange", choices=list(self.exchanges.keys()), default="binance")
            
            # Symbol selection con categorías
            self.console.print("\n[cyan]💰 Selección de símbolo:[/cyan]")
            show_symbols = Confirm.ask("¿Mostrar símbolos populares?", default=True)
            
            if show_symbols:
                self._show_popular_symbols()
            
            config['symbol'] = Prompt.ask("Símbolo (ej: BTC/USDT)", default="BTC/USDT")
            
            # Timeframe selection
            self.console.print("\n[cyan]⏰ Timeframes disponibles:[/cyan]")
            for tf, desc in self.timeframes.items():
                self.console.print(f"  {tf}: {desc}")
            
            config['timeframe'] = Prompt.ask("Timeframe", choices=list(self.timeframes.keys()), default="1h")
            
            # Enhanced time range selection
            self.console.print("\n[cyan]📅 Selección de rango de fechas:[/cyan]")
            date_method = Prompt.ask(
                "Método de selección", 
                choices=["days", "dates", "preset"], 
                default="days"
            )
            
            if date_method == "days":
                config['days'] = IntPrompt.ask("Días de datos históricos", default=30)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=config['days'])
            elif date_method == "dates":
                # Selección manual de fechas
                start_date_str = Prompt.ask(
                    "Fecha inicio (YYYY-MM-DD)", 
                    default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                )
                end_date_str = Prompt.ask(
                    "Fecha fin (YYYY-MM-DD)", 
                    default=datetime.now().strftime("%Y-%m-%d")
                )
                try:
                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                    config['days'] = (end_date - start_date).days
                except ValueError:
                    self.console.print("[red]❌ Formato de fecha inválido, usando valores por defecto[/red]")
                    config['days'] = 30
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=config['days'])
            else:  # preset
                presets = {
                    "1": ("1 semana", 7),
                    "2": ("2 semanas", 14),
                    "3": ("1 mes", 30),
                    "4": ("3 meses", 90),
                    "5": ("6 meses", 180),
                    "6": ("1 año", 365),
                    "7": ("2 años", 730)
                }
                
                self.console.print("\n[cyan]Presets disponibles:[/cyan]")
                for key, (desc, days_val) in presets.items():
                    self.console.print(f"  {key}: {desc} ({days_val} días)")
                
                preset_choice = Prompt.ask("Selecciona preset", choices=list(presets.keys()), default="3")
                config['days'] = presets[preset_choice][1]
                end_date = datetime.now()
                start_date = end_date - timedelta(days=config['days'])
            
            config['start_date'] = start_date
            config['end_date'] = end_date
            
            # Options
            config['include_volume'] = Confirm.ask("¿Incluir datos de volumen?", default=True)
            config['clean_data'] = Confirm.ask("¿Limpiar datos automáticamente?", default=True)
            config['save_raw'] = Confirm.ask("¿Guardar datos sin procesar también?", default=False)
            
            self.console.print(f"\n[green]📊 Descargando {config['symbol']} ({config['timeframe']}) desde {config['exchange']}...[/green]")
            self.console.print(f"[dim]Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')} ({config['days']} días)[/dim]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Descargando datos...", total=100)
                result = self._download_data(config, progress, task)
            
            self._show_download_result(result)
            
        else:
            # Versión simple mejorada
            print("\n📅 CONFIGURACIÓN DE DESCARGA")
            config['exchange'] = input("Exchange [binance]: ") or "binance"
            config['symbol'] = input("Símbolo [BTC/USDT]: ") or "BTC/USDT"
            config['timeframe'] = input("Timeframe [1h]: ") or "1h"
            
            # Selección de fechas simplificada
            print("\nMétodos de selección de fecha:")
            print("1. Días hacia atrás (ej: 30)")
            print("2. Fecha específica (ej: 2024-01-01 a 2024-02-01)")
            method = input("Método [1]: ") or "1"
            
            if method == "2":
                start_str = input("Fecha inicio (YYYY-MM-DD): ")
                end_str = input("Fecha fin (YYYY-MM-DD): ")
                try:
                    start_date = datetime.strptime(start_str, "%Y-%m-%d")
                    end_date = datetime.strptime(end_str, "%Y-%m-%d")
                    config['start_date'] = start_date
                    config['end_date'] = end_date
                    config['days'] = (end_date - start_date).days
                except:
                    print("❌ Formato inválido, usando 30 días por defecto")
                    config['days'] = 30
            else:
                config['days'] = int(input("Días [30]: ") or "30")
            
            config['include_volume'] = True
            config['clean_data'] = True
            config['save_raw'] = False
            
            print("Descargando...")
            result = self._download_data(config)
            self._show_download_result(result)
    
    def _show_popular_symbols(self):
        """Mostrar símbolos populares por categoría"""
        if self.console:
            symbols_table = Table(title="💰 Símbolos Populares", show_header=True, header_style="bold yellow")
            symbols_table.add_column("Categoría", style="cyan", width=12)
            symbols_table.add_column("Símbolos", style="white", width=50)
            
            for category, symbols in self.popular_symbols.items():
                symbols_table.add_row(category, ", ".join(symbols))
            
            self.console.print(symbols_table)
    
    def _multi_timeframe_download(self):
        """Descarga con múltiples timeframes"""
        if self.console:
            # Configuración base
            exchange = Prompt.ask("Exchange", choices=list(self.exchanges.keys()), default="binance")
            symbol = Prompt.ask("Símbolo", default="BTC/USDT")
            days = IntPrompt.ask("Días de datos", default=30)
            
            # Seleccionar timeframes
            self.console.print("\n[cyan]Timeframes disponibles:[/cyan]")
            for tf, desc in self.timeframes.items():
                self.console.print(f"  {tf}: {desc}")
            
            timeframes_input = Prompt.ask("Timeframes (separados por coma)", default="1h,4h,1d")
            timeframes = [tf.strip() for tf in timeframes_input.split(",")]
            
            # Validar timeframes
            valid_timeframes = [tf for tf in timeframes if tf in self.timeframes]
            
            if not valid_timeframes:
                self.console.print("[red]❌ No se seleccionaron timeframes válidos[/red]")
                return
            
            self.console.print(f"\n[green]📊 Descargando {symbol} en {len(valid_timeframes)} timeframes...[/green]")
            
            results = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Descargando...", total=len(valid_timeframes))
                
                for timeframe in valid_timeframes:
                    progress.update(task, description=f"Descargando {timeframe}")
                    
                    config = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'days': days,
                        'include_volume': True,
                        'clean_data': True
                    }
                    
                    result = self._download_data(config)
                    results.append(result)
                    progress.advance(task)
            
            self._show_multi_download_results(results)
        else:
            # Versión simple
            symbol = input("Símbolo [BTC/USDT]: ") or "BTC/USDT"
            timeframes = ["1h", "4h", "1d"]  # Por defecto
            
            print(f"Descargando {symbol} en múltiples timeframes...")
            results = []
            for tf in timeframes:
                config = {
                    'exchange': 'binance',
                    'symbol': symbol,
                    'timeframe': tf,
                    'days': 30,
                    'include_volume': True,
                    'clean_data': True
                }
                result = self._download_data(config)
                results.append(result)
            
            self._show_multi_download_results(results)
    
    def _multi_symbol_download(self):
        """Descarga múltiples símbolos"""
        if self.console:
            # Mostrar categorías
            self._show_popular_symbols()
            
            category = Prompt.ask("Categoría", choices=list(self.popular_symbols.keys()), default="Major")
            symbols = self.popular_symbols[category]
            
            timeframe = Prompt.ask("Timeframe", choices=list(self.timeframes.keys()), default="1h")
            days = IntPrompt.ask("Días de datos", default=30)
            
            self.console.print(f"\n[green]📊 Descargando {len(symbols)} símbolos de {category}...[/green]")
            
            results = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Descargando...", total=len(symbols))
                
                for symbol in symbols:
                    progress.update(task, description=f"Descargando {symbol}")
                    
                    config = {
                        'exchange': 'binance',
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'days': days,
                        'include_volume': True,
                        'clean_data': True
                    }
                    
                    result = self._download_data(config)
                    results.append(result)
                    progress.advance(task)
                    
                    # Pausa para evitar rate limits
                    time.sleep(0.5)
            
            self._show_multi_download_results(results)
        else:
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
            print(f"Descargando {len(symbols)} símbolos...")
            
            results = []
            for symbol in symbols:
                config = {
                    'exchange': 'binance',
                    'symbol': symbol,
                    'timeframe': '1h',
                    'days': 30,
                    'include_volume': True,
                    'clean_data': True
                }
                result = self._download_data(config)
                results.append(result)
                time.sleep(0.5)
            
            self._show_multi_download_results(results)
    
    def _complete_dataset_download(self):
        """Crear dataset completo para entrenamiento"""
        if self.console:
            dataset_panel = Panel.fit(
                "[bold green]📊 CREACIÓN DE DATASET COMPLETO[/bold green]\n"
                "[dim]Este proceso descargará múltiples símbolos y timeframes para crear un dataset completo[/dim]",
                border_style="green"
            )
            self.console.print(dataset_panel)
            
            # Configuración del dataset
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
            timeframes = ["1h", "4h", "1d"]
            days = IntPrompt.ask("Días de datos históricos", default=90)
            
            total_downloads = len(symbols) * len(timeframes)
            
            self.console.print(f"\n[yellow]⚠️ Se descargarán {total_downloads} archivos de datos[/yellow]")
            if not Confirm.ask("¿Continuar?", default=True):
                return
            
            results = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Creando dataset...", total=total_downloads)
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        progress.update(task, description=f"{symbol} - {timeframe}")
                        
                        config = {
                            'exchange': 'binance',
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'days': days,
                            'include_volume': True,
                            'clean_data': True
                        }
                        
                        result = self._download_data(config)
                        results.append(result)
                        progress.advance(task)
                        
                        # Pausa para rate limits
                        time.sleep(1)
            
            # Crear índice del dataset
            self._create_dataset_index(results)
            self._show_multi_download_results(results)
        else:
            print("📊 Creando dataset completo...")
            # Implementación simple
            symbols = ["BTC/USDT", "ETH/USDT"]
            timeframes = ["1h", "1d"]
            
            results = []
            for symbol in symbols:
                for tf in timeframes:
                    config = {
                        'exchange': 'binance',
                        'symbol': symbol,
                        'timeframe': tf,
                        'days': 30,
                        'include_volume': True,
                        'clean_data': True
                    }
                    result = self._download_data(config)
                    results.append(result)
                    time.sleep(1)
            
            self._show_multi_download_results(results)
    
    def _download_data(self, config: Dict, progress=None, task=None) -> Dict:
        """Descargar datos con la configuración especificada"""
        try:
            # Validar configuración
            exchange_name = config.get('exchange', 'binance')
            symbol = config.get('symbol', 'BTC/USDT')
            timeframe = config.get('timeframe', '1h')
            days = config.get('days', 30)
            
            if exchange_name not in self.exchanges:
                return {'error': f'Exchange {exchange_name} no soportado', 'config': config}
            
            if timeframe not in self.timeframes:
                return {'error': f'Timeframe {timeframe} no soportado', 'config': config}
            
            # Configurar exchange
            exchange = self.exchanges[exchange_name]
            
            # Calcular fechas
            if 'start_date' in config and 'end_date' in config:
                start_time = config['start_date']
                end_time = config['end_date']
            else:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days)
            
            if progress and task:
                progress.update(task, advance=20, description="Configurando exchange...")
            
            # Descargar datos
            since = int(start_time.timestamp() * 1000)
            
            if progress and task:
                progress.update(task, advance=30, description="Descargando datos OHLCV...")
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            
            if not ohlcv:
                return {'error': 'No se pudieron descargar datos', 'config': config}
            
            if progress and task:
                progress.update(task, advance=20, description="Procesando datos...")
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Filtrar por rango de fechas si se especificó
            if 'start_date' in config and 'end_date' in config:
                df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
            
            # Limpiar datos si se requiere
            if config.get('clean_data', True):
                df = self._clean_data(df)
            
            if progress and task:
                progress.update(task, advance=20, description="Guardando archivos...")
            
            # Generar nombre de archivo
            symbol_clean = symbol.replace('/', '_')
            date_suffix = start_time.strftime('%Y%m%d')
            if 'end_date' in config:
                date_suffix += f"_{config['end_date'].strftime('%Y%m%d')}"
            filename = f"{symbol_clean}_{timeframe}_{date_suffix}.csv"
            filepath = self.data_dir / filename
            
            # Guardar archivo procesado
            df.to_csv(filepath, index=False)
            
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange_name,
                'rows': len(df),
                'start_date': start_time.strftime('%Y-%m-%d %H:%M'),
                'end_date': end_time.strftime('%Y-%m-%d %H:%M'),
                'file': str(filepath),  # Keep 'file' key for compatibility
                'filepath': str(filepath),
                'size_mb': filepath.stat().st_size / (1024 * 1024),
                'date_range': f"{df['timestamp'].min()} - {df['timestamp'].max()}",
                'config': config
            }
            
            # Guardar datos sin procesar si se requiere
            if config.get('save_raw', False):
                raw_filename = f"raw_{filename}"
                raw_filepath = self.data_dir / raw_filename
                # Recrear DataFrame original sin limpiar
                raw_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], unit='ms')
                raw_df.to_csv(raw_filepath, index=False)
                result['raw_filepath'] = str(raw_filepath)
            
            if progress and task:
                progress.update(task, advance=10, description="¡Completado!")
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'symbol': config.get('symbol', 'N/A'),
                'timeframe': config.get('timeframe', 'N/A'),
                'config': config
            }
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar y validar datos"""
        # Eliminar duplicados
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp')
        
        # Eliminar filas con valores nulos en precios
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Validar que high >= low y que open/close estén en el rango
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['volume'] >= 0)
        ]
        
        # Eliminar outliers extremos (precio 0 o negativo)
        df = df[df['close'] > 0]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _show_download_result(self, result: Dict):
        """Mostrar resultado de descarga individual"""
        if 'error' in result:
            if self.console:
                self.console.print(f"[red]❌ Error: {result['error']}[/red]")
            else:
                print(f"❌ Error: {result['error']}")
        else:
            if self.console:
                content = f"[green]✅ Descarga completada[/green]\n"
                content += f"[cyan]Símbolo:[/cyan] {result['symbol']}\n"
                content += f"[cyan]Timeframe:[/cyan] {result['timeframe']}\n"
                content += f"[cyan]Datos:[/cyan] {result['rows']:,} filas\n"
                content += f"[cyan]Archivo:[/cyan] {result['file']}\n"
                content += f"[cyan]Tamaño:[/cyan] {result['size_mb']:.2f} MB\n"
                content += f"[cyan]Rango:[/cyan] {result['date_range']}"
                
                self.console.print(Panel(content, title="📊 Resultado", border_style="green"))
            else:
                print(f"✅ Descarga completada: {result['symbol']} {result['timeframe']}")
                print(f"   Filas: {result['rows']:,}")
                print(f"   Archivo: {result['file']}")
    
    def _show_multi_download_results(self, results: List[Dict]):
        """Mostrar resultados de descargas múltiples"""
        successful = [r for r in results if 'success' in r]
        failed = [r for r in results if 'error' in r]
        
        if self.console:
            # Tabla de resumen
            summary_table = Table(title="📊 Resumen de Descargas", show_header=True, header_style="bold blue")
            summary_table.add_column("Símbolo", style="cyan", width=12)
            summary_table.add_column("Timeframe", style="white", width=10)
            summary_table.add_column("Filas", style="green", width=10)
            summary_table.add_column("Tamaño", style="yellow", width=10)
            summary_table.add_column("Estado", style="blue", width=10)
            
            for result in results:
                if 'success' in result:
                    summary_table.add_row(
                        result['symbol'],
                        result['timeframe'],
                        f"{result['rows']:,}",
                        f"{result['size_mb']:.1f}MB",
                        "✅ OK"
                    )
                else:
                    summary_table.add_row(
                        result.get('symbol', 'N/A'),
                        result.get('timeframe', 'N/A'),
                        "N/A",
                        "N/A",
                        "❌ Error"
                    )
            
            self.console.print(summary_table)
            
            # Estadísticas
            total_rows = sum(r.get('rows', 0) for r in successful)
            total_size = sum(r.get('size_mb', 0) for r in successful)
            
            stats_text = f"[green]📈 Estadísticas:[/green]\n"
            stats_text += f"• Exitosas: {len(successful)}\n"
            stats_text += f"• Fallidas: {len(failed)}\n"
            stats_text += f"• Total filas: {total_rows:,}\n"
            stats_text += f"• Tamaño total: {total_size:.2f} MB"
            
            self.console.print(Panel(stats_text, title="📊 Resumen", border_style="green"))
        else:
            print(f"📊 RESUMEN: {len(successful)} exitosas, {len(failed)} fallidas")
            for result in successful:
                print(f"  ✅ {result['symbol']} {result['timeframe']}: {result['rows']:,} filas")
    
    def _explore_downloaded_data(self):
        """Explorar datos descargados"""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            if self.console:
                self.console.print("[yellow]⚠️ No hay datos descargados[/yellow]")
            else:
                print("⚠️ No hay datos descargados")
            return
        
        if self.console:
            files_table = Table(title="📂 Datos Descargados", show_header=True, header_style="bold green")
            files_table.add_column("Archivo", style="cyan", width=30)
            files_table.add_column("Tamaño", style="white", width=10)
            files_table.add_column("Modificado", style="dim", width=20)
            
            for file in csv_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                files_table.add_row(file.name, f"{size_mb:.1f}MB", mod_time)
            
            self.console.print(files_table)
        else:
            print("📂 DATOS DESCARGADOS:")
            for file in csv_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.name} ({size_mb:.1f}MB)")
    
    def _configure_defaults(self):
        """Configurar valores por defecto"""
        if self.console:
            self.console.print("[cyan]⚙️ Configurando valores por defecto[/cyan]")
            
            # Mostrar configuración actual
            current_table = Table(title="⚙️ Configuración Actual", show_header=True)
            current_table.add_column("Parámetro", style="cyan")
            current_table.add_column("Valor", style="white")
            
            for key, value in self.default_config.items():
                current_table.add_row(key, str(value))
            
            self.console.print(current_table)
            
            # Permitir modificar
            if Confirm.ask("¿Modificar configuración?", default=False):
                self.default_config['exchange'] = Prompt.ask("Exchange", choices=list(self.exchanges.keys()), default=self.default_config['exchange'])
                self.default_config['symbol'] = Prompt.ask("Símbolo por defecto", default=self.default_config['symbol'])
                self.default_config['timeframe'] = Prompt.ask("Timeframe", choices=list(self.timeframes.keys()), default=self.default_config['timeframe'])
                self.default_config['days'] = IntPrompt.ask("Días por defecto", default=self.default_config['days'])
                
                self.console.print("[green]✅ Configuración actualizada[/green]")
        else:
            print("⚙️ Configuración por defecto:")
            for key, value in self.default_config.items():
                print(f"  {key}: {value}")
    
    def _create_dataset_index(self, results: List[Dict]):
        """Crear índice del dataset descargado"""
        successful_results = [r for r in results if 'success' in r]
        
        index_data = {
            'created_at': datetime.now().isoformat(),
            'total_files': len(successful_results),
            'files': []
        }
        
        for result in successful_results:
            file_info = {
                'symbol': result['symbol'],
                'timeframe': result['timeframe'],
                'days': result['days'],
                'rows': result['rows'],
                'file': result['file'],
                'size_mb': result['size_mb']
            }
            index_data['files'].append(file_info)
        
        index_file = self.data_dir / "dataset_index.json"
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        if self.console:
            self.console.print(f"[green]✅ Índice de dataset creado: {index_file}[/green]")
        else:
            print(f"✅ Índice creado: {index_file}")


def main():
    """Función principal para pruebas"""
    downloader = CryptoDataDownloader()
    downloader.show_download_menu()


if __name__ == "__main__":
    main()
