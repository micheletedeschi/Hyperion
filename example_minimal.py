#!/usr/bin/env python3
"""
🎯 HYPERION3 - EJEMPLO MÍNIMO FUNCIONAL
Tutorial: Tu primer bot en 5 minutos (sin APIs, datos incluidos)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_data():
    """Crear datos de ejemplo para testing"""
    print("📊 Creando datos de ejemplo...")
    
    # Generar 1000 puntos de datos simulados de BTC
    np.random.seed(42)  # Para reproducibilidad
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
    
    # Simular precio de BTC con tendencia y volatilidad
    price = 50000  # Precio inicial
    returns = np.random.normal(0.0001, 0.02, 1000)  # Returns con drift positivo
    prices = []
    
    for i, ret in enumerate(returns):
        price = price * (1 + ret)
        prices.append(price)
    
    # Crear OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000000, 5000000, 1000)
    })
    
    # Asegurar que high >= low >= close y open
    for i in range(len(df)):
        high = max(df.iloc[i]['open'], df.iloc[i]['close'])
        low = min(df.iloc[i]['open'], df.iloc[i]['close'])
        df.at[i, 'high'] = max(df.iloc[i]['high'], high)
        df.at[i, 'low'] = min(df.iloc[i]['low'], low)
    
    return df

def simple_strategy(df):
    """Estrategia simple: comprar cuando RSI < 30, vender cuando RSI > 70"""
    print("🧠 Calculando indicadores técnicos...")
    
    # Calcular RSI simple
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calcular SMA
    sma_20 = df['close'].rolling(window=20).mean()
    
    # Generar señales
    signals = []
    position = 0  # 0 = sin posición, 1 = long, -1 = short
    
    for i in range(len(df)):
        if i < 20:  # Esperar datos suficientes
            signals.append(0)
            continue
            
        current_rsi = rsi.iloc[i]
        current_price = df.iloc[i]['close']
        current_sma = sma_20.iloc[i]
        
        # Lógica simple de trading
        if current_rsi < 30 and current_price > current_sma and position != 1:
            signals.append(1)  # Señal de compra
            position = 1
        elif current_rsi > 70 and position == 1:
            signals.append(-1)  # Señal de venta
            position = 0
        else:
            signals.append(0)  # Sin señal
    
    return signals, rsi

def backtest_simple(df, signals):
    """Backtest simple de la estrategia"""
    print("🧪 Ejecutando backtest...")
    
    initial_balance = 10000
    balance = initial_balance
    position = 0
    trades = []
    equity_curve = [initial_balance]
    
    for i, signal in enumerate(signals):
        current_price = df.iloc[i]['close']
        
        if signal == 1 and position == 0:  # Comprar
            position = balance / current_price
            balance = 0
            trades.append(f"COMPRA: {position:.4f} BTC a ${current_price:.2f}")
            
        elif signal == -1 and position > 0:  # Vender
            balance = position * current_price
            trades.append(f"VENTA: {position:.4f} BTC a ${current_price:.2f} - Profit: ${balance - initial_balance:.2f}")
            position = 0
        
        # Calcular equity actual
        current_equity = balance if position == 0 else position * current_price
        equity_curve.append(current_equity)
    
    # Cerrar posición final si es necesario
    if position > 0:
        final_price = df.iloc[-1]['close']
        balance = position * final_price
        trades.append(f"CIERRE FINAL: {position:.4f} BTC a ${final_price:.2f}")
    
    return balance, trades, equity_curve

def analyze_results(initial_balance, final_balance, trades, equity_curve):
    """Analizar resultados del backtest"""
    print("\n" + "="*50)
    print("📊 RESULTADOS DEL BACKTEST")
    print("="*50)
    
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    print(f"💰 Balance inicial: ${initial_balance:,.2f}")
    print(f"💰 Balance final: ${final_balance:,.2f}")
    print(f"📈 Retorno total: {total_return:.2f}%")
    print(f"🔄 Número de trades: {len([t for t in trades if 'VENTA' in t])}")
    
    # Calcular máximo drawdown
    peak = initial_balance
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown
    
    print(f"📉 Máximo Drawdown: {max_dd:.2f}%")
    
    print("\n🔄 Historial de trades:")
    for trade in trades[:5]:  # Mostrar primeros 5
        print(f"  {trade}")
    if len(trades) > 5:
        print(f"  ... y {len(trades) - 5} trades más")

def main():
    """Ejemplo principal"""
    print("🚀 HYPERION3 - EJEMPLO MÍNIMO")
    print("=" * 40)
    print("🎯 Este ejemplo NO requiere APIs ni configuración compleja")
    print("📊 Usa datos simulados y una estrategia simple")
    print("⚠️  Es solo para demostración - NO uses en dinero real")
    print()
    
    try:
        # 1. Crear datos de ejemplo
        df = create_sample_data()
        
        # 2. Aplicar estrategia simple
        signals, rsi = simple_strategy(df)
        
        # 3. Ejecutar backtest
        final_balance, trades, equity_curve = backtest_simple(df, signals)
        
        # 4. Analizar resultados
        analyze_results(10000, final_balance, trades, equity_curve)
        
        print("\n✅ ¡Ejemplo completado exitosamente!")
        print("🎓 Próximos pasos:")
        print("   1. Explora la configuración avanzada con: python main.py")
        print("   2. Lee la documentación en docs/")
        print("   3. Experimenta con datos reales (con MUCHO cuidado)")
        
    except Exception as e:
        print(f"❌ Error en el ejemplo: {e}")
        print("🔧 Verifica que tienes numpy y pandas instalados:")
        print("   pip install numpy pandas")

if __name__ == "__main__":
    main()
