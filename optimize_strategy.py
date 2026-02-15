#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Strategy Optimization - Polynomial Surface Fitting
Finds optimal coefficients for predicting forward returns based on F&G and Price/SMA ratio
"""

import sys
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import json
import time

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("BTC STRATEGY OPTIMIZATION - POLYNOMIAL SURFACE FITTING")
print("="*80)

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_daily_btc_data():
    """Fetch daily BTC/USD price data from 2018 to today"""
    print("\n[1/5] Fetching daily BTC price data...")
    try:
        import yfinance as yf
        btc = yf.Ticker("BTC-USD")
        df = btc.history(start="2018-01-01", interval="1d")
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.rename(columns={'Date': 'date', 'Close': 'price'})
        df = df[['date', 'price']].copy()
        print(f"   Fetched {len(df)} days of BTC price data")
        return df
    except Exception as e:
        print(f"   ERROR: {e}")
        return None

def calculate_daily_200w_sma():
    """Calculate 200-week SMA and resample to daily"""
    print("\n[2/5] Calculating 200-week SMA...")
    try:
        import yfinance as yf
        btc = yf.Ticker("BTC-USD")
        # Fetch weekly data from earlier to ensure 200 weeks of history
        weekly = btc.history(start="2014-01-01", interval="1wk")
        weekly = weekly.reset_index()
        weekly['Date'] = pd.to_datetime(weekly['Date']).dt.tz_localize(None)
        weekly['sma_200'] = weekly['Close'].rolling(window=200, min_periods=200).mean()
        weekly = weekly[['Date', 'sma_200']].copy()
        weekly = weekly.dropna()

        # Forward fill to daily frequency
        weekly.set_index('Date', inplace=True)
        daily_sma = weekly.resample('D').ffill()
        daily_sma = daily_sma.reset_index()
        daily_sma = daily_sma.rename(columns={'Date': 'date'})
        print(f"   Calculated 200W SMA for {len(daily_sma)} days")
        return daily_sma
    except Exception as e:
        print(f"   ERROR: {e}")
        return None

def fetch_daily_fear_greed():
    """Fetch Fear & Greed index data"""
    print("\n[3/5] Fetching Fear & Greed index...")
    try:
        url = 'https://api.alternative.me/fng/?limit=0'
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        records = []
        for item in data['data']:
            records.append({
                'date': pd.to_datetime(int(item['timestamp']), unit='s'),
                'fg_index': int(item['value'])
            })

        df = pd.DataFrame(records)
        print(f"   Fetched {len(df)} days of Fear & Greed data")
        return df
    except Exception as e:
        print(f"   ERROR: {e}")
        return None

def prepare_training_data():
    """Combine all data sources and calculate features"""
    print("\n[4/5] Preparing training dataset...")

    # Fetch all data
    btc_df = fetch_daily_btc_data()
    sma_df = calculate_daily_200w_sma()
    fg_df = fetch_daily_fear_greed()

    if btc_df is None or sma_df is None or fg_df is None:
        print("   ERROR: Failed to fetch required data")
        return None

    # Merge all datasets
    df = btc_df.merge(sma_df, on='date', how='inner')
    df = df.merge(fg_df, on='date', how='inner')

    # Calculate features
    df['price_to_sma'] = df['price'] / df['sma_200']

    # Calculate 30-day, 60-day, and 90-day forward returns
    df['price_30d_future'] = df['price'].shift(-30)
    df['price_60d_future'] = df['price'].shift(-60)
    df['price_90d_future'] = df['price'].shift(-90)

    df['return_30d'] = ((df['price_30d_future'] - df['price']) / df['price']) * 100
    df['return_60d'] = ((df['price_60d_future'] - df['price']) / df['price']) * 100
    df['return_90d'] = ((df['price_90d_future'] - df['price']) / df['price']) * 100

    # Drop rows with NaN (last 90 days won't have future returns)
    df = df.dropna()

    print(f"   Prepared {len(df)} training samples")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Features: F&G index [{df['fg_index'].min():.0f}-{df['fg_index'].max():.0f}], "
          f"Price/SMA [{df['price_to_sma'].min():.2f}-{df['price_to_sma'].max():.2f}]")

    return df

# ============================================================================
# POLYNOMIAL FITTING
# ============================================================================

def polynomial_3d(xy, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    """
    3rd degree 2D polynomial:
    z = a0 + a1*x + a2*y + a3*x² + a4*x*y + a5*y² + a6*x³ + a7*x²*y + a8*x*y² + a9*y³
    where x = F&G index, y = price/SMA ratio, z = forward return %
    """
    x, y = xy
    return (a0 + a1*x + a2*y + a3*x**2 + a4*x*y + a5*y**2 +
            a6*x**3 + a7*x**2*y + a8*x*y**2 + a9*y**3)

def fit_polynomial(df, return_col='return_30d'):
    """Fit polynomial surface to predict forward returns"""
    X = df['fg_index'].values
    Y = df['price_to_sma'].values
    Z = df[return_col].values

    # Initial guess for coefficients
    p0 = [0] * 10

    try:
        # Fit the polynomial
        popt, pcov = curve_fit(polynomial_3d, (X, Y), Z, p0=p0, maxfev=10000)

        # Calculate R² score
        z_pred = polynomial_3d((X, Y), *popt)
        ss_res = np.sum((Z - z_pred) ** 2)
        ss_tot = np.sum((Z - np.mean(Z)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return popt, r_squared
    except Exception as e:
        print(f"   ERROR fitting polynomial: {e}")
        return None, 0

def optimize_strategy(df):
    """Fit polynomials for 30d, 60d, and 90d returns"""
    print("\n[5/5] Fitting polynomial surfaces...")

    results = {}

    for horizon, col in [('30d', 'return_30d'), ('60d', 'return_60d'), ('90d', 'return_90d')]:
        print(f"\n   Fitting {horizon} forward return polynomial...")
        coeffs, r2 = fit_polynomial(df, col)

        if coeffs is not None:
            print(f"   R² score: {r2:.4f}")
            print(f"   Coefficients: {coeffs}")

            # Test signal distribution
            X = df['fg_index'].values
            Y = df['price_to_sma'].values
            predicted = polynomial_3d((X, Y), *coeffs)

            buy_signals = np.sum(predicted > 5)
            sell_signals = np.sum(predicted < -5)
            hold_signals = np.sum((predicted >= -5) & (predicted <= 5))

            print(f"   Signal distribution: BUY={buy_signals} ({buy_signals/len(df)*100:.1f}%), "
                  f"SELL={sell_signals} ({sell_signals/len(df)*100:.1f}%), "
                  f"HOLD={hold_signals} ({hold_signals/len(df)*100:.1f}%)")

            results[horizon] = {
                'coefficients': coeffs.tolist(),
                'r_squared': float(r2),
                'signal_distribution': {
                    'buy': int(buy_signals),
                    'sell': int(sell_signals),
                    'hold': int(hold_signals)
                }
            }
        else:
            print(f"   FAILED to fit {horizon} polynomial")

    return results

# ============================================================================
# BACKTESTING
# ============================================================================

def backtest_strategy(df, coeffs_30d):
    """Backtest the strategy vs baseline DCA"""
    print("\n" + "="*80)
    print("BACKTESTING RESULTS")
    print("="*80)

    # Resample to weekly for DCA simulation
    df_weekly = df.set_index('date').resample('W-MON').first().dropna().reset_index()

    strategy_value = 0
    strategy_cash = 0
    strategy_btc = 0

    baseline_value = 0
    baseline_cash = 0
    baseline_btc = 0

    weekly_investment = 100

    buy_count = 0
    sell_count = 0
    hold_count = 0

    for idx, row in df_weekly.iterrows():
        x = row['fg_index']
        y = row['price_to_sma']
        price = row['price']

        # Predict 30-day return
        predicted_return = polynomial_3d((x, y), *coeffs_30d)

        # Strategy signal
        if predicted_return > 5:
            # BUY
            strategy_btc += weekly_investment / price
            buy_count += 1
        elif predicted_return < -5:
            # SELL (reduce position by equivalent amount)
            btc_to_sell = min(strategy_btc, weekly_investment / price)
            strategy_btc -= btc_to_sell
            strategy_cash += btc_to_sell * price
            sell_count += 1
        else:
            # HOLD
            hold_count += 1

        # Baseline: always buy
        baseline_btc += weekly_investment / price

    # Final values
    final_price = df_weekly.iloc[-1]['price']
    strategy_value = strategy_btc * final_price + strategy_cash
    baseline_value = baseline_btc * final_price

    total_weeks = len(df_weekly)
    strategy_invested = (buy_count * weekly_investment) - strategy_cash
    baseline_invested = total_weeks * weekly_investment

    strategy_return = ((strategy_value - strategy_invested) / strategy_invested) * 100 if strategy_invested > 0 else 0
    baseline_return = ((baseline_value - baseline_invested) / baseline_invested) * 100
    outperformance = strategy_return - baseline_return

    print(f"\nBacktest period: {df_weekly.iloc[0]['date'].date()} to {df_weekly.iloc[-1]['date'].date()}")
    print(f"Total weeks: {total_weeks}")
    print(f"\nSignal counts: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
    print(f"\nSTRATEGY:")
    print(f"  Invested: ${strategy_invested:,.2f}")
    print(f"  Final value: ${strategy_value:,.2f}")
    print(f"  Return: {strategy_return:.2f}%")
    print(f"\nBASELINE (DCA every week):")
    print(f"  Invested: ${baseline_invested:,.2f}")
    print(f"  Final value: ${baseline_value:,.2f}")
    print(f"  Return: {baseline_return:.2f}%")
    print(f"\nOUTPERFORMANCE: {outperformance:+.2f}%")

    return {
        'strategy_return': float(strategy_return),
        'baseline_return': float(baseline_return),
        'outperformance': float(outperformance),
        'signal_counts': {'buy': buy_count, 'sell': sell_count, 'hold': hold_count}
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()

    # Prepare data
    df = prepare_training_data()
    if df is None:
        print("\nFailed to prepare data. Exiting.")
        return

    # Optimize
    results = optimize_strategy(df)

    if '30d' not in results:
        print("\nFailed to optimize strategy. Exiting.")
        return

    # Backtest
    backtest_results = backtest_strategy(df, results['30d']['coefficients'])

    # Save to JSON
    output = {
        'generated_at': datetime.now().isoformat(),
        'data_quality': {
            'start_date': df['date'].min().isoformat(),
            'end_date': df['date'].max().isoformat(),
            'total_days': len(df)
        },
        'polynomials': results,
        'thresholds': {
            'buy': 5.0,
            'sell': -5.0
        },
        'backtest': backtest_results
    }

    output_path = 'optimal_strategy.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print(f"Optimization complete in {time.time() - start_time:.1f} seconds")
    print(f"Results saved to: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
