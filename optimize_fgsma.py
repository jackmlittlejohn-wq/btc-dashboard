#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FGSMA Strategy Optimizer
Compares DCA vs Fear & Greed + SMA polynomial-based strategy
"""

import sys
import io
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("FGSMA STRATEGY OPTIMIZATION")
print("="*80)
sys.stdout.flush()

# ============================================================================
# DATA LOADING
# ============================================================================

def fetch_daily_btc_data():
    """Fetch daily BTC/USD price data"""
    print("\n[1/4] Fetching daily BTC price data...")
    sys.stdout.flush()
    try:
        import yfinance as yf
        btc = yf.Ticker("BTC-USD")
        df = btc.history(start="2015-01-01", interval="1d")
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.rename(columns={'Date': 'date', 'Close': 'price'})
        df = df[['date', 'price']].copy()
        print(f"   Fetched {len(df)} days of BTC price data")
        sys.stdout.flush()
        return df
    except Exception as e:
        print(f"   ERROR: {e}")
        return None

def calculate_200w_sma_from_daily(daily_df):
    """Calculate 200-week SMA from daily data (1400 day rolling average)"""
    print("\n[2/4] Calculating 200-week SMA...")
    sys.stdout.flush()
    try:
        df = daily_df.copy()
        df['sma_200w'] = df['price'].rolling(window=1400, min_periods=1400).mean()
        df = df[['date', 'sma_200w']].copy()
        df = df.dropna()
        print(f"   Calculated 200W SMA for {len(df)} days")
        sys.stdout.flush()
        return df
    except Exception as e:
        print(f"   ERROR: {e}")
        return None

def fetch_daily_fear_greed():
    """Fetch Fear & Greed index data"""
    print("\n[3/4] Fetching Fear & Greed index...")
    sys.stdout.flush()
    try:
        import requests
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
        sys.stdout.flush()
        return df
    except Exception as e:
        print(f"   ERROR: {e}")
        return None

def prepare_data():
    """Load and merge all data sources"""
    btc_df = fetch_daily_btc_data()
    if btc_df is None:
        return None

    sma_df = calculate_200w_sma_from_daily(btc_df)
    fg_df = fetch_daily_fear_greed()

    if sma_df is None or fg_df is None:
        print("\n[ERROR] Failed to fetch required data")
        return None

    print("\n[4/4] Merging datasets...")
    sys.stdout.flush()
    # Merge all datasets
    df = btc_df.merge(sma_df, on='date', how='inner')
    df = df.merge(fg_df, on='date', how='inner')

    # Calculate price to SMA ratio
    df['price_to_sma'] = df['price'] / df['sma_200w']

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Total days: {len(df)}")
    print(f"   F&G range: [{df['fg_index'].min()}, {df['fg_index'].max()}]")
    print(f"   Price/SMA range: [{df['price_to_sma'].min():.2f}, {df['price_to_sma'].max():.2f}]")
    sys.stdout.flush()

    return df

# ============================================================================
# DCA STRATEGY
# ============================================================================

def simulate_dca(df):
    """
    DCA Model:
    - Starting capital = total_days × $1
    - Every day: buy $1 worth of BTC
    """
    total_days = len(df)
    starting_capital = total_days * 1.0
    daily_investment = 1.0

    btc_holdings = 0.0
    cash = starting_capital

    portfolio_value = []

    for idx, row in df.iterrows():
        price = row['price']

        # Buy $1 worth of BTC
        if cash >= daily_investment:
            btc_purchased = daily_investment / price
            btc_holdings += btc_purchased
            cash -= daily_investment

        # Track portfolio value
        current_value = btc_holdings * price + cash
        portfolio_value.append(current_value)

    final_price = df.iloc[-1]['price']
    final_value = btc_holdings * final_price + cash
    total_return = ((final_value - starting_capital) / starting_capital) * 100

    return {
        'final_value': final_value,
        'total_return': total_return,
        'btc_holdings': btc_holdings,
        'cash_remaining': cash,
        'portfolio_value': portfolio_value,
        'starting_capital': starting_capital
    }

# ============================================================================
# FGSMA STRATEGY
# ============================================================================

def calculate_signal_score(x, y, coeffs):
    """
    Polynomial: signal_score = a*x + b*y + c*x² + d*y² + e*x*y + f
    where x = F&G index (0-100), y = price/SMA ratio
    """
    a, b, c, d, e, f = coeffs
    score = a*x + b*y + c*x**2 + d*y**2 + e*x*y + f
    # Clamp to 0-100
    return np.clip(score, 0, 100)

def simulate_fgsma(df, coeffs, buy_sell_pct):
    """
    FGSMA Model:
    - Signal score 0-30: SELL buy_sell_pct% of BTC holdings
    - Signal score 70-100: BUY with buy_sell_pct% of cash
    - Signal score 31-69: HOLD
    """
    total_days = len(df)
    starting_capital = total_days * 1.0

    btc_holdings = 0.0
    cash = starting_capital

    portfolio_value = []

    for idx, row in df.iterrows():
        price = row['price']
        x = row['fg_index']
        y = row['price_to_sma']

        # Calculate signal score
        signal_score = calculate_signal_score(x, y, coeffs)

        # Execute signal
        if signal_score <= 30:
            # SELL signal
            btc_to_sell = btc_holdings * (buy_sell_pct / 100.0)
            cash_received = btc_to_sell * price
            btc_holdings -= btc_to_sell
            cash += cash_received

        elif signal_score >= 70:
            # BUY signal
            cash_to_spend = cash * (buy_sell_pct / 100.0)
            btc_purchased = cash_to_spend / price
            btc_holdings += btc_purchased
            cash -= cash_to_spend

        # HOLD: do nothing for 31-69

        # Track portfolio value
        current_value = btc_holdings * price + cash
        portfolio_value.append(current_value)

    final_price = df.iloc[-1]['price']
    final_value = btc_holdings * final_price + cash
    total_return = ((final_value - starting_capital) / starting_capital) * 100

    return {
        'final_value': final_value,
        'total_return': total_return,
        'btc_holdings': btc_holdings,
        'cash_remaining': cash,
        'portfolio_value': portfolio_value,
        'starting_capital': starting_capital
    }

# ============================================================================
# OPTIMIZATION
# ============================================================================

def objective_function(params, df, buy_sell_pct=1.0):
    """
    Objective: maximize final portfolio value of FGSMA
    params = [a, b, c, d, e, f]
    buy_sell_pct is locked at 1.0%
    """
    coeffs = params

    result = simulate_fgsma(df, coeffs, buy_sell_pct)
    # Return negative because we're minimizing
    return -result['final_value']

def optimize_fgsma(df, buy_sell_pct=1.0):
    """
    Optimize polynomial coefficients with locked buy/sell percentage
    """
    print("\n" + "="*80)
    print("OPTIMIZATION")
    print("="*80)
    print(f"\nOptimizing FGSMA strategy with buy/sell % locked at {buy_sell_pct}%...")
    print("(this should be faster with fewer parameters to optimize)")
    sys.stdout.flush()

    # Define bounds for optimization
    # coeffs: [a, b, c, d, e, f]
    # a, b: linear terms, can be positive or negative
    # c, d: quadratic terms for x² and y²
    # e: interaction term x*y
    # f: constant offset

    bounds = [
        (-10, 10),     # a: coefficient for x (F&G)
        (-10, 10),     # b: coefficient for y (price/SMA)
        (-1, 1),       # c: coefficient for x²
        (-1, 1),       # d: coefficient for y²
        (-1, 1),       # e: coefficient for x*y
        (-100, 100)    # f: constant offset
    ]

    # Use differential evolution optimizer
    result = differential_evolution(
        objective_function,
        bounds,
        args=(df, buy_sell_pct),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        seed=42,
        disp=True
    )

    optimal_coeffs = result.x

    print(f"\nOptimization complete!")
    print(f"Final objective value: {-result.fun:,.2f}")
    sys.stdout.flush()

    return optimal_coeffs, buy_sell_pct

# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()

    # Load data
    df = prepare_data()
    if df is None:
        print("\n[ERROR] Failed to prepare data. Exiting.")
        return

    # Run DCA simulation
    print("\n" + "="*80)
    print("DCA SIMULATION")
    print("="*80)
    dca_result = simulate_dca(df)
    print(f"\nStarting capital: ${dca_result['starting_capital']:,.2f}")
    print(f"Final value: ${dca_result['final_value']:,.2f}")
    print(f"Total return: {dca_result['total_return']:.2f}%")
    print(f"BTC accumulated: {dca_result['btc_holdings']:.8f} BTC")
    print(f"Cash remaining: ${dca_result['cash_remaining']:.2f}")

    # Optimize FGSMA with buy/sell % locked at 1%
    LOCKED_BUY_SELL_PCT = 1.0
    optimal_coeffs, optimal_buy_sell_pct = optimize_fgsma(df, LOCKED_BUY_SELL_PCT)

    # Run FGSMA simulation with optimal parameters
    print("\n" + "="*80)
    print("FGSMA SIMULATION (OPTIMIZED)")
    print("="*80)
    fgsma_result = simulate_fgsma(df, optimal_coeffs, optimal_buy_sell_pct)

    print(f"\nOptimal polynomial coefficients:")
    print(f"  a (F&G linear):     {optimal_coeffs[0]:.6f}")
    print(f"  b (SMA linear):     {optimal_coeffs[1]:.6f}")
    print(f"  c (F&G squared):    {optimal_coeffs[2]:.6f}")
    print(f"  d (SMA squared):    {optimal_coeffs[3]:.6f}")
    print(f"  e (F&G × SMA):      {optimal_coeffs[4]:.6f}")
    print(f"  f (constant):       {optimal_coeffs[5]:.6f}")
    print(f"\nOptimal buy/sell %: {optimal_buy_sell_pct:.2f}%")

    print(f"\nStarting capital: ${fgsma_result['starting_capital']:,.2f}")
    print(f"Final value: ${fgsma_result['final_value']:,.2f}")
    print(f"Total return: {fgsma_result['total_return']:.2f}%")
    print(f"BTC holdings: {fgsma_result['btc_holdings']:.8f} BTC")
    print(f"Cash remaining: ${fgsma_result['cash_remaining']:.2f}")

    # Compare strategies
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    outperformance = fgsma_result['total_return'] - dca_result['total_return']
    print(f"\nDCA total return:   {dca_result['total_return']:.2f}%")
    print(f"FGSMA total return: {fgsma_result['total_return']:.2f}%")
    print(f"Outperformance:     {outperformance:+.2f}%")

    # Save coefficients to JSON
    output = {
        'generated_at': datetime.now().isoformat(),
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat(),
            'total_days': len(df)
        },
        'optimal_coefficients': {
            'a_fg_linear': float(optimal_coeffs[0]),
            'b_sma_linear': float(optimal_coeffs[1]),
            'c_fg_squared': float(optimal_coeffs[2]),
            'd_sma_squared': float(optimal_coeffs[3]),
            'e_fg_sma_interaction': float(optimal_coeffs[4]),
            'f_constant': float(optimal_coeffs[5])
        },
        'optimal_buy_sell_pct': float(optimal_buy_sell_pct),
        'dca_results': {
            'final_value': float(dca_result['final_value']),
            'total_return': float(dca_result['total_return']),
            'btc_holdings': float(dca_result['btc_holdings'])
        },
        'fgsma_results': {
            'final_value': float(fgsma_result['final_value']),
            'total_return': float(fgsma_result['total_return']),
            'btc_holdings': float(fgsma_result['btc_holdings'])
        },
        'outperformance': float(outperformance)
    }

    output_path = 'fgsma_coefficients.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nCoefficients saved to: {output_path}")

    # Generate comparison chart
    print("\nGenerating portfolio comparison chart...")
    plt.figure(figsize=(14, 8))

    dates = df['date'].values

    plt.subplot(2, 1, 1)
    plt.plot(dates, dca_result['portfolio_value'], label='DCA Strategy', color='#3fb950', linewidth=2)
    plt.plot(dates, fgsma_result['portfolio_value'], label='FGSMA Strategy', color='#9333ea', linewidth=2)
    plt.title('Portfolio Value Over Time: DCA vs FGSMA', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Plot BTC price for reference
    plt.subplot(2, 1, 2)
    plt.plot(dates, df['price'].values, label='BTC Price', color='#d29922', linewidth=2)
    plt.title('BTC Price', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = 'fgsma_comparison.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")

    print("\n" + "="*80)
    print(f"Optimization complete in {time.time() - start_time:.1f} seconds")
    print("="*80)

if __name__ == "__main__":
    main()
