#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FGSMA Final Optimizer
5-level threshold system with EMA smoothing and time adjustment
"""

import sys
import io
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("FGSMA FINAL OPTIMIZATION")
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
        sys.stdout.flush()
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
        sys.stdout.flush()
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
        sys.stdout.flush()
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
        sys.stdout.flush()
        return None

    print("\n[4/4] Merging datasets...")
    sys.stdout.flush()

    # Merge all datasets
    df = btc_df.merge(sma_df, on='date', how='inner')
    df = df.merge(fg_df, on='date', how='inner')

    # Calculate price to SMA ratio
    df['price_to_sma'] = df['price'] / df['sma_200w']

    # Calculate days since start (time factor z)
    df = df.sort_values('date').reset_index(drop=True)
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Total days: {len(df)}")
    print(f"   F&G range: [{df['fg_index'].min()}, {df['fg_index'].max()}]")
    print(f"   Price/SMA range: [{df['price_to_sma'].min():.2f}, {df['price_to_sma'].max():.2f}]")
    sys.stdout.flush()

    return df

# ============================================================================
# EMA CALCULATION
# ============================================================================

def calculate_ema(series, period):
    """Calculate exponential moving average"""
    return series.ewm(span=period, adjust=False).mean()

# ============================================================================
# SIGNAL CALCULATION
# ============================================================================

def get_fg_signal(fg_ema, t1, t2, t3, t4):
    """
    Get F&G EMA signal based on thresholds (CORRECTED: fear = buy, greed = sell)
    0 to T1: Strong BUY (4) - extreme fear is buying opportunity
    T1 to T2: Buy (3) - moderate fear
    T2 to T3: Hold (2) - neutral
    T3 to T4: Sell (1) - moderate greed
    T4 to 100: Strong Sell (0) - extreme greed
    """
    if fg_ema <= t1:
        return 4  # Strong Buy (extreme fear)
    elif fg_ema <= t2:
        return 3  # Buy (moderate fear)
    elif fg_ema <= t3:
        return 2  # Hold (neutral)
    elif fg_ema <= t4:
        return 1  # Sell (moderate greed)
    else:
        return 0  # Strong Sell (extreme greed)

def get_sma_signal(price_to_sma, r1, r2, r3, r4):
    """
    Get SMA Ratio signal based on thresholds
    0 to R1: Strong Buy (4)
    R1 to R2: Buy (3)
    R2 to R3: Hold (2)
    R3 to R4: Sell (1)
    R4 to inf: Strong Sell (0)
    """
    if price_to_sma <= r1:
        return 4  # Strong Buy
    elif price_to_sma <= r2:
        return 3  # Buy
    elif price_to_sma <= r3:
        return 2  # Hold
    elif price_to_sma <= r4:
        return 1  # Sell
    else:
        return 0  # Strong Sell

def apply_time_adjustment(r1, r2, r3, r4, days_since_start, decay_rate):
    """
    Apply time adjustment to SMA ratio thresholds with SQUARE ROOT DECAY
    As Bitcoin matures, price/SMA ratio compresses (peaks get lower, bottoms get lower)
    So thresholds must DECREASE over time, not increase

    SQUARE ROOT DECAY provides gentler, more realistic long-term thresholds
    Starting around R1=1.10, decaying to ~0.90 by 2026
    decay_rate: controls the decay speed
    """
    # Calculate years since start (2018-10-31)
    years_elapsed = days_since_start / 365.25

    # Square root decay: fast initial decay, then slows down
    # Floor at 30% to keep thresholds realistic long-term
    decay_factor = max(0.3, 1.0 - (decay_rate * np.sqrt(years_elapsed)))

    return r1 * decay_factor, r2 * decay_factor, r3 * decay_factor, r4 * decay_factor

def combine_signals(fg_signal, sma_signal):
    """
    Combine F&G and SMA signals using boolean logic
    Returns: 0=Strong Sell, 1=Sell, 2=Hold, 3=Buy, 4=Strong Buy
    """
    # Both agree on extreme
    if fg_signal == 4 and sma_signal == 4:
        return 4  # Strong Buy
    if fg_signal == 0 and sma_signal == 0:
        return 0  # Strong Sell

    # One strong buy, other buy or hold
    if (fg_signal == 4 and sma_signal >= 2) or (sma_signal == 4 and fg_signal >= 2):
        return 3  # Buy

    # One strong sell, other sell or hold
    if (fg_signal == 0 and sma_signal <= 2) or (sma_signal == 0 and fg_signal <= 2):
        return 1  # Sell

    # Both say hold
    if fg_signal == 2 and sma_signal == 2:
        return 2  # Hold

    # Direct conflict (one buy/strong buy, other sell/strong sell)
    if (fg_signal >= 3 and sma_signal <= 1) or (sma_signal >= 3 and fg_signal <= 1):
        return 2  # Hold

    # Both say buy (but not strong buy)
    if fg_signal == 3 and sma_signal == 3:
        return 3  # Buy

    # Both say sell (but not strong sell)
    if fg_signal == 1 and sma_signal == 1:
        return 1  # Sell

    # Mixed signals lean toward more conservative
    if fg_signal >= 3 or sma_signal >= 3:
        return 3  # Buy
    if fg_signal <= 1 or sma_signal <= 1:
        return 1  # Sell

    # Default to hold
    return 2  # Hold

# ============================================================================
# SIMULATION
# ============================================================================

def simulate_strategy(df, ema_period, t1, t2, t3, t4, r1, r2, r3, r4, decay_rate):
    """
    Simulate the FGSMA strategy with given parameters
    """
    total_days = len(df)
    starting_capital = total_days * 1.0

    # Calculate F&G EMA
    df_sim = df.copy()
    df_sim['fg_ema'] = calculate_ema(df_sim['fg_index'], ema_period)

    btc_holdings = 0.0
    cash = starting_capital

    portfolio_value = []
    fg_signals = []
    sma_signals = []
    overall_signals = []

    buy_count = 0
    sell_count = 0
    hold_count = 0

    for idx, row in df_sim.iterrows():
        price = row['price']
        fg_ema = row['fg_ema']
        price_to_sma = row['price_to_sma']
        days_since_start = row['days_since_start']

        # Apply time adjustment to SMA thresholds with DECAY
        r1_adj, r2_adj, r3_adj, r4_adj = apply_time_adjustment(
            r1, r2, r3, r4, days_since_start, decay_rate
        )

        # Get individual signals
        fg_signal = get_fg_signal(fg_ema, t1, t2, t3, t4)
        sma_signal = get_sma_signal(price_to_sma, r1_adj, r2_adj, r3_adj, r4_adj)
        overall_signal = combine_signals(fg_signal, sma_signal)

        fg_signals.append(fg_signal)
        sma_signals.append(sma_signal)
        overall_signals.append(overall_signal)

        # Execute signal (1% of cash/BTC)
        if overall_signal >= 3:  # Buy or Strong Buy
            cash_to_spend = cash * 0.01
            btc_purchased = cash_to_spend / price
            btc_holdings += btc_purchased
            cash -= cash_to_spend
            if overall_signal == 4:
                buy_count += 1
            else:
                buy_count += 1

        elif overall_signal <= 1:  # Sell or Strong Sell
            btc_to_sell = btc_holdings * 0.01
            cash_received = btc_to_sell * price
            btc_holdings -= btc_to_sell
            cash += cash_received
            if overall_signal == 0:
                sell_count += 1
            else:
                sell_count += 1
        else:  # Hold
            hold_count += 1

        # Track portfolio value
        current_value = btc_holdings * price + cash
        portfolio_value.append(current_value)

    final_price = df_sim.iloc[-1]['price']
    final_value = btc_holdings * final_price + cash
    total_return = ((final_value - starting_capital) / starting_capital) * 100

    return {
        'final_value': final_value,
        'total_return': total_return,
        'btc_holdings': btc_holdings,
        'cash_remaining': cash,
        'portfolio_value': portfolio_value,
        'starting_capital': starting_capital,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'hold_count': hold_count,
        'fg_signals': fg_signals,
        'sma_signals': sma_signals,
        'overall_signals': overall_signals,
        'df': df_sim
    }

# ============================================================================
# OPTIMIZATION
# ============================================================================

def objective_function(params, df, ema_period):
    """
    Objective: maximize final portfolio value
    params = [t1, t2, t3, t4, r1, r2, r3, r4, decay_rate]
    """
    t1, t2, t3, t4, r1, r2, r3, r4, decay_rate = params

    # Ensure thresholds are ordered
    if not (0 < t1 < t2 < t3 < t4 < 100):
        return 1e9  # Invalid
    if not (0 < r1 < r2 < r3 < r4):
        return 1e9  # Invalid

    result = simulate_strategy(df, ema_period, t1, t2, t3, t4, r1, r2, r3, r4, decay_rate)

    # Return negative because we're minimizing
    return -result['final_value']

def optimize_strategy(df):
    """
    Optimize all parameters including EMA period
    """
    print("\n" + "="*80)
    print("OPTIMIZATION")
    print("="*80)
    print("\nOptimizing FGSMA strategy...")
    print("This will test different EMA periods and threshold combinations...")
    sys.stdout.flush()

    # Test different EMA periods
    ema_periods = [7, 14, 21, 30, 60]
    best_result = None
    best_ema_period = None
    best_params = None

    for ema_period in ema_periods:
        print(f"\n[Testing EMA period: {ema_period} days]")
        sys.stdout.flush()

        # Define bounds for optimization
        # [t1, t2, t3, t4, r1, r2, r3, r4, decay_rate]
        # NOTE: R thresholds are now BASE values (2018), they will DECAY over time
        # Targeting R1: ~1.10 (2018) → ~0.90 (2026) with square root decay
        bounds = [
            (10, 30),      # t1: F&G threshold 1
            (25, 40),      # t2: F&G threshold 2
            (60, 75),      # t3: F&G threshold 3
            (75, 90),      # t4: F&G threshold 4
            (0.9, 1.4),    # r1: SMA ratio threshold 1 (BASE, targeting ~1.10)
            (1.5, 2.5),    # r2: SMA ratio threshold 2 (BASE, will decay)
            (2.5, 4.5),    # r3: SMA ratio threshold 3 (BASE, will decay)
            (4.0, 7.0),    # r4: SMA ratio threshold 4 (BASE, will decay)
            (0.05, 0.15)   # decay_rate: controls sqrt decay speed
        ]

        # Use differential evolution optimizer
        result = differential_evolution(
            objective_function,
            bounds,
            args=(df, ema_period),
            strategy='best1bin',
            maxiter=50,
            popsize=10,
            tol=0.01,
            seed=42,
            disp=False
        )

        final_value = -result.fun
        print(f"   EMA {ema_period}: Final value = ${final_value:,.2f}")
        sys.stdout.flush()

        if best_result is None or final_value > best_result:
            best_result = final_value
            best_ema_period = ema_period
            best_params = result.x

    print(f"\n[BEST RESULT]")
    print(f"   EMA Period: {best_ema_period} days")
    print(f"   Final Value: ${best_result:,.2f}")
    sys.stdout.flush()

    return best_ema_period, best_params

# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()

    # Load data
    df = prepare_data()
    if df is None:
        print("\n[ERROR] Failed to prepare data. Exiting.")
        sys.stdout.flush()
        return

    # Optimize strategy
    best_ema_period, best_params = optimize_strategy(df)
    t1, t2, t3, t4, r1, r2, r3, r4, decay_rate = best_params

    # Run simulation with optimal parameters
    result = simulate_strategy(df, best_ema_period, t1, t2, t3, t4, r1, r2, r3, r4, decay_rate)

    # Calculate what thresholds look like over time with sqrt decay
    def get_decay_factor(years):
        return max(0.3, 1.0 - (decay_rate * np.sqrt(years)))

    # 2026 (7.28 years)
    factor_2026 = get_decay_factor(7.28)
    r1_2026 = r1 * factor_2026
    r2_2026 = r2 * factor_2026
    r3_2026 = r3 * factor_2026
    r4_2026 = r4 * factor_2026

    # 2030 (11.17 years)
    factor_2030 = get_decay_factor(11.17)
    r1_2030 = r1 * factor_2030
    r2_2030 = r2 * factor_2030
    r3_2030 = r3 * factor_2030
    r4_2030 = r4 * factor_2030

    # 2035 (16.17 years)
    factor_2035 = get_decay_factor(16.17)
    r1_2035 = r1 * factor_2035
    r2_2035 = r2 * factor_2035
    r3_2035 = r3 * factor_2035
    r4_2035 = r4 * factor_2035

    # 2040 (21.17 years)
    factor_2040 = get_decay_factor(21.17)
    r1_2040 = r1 * factor_2040
    r2_2040 = r2 * factor_2040
    r3_2040 = r3 * factor_2040
    r4_2040 = r4 * factor_2040

    print("\n" + "="*80)
    print("OPTIMIZED STRATEGY RESULTS")
    print("="*80)
    print(f"\nOptimal Parameters:")
    print(f"  EMA Period: {best_ema_period} days")
    print(f"\n  F&G Thresholds:")
    print(f"    T1 (Strong Buy → Buy):       {t1:.2f}")
    print(f"    T2 (Buy → Hold):             {t2:.2f}")
    print(f"    T3 (Hold → Sell):            {t3:.2f}")
    print(f"    T4 (Sell → Strong Sell):     {t4:.2f}")
    print(f"\n  SMA Ratio Thresholds (BASE - 2018):")
    print(f"    R1 (Strong Buy → Buy):       {r1:.2f}")
    print(f"    R2 (Buy → Hold):             {r2:.2f}")
    print(f"    R3 (Hold → Sell):            {r3:.2f}")
    print(f"    R4 (Sell → Strong Sell):     {r4:.2f}")
    print(f"\n  SQUARE ROOT DECAY PROJECTIONS:")
    print(f"\n    2026 (7.28 years):")
    print(f"      R1={r1_2026:.2f}, R2={r2_2026:.2f}, R3={r3_2026:.2f}, R4={r4_2026:.2f}")
    print(f"      Decay factor: {factor_2026:.4f} ({(1-factor_2026)*100:.1f}% reduction)")
    print(f"\n    2030 (11.17 years):")
    print(f"      R1={r1_2030:.2f}, R2={r2_2030:.2f}, R3={r3_2030:.2f}, R4={r4_2030:.2f}")
    print(f"      Decay factor: {factor_2030:.4f} ({(1-factor_2030)*100:.1f}% reduction)")
    print(f"\n    2035 (16.17 years):")
    print(f"      R1={r1_2035:.2f}, R2={r2_2035:.2f}, R3={r3_2035:.2f}, R4={r4_2035:.2f}")
    print(f"      Decay factor: {factor_2035:.4f} ({(1-factor_2035)*100:.1f}% reduction)")
    print(f"\n    2040 (21.17 years):")
    print(f"      R1={r1_2040:.2f}, R2={r2_2040:.2f}, R3={r3_2040:.2f}, R4={r4_2040:.2f}")
    print(f"      Decay factor: {factor_2040:.4f} ({(1-factor_2040)*100:.1f}% reduction)")
    print(f"\n  Decay Parameter: {decay_rate:.4f}")

    print(f"\nPortfolio Performance:")
    print(f"  Starting capital: ${result['starting_capital']:,.2f}")
    print(f"  Final value:      ${result['final_value']:,.2f}")
    print(f"  Total return:     {result['total_return']:.2f}%")
    print(f"  BTC holdings:     {result['btc_holdings']:.8f} BTC")
    print(f"  Cash remaining:   ${result['cash_remaining']:,.2f}")

    print(f"\nSignal Distribution:")
    print(f"  Buy signals:      {result['buy_count']} days")
    print(f"  Sell signals:     {result['sell_count']} days")
    print(f"  Hold days:        {result['hold_count']} days")

    # Save optimized parameters to JSON
    output = {
        'generated_at': datetime.now().isoformat(),
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat(),
            'total_days': len(df)
        },
        'optimal_parameters': {
            'ema_period': int(best_ema_period),
            'fg_thresholds': {
                't1_strong_buy_to_buy': float(t1),
                't2_buy_to_hold': float(t2),
                't3_hold_to_sell': float(t3),
                't4_sell_to_strong_sell': float(t4)
            },
            'sma_ratio_thresholds_base': {
                'r1_strong_buy_to_buy': float(r1),
                'r2_buy_to_hold': float(r2),
                'r3_hold_to_sell': float(r3),
                'r4_sell_to_strong_sell': float(r4)
            },
            'decay_rate': float(decay_rate),
            'decay_explanation': f"Square root decay with parameter {decay_rate:.4f}. After 7.28 years (2018-2026): {(1-factor_2026)*100:.1f}% reduction"
        },
        'performance': {
            'final_value': float(result['final_value']),
            'total_return': float(result['total_return']),
            'btc_holdings': float(result['btc_holdings']),
            'buy_signals': int(result['buy_count']),
            'sell_signals': int(result['sell_count']),
            'hold_days': int(result['hold_count'])
        }
    }

    output_path = 'fgsma_optimized.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nOptimized parameters saved to: {output_path}")
    sys.stdout.flush()

    # Generate portfolio value chart
    print("\nGenerating portfolio value chart...")
    sys.stdout.flush()

    plt.figure(figsize=(14, 8))
    dates = result['df']['date'].values

    plt.subplot(2, 1, 1)
    plt.plot(dates, result['portfolio_value'], label='FGSMA Strategy', color='#10b981', linewidth=2)
    plt.title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Plot BTC price for reference
    plt.subplot(2, 1, 2)
    plt.plot(dates, result['df']['price'].values, label='BTC Price', color='#d29922', linewidth=2)
    plt.title('BTC Price', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = 'fgsma_portfolio.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Portfolio chart saved to: {chart_path}")
    sys.stdout.flush()

    # Generate signal chart
    print("\nGenerating signal chart...")
    sys.stdout.flush()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # F&G EMA Signal
    signal_colors = ['#dc2626', '#f97316', '#94a3b8', '#22c55e', '#16a34a']
    signal_names = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']

    ax1.plot(dates, result['df']['fg_ema'].values, color='#000', linewidth=2, label='F&G EMA')
    ax1.axhline(y=t1, color='#dc2626', linestyle='--', alpha=0.5, label=f'T1={t1:.1f}')
    ax1.axhline(y=t2, color='#f97316', linestyle='--', alpha=0.5, label=f'T2={t2:.1f}')
    ax1.axhline(y=t3, color='#22c55e', linestyle='--', alpha=0.5, label=f'T3={t3:.1f}')
    ax1.axhline(y=t4, color='#16a34a', linestyle='--', alpha=0.5, label=f'T4={t4:.1f}')
    ax1.set_title('F&G EMA Signal Levels', fontsize=16, fontweight='bold')
    ax1.set_ylabel('F&G EMA Value', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # SMA Ratio Signal
    ax2.plot(dates, result['df']['price_to_sma'].values, color='#000', linewidth=2, label='Price/SMA Ratio')

    # Show time-decaying thresholds
    # Plot start (2018) and end (2026) thresholds
    ax2.axhline(y=r1, color='#16a34a', linestyle='--', alpha=0.5, label=f'R1: {r1:.2f} (2018) → {r1_2026:.2f} (2026)')
    ax2.axhline(y=r2, color='#22c55e', linestyle='--', alpha=0.5, label=f'R2: {r2:.2f} (2018) → {r2_2026:.2f} (2026)')
    ax2.axhline(y=r3, color='#f97316', linestyle='--', alpha=0.5, label=f'R3: {r3:.2f} (2018) → {r3_2026:.2f} (2026)')
    ax2.axhline(y=r4, color='#dc2626', linestyle='--', alpha=0.5, label=f'R4: {r4:.2f} (2018) → {r4_2026:.2f} (2026)')

    # Show 2026 values with lighter lines
    ax2.axhline(y=r1_2026, color='#16a34a', linestyle=':', alpha=0.3)
    ax2.axhline(y=r2_2026, color='#22c55e', linestyle=':', alpha=0.3)
    ax2.axhline(y=r3_2026, color='#f97316', linestyle=':', alpha=0.3)
    ax2.axhline(y=r4_2026, color='#dc2626', linestyle=':', alpha=0.3)

    ax2.set_title(f'SMA Ratio Signal Levels (DECAYING at {decay_rate*100:.2f}%/year)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price / 200W SMA', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    signal_chart_path = 'fgsma_signals.png'
    plt.savefig(signal_chart_path, dpi=150, bbox_inches='tight')
    print(f"Signal chart saved to: {signal_chart_path}")
    sys.stdout.flush()

    print("\n" + "="*80)
    print(f"Optimization complete in {time.time() - start_time:.1f} seconds")
    print("="*80)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
