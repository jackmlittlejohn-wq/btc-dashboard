#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Different Decay Functions for FGSMA Optimization
Compare exponential, linear, logarithmic, square root, and polynomial decay
"""

import sys
import io
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
from scipy.optimize import differential_evolution

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("TESTING DECAY FUNCTIONS FOR FGSMA")
print("="*80)
sys.stdout.flush()

# ============================================================================
# DATA LOADING (same as before)
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
    sys.stdout.flush()

    return df

# ============================================================================
# EMA & SIGNAL FUNCTIONS
# ============================================================================

def calculate_ema(series, period):
    """Calculate exponential moving average"""
    return series.ewm(span=period, adjust=False).mean()

def get_fg_signal(fg_ema, t1, t2, t3, t4):
    """Get F&G EMA signal (fear = buy, greed = sell)"""
    if fg_ema <= t1:
        return 4  # Strong Buy
    elif fg_ema <= t2:
        return 3  # Buy
    elif fg_ema <= t3:
        return 2  # Hold
    elif fg_ema <= t4:
        return 1  # Sell
    else:
        return 0  # Strong Sell

def get_sma_signal(price_to_sma, r1, r2, r3, r4):
    """Get SMA Ratio signal"""
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

def combine_signals(fg_signal, sma_signal):
    """Combine F&G and SMA signals"""
    if fg_signal == 4 and sma_signal == 4:
        return 4
    if fg_signal == 0 and sma_signal == 0:
        return 0
    if (fg_signal == 4 and sma_signal >= 2) or (sma_signal == 4 and fg_signal >= 2):
        return 3
    if (fg_signal == 0 and sma_signal <= 2) or (sma_signal == 0 and fg_signal <= 2):
        return 1
    if fg_signal == 2 and sma_signal == 2:
        return 2
    if (fg_signal >= 3 and sma_signal <= 1) or (sma_signal >= 3 and fg_signal <= 1):
        return 2
    if fg_signal == 3 and sma_signal == 3:
        return 3
    if fg_signal == 1 and sma_signal == 1:
        return 1
    if fg_signal >= 3 or sma_signal >= 3:
        return 3
    if fg_signal <= 1 or sma_signal <= 1:
        return 1
    return 2

# ============================================================================
# DECAY FUNCTIONS
# ============================================================================

def exponential_decay(r_base, years, decay_param):
    """Current: exponential decay"""
    return r_base * ((1.0 - decay_param) ** years)

def linear_decay(r_base, years, decay_param):
    """Linear decay: decreases linearly with time"""
    factor = max(0.1, 1.0 - (decay_param * years))  # Floor at 10%
    return r_base * factor

def logarithmic_decay(r_base, years, decay_param):
    """Logarithmic decay: fast initial decay, then slows"""
    factor = max(0.1, 1.0 - (decay_param * np.log(1 + years)))
    return r_base * factor

def sqrt_decay(r_base, years, decay_param):
    """Square root decay: moderate initial decay"""
    factor = max(0.1, 1.0 - (decay_param * np.sqrt(years)))
    return r_base * factor

def polynomial_decay(r_base, years, decay_param, power):
    """Polynomial decay: adjustable power"""
    factor = max(0.1, 1.0 - (decay_param * (years ** power)))
    return r_base * factor

# ============================================================================
# SIMULATION
# ============================================================================

def simulate_strategy_with_decay_function(df, ema_period, t1, t2, t3, t4, r1, r2, r3, r4,
                                          decay_param, decay_function, decay_name):
    """Simulate strategy with specified decay function"""
    total_days = len(df)
    starting_capital = total_days * 1.0

    df_sim = df.copy()
    df_sim['fg_ema'] = calculate_ema(df_sim['fg_index'], ema_period)

    btc_holdings = 0.0
    cash = starting_capital

    buy_count = 0
    sell_count = 0
    hold_count = 0

    for idx, row in df_sim.iterrows():
        price = row['price']
        fg_ema = row['fg_ema']
        price_to_sma = row['price_to_sma']
        days_since_start = row['days_since_start']
        years_elapsed = days_since_start / 365.25

        # Apply decay function
        if decay_name == 'exponential':
            r1_adj = exponential_decay(r1, years_elapsed, decay_param)
            r2_adj = exponential_decay(r2, years_elapsed, decay_param)
            r3_adj = exponential_decay(r3, years_elapsed, decay_param)
            r4_adj = exponential_decay(r4, years_elapsed, decay_param)
        elif decay_name == 'linear':
            r1_adj = linear_decay(r1, years_elapsed, decay_param)
            r2_adj = linear_decay(r2, years_elapsed, decay_param)
            r3_adj = linear_decay(r3, years_elapsed, decay_param)
            r4_adj = linear_decay(r4, years_elapsed, decay_param)
        elif decay_name == 'logarithmic':
            r1_adj = logarithmic_decay(r1, years_elapsed, decay_param)
            r2_adj = logarithmic_decay(r2, years_elapsed, decay_param)
            r3_adj = logarithmic_decay(r3, years_elapsed, decay_param)
            r4_adj = logarithmic_decay(r4, years_elapsed, decay_param)
        elif decay_name == 'sqrt':
            r1_adj = sqrt_decay(r1, years_elapsed, decay_param)
            r2_adj = sqrt_decay(r2, years_elapsed, decay_param)
            r3_adj = sqrt_decay(r3, years_elapsed, decay_param)
            r4_adj = sqrt_decay(r4, years_elapsed, decay_param)
        else:  # polynomial
            power = decay_function  # Pass power as function parameter
            r1_adj = polynomial_decay(r1, years_elapsed, decay_param, power)
            r2_adj = polynomial_decay(r2, years_elapsed, decay_param, power)
            r3_adj = polynomial_decay(r3, years_elapsed, decay_param, power)
            r4_adj = polynomial_decay(r4, years_elapsed, decay_param, power)

        # Get signals
        fg_signal = get_fg_signal(fg_ema, t1, t2, t3, t4)
        sma_signal = get_sma_signal(price_to_sma, r1_adj, r2_adj, r3_adj, r4_adj)
        overall_signal = combine_signals(fg_signal, sma_signal)

        # Execute signal
        if overall_signal >= 3:
            cash_to_spend = cash * 0.01
            btc_purchased = cash_to_spend / price
            btc_holdings += btc_purchased
            cash -= cash_to_spend
            buy_count += 1
        elif overall_signal <= 1:
            btc_to_sell = btc_holdings * 0.01
            cash_received = btc_to_sell * price
            btc_holdings -= btc_to_sell
            cash += cash_received
            sell_count += 1
        else:
            hold_count += 1

    final_price = df_sim.iloc[-1]['price']
    final_value = btc_holdings * final_price + cash

    return {
        'final_value': final_value,
        'btc_holdings': btc_holdings,
        'cash_remaining': cash,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'hold_count': hold_count
    }

# ============================================================================
# OPTIMIZATION FOR EACH DECAY FUNCTION
# ============================================================================

def optimize_with_decay_function(df, decay_name, decay_function=None):
    """Optimize parameters for a specific decay function"""
    print(f"\n{'='*80}")
    print(f"TESTING: {decay_name.upper()} DECAY")
    print(f"{'='*80}")
    sys.stdout.flush()

    ema_period = 60  # Fixed based on previous optimization
    best_result = None
    best_params = None

    # Define bounds
    # [t1, t2, t3, t4, r1, r2, r3, r4, decay_param]
    bounds = [
        (10, 30),      # t1
        (25, 40),      # t2
        (60, 75),      # t3
        (75, 90),      # t4
        (0.5, 1.5),    # r1 (base)
        (1.0, 2.5),    # r2 (base)
        (2.5, 5.0),    # r3 (base)
        (3.0, 8.0),    # r4 (base)
        (0.02, 0.30)   # decay_param (wider range for testing)
    ]

    def objective(params):
        t1, t2, t3, t4, r1, r2, r3, r4, decay_param = params
        if not (0 < t1 < t2 < t3 < t4 < 100):
            return 1e9
        if not (0 < r1 < r2 < r3 < r4):
            return 1e9
        result = simulate_strategy_with_decay_function(
            df, ema_period, t1, t2, t3, t4, r1, r2, r3, r4,
            decay_param, decay_function, decay_name
        )
        return -result['final_value']

    print(f"Optimizing {decay_name} decay function...")
    sys.stdout.flush()

    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=50,
        popsize=10,
        tol=0.01,
        seed=42,
        disp=False
    )

    final_value = -result.fun
    best_params = result.x
    t1, t2, t3, t4, r1, r2, r3, r4, decay_param = best_params

    # Calculate 2026 thresholds
    years_elapsed = 7.28
    if decay_name == 'exponential':
        factor = exponential_decay(1.0, years_elapsed, decay_param)
    elif decay_name == 'linear':
        factor = linear_decay(1.0, years_elapsed, decay_param)
    elif decay_name == 'logarithmic':
        factor = logarithmic_decay(1.0, years_elapsed, decay_param)
    elif decay_name == 'sqrt':
        factor = sqrt_decay(1.0, years_elapsed, decay_param)
    else:
        factor = polynomial_decay(1.0, years_elapsed, decay_param, decay_function)

    r1_2026 = r1 * factor
    r2_2026 = r2 * factor
    r3_2026 = r3 * factor
    r4_2026 = r4 * factor

    print(f"\nResults for {decay_name} decay:")
    print(f"  Final Value:        ${final_value:,.2f}")
    print(f"  Decay Parameter:    {decay_param:.4f}")
    print(f"  R1 (2018→2026):     {r1:.2f} → {r1_2026:.2f}")
    print(f"  R2 (2018→2026):     {r2:.2f} → {r2_2026:.2f}")
    print(f"  R3 (2018→2026):     {r3:.2f} → {r3_2026:.2f}")
    print(f"  R4 (2018→2026):     {r4:.2f} → {r4_2026:.2f}")
    print(f"  Reduction Factor:   {(1-factor)*100:.1f}%")
    sys.stdout.flush()

    return {
        'decay_name': decay_name,
        'final_value': final_value,
        'params': best_params,
        'r1_2026': r1_2026,
        'factor': factor
    }

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

    # Test different decay functions
    results = []

    # 1. Exponential (current)
    results.append(optimize_with_decay_function(df, 'exponential'))

    # 2. Linear
    results.append(optimize_with_decay_function(df, 'linear'))

    # 3. Logarithmic
    results.append(optimize_with_decay_function(df, 'logarithmic'))

    # 4. Square root
    results.append(optimize_with_decay_function(df, 'sqrt'))

    # 5. Polynomial (power = 0.5, gentler than linear)
    results.append(optimize_with_decay_function(df, 'polynomial_0.5', decay_function=0.5))

    # 6. Polynomial (power = 1.5, steeper than linear)
    results.append(optimize_with_decay_function(df, 'polynomial_1.5', decay_function=1.5))

    # Compare all results
    print("\n" + "="*80)
    print("COMPARISON OF DECAY FUNCTIONS")
    print("="*80)
    print(f"\n{'Decay Function':<25} {'Final Value':<20} {'R1 (2026)':<15} {'Reduction %':<15}")
    print("-" * 80)

    best_result = None
    for res in results:
        print(f"{res['decay_name']:<25} ${res['final_value']:>15,.2f}   {res['r1_2026']:>10.2f}      {(1-res['factor'])*100:>8.1f}%")
        if best_result is None or res['final_value'] > best_result['final_value']:
            best_result = res

    print("\n" + "="*80)
    print(f"BEST PERFORMING: {best_result['decay_name'].upper()}")
    print(f"Final Value: ${best_result['final_value']:,.2f}")
    print("="*80)

    print(f"\nTotal execution time: {time.time() - start_time:.1f} seconds")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
