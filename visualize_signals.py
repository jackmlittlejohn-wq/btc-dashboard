#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize FGSMA Buy/Sell Signals on BTC Price Chart
"""

import sys
import io
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("FGSMA SIGNAL VISUALIZATION")
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
        print(f"   ✓ Fetched {len(df)} days")
        sys.stdout.flush()
        return df
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
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
        print(f"   ✓ Calculated for {len(df)} days")
        sys.stdout.flush()
        return df
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
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
        print(f"   ✓ Fetched {len(df)} days")
        sys.stdout.flush()
        return df
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
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

    print(f"   ✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   ✓ Total days: {len(df)}")
    sys.stdout.flush()

    return df

# ============================================================================
# SIGNAL CALCULATION
# ============================================================================

def calculate_ema(series, period):
    """Calculate exponential moving average"""
    return series.ewm(span=period, adjust=False).mean()

def get_fg_signal(fg_ema, t1, t2, t3, t4):
    """Get F&G EMA signal based on thresholds (CORRECTED: fear = buy, greed = sell)"""
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
    """Get SMA Ratio signal based on thresholds"""
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
    """Apply time adjustment to SMA ratio thresholds with SQUARE ROOT DECAY"""
    years_elapsed = days_since_start / 365.25
    decay_factor = max(0.3, 1.0 - (decay_rate * np.sqrt(years_elapsed)))  # Sqrt decay with 30% floor
    return r1 * decay_factor, r2 * decay_factor, r3 * decay_factor, r4 * decay_factor

def combine_signals(fg_signal, sma_signal):
    """Combine F&G and SMA signals using boolean logic"""
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

    # Direct conflict
    if (fg_signal >= 3 and sma_signal <= 1) or (sma_signal >= 3 and fg_signal <= 1):
        return 2  # Hold

    # Both say buy
    if fg_signal == 3 and sma_signal == 3:
        return 3  # Buy

    # Both say sell
    if fg_signal == 1 and sma_signal == 1:
        return 1  # Sell

    # Mixed signals lean toward more conservative
    if fg_signal >= 3 or sma_signal >= 3:
        return 3  # Buy
    if fg_signal <= 1 or sma_signal <= 1:
        return 1  # Sell

    return 2  # Hold

# ============================================================================
# GENERATE SIGNALS
# ============================================================================

def generate_signals(df, params):
    """Generate buy/sell signals for visualization"""
    print("\n[Generating signals with optimal parameters...]")
    sys.stdout.flush()

    ema_period = params['ema_period']
    t1 = params['fg_thresholds']['t1_strong_buy_to_buy']
    t2 = params['fg_thresholds']['t2_buy_to_hold']
    t3 = params['fg_thresholds']['t3_hold_to_sell']
    t4 = params['fg_thresholds']['t4_sell_to_strong_sell']
    r1 = params['sma_ratio_thresholds_base']['r1_strong_buy_to_buy']
    r2 = params['sma_ratio_thresholds_base']['r2_buy_to_hold']
    r3 = params['sma_ratio_thresholds_base']['r3_hold_to_sell']
    r4 = params['sma_ratio_thresholds_base']['r4_sell_to_strong_sell']
    decay_rate = params['decay_rate']

    # Calculate F&G EMA
    df_signals = df.copy()
    df_signals['fg_ema'] = calculate_ema(df_signals['fg_index'], ema_period)

    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []

    for idx, row in df_signals.iterrows():
        date = row['date']
        price = row['price']
        fg_ema = row['fg_ema']
        price_to_sma = row['price_to_sma']
        days_since_start = row['days_since_start']

        # Apply time adjustment with DECAY
        r1_adj, r2_adj, r3_adj, r4_adj = apply_time_adjustment(
            r1, r2, r3, r4, days_since_start, decay_rate
        )

        # Get individual signals
        fg_signal = get_fg_signal(fg_ema, t1, t2, t3, t4)
        sma_signal = get_sma_signal(price_to_sma, r1_adj, r2_adj, r3_adj, r4_adj)
        overall_signal = combine_signals(fg_signal, sma_signal)

        # Record buy/sell signals
        if overall_signal >= 3:  # Buy or Strong Buy
            buy_dates.append(date)
            buy_prices.append(price)
        elif overall_signal <= 1:  # Sell or Strong Sell
            sell_dates.append(date)
            sell_prices.append(price)

    print(f"   ✓ Buy signals: {len(buy_dates)}")
    print(f"   ✓ Sell signals: {len(sell_dates)}")
    sys.stdout.flush()

    return df_signals, buy_dates, buy_prices, sell_dates, sell_prices

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(df, buy_dates, buy_prices, sell_dates, sell_prices):
    """Create overlay chart with buy/sell signals"""
    print("\n[Creating visualization...]")
    sys.stdout.flush()

    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot BTC price
    ax.plot(df['date'], df['price'], color='#1f2937', linewidth=1.5, label='BTC Price', zorder=1)

    # Plot buy signals (green dots)
    if len(buy_dates) > 0:
        ax.scatter(buy_dates, buy_prices, color='#10b981', s=30, alpha=0.6,
                   label=f'Buy Signals ({len(buy_dates)})', zorder=3, edgecolors='none')

    # Plot sell signals (red dots)
    if len(sell_dates) > 0:
        ax.scatter(sell_dates, sell_prices, color='#ef4444', s=30, alpha=0.6,
                   label=f'Sell Signals ({len(sell_dates)})', zorder=3, edgecolors='none')

    # Formatting
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('BTC Price (USD)', fontsize=14, fontweight='bold')
    ax.set_title('FGSMA Strategy: Buy/Sell Signals on BTC Price',
                 fontsize=18, fontweight='bold', pad=20)

    # Log scale for better visibility
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95)

    # Add date range annotation
    date_range_text = f"Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    ax.text(0.99, 0.01, date_range_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_path = 'fgsma_signals_overlay.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Chart saved to: {output_path}")
    sys.stdout.flush()

# ============================================================================
# MAIN
# ============================================================================

def main():
    import time
    start_time = time.time()

    # Load optimized parameters
    print("\n[Loading optimized parameters...]")
    sys.stdout.flush()
    try:
        with open('fgsma_optimized.json', 'r') as f:
            data = json.load(f)
            params = data['optimal_parameters']
        print(f"   ✓ Loaded parameters (EMA period: {params['ema_period']} days)")
        sys.stdout.flush()
    except Exception as e:
        print(f"   ✗ ERROR: Could not load fgsma_optimized.json: {e}")
        sys.stdout.flush()
        return

    # Load data
    df = prepare_data()
    if df is None:
        print("\n[ERROR] Failed to prepare data. Exiting.")
        sys.stdout.flush()
        return

    # Generate signals
    df_signals, buy_dates, buy_prices, sell_dates, sell_prices = generate_signals(df, params)

    # Create visualization
    create_visualization(df_signals, buy_dates, buy_prices, sell_dates, sell_prices)

    print("\n" + "="*80)
    print(f"Visualization complete in {time.time() - start_time:.1f} seconds")
    print("="*80)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
