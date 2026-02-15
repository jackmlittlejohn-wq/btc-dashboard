#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin FGSMA Model Dashboard
5-level threshold system with buy/sell signal visualization
"""

import sys
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string, send_from_directory, send_file
import threading
import webbrowser
import time
import json
import os

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__, static_folder='static', static_url_path='/static')

DATA_CACHE = {'chart_data': None, 'last_update': None, 'cache_duration': 300}
SIGNALS_CACHE = {'buy_dates': None, 'buy_prices': None, 'sell_dates': None, 'sell_prices': None, 'last_update': None}
FGSMA_PARAMS = None

# ============================================================================
# LOAD FGSMA PARAMETERS
# ============================================================================

def load_fgsma_parameters():
    """Load FGSMA optimized parameters from JSON file"""
    global FGSMA_PARAMS
    try:
        # Use script directory as base path for better Railway compatibility
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'fgsma_optimized.json')

        print(f"[INFO] Looking for config at: {json_path}")
        print(f"[INFO] Current working directory: {os.getcwd()}")
        print(f"[INFO] Script directory: {script_dir}")

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                FGSMA_PARAMS = data['optimal_parameters']
            print("[INFO] ✓ Loaded FGSMA optimized parameters")
            print(f"[INFO] EMA Period: {FGSMA_PARAMS['ema_period']} days")
        else:
            print(f"[ERROR] fgsma_optimized.json not found at {json_path}")
            print(f"[ERROR] Files in directory: {os.listdir(script_dir)[:20]}")
            FGSMA_PARAMS = None
    except Exception as e:
        print(f"[ERROR] Loading FGSMA parameters: {e}")
        import traceback
        traceback.print_exc()
        FGSMA_PARAMS = None

# ============================================================================
# SIGNAL CALCULATION
# ============================================================================

def calculate_ema(series, period):
    """Calculate exponential moving average"""
    return series.ewm(span=period, adjust=False).mean()

def get_fg_signal(fg_ema, t1, t2, t3, t4):
    """Get F&G EMA signal based on thresholds (CORRECTED: fear = buy, greed = sell)"""
    if fg_ema <= t1:
        return 4, "Strong Buy"  # Extreme fear
    elif fg_ema <= t2:
        return 3, "Buy"  # Moderate fear
    elif fg_ema <= t3:
        return 2, "Hold"  # Neutral
    elif fg_ema <= t4:
        return 1, "Sell"  # Moderate greed
    else:
        return 0, "Strong Sell"  # Extreme greed

def get_sma_signal(price_to_sma, r1, r2, r3, r4):
    """Get SMA Ratio signal based on thresholds (0-4)"""
    if price_to_sma <= r1:
        return 4, "Strong Buy"
    elif price_to_sma <= r2:
        return 3, "Buy"
    elif price_to_sma <= r3:
        return 2, "Hold"
    elif price_to_sma <= r4:
        return 1, "Sell"
    else:
        return 0, "Strong Sell"

def apply_time_adjustment(r1, r2, r3, r4, days_since_start, decay_rate):
    """Apply time adjustment to SMA ratio thresholds with EXPONENTIAL DECAY"""
    years_elapsed = days_since_start / 365.25
    decay_factor = (1.0 - decay_rate) ** years_elapsed
    return r1 * decay_factor, r2 * decay_factor, r3 * decay_factor, r4 * decay_factor

def combine_signals(fg_signal, sma_signal):
    """Combine F&G and SMA signals using boolean logic (returns 0-4)"""
    # Both agree on extreme
    if fg_signal == 4 and sma_signal == 4:
        return 4, "Strong Buy"
    if fg_signal == 0 and sma_signal == 0:
        return 0, "Strong Sell"

    # One strong buy, other buy or hold
    if (fg_signal == 4 and sma_signal >= 2) or (sma_signal == 4 and fg_signal >= 2):
        return 3, "Buy"

    # One strong sell, other sell or hold
    if (fg_signal == 0 and sma_signal <= 2) or (sma_signal == 0 and fg_signal <= 2):
        return 1, "Sell"

    # Both say hold
    if fg_signal == 2 and sma_signal == 2:
        return 2, "Hold"

    # Direct conflict
    if (fg_signal >= 3 and sma_signal <= 1) or (sma_signal >= 3 and fg_signal <= 1):
        return 2, "Hold"

    # Both say buy
    if fg_signal == 3 and sma_signal == 3:
        return 3, "Buy"

    # Both say sell
    if fg_signal == 1 and sma_signal == 1:
        return 1, "Sell"

    # Mixed signals lean toward more conservative
    if fg_signal >= 3 or sma_signal >= 3:
        return 3, "Buy"
    if fg_signal <= 1 or sma_signal <= 1:
        return 1, "Sell"

    return 2, "Hold"

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_recent_data_coingecko(days=90):
    """Fetch recent Bitcoin OHLC data from CoinGecko API (free, no API key needed)"""
    try:
        print(f"[INFO] Fetching last {days} days OHLC from CoinGecko...")
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
        params = {'vs_currency': 'usd', 'days': days}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data and len(data) > 0:
                # CoinGecko OHLC format: [timestamp, open, high, low, close]
                df_recent = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df_recent['time'] = pd.to_datetime(df_recent['timestamp'], unit='ms').dt.tz_localize(None)
                df_recent['volume'] = 0  # Not provided by OHLC endpoint

                # Reorder columns to match historical data format
                df_recent = df_recent[['time', 'open', 'high', 'low', 'close', 'volume']]

                print(f"[OK] Fetched {len(df_recent)} days of OHLC data from CoinGecko")
                print(f"[OK] Date range: {df_recent['time'].iloc[0].strftime('%Y-%m-%d')} to {df_recent['time'].iloc[-1].strftime('%Y-%m-%d')}")
                return df_recent

        print(f"[WARNING] CoinGecko returned status {response.status_code}")
        return None
    except Exception as e:
        print(f"[WARNING] CoinGecko API failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_daily_btc_data():
    """Fetch daily BTC/USD price data: historical baseline + live updates"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'btc_historical_data.csv')

    # Load historical baseline from CSV
    df_historical = None
    try:
        if os.path.exists(csv_path):
            print(f"[INFO] Loading historical baseline from CSV...")
            df_historical = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df_historical = df_historical.reset_index()
            df_historical['Date'] = pd.to_datetime(df_historical['Date']).dt.tz_localize(None)
            df_historical = df_historical.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high',
                                     'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            df_historical = df_historical[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
            print(f"[OK] Loaded {len(df_historical)} days of historical data")
    except Exception as e:
        print(f"[WARNING] Could not load CSV baseline: {e}")

    # Fetch recent live data from CoinGecko
    df_recent = fetch_recent_data_coingecko(days=90)

    # Combine historical + recent
    if df_historical is not None and df_recent is not None:
        # Remove overlap: keep historical up to 90 days ago, then append recent
        cutoff_date = df_recent['time'].min() - timedelta(days=1)
        df_historical_filtered = df_historical[df_historical['time'] <= cutoff_date]

        # Combine
        df = pd.concat([df_historical_filtered, df_recent], ignore_index=True)
        df = df.drop_duplicates(subset=['time'], keep='last')
        df = df.sort_values('time').reset_index(drop=True)

        print(f"[OK] Combined dataset: {len(df)} days total (historical + live)")
        print(f"[OK] Latest data: {df['time'].iloc[-1].strftime('%Y-%m-%d')}")
        return df

    elif df_recent is not None:
        print(f"[WARNING] Using CoinGecko data only ({len(df_recent)} days)")
        return df_recent

    elif df_historical is not None:
        print(f"[WARNING] Using cached data only (may be stale)")
        return df_historical

    else:
        print("[ERROR] No data source available!")
        return None

def calculate_200w_sma_from_daily(daily_df):
    """Calculate 200-week SMA from daily data (1400 day rolling average)"""
    try:
        if daily_df is None or len(daily_df) < 1400:
            print(f"[WARNING] Not enough data for 200W SMA")
            return None

        df = daily_df.copy()
        df['sma_200w'] = df['close'].rolling(window=1400, min_periods=1400).mean()
        sma_df = df[['time', 'sma_200w']].copy()
        sma_df = sma_df.rename(columns={'sma_200w': 'value'})
        sma_df = sma_df.dropna()
        return sma_df
    except Exception as e:
        print(f"[ERROR] SMA calculation: {e}")
        return None

def fetch_fear_greed():
    """Fetch daily Fear & Greed index data"""
    try:
        url = 'https://api.alternative.me/fng/?limit=0'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        records = []
        for item in data['data']:
            records.append({
                'time': pd.to_datetime(int(item['timestamp']), unit='s'),
                'value': int(item['value'])
            })

        df = pd.DataFrame(records)
        return df
    except Exception as e:
        print(f"[ERROR] F&G: {e}")
        return None

def fetch_current_price():
    try:
        response = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', timeout=5)
        response.raise_for_status()
        return float(response.json()['price'])
    except:
        return None

def prepare_all_data():
    print("\n[INFO] Fetching daily data...")
    start_time = time.time()

    daily_df = fetch_daily_btc_data()
    if daily_df is None:
        return None

    sma_df = calculate_200w_sma_from_daily(daily_df)
    fg_df = fetch_fear_greed()

    # Merge datasets
    if sma_df is not None:
        daily_df = daily_df.merge(sma_df, on='time', how='left')
    else:
        daily_df['value'] = None

    if fg_df is not None:
        daily_df = daily_df.merge(fg_df, on='time', how='left')
        daily_df['fg_index'] = daily_df['value_y'].ffill()
        daily_df = daily_df.drop(columns=['value_y'])
        daily_df = daily_df.rename(columns={'value_x': 'sma_200w'})
    else:
        daily_df['fg_index'] = 50
        daily_df = daily_df.rename(columns={'value': 'sma_200w'})

    daily_df['price_to_sma'] = daily_df['close'] / daily_df['sma_200w']

    # Calculate days since start for time adjustment
    daily_df = daily_df.sort_values('time').reset_index(drop=True)
    first_valid_idx = daily_df[daily_df['sma_200w'].notna() & daily_df['fg_index'].notna()].index.min()
    if pd.notna(first_valid_idx):
        start_date = daily_df.loc[first_valid_idx, 'time']
        daily_df['days_since_start'] = (daily_df['time'] - start_date).dt.days
    else:
        daily_df['days_since_start'] = 0

    # Calculate F&G EMA if parameters are loaded
    if FGSMA_PARAMS is not None:
        ema_period = FGSMA_PARAMS['ema_period']
        daily_df['fg_ema'] = calculate_ema(daily_df['fg_index'], ema_period)

    # Get current values
    current_price = fetch_current_price()

    if current_price is None and len(daily_df) > 0:
        current_price = float(daily_df['close'].iloc[-1])

    current_fg = None
    if fg_df is not None and len(fg_df) > 0:
        try:
            url = 'https://api.alternative.me/fng/?limit=1'
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            current_fg = int(response.json()['data'][0]['value'])
        except:
            current_fg = int(daily_df['fg_index'].iloc[-1]) if len(daily_df) > 0 else 50

    current_sma = float(daily_df['sma_200w'].iloc[-1]) if len(daily_df) > 0 and pd.notna(daily_df['sma_200w'].iloc[-1]) else None
    current_ratio = float(daily_df['price_to_sma'].iloc[-1]) if len(daily_df) > 0 and pd.notna(daily_df['price_to_sma'].iloc[-1]) else None
    current_fg_ema = float(daily_df['fg_ema'].iloc[-1]) if len(daily_df) > 0 and 'fg_ema' in daily_df.columns and pd.notna(daily_df['fg_ema'].iloc[-1]) else None
    current_days_since_start = float(daily_df['days_since_start'].iloc[-1]) if len(daily_df) > 0 else 0

    print(f"[OK] Data fetched in {time.time() - start_time:.2f}s")

    return {
        'daily': daily_df,
        'current_price': current_price,
        'current_sma': current_sma,
        'current_fg': current_fg,
        'current_ratio': current_ratio,
        'current_fg_ema': current_fg_ema,
        'current_days_since_start': current_days_since_start
    }

# ============================================================================
# GENERATE BUY/SELL SIGNALS
# ============================================================================

def generate_signals(df, params):
    """Generate buy/sell signals for entire dataset"""
    if params is None:
        return [], [], [], []

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

    # Only calculate for rows with complete data
    valid_df = df[(df['sma_200w'].notna()) & (df['fg_ema'].notna()) & (df['price_to_sma'].notna())].copy()

    if len(valid_df) == 0:
        return [], [], [], []

    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []

    for idx, row in valid_df.iterrows():
        date = row['time']
        price = row['close']
        fg_ema = row['fg_ema']
        price_to_sma = row['price_to_sma']
        days_since_start = row['days_since_start']

        # Apply time adjustment with DECAY
        r1_adj, r2_adj, r3_adj, r4_adj = apply_time_adjustment(
            r1, r2, r3, r4, days_since_start, decay_rate
        )

        # Get signals
        fg_signal_val, _ = get_fg_signal(fg_ema, t1, t2, t3, t4)
        sma_signal_val, _ = get_sma_signal(price_to_sma, r1_adj, r2_adj, r3_adj, r4_adj)
        overall_signal_val, _ = combine_signals(fg_signal_val, sma_signal_val)

        # Record buy/sell signals
        if overall_signal_val >= 3:  # Buy or Strong Buy
            buy_dates.append(date.strftime('%Y-%m-%d'))
            buy_prices.append(float(price))
        elif overall_signal_val <= 1:  # Sell or Strong Sell
            sell_dates.append(date.strftime('%Y-%m-%d'))
            sell_prices.append(float(price))

    return buy_dates, buy_prices, sell_dates, sell_prices

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/service-worker.js')
def service_worker():
    """Serve the service worker file"""
    return send_file('service-worker.js', mimetype='application/javascript')

@app.route('/api/data')
def api_data():
    global DATA_CACHE
    now = time.time()

    if (DATA_CACHE['chart_data'] and DATA_CACHE['last_update'] and
        now - DATA_CACHE['last_update'] < DATA_CACHE['cache_duration']):
        print("[INFO] Serving from cache")
        return jsonify(DATA_CACHE['chart_data'])

    print("[INFO] Fetching fresh data")
    data = prepare_all_data()
    if data is None:
        return jsonify({'error': 'Failed to fetch data'}), 500

    # Convert daily DataFrame to JSON
    daily_records = []
    for _, row in data['daily'].iterrows():
        daily_records.append({
            'time': row['time'].strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
            'sma_200w': float(row['sma_200w']) if pd.notna(row['sma_200w']) else None,
            'fg_index': float(row['fg_index']) if pd.notna(row['fg_index']) else None,
            'fg_ema': float(row['fg_ema']) if 'fg_ema' in row and pd.notna(row['fg_ema']) else None,
            'price_to_sma': float(row['price_to_sma']) if pd.notna(row['price_to_sma']) else None
        })

    # Calculate current signals
    current_overall_signal = None
    current_fg_signal = None
    current_sma_signal = None

    if FGSMA_PARAMS is not None and data['current_fg_ema'] is not None and data['current_ratio'] is not None:
        t1 = FGSMA_PARAMS['fg_thresholds']['t1_strong_buy_to_buy']
        t2 = FGSMA_PARAMS['fg_thresholds']['t2_buy_to_hold']
        t3 = FGSMA_PARAMS['fg_thresholds']['t3_hold_to_sell']
        t4 = FGSMA_PARAMS['fg_thresholds']['t4_sell_to_strong_sell']
        r1 = FGSMA_PARAMS['sma_ratio_thresholds_base']['r1_strong_buy_to_buy']
        r2 = FGSMA_PARAMS['sma_ratio_thresholds_base']['r2_buy_to_hold']
        r3 = FGSMA_PARAMS['sma_ratio_thresholds_base']['r3_hold_to_sell']
        r4 = FGSMA_PARAMS['sma_ratio_thresholds_base']['r4_sell_to_strong_sell']
        decay_rate = FGSMA_PARAMS['decay_rate']

        # Apply time adjustment with DECAY
        r1_adj, r2_adj, r3_adj, r4_adj = apply_time_adjustment(
            r1, r2, r3, r4, data['current_days_since_start'], decay_rate
        )

        # Get signals
        fg_signal_val, fg_signal_text = get_fg_signal(data['current_fg_ema'], t1, t2, t3, t4)
        sma_signal_val, sma_signal_text = get_sma_signal(data['current_ratio'], r1_adj, r2_adj, r3_adj, r4_adj)
        overall_signal_val, overall_signal_text = combine_signals(fg_signal_val, sma_signal_val)

        current_overall_signal = {'value': overall_signal_val, 'text': overall_signal_text}
        current_fg_signal = {'value': fg_signal_val, 'text': fg_signal_text}
        current_sma_signal = {'value': sma_signal_val, 'text': sma_signal_text}

    # Generate buy/sell signals for visualization (cached for 1 hour)
    global SIGNALS_CACHE
    signals_age = time.time() - SIGNALS_CACHE['last_update'] if SIGNALS_CACHE['last_update'] else float('inf')
    if signals_age > 3600 or SIGNALS_CACHE['buy_dates'] is None:
        print("[INFO] Generating signals (cached for 1 hour)")
        buy_dates, buy_prices, sell_dates, sell_prices = generate_signals(data['daily'], FGSMA_PARAMS)
        SIGNALS_CACHE = {
            'buy_dates': buy_dates,
            'buy_prices': buy_prices,
            'sell_dates': sell_dates,
            'sell_prices': sell_prices,
            'last_update': time.time()
        }
    else:
        print("[INFO] Using cached signals")
        buy_dates = SIGNALS_CACHE['buy_dates']
        buy_prices = SIGNALS_CACHE['buy_prices']
        sell_dates = SIGNALS_CACHE['sell_dates']
        sell_prices = SIGNALS_CACHE['sell_prices']

    response = {
        'daily': daily_records,
        'current_price': float(data['current_price']) if data['current_price'] else 0,
        'current_sma': float(data['current_sma']) if data['current_sma'] else 0,
        'current_fg': int(data['current_fg']) if data['current_fg'] else 50,
        'current_fg_ema': float(data['current_fg_ema']) if data['current_fg_ema'] else None,
        'current_ratio': float(data['current_ratio']) if data['current_ratio'] else 1.0,
        'current_days_since_start': float(data['current_days_since_start']) if data['current_days_since_start'] else 0,
        'current_overall_signal': current_overall_signal,
        'current_fg_signal': current_fg_signal,
        'current_sma_signal': current_sma_signal,
        'buy_signals': {'dates': buy_dates, 'prices': buy_prices},
        'sell_signals': {'dates': sell_dates, 'prices': sell_prices},
        'fgsma_params': FGSMA_PARAMS
    }

    DATA_CACHE['chart_data'] = response
    DATA_CACHE['last_update'] = now
    return jsonify(response)

# ============================================================================
# FRONTEND
# ============================================================================

@app.route('/')
def index():
    html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <title>Bitcoin FGSMA Model</title>

    <!-- PWA Meta Tags -->
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#000000">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="BTC FGSMA">
    <link rel="apple-touch-icon" href="/static/icon-192.png">
    <meta name="description" content="Fear & Greed + 200W SMA trading signals for Bitcoin">

    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
            background: #f8f9fa;
            color: #1a1a1a;
            overflow: hidden;
        }
        .container { display: flex; height: 100vh; }
        .chart-section {
            width: 70%;
            display: flex;
            flex-direction: column;
            padding: 24px;
            background: white;
            overflow-y: auto;
        }
        .chart-title {
            font-size: 28px;
            font-weight: 700;
            color: #000;
            margin-bottom: 20px;
            letter-spacing: -0.5px;
        }
        .chart-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e5e7eb;
            align-items: center;
        }
        .chart-tab {
            padding: 12px 24px;
            background: none;
            border: none;
            font-size: 14px;
            font-weight: 600;
            color: #6b7280;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }
        .chart-tab:hover {
            color: #000;
        }
        .chart-tab.active {
            color: #10b981;
            border-bottom-color: #10b981;
        }
        .log-toggle {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: #f3f4f6;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .log-toggle:hover {
            background: #e5e7eb;
        }
        .log-toggle input {
            cursor: pointer;
            width: 16px;
            height: 16px;
            accent-color: #10b981;
        }
        .log-toggle label {
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            color: #374151;
        }
        .chart-controls {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            padding: 12px 16px;
            background: #f8f9fa;
            border-radius: 12px;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .checkbox-group input {
            cursor: pointer;
            width: 18px;
            height: 18px;
            accent-color: #10b981;
        }
        .checkbox-group label {
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #374151;
        }
        .chart-wrapper {
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .chart-wrapper.hidden {
            display: none;
        }
        #chart { height: 650px; }
        #signal-chart { height: 650px; }
        .data-panel {
            width: 30%;
            padding: 24px;
            background: #ffffff;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
            border-left: 1px solid #e5e7eb;
        }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #000;
            margin-bottom: 12px;
        }
        .data-card {
            background: #f8f9fa;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            transition: all 0.2s;
        }
        .data-card:hover {
            background: #f3f4f6;
            border-color: #d1d5db;
        }
        .data-label {
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .data-value {
            font-size: 24px;
            font-weight: 700;
            color: #000;
        }
        .data-value.green { color: #10b981; }
        .data-value.red { color: #ef4444; }
        .data-value.orange { color: #f59e0b; }

        .fgsma-card {
            border: none;
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
        }
        .fgsma-card.strong-buy {
            background: #059669;
            color: #ffffff;
            box-shadow: 0 10px 30px rgba(5, 150, 105, 0.3);
        }
        .fgsma-card.buy {
            background: #d1fae5;
            color: #065f46;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
        }
        .fgsma-card.sell {
            background: #fecdd3;
            color: #881337;
            box-shadow: 0 10px 30px rgba(225, 29, 72, 0.2);
        }
        .fgsma-card.strong-sell {
            background: #dc2626;
            color: #ffffff;
            box-shadow: 0 10px 30px rgba(220, 38, 38, 0.3);
        }
        .fgsma-card.hold {
            background: #e5e7eb;
            color: #374151;
            box-shadow: 0 10px 30px rgba(107, 114, 128, 0.2);
        }
        .fgsma-date {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            opacity: 0.9;
        }
        .fgsma-signal-main {
            text-align: center;
            margin: 20px 0;
        }
        .fgsma-signal-text {
            font-size: 48px;
            font-weight: 900;
            margin-bottom: 8px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .fgsma-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.7;
        }
        .fgsma-subsignals {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 16px;
        }
        .fgsma-subsignal {
            background: rgba(255,255,255,0.3);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .fgsma-subsignal-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.8;
            margin-bottom: 6px;
        }
        .fgsma-subsignal-value {
            font-size: 16px;
            font-weight: 700;
        }
        .signal-strength-bar {
            margin-top: 8px;
            height: 4px;
            background: rgba(0,0,0,0.1);
            border-radius: 2px;
            overflow: hidden;
        }
        .signal-strength-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.5s ease, background 0.3s ease;
        }
        .faq-content {
            padding: 24px;
            max-width: 1000px;
            margin: 0 auto;
        }
        .faq-section {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .faq-section-title {
            font-size: 20px;
            font-weight: 700;
            color: #000;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .faq-section-icon {
            font-size: 24px;
        }
        .faq-section-content {
            color: #374151;
            line-height: 1.6;
        }
        .faq-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }
        .faq-table th {
            text-align: left;
            padding: 10px;
            background: #f8f9fa;
            border: 1px solid #e5e7eb;
            font-weight: 600;
            color: #1f2937;
        }
        .faq-table td {
            padding: 10px;
            border: 1px solid #e5e7eb;
            color: #374151;
        }
        .faq-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }
        .faq-stat {
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .faq-stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #10b981;
            margin-bottom: 4px;
        }
        .faq-stat-label {
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .loading {
            position: fixed;
            inset: 0;
            background: rgba(255, 255, 255, 0.95);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            flex-direction: column;
            gap: 20px;
        }
        .loading.hidden { display: none; }
        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid #e5e7eb;
            border-top-color: #10b981;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text { color: #6b7280; font-size: 14px; font-weight: 500; }

        /* Collapsible FAQ sections */
        .faq-section-header {
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 0;
        }
        .faq-section-header:hover {
            opacity: 0.8;
        }
        .faq-collapse-icon {
            font-size: 20px;
            transition: transform 0.3s;
        }
        .faq-section-header.collapsed .faq-collapse-icon {
            transform: rotate(-90deg);
        }
        .faq-section-body {
            max-height: 5000px;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        .faq-section-body.collapsed {
            max-height: 0;
        }

        /* MOBILE RESPONSIVE STYLES */
        @media (max-width: 768px) {
            body {
                overflow-x: hidden;
                overflow-y: auto;
            }

            .container {
                flex-direction: column;
                height: auto;
                min-height: 100vh;
            }

            .chart-section {
                width: 100%;
                padding: 16px;
                overflow-y: visible;
            }

            .chart-title {
                font-size: 22px;
                margin-bottom: 16px;
            }

            .chart-tabs {
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 16px;
            }

            .chart-tab {
                flex: 1;
                min-width: calc(50% - 4px);
                padding: 10px 12px;
                font-size: 13px;
                text-align: center;
            }

            .log-toggle {
                width: 100%;
                justify-content: center;
                padding: 12px 16px;
                margin-top: 8px;
                min-height: 44px;
            }

            .log-toggle input {
                width: 20px;
                height: 20px;
            }

            .log-toggle label {
                font-size: 14px;
            }

            .chart-controls {
                flex-direction: column;
                gap: 12px;
                padding: 12px;
            }

            .checkbox-group {
                width: 100%;
                min-height: 44px;
                padding: 8px;
            }

            .checkbox-group input {
                width: 20px;
                height: 20px;
            }

            .checkbox-group label {
                font-size: 15px;
            }

            #chart, #signal-chart {
                height: 350px !important;
            }

            .data-panel {
                width: 100%;
                padding: 16px;
                border-left: none;
                border-top: 1px solid #e5e7eb;
            }

            .section-title {
                font-size: 16px;
                margin-bottom: 12px;
            }

            .fgsma-card {
                padding: 20px;
            }

            .fgsma-date {
                font-size: 13px;
                margin-bottom: 12px;
            }

            .fgsma-signal-text {
                font-size: 36px;
            }

            .fgsma-subsignals {
                grid-template-columns: 1fr;
                gap: 12px;
            }

            .fgsma-subsignal {
                padding: 14px;
            }

            .fgsma-subsignal-label {
                font-size: 11px;
            }

            .fgsma-subsignal-value {
                font-size: 18px;
            }

            .data-card {
                padding: 14px;
                margin-bottom: 10px;
            }

            .data-label {
                font-size: 11px;
            }

            .data-value {
                font-size: 20px;
            }

            .faq-content {
                padding: 16px;
            }

            .faq-section {
                padding: 16px;
                margin-bottom: 12px;
            }

            .faq-section-title {
                font-size: 18px;
                margin-bottom: 12px;
            }

            .faq-table {
                font-size: 12px;
                display: block;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }

            .faq-table th,
            .faq-table td {
                padding: 8px;
                white-space: nowrap;
            }

            .faq-stats {
                grid-template-columns: 1fr;
                gap: 12px;
            }

            .faq-stat {
                padding: 14px;
            }

            .faq-stat-value {
                font-size: 24px;
            }

            .faq-stat-label {
                font-size: 11px;
            }

            /* Make FAQ collapsible on mobile */
            .faq-section-header {
                display: flex !important;
            }
        }

        /* Extra small devices (phones in portrait, less than 375px) */
        @media (max-width: 374px) {
            .chart-title {
                font-size: 20px;
            }

            .chart-tab {
                font-size: 12px;
                padding: 8px 10px;
            }

            .fgsma-signal-text {
                font-size: 32px;
            }

            .data-value {
                font-size: 18px;
            }

            #chart, #signal-chart {
                height: 300px !important;
            }
        }

        /* Landscape orientation on mobile */
        @media (max-width: 768px) and (orientation: landscape) {
            #chart, #signal-chart {
                height: 250px !important;
            }

            .container {
                overflow-y: auto;
            }
        }
    </style>
</head>
<body>
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div class="loading-text">Loading data...</div>
    </div>

    <div class="container">
        <div class="chart-section">
            <h1 class="chart-title">Bitcoin FGSMA Model</h1>

            <div class="chart-tabs">
                <button class="chart-tab active" onclick="switchChart('price')">Price Chart</button>
                <button class="chart-tab" onclick="switchChart('signals')">Historical Signals</button>
                <button class="chart-tab" onclick="switchChart('faq')">FAQ</button>
                <div class="log-toggle" id="log-toggle">
                    <input type="checkbox" id="toggle-log">
                    <label for="toggle-log">Log Scale</label>
                </div>
            </div>

            <div class="chart-controls" id="price-controls">
                <div class="checkbox-group">
                    <input type="checkbox" id="toggle-price" checked>
                    <label for="toggle-price">BTC Price</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="toggle-sma" checked>
                    <label for="toggle-sma">200W SMA</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="toggle-fg">
                    <label for="toggle-fg">Fear & Greed</label>
                </div>
            </div>

            <div class="chart-wrapper" id="chart"></div>
            <div class="chart-wrapper hidden" id="signal-chart"></div>
            <div class="chart-wrapper hidden" id="faq-content"></div>
        </div>

        <div class="data-panel">
            <div>
                <div class="section-title">FGSMA Signal</div>
                <div class="fgsma-card" id="fgsma-card">
                    <div class="fgsma-date" id="fgsma-date">--</div>
                    <div class="fgsma-signal-main">
                        <div class="fgsma-label">Overall Signal</div>
                        <div class="fgsma-signal-text" id="fgsma-signal">--</div>
                        <div class="signal-strength-bar" style="margin-top: 12px;">
                            <div class="signal-strength-fill" id="overall-strength"></div>
                        </div>
                    </div>
                    <div class="fgsma-subsignals">
                        <div class="fgsma-subsignal">
                            <div class="fgsma-subsignal-label">F&G EMA</div>
                            <div class="fgsma-subsignal-value" id="fg-signal">--</div>
                            <div class="fgsma-subsignal-numeric" id="fg-numeric" style="font-size: 11px; opacity: 0.8; margin-top: 2px;">--</div>
                            <div class="signal-strength-bar">
                                <div class="signal-strength-fill" id="fg-strength"></div>
                            </div>
                        </div>
                        <div class="fgsma-subsignal">
                            <div class="fgsma-subsignal-label">SMA Ratio</div>
                            <div class="fgsma-subsignal-value" id="sma-signal">--</div>
                            <div class="fgsma-subsignal-numeric" id="sma-numeric" style="font-size: 11px; opacity: 0.8; margin-top: 2px;">--</div>
                            <div class="signal-strength-bar">
                                <div class="signal-strength-fill" id="sma-strength"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div>
                <div class="section-title">Live Market Data</div>
                <div class="data-card">
                    <div class="data-label">Bitcoin Price</div>
                    <div class="data-value" id="price">$--,---</div>
                </div>
                <div class="data-card">
                    <div class="data-label">200-Week SMA</div>
                    <div class="data-value orange" id="sma">$--,---</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Price to SMA Ratio</div>
                    <div class="data-value" id="ratio">-.--x</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Fear & Greed Index</div>
                    <div class="data-value" id="fg">--</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let chartData = null;
        let currentLayout = null;
        let currentSignalLayout = null;
        let currentChartView = 'price';

        function toggleFaqSection(header) {
            if (window.innerWidth > 768) return; // Only work on mobile

            const body = header.nextElementSibling;
            const icon = header.querySelector('.faq-collapse-icon');

            header.classList.toggle('collapsed');
            body.classList.toggle('collapsed');
        }

        function switchChart(view) {
            currentChartView = view;
            const priceChart = document.getElementById('chart');
            const signalChart = document.getElementById('signal-chart');
            const faqContent = document.getElementById('faq-content');
            const priceControls = document.getElementById('price-controls');
            const logToggle = document.getElementById('log-toggle');
            const tabs = document.querySelectorAll('.chart-tab');

            tabs.forEach(tab => tab.classList.remove('active'));

            if (view === 'price') {
                priceChart.classList.remove('hidden');
                signalChart.classList.add('hidden');
                faqContent.classList.add('hidden');
                priceControls.classList.remove('hidden');
                logToggle.style.display = 'flex';
                tabs[0].classList.add('active');
                if (chartData) {
                    setTimeout(() => Plotly.Plots.resize('chart'), 100);
                }
            } else if (view === 'signals') {
                priceChart.classList.add('hidden');
                signalChart.classList.remove('hidden');
                faqContent.classList.add('hidden');
                priceControls.classList.add('hidden');
                logToggle.style.display = 'flex';
                tabs[1].classList.add('active');
                if (chartData) {
                    setTimeout(() => Plotly.Plots.resize('signal-chart'), 100);
                }
            } else if (view === 'faq') {
                priceChart.classList.add('hidden');
                signalChart.classList.add('hidden');
                faqContent.classList.remove('hidden');
                priceControls.classList.add('hidden');
                logToggle.style.display = 'none';
                tabs[2].classList.add('active');
                if (chartData) {
                    renderFAQ(chartData);
                }
            }
        }

        // Load toggle states from localStorage
        function loadToggleStates() {
            const priceState = localStorage.getItem('toggle-price');
            const smaState = localStorage.getItem('toggle-sma');
            const fgState = localStorage.getItem('toggle-fg');
            const logState = localStorage.getItem('toggle-log');

            document.getElementById('toggle-price').checked = priceState !== 'false';
            document.getElementById('toggle-sma').checked = smaState !== 'false';
            document.getElementById('toggle-fg').checked = fgState === 'true';
            document.getElementById('toggle-log').checked = logState === 'true';
        }

        function saveToggleState(id, checked) {
            localStorage.setItem(id, checked);
        }

        async function loadData() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();
                chartData = data;

                renderChart(data);
                renderSignalChart(data);
                updateLiveData(data);
                updateFGSMASignal(data);

                document.getElementById('loading').classList.add('hidden');
            } catch (err) {
                console.error('Error loading data:', err);
                document.getElementById('loading').classList.add('hidden');
                alert('Error loading data. Please refresh the page.');
            }
        }

        function renderChart(data) {
            try {
                const traces = [];
                const showPrice = document.getElementById('toggle-price').checked;
                const showSMA = document.getElementById('toggle-sma').checked;
                const showFG = document.getElementById('toggle-fg').checked;
                const useLogScale = document.getElementById('toggle-log').checked;

            if (showPrice && data.daily) {
                traces.push({
                    type: 'candlestick',
                    x: data.daily.map(d => d.time),
                    open: data.daily.map(d => d.open),
                    high: data.daily.map(d => d.high),
                    low: data.daily.map(d => d.low),
                    close: data.daily.map(d => d.close),
                    name: 'BTC/USD',
                    increasing: {line: {color: '#10b981'}},
                    decreasing: {line: {color: '#ef4444'}},
                    yaxis: 'y',
                    showlegend: true
                });

                traces.push({
                    type: 'bar',
                    x: data.daily.map(d => d.time),
                    y: data.daily.map(d => d.volume),
                    name: 'Volume',
                    marker: {color: data.daily.map(d => d.close >= d.open ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)')},
                    yaxis: 'y2',
                    showlegend: true
                });
            }

            if (showSMA && data.daily) {
                const smaData = data.daily.filter(d => d.sma_200w !== null);
                traces.push({
                    type: 'scatter',
                    mode: 'lines',
                    x: smaData.map(d => d.time),
                    y: smaData.map(d => d.sma_200w),
                    name: '200W SMA',
                    line: {color: '#f59e0b', width: 2},
                    yaxis: 'y',
                    showlegend: true
                });
            }

            if (showFG && data.daily) {
                const fgData = data.daily.filter(d => d.fg_index !== null);
                traces.push({
                    type: 'scatter',
                    mode: 'lines',
                    x: fgData.map(d => d.time),
                    y: fgData.map(d => d.fg_index),
                    name: 'Fear & Greed',
                    line: {color: '#8b5cf6', width: 2, shape: 'spline', smoothing: 0.3},
                    yaxis: 'y3',
                    showlegend: true
                });
            }

            // Extend x-axis ~2 years into future
            const lastDate = new Date(data.daily[data.daily.length - 1].time);
            const futureDate = new Date(lastDate);
            futureDate.setFullYear(futureDate.getFullYear() + 2);

            const layout = {
                paper_bgcolor: '#ffffff',
                plot_bgcolor: '#f8f9fa',
                font: {color: '#374151', family: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif'},
                showlegend: true,
                legend: {
                    font: {color: '#1f2937', size: 12, family: '-apple-system, BlinkMacSystemFont, sans-serif'},
                    bgcolor: 'rgba(248,249,250,0.95)',
                    bordercolor: '#d1d5db',
                    borderwidth: 1,
                    itemclick: false,
                    itemdoubleclick: false,
                    orientation: 'h',
                    x: 0,
                    y: 1.08,
                    xanchor: 'left',
                    yanchor: 'top'
                },
                hovermode: 'x unified',
                dragmode: 'zoom',
                xaxis: {
                    type: 'date',
                    gridcolor: '#e5e7eb',
                    fixedrange: false,
                    rangeslider: {visible: false},
                    range: [data.daily[0].time, futureDate.toISOString().split('T')[0]]
                },
                yaxis: {
                    title: {text: 'Price (USD)', font: {size: 12, color: '#6b7280'}},
                    gridcolor: '#e5e7eb',
                    side: 'left',
                    domain: [0.25, 1],
                    fixedrange: false,
                    type: useLogScale ? 'log' : 'linear'
                },
                yaxis2: {
                    gridcolor: 'transparent',
                    showticklabels: false,
                    domain: [0, 0.2],
                    fixedrange: false
                },
                yaxis3: {
                    title: {text: 'Fear & Greed', font: {size: 12, color: '#6b7280'}},
                    gridcolor: 'transparent',
                    side: 'right',
                    overlaying: 'y',
                    range: [0, 100],
                    fixedrange: false
                },
                margin: {l: 60, r: 80, t: 60, b: 60}
            };

            const config = {
                responsive: true,
                displayModeBar: window.innerWidth > 768,
                displaylogo: false,
                scrollZoom: true,
                modeBarButtonsToRemove: window.innerWidth <= 768 ? ['lasso2d', 'select2d'] : [],
                doubleClick: 'reset',
                showTips: true,
                touchAction: 'pan'
            };

            // Preserve zoom/pan state on refresh, but reset when toggling scale
            const chartDiv = document.getElementById('chart');
            const previousScale = currentLayout ? currentLayout.yaxis.type : 'linear';
            const scaleChanged = previousScale !== (useLogScale ? 'log' : 'linear');

            if (!scaleChanged && currentLayout && chartDiv.layout && chartDiv.layout.xaxis) {
                layout.xaxis.range = chartDiv.layout.xaxis.range;
                if (!useLogScale && chartDiv.layout.yaxis) {
                    layout.yaxis.range = chartDiv.layout.yaxis.range;
                }
            }
            // If scale changed, let Plotly auto-calculate the axis range

            Plotly.newPlot('chart', traces, layout, config);
            currentLayout = layout;
            } catch (err) {
                console.error('Error rendering chart:', err);
            }
        }

        function renderSignalChart(data) {
            try {
            const traces = [];
            const useLogScale = document.getElementById('toggle-log').checked;

            // BTC price line
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: data.daily.map(d => d.time),
                y: data.daily.map(d => d.close),
                name: 'BTC Price',
                line: {color: '#1f2937', width: 1.5},
                showlegend: true
            });

            // Buy signals (green dots)
            if (data.buy_signals && data.buy_signals.dates.length > 0) {
                traces.push({
                    type: 'scatter',
                    mode: 'markers',
                    x: data.buy_signals.dates,
                    y: data.buy_signals.prices,
                    name: `Buy (${data.buy_signals.dates.length})`,
                    marker: {color: '#10b981', size: 6, opacity: 0.6},
                    showlegend: true
                });
            }

            // Sell signals (red dots)
            if (data.sell_signals && data.sell_signals.dates.length > 0) {
                traces.push({
                    type: 'scatter',
                    mode: 'markers',
                    x: data.sell_signals.dates,
                    y: data.sell_signals.prices,
                    name: `Sell (${data.sell_signals.dates.length})`,
                    marker: {color: '#ef4444', size: 6, opacity: 0.6},
                    showlegend: true
                });
            }

            const layout = {
                paper_bgcolor: '#ffffff',
                plot_bgcolor: '#f8f9fa',
                font: {color: '#374151', family: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif'},
                showlegend: true,
                legend: {
                    font: {color: '#1f2937', size: 12, family: '-apple-system, BlinkMacSystemFont, sans-serif'},
                    bgcolor: 'rgba(248,249,250,0.95)',
                    bordercolor: '#d1d5db',
                    borderwidth: 1,
                    orientation: 'h',
                    x: 0,
                    y: 1.08,
                    xanchor: 'left',
                    yanchor: 'top'
                },
                hovermode: 'closest',
                dragmode: 'zoom',
                xaxis: {
                    type: 'date',
                    gridcolor: '#e5e7eb',
                    fixedrange: false
                },
                yaxis: {
                    title: {text: 'Price (USD)', font: {size: 12, color: '#6b7280'}},
                    gridcolor: '#e5e7eb',
                    fixedrange: false,
                    type: useLogScale ? 'log' : 'linear'
                },
                margin: {l: 60, r: 40, t: 60, b: 60}
            };

            const config = {
                responsive: true,
                displayModeBar: window.innerWidth > 768,
                displaylogo: false,
                scrollZoom: true,
                modeBarButtonsToRemove: window.innerWidth <= 768 ? ['lasso2d', 'select2d'] : [],
                doubleClick: 'reset',
                showTips: true,
                touchAction: 'pan'
            };

            // Preserve zoom/pan state, but reset when toggling scale
            const chartDiv = document.getElementById('signal-chart');
            const previousScale = currentSignalLayout ? currentSignalLayout.yaxis.type : 'linear';
            const scaleChanged = previousScale !== (useLogScale ? 'log' : 'linear');

            if (!scaleChanged && currentSignalLayout && chartDiv.layout && chartDiv.layout.xaxis) {
                layout.xaxis.range = chartDiv.layout.xaxis.range;
                if (!useLogScale && chartDiv.layout.yaxis) {
                    layout.yaxis.range = chartDiv.layout.yaxis.range;
                }
            }
            // If scale changed, let Plotly auto-calculate the axis range

            Plotly.newPlot('signal-chart', traces, layout, config);
            currentSignalLayout = layout;
            } catch (err) {
                console.error('Error rendering signal chart:', err);
            }
        }

        function updateLiveData(data) {
            document.getElementById('price').textContent = '$' + data.current_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
            document.getElementById('sma').textContent = '$' + data.current_sma.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
            document.getElementById('ratio').textContent = data.current_ratio.toFixed(2) + 'x';
            document.getElementById('fg').textContent = data.current_fg;

            // Update numeric values in signal card
            if (data.current_fg_ema !== null && data.current_fg_ema !== undefined) {
                document.getElementById('fg-numeric').textContent = data.current_fg_ema.toFixed(1);
            }
            if (data.current_ratio !== null && data.current_ratio !== undefined) {
                document.getElementById('sma-numeric').textContent = data.current_ratio.toFixed(2) + 'x';
            }
        }

        function renderPerformanceChart() {
            if (!chartData || !chartData.daily || !chartData.fgsma_params) {
                console.log('Missing data for performance chart');
                return;
            }

            const daily = chartData.daily;
            const params = chartData.fgsma_params;

            // Find start index (where we have both F&G EMA and SMA)
            let startIdx = 0;
            for (let i = 0; i < daily.length; i++) {
                if (daily[i].fg_ema !== null && daily[i].sma_200w !== null && daily[i].price_to_sma !== null) {
                    startIdx = i;
                    break;
                }
            }

            const totalDays = daily.length - startIdx;
            const dailyInvestment = 1;
            const totalCapital = totalDays * dailyInvestment;

            // Calculate DCA portfolio (buy $1 of BTC every day)
            let dcaCash = 0;
            let dcaBtc = 0;
            const dcaPortfolioValues = [];
            const dcaDates = [];

            // Calculate FGSMA portfolio (start with total capital upfront)
            let fgsmaCash = totalCapital;
            let fgsmaBtc = 0;
            const fgsmaPortfolioValues = [];
            const fgsmaDates = [];

            const t1 = params.fg_thresholds.t1_strong_buy_to_buy;
            const t2 = params.fg_thresholds.t2_buy_to_hold;
            const t3 = params.fg_thresholds.t3_hold_to_sell;
            const t4 = params.fg_thresholds.t4_sell_to_strong_sell;
            const r1_base = params.sma_ratio_thresholds_base.r1_strong_buy_to_buy;
            const r2_base = params.sma_ratio_thresholds_base.r2_buy_to_hold;
            const r3_base = params.sma_ratio_thresholds_base.r3_hold_to_sell;
            const r4_base = params.sma_ratio_thresholds_base.r4_sell_to_strong_sell;
            const decayRate = params.decay_rate;

            for (let i = startIdx; i < daily.length; i++) {
                const row = daily[i];
                if (!row.fg_ema || !row.price_to_sma || !row.close) continue;

                const date = row.time;
                const price = row.close;
                const fgEma = row.fg_ema;
                const priceToSma = row.price_to_sma;
                const daysFromStart = i - startIdx;

                // DCA: buy $1 every day
                dcaCash += dailyInvestment;
                const btcBought = dailyInvestment / price;
                dcaBtc += btcBought;
                dcaPortfolioValues.push(dcaBtc * price);
                dcaDates.push(date);

                // FGSMA: get signal (no daily investment, started with all capital)

                // Calculate decayed thresholds (EXPONENTIAL DECAY)
                const years = daysFromStart / 365.25;
                const decayFactor = Math.pow(1 - decayRate, years);
                const r1 = r1_base * decayFactor;
                const r2 = r2_base * decayFactor;
                const r3 = r3_base * decayFactor;
                const r4 = r4_base * decayFactor;

                // Get F&G signal
                let fgSignal = 2;
                if (fgEma <= t1) fgSignal = 4;
                else if (fgEma <= t2) fgSignal = 3;
                else if (fgEma <= t3) fgSignal = 2;
                else if (fgEma <= t4) fgSignal = 1;
                else fgSignal = 0;

                // Get SMA signal
                let smaSignal = 2;
                if (priceToSma <= r1) smaSignal = 4;
                else if (priceToSma <= r2) smaSignal = 3;
                else if (priceToSma <= r3) smaSignal = 2;
                else if (priceToSma <= r4) smaSignal = 1;
                else smaSignal = 0;

                // Combine signals (simplified)
                let overallSignal = 2;
                if (fgSignal === 4 && smaSignal === 4) overallSignal = 4;
                else if (fgSignal === 0 && smaSignal === 0) overallSignal = 0;
                else if ((fgSignal === 4 && smaSignal >= 2) || (smaSignal === 4 && fgSignal >= 2)) overallSignal = 3;
                else if ((fgSignal === 0 && smaSignal <= 2) || (smaSignal === 0 && fgSignal <= 2)) overallSignal = 1;
                else if (fgSignal === 2 && smaSignal === 2) overallSignal = 2;
                else if ((fgSignal >= 3 && smaSignal <= 1) || (smaSignal >= 3 && fgSignal <= 1)) overallSignal = 2;
                else if (fgSignal === 3 && smaSignal === 3) overallSignal = 3;
                else if (fgSignal === 1 && smaSignal === 1) overallSignal = 1;
                else if (fgSignal >= 3 || smaSignal >= 3) overallSignal = 3;
                else if (fgSignal <= 1 || smaSignal <= 1) overallSignal = 1;

                // Execute trades
                if (overallSignal >= 3 && fgsmaCash > 0) {
                    // Buy: invest available cash
                    const buyAmount = fgsmaCash;
                    fgsmaBtc += buyAmount / price;
                    fgsmaCash -= buyAmount;
                } else if (overallSignal <= 1 && fgsmaBtc > 0) {
                    // Sell: liquidate all BTC
                    fgsmaCash += fgsmaBtc * price;
                    fgsmaBtc = 0;
                }

                const fgsmaValue = fgsmaBtc * price + fgsmaCash;
                fgsmaPortfolioValues.push(fgsmaValue);
                fgsmaDates.push(date);
            }

            console.log(`DCA final: $${dcaPortfolioValues[dcaPortfolioValues.length-1].toFixed(2)}`);
            console.log(`FGSMA final: $${fgsmaPortfolioValues[fgsmaPortfolioValues.length-1].toFixed(2)}`);

            // Create Plotly chart
            const traces = [
                {
                    type: 'scatter',
                    mode: 'lines',
                    x: fgsmaDates,
                    y: fgsmaPortfolioValues,
                    name: 'FGSMA Strategy',
                    line: {color: '#10b981', width: 3},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(16, 185, 129, 0.1)'
                },
                {
                    type: 'scatter',
                    mode: 'lines',
                    x: dcaDates,
                    y: dcaPortfolioValues,
                    name: 'DCA Baseline',
                    line: {color: '#6b7280', width: 2, dash: 'dash'}
                }
            ];

            const layout = {
                paper_bgcolor: '#ffffff',
                plot_bgcolor: '#f8f9fa',
                font: {color: '#374151', family: '-apple-system, BlinkMacSystemFont, sans-serif', size: 11},
                showlegend: true,
                legend: {
                    font: {color: '#1f2937', size: 12, family: '-apple-system, BlinkMacSystemFont, sans-serif'},
                    bgcolor: 'rgba(248,249,250,0.95)',
                    bordercolor: '#d1d5db',
                    borderwidth: 1,
                    x: 0.02,
                    y: 0.98,
                    xanchor: 'left',
                    yanchor: 'top',
                    orientation: 'v'
                },
                hovermode: 'x unified',
                xaxis: {
                    type: 'date',
                    gridcolor: '#e5e7eb',
                    title: {text: 'Date', font: {size: 11, color: '#6b7280'}}
                },
                yaxis: {
                    title: {text: 'Portfolio Value (USD)', font: {size: 11, color: '#6b7280'}},
                    gridcolor: '#e5e7eb',
                    type: 'linear'
                },
                margin: {l: 60, r: 20, t: 30, b: 40}
            };

            const config = {
                responsive: true,
                displayModeBar: window.innerWidth > 768,
                displaylogo: false,
                modeBarButtonsToRemove: window.innerWidth <= 768 ? ['lasso2d', 'select2d'] : [],
                touchAction: 'pan'
            };

            Plotly.newPlot('performance-chart', traces, layout, config);

            console.log('Performance chart rendered');
        }

        function renderFAQ(data) {
            const faqDiv = document.getElementById('faq-content');
            const params = data.fgsma_params;

            if (!params) {
                faqDiv.innerHTML = '<div class="faq-content"><p>Loading parameters...</p></div>';
                return;
            }

            const t1 = params.fg_thresholds.t1_strong_buy_to_buy.toFixed(2);
            const t2 = params.fg_thresholds.t2_buy_to_hold.toFixed(2);
            const t3 = params.fg_thresholds.t3_hold_to_sell.toFixed(2);
            const t4 = params.fg_thresholds.t4_sell_to_strong_sell.toFixed(2);
            const r1 = params.sma_ratio_thresholds_base.r1_strong_buy_to_buy.toFixed(2);
            const r2 = params.sma_ratio_thresholds_base.r2_buy_to_hold.toFixed(2);
            const r3 = params.sma_ratio_thresholds_base.r3_hold_to_sell.toFixed(2);
            const r4 = params.sma_ratio_thresholds_base.r4_sell_to_strong_sell.toFixed(2);
            const decay = (params.decay_rate * 100).toFixed(2);

            // Calculate 2026 values (EXPONENTIAL DECAY)
            const years = 7.28;
            const decayFactor = Math.pow(1 - params.decay_rate, years);
            const r1_2026 = (r1 * decayFactor).toFixed(2);
            const r2_2026 = (r2 * decayFactor).toFixed(2);
            const r3_2026 = (r3 * decayFactor).toFixed(2);
            const r4_2026 = (r4 * decayFactor).toFixed(2);

            faqDiv.innerHTML = `
                <div class="faq-content">
                    <div class="faq-section">
                        <div class="faq-section-header" onclick="toggleFaqSection(this)">
                            <div class="faq-section-title" style="margin-bottom: 0;">Strategy Overview</div>
                            <span class="faq-collapse-icon">▼</span>
                        </div>
                        <div class="faq-section-body" id="faq-section-1">
                            <p style="margin-bottom: 16px;">The FGSMA model generates trading signals by combining sentiment analysis with technical price positioning. Both indicators must align for strong buy/sell signals, reducing false positives.</p>

                            <div style="margin-bottom: 24px;">
                                <div style="font-weight: 600; margin-bottom: 8px; color: #1f2937;">Fear & Greed EMA (60-day)</div>
                                <p style="margin-bottom: 12px; font-size: 13px; color: #4b5563;">Smoothed sentiment indicator. Low values indicate extreme fear (contrarian buying opportunity), high values indicate extreme greed (potential market top).</p>
                                <table class="faq-table">
                                    <thead>
                                        <tr>
                                            <th>Range</th>
                                            <th>Signal</th>
                                            <th>Interpretation</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>0 - ${t1}</td>
                                            <td><strong>Strong Buy</strong></td>
                                            <td>Extreme fear</td>
                                        </tr>
                                        <tr>
                                            <td>${t1} - ${t2}</td>
                                            <td><strong>Buy</strong></td>
                                            <td>Moderate fear</td>
                                        </tr>
                                        <tr>
                                            <td>${t2} - ${t3}</td>
                                            <td><strong>Hold</strong></td>
                                            <td>Neutral sentiment</td>
                                        </tr>
                                        <tr>
                                            <td>${t3} - ${t4}</td>
                                            <td><strong>Sell</strong></td>
                                            <td>Moderate greed</td>
                                        </tr>
                                        <tr>
                                            <td>${t4} - 100</td>
                                            <td><strong>Strong Sell</strong></td>
                                            <td>Extreme greed</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>

                            <div style="margin-bottom: 24px;">
                                <div style="font-weight: 600; margin-bottom: 8px; color: #1f2937;">Price / 200-Week SMA Ratio</div>
                                <p style="margin-bottom: 12px; font-size: 13px; color: #4b5563;">Measures how far price has deviated from its long-term average. Thresholds decay ${decay}% annually as Bitcoin matures and volatility compresses.</p>
                                <table class="faq-table">
                                    <thead>
                                        <tr>
                                            <th>2018 Threshold</th>
                                            <th>2026 Threshold</th>
                                            <th>Signal</th>
                                            <th>Interpretation</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>&lt; ${r1}x</td>
                                            <td>&lt; ${r1_2026}x</td>
                                            <td><strong>Strong Buy</strong></td>
                                            <td>Deep below SMA</td>
                                        </tr>
                                        <tr>
                                            <td>${r1}x - ${r2}x</td>
                                            <td>${r1_2026}x - ${r2_2026}x</td>
                                            <td><strong>Buy</strong></td>
                                            <td>Below SMA</td>
                                        </tr>
                                        <tr>
                                            <td>${r2}x - ${r3}x</td>
                                            <td>${r2_2026}x - ${r3_2026}x</td>
                                            <td><strong>Hold</strong></td>
                                            <td>Near fair value</td>
                                        </tr>
                                        <tr>
                                            <td>${r3}x - ${r4}x</td>
                                            <td>${r3_2026}x - ${r4_2026}x</td>
                                            <td><strong>Sell</strong></td>
                                            <td>Extended above SMA</td>
                                        </tr>
                                        <tr>
                                            <td>&gt; ${r4}x</td>
                                            <td>&gt; ${r4_2026}x</td>
                                            <td><strong>Strong Sell</strong></td>
                                            <td>Extreme overvaluation</td>
                                        </tr>
                                    </tbody>
                                </table>
                                <div style="margin-top: 8px; font-size: 11px; color: #6b7280;">
                                    Decay factor (2018→2026): ${((1 - decayFactor) * 100).toFixed(1)}% reduction
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="faq-section">
                        <div class="faq-section-header" onclick="toggleFaqSection(this)">
                            <div class="faq-section-title" style="margin-bottom: 0;">Back-Tested Performance</div>
                            <span class="faq-collapse-icon">▼</span>
                        </div>
                        <div class="faq-section-body" id="faq-section-2">
                            <div id="performance-chart" style="width: 100%; height: 350px; margin-top: 12px;"></div>
                            <p style="margin-top: 8px; font-size: 10px; color: #9ca3af;"><em>Past performance does not guarantee future results. For educational purposes only.</em></p>
                        </div>
                    </div>
                </div>
            `;

            // Make FAQ sections collapsible on mobile only
            if (window.innerWidth <= 768) {
                document.querySelectorAll('.faq-section-header').forEach(header => {
                    header.style.display = 'flex';
                });
            }

            // Render performance comparison chart
            setTimeout(() => renderPerformanceChart(), 100);
        }

        function calculateSignalStrength(value, thresholds, isInverted = false) {
            // Calculate how strong a signal is based on distance from boundaries
            // Returns {strength: 0-100, color: '#color'}
            const [t1, t2, t3, t4] = thresholds;

            let strength = 0;
            let color = '#9ca3af';

            if (isInverted) {
                // For F&G: low = buy, high = sell
                if (value <= t1) {
                    strength = 100 - (value / t1) * 30; // Strong Buy: 70-100%
                    color = '#059669';
                } else if (value <= t2) {
                    const range = t2 - t1;
                    const pos = (value - t1) / range;
                    strength = 70 - pos * 20; // Buy: 50-70%
                    color = '#10b981';
                } else if (value <= t3) {
                    strength = 30; // Hold: 30%
                    color = '#6b7280';
                } else if (value <= t4) {
                    const range = t4 - t3;
                    const pos = (value - t3) / range;
                    strength = 30 + pos * 20; // Sell: 30-50%
                    color = '#ef4444';
                } else {
                    strength = 50 + ((value - t4) / (100 - t4)) * 50; // Strong Sell: 50-100%
                    color = '#dc2626';
                }
            } else {
                // For SMA: low = buy, high = sell (same as inverted F&G logic actually)
                if (value <= t1) {
                    strength = 100 - (value / t1) * 30;
                    color = '#059669';
                } else if (value <= t2) {
                    const range = t2 - t1;
                    const pos = (value - t1) / range;
                    strength = 70 - pos * 20;
                    color = '#10b981';
                } else if (value <= t3) {
                    strength = 30;
                    color = '#6b7280';
                } else if (value <= t4) {
                    const range = t4 - t3;
                    const pos = (value - t3) / range;
                    strength = 30 + pos * 20;
                    color = '#ef4444';
                } else {
                    const maxRatio = 10; // Assume max ratio of 10x for calculation
                    strength = 50 + Math.min(((value - t4) / (maxRatio - t4)) * 50, 50);
                    color = '#dc2626';
                }
            }

            return {strength: Math.min(100, Math.max(0, strength)), color};
        }

        function updateSignalStrengthBars(data) {
            if (!data.fgsma_params || !data.current_fg_ema || !data.current_ratio ||
                !data.current_overall_signal || data.current_days_since_start === undefined) {
                console.warn('Missing data for strength bars');
                return;
            }

            const params = data.fgsma_params;
            const fgEma = data.current_fg_ema;
            const ratio = data.current_ratio;

            // F&G thresholds
            const t1 = params.fg_thresholds.t1_strong_buy_to_buy;
            const t2 = params.fg_thresholds.t2_buy_to_hold;
            const t3 = params.fg_thresholds.t3_hold_to_sell;
            const t4 = params.fg_thresholds.t4_sell_to_strong_sell;

            // Calculate decayed SMA thresholds (EXPONENTIAL DECAY)
            const years = data.current_days_since_start / 365.25;
            const decayFactor = Math.pow(1 - params.decay_rate, years);
            const r1 = params.sma_ratio_thresholds_base.r1_strong_buy_to_buy * decayFactor;
            const r2 = params.sma_ratio_thresholds_base.r2_buy_to_hold * decayFactor;
            const r3 = params.sma_ratio_thresholds_base.r3_hold_to_sell * decayFactor;
            const r4 = params.sma_ratio_thresholds_base.r4_sell_to_strong_sell * decayFactor;

            // Calculate strengths
            const fgStrength = calculateSignalStrength(fgEma, [t1, t2, t3, t4], true);
            const smaStrength = calculateSignalStrength(ratio, [r1, r2, r3, r4], false);

            // Overall strength is average of both
            const overallStrength = {
                strength: (fgStrength.strength + smaStrength.strength) / 2,
                color: data.current_overall_signal.value >= 3 ? '#10b981' :
                       data.current_overall_signal.value <= 1 ? '#ef4444' : '#6b7280'
            };

            // Update bars
            const fgBar = document.getElementById('fg-strength');
            const smaBar = document.getElementById('sma-strength');
            const overallBar = document.getElementById('overall-strength');

            fgBar.style.width = fgStrength.strength + '%';
            fgBar.style.background = `linear-gradient(90deg, ${fgStrength.color}, ${fgStrength.color}99)`;

            smaBar.style.width = smaStrength.strength + '%';
            smaBar.style.background = `linear-gradient(90deg, ${smaStrength.color}, ${smaStrength.color}99)`;

            overallBar.style.width = overallStrength.strength + '%';
            overallBar.style.background = `linear-gradient(90deg, ${overallStrength.color}, ${overallStrength.color}99)`;
        }

        function updateFGSMASignal(data) {
            const dateEl = document.getElementById('fgsma-date');
            const signalEl = document.getElementById('fgsma-signal');
            const cardEl = document.getElementById('fgsma-card');
            const fgSignalEl = document.getElementById('fg-signal');
            const smaSignalEl = document.getElementById('sma-signal');

            // Update date
            const now = new Date();
            const dateStr = now.toLocaleDateString('en-US', {year: 'numeric', month: 'long', day: 'numeric'});
            dateEl.textContent = dateStr;

            if (data.current_overall_signal) {
                // Overall signal
                signalEl.textContent = data.current_overall_signal.text;

                // Update card class based on signal
                const signalClass = data.current_overall_signal.text.toLowerCase().replace(' ', '-');
                cardEl.className = 'fgsma-card ' + signalClass;

                // Subsignals
                fgSignalEl.textContent = data.current_fg_signal.text;
                smaSignalEl.textContent = data.current_sma_signal.text;

                // Update strength bars
                updateSignalStrengthBars(data);
            } else {
                signalEl.textContent = '--';
                cardEl.className = 'fgsma-card hold';
                fgSignalEl.textContent = '--';
                smaSignalEl.textContent = '--';
            }
        }

        document.getElementById('toggle-price').onchange = function() {
            saveToggleState('toggle-price', this.checked);
            renderChart(chartData);
        };
        document.getElementById('toggle-sma').onchange = function() {
            saveToggleState('toggle-sma', this.checked);
            renderChart(chartData);
        };
        document.getElementById('toggle-fg').onchange = function() {
            saveToggleState('toggle-fg', this.checked);
            renderChart(chartData);
        };
        document.getElementById('toggle-log').onchange = function() {
            saveToggleState('toggle-log', this.checked);
            // Apply to whichever chart is currently active
            if (currentChartView === 'price') {
                renderChart(chartData);
            } else if (currentChartView === 'signals') {
                renderSignalChart(chartData);
            }
        };

        loadToggleStates();
        loadData();
        setInterval(loadData, 30000);

        // Handle window resize for responsive charts
        let resizeTimer;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function() {
                if (chartData) {
                    if (currentChartView === 'price') {
                        Plotly.Plots.resize('chart');
                    } else if (currentChartView === 'signals') {
                        Plotly.Plots.resize('signal-chart');
                    }
                }
            }, 250);
        });

        // Handle orientation change on mobile
        window.addEventListener('orientationchange', function() {
            setTimeout(function() {
                if (chartData) {
                    if (currentChartView === 'price') {
                        Plotly.Plots.resize('chart');
                    } else if (currentChartView === 'signals') {
                        Plotly.Plots.resize('signal-chart');
                    }
                }
            }, 300);
        });

        // Register Service Worker for PWA functionality
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                navigator.serviceWorker.register('/service-worker.js')
                    .then(function(registration) {
                        console.log('[PWA] Service Worker registered successfully:', registration.scope);
                    })
                    .catch(function(error) {
                        console.log('[PWA] Service Worker registration failed:', error);
                    });
            });
        }

        // PWA install prompt
        let deferredPrompt;
        window.addEventListener('beforeinstallprompt', (e) => {
            console.log('[PWA] Install prompt available');
            e.preventDefault();
            deferredPrompt = e;
            // You could show a custom install button here if desired
        });

        window.addEventListener('appinstalled', () => {
            console.log('[PWA] App installed successfully');
            deferredPrompt = null;
        });
    </script>
</body>
</html>
    '''
    return render_template_string(html)

def open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

def main():
    load_fgsma_parameters()

    # Get port from environment variable (for cloud deployment) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')

    print("="*70)
    print("BITCOIN FGSMA MODEL DASHBOARD")
    print("="*70)
    print(f"\nServer: http://{host}:{port}")
    print(f"API: http://{host}:{port}/api/data")
    print("\nPress Ctrl+C to stop")
    print("="*70)

    # Only open browser if running locally
    if host == 'localhost' or host == '127.0.0.1':
        threading.Thread(target=open_browser, daemon=True).start()

    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
