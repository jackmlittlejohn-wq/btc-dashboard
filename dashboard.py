#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin 4-Indicator Optimized Dashboard
Uses 200W SMA, 50W MA (regime-based), F&G, and RSI
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
import gc

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__, static_folder='static', static_url_path='/static')

DATA_CACHE = {'chart_data': None, 'last_update': None, 'cache_duration': 300}  # 5 minutes
CONFIG = None
START_YEAR = 2017  # Reference year for decay calculation

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

def load_config():
    """Load optimized parameters from JSON file"""
    global CONFIG
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'final_5stage_config.json')

        print(f"[INFO] Looking for config at: {json_path}")

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                CONFIG = json.load(f)
            print("[INFO] ✓ Loaded optimized 4-indicator configuration")
            print(f"[INFO] Total Return: {CONFIG.get('total_return', 0):.2f}%")
        else:
            print(f"[ERROR] final_5stage_config.json not found at {json_path}")
            CONFIG = None
    except Exception as e:
        print(f"[ERROR] Loading configuration: {e}")
        import traceback
        traceback.print_exc()
        CONFIG = None

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_daily_btc_data():
    """Fetch daily BTC/USD price data using ccxt"""
    import ccxt

    exchanges_to_try = [('bitstamp', 'BTC/USD', 'Bitstamp')]

    for exchange_id, symbol, exchange_name in exchanges_to_try:
        try:
            print(f"[INFO] Fetching Bitcoin data from {exchange_name}...")

            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000
            })

            timeframe = '1d'
            limit = 500
            all_candles = []
            since = exchange.parse8601('2016-01-01T00:00:00Z')

            print(f"[INFO] Fetching historical data from {exchange_name}...")
            batch_count = 0
            max_batches = 10

            while batch_count < max_batches:
                batch_count += 1
                print(f"[INFO] Batch {batch_count}...")

                try:
                    candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

                    if not candles:
                        print(f"[INFO] No more data available")
                        break

                    all_candles.extend(candles)

                    if len(candles) < limit:
                        print(f"[INFO] Got final batch with {len(candles)} candles")
                        break

                    since = candles[-1][0] + 86400000
                    time.sleep(0.5)

                except Exception as e:
                    print(f"[WARNING] Batch {batch_count} from {exchange_name} failed: {e}")
                    break

            print(f"[INFO] Total candles fetched: {len(all_candles)}")

            if len(all_candles) > 1400:
                df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize(None)
                df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

                df = df.drop_duplicates(subset=['time'], keep='last')
                df = df.sort_values('time').reset_index(drop=True)

                print(f"[OK] ✓ Fetched {len(df)} days from {exchange_name}")
                print(f"[OK] Date range: {df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}")
                gc.collect()
                return df
            else:
                print(f"[WARNING] {exchange_name} returned insufficient data ({len(all_candles)} days)")

        except Exception as e:
            print(f"[WARNING] {exchange_name} failed: {str(e)[:200]}")
            continue

    print("[ERROR] All exchange attempts failed")
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
    """Get current BTC price"""
    try:
        response = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', timeout=5)
        response.raise_for_status()
        return float(response.json()['price'])
    except:
        return None

# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

def calculate_200w_sma(df):
    """Calculate 200-week SMA from daily data"""
    try:
        if len(df) < 1400:
            return None

        df_copy = df.copy()
        df_copy['sma_200w'] = df_copy['close'].rolling(window=1400, min_periods=1400).mean()
        df_copy['sma_ratio'] = df_copy['close'] / df_copy['sma_200w']
        return df_copy
    except Exception as e:
        print(f"[ERROR] 200W SMA calculation: {e}")
        return None

def calculate_50w_ma_with_regime(df):
    """Calculate 50-week MA with regime detection"""
    try:
        # Resample to weekly data
        df_weekly = df.set_index('time').resample('W-SUN')['close'].last().dropna()
        ma_50w = df_weekly.rolling(window=50, min_periods=1).mean()

        ma_df = pd.DataFrame({
            'weekly_close': df_weekly,
            'ma_50w': ma_50w
        })

        # Detect regime
        regimes = []
        for i in range(len(ma_df)):
            if i < 2:
                regimes.append('bull')
                continue

            last_3 = ma_df.iloc[max(0, i-2):i+1]
            above_count = (last_3['weekly_close'] > last_3['ma_50w']).sum()

            if above_count == 3:
                regimes.append('bull')
            elif above_count == 0:
                regimes.append('bear')
            else:
                regimes.append(regimes[-1] if regimes else 'bull')

        ma_df['regime'] = regimes
        ma_df['ma50w_ratio'] = ma_df['weekly_close'] / ma_df['ma_50w']

        return ma_df
    except Exception as e:
        print(f"[ERROR] 50W MA calculation: {e}")
        return None

def calculate_rsi(series, period):
    """Calculate RSI indicator"""
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        print(f"[ERROR] RSI calculation: {e}")
        return None

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def get_signals(latest_row, config):
    """Calculate signals for all 4 indicators"""
    if config is None:
        return None

    try:
        p = config['parameters']
        w = config['weights']

        years = latest_row['time'].year - START_YEAR

        # 1. 200W SMA Signal
        decay = p['sma_200w']['decay'] ** years
        adj_buy = p['sma_200w']['buy'] * decay
        adj_sell = p['sma_200w']['sell'] * decay
        sig_200w = 1 if latest_row['sma_ratio'] < adj_buy else (-1 if latest_row['sma_ratio'] > adj_sell else 0)

        # 2. 50W MA Signal (regime-dependent)
        if latest_row['regime'] == 'bull':
            sig_50w = -1 if latest_row['ma50w_ratio'] > p['ma_50w']['bull_ext'] else (
                1 if latest_row['ma50w_ratio'] < p['ma_50w']['bull_supp'] else 0
            )
        else:
            sig_50w = -1 if latest_row['ma50w_ratio'] > p['ma_50w']['bear_res'] else (
                1 if latest_row['ma50w_ratio'] < p['ma_50w']['bear_deep'] else 0
            )

        # 3. F&G Signal
        sig_fg = 1 if latest_row['fg_ema'] < p['fg']['buy'] else (
            -1 if latest_row['fg_ema'] > p['fg']['sell'] else 0
        )

        # 4. RSI Signal
        sig_rsi = 1 if latest_row['rsi'] < p['rsi']['buy'] else (
            -1 if latest_row['rsi'] > p['rsi']['sell'] else 0
        )

        # Weighted combined signal
        combined = (sig_200w * w['w_200w']) + (sig_50w * w['w_50w']) + (sig_fg * w['w_fg']) + (sig_rsi * w['w_rsi'])

        # Determine action
        if combined >= 2:
            action = 'STRONG BUY'
            action_class = 'buy'
        elif combined <= -2:
            action = 'STRONG SELL'
            action_class = 'sell'
        else:
            action = 'HOLD'
            action_class = 'hold'

        return {
            '200w': {
                'signal': sig_200w,
                'signal_text': 'BUY' if sig_200w == 1 else ('SELL' if sig_200w == -1 else 'NEUTRAL'),
                'weighted': round(sig_200w * w['w_200w'], 2),
                'weight': round(w['w_200w'], 2),
                'ratio': round(latest_row['sma_ratio'], 3),
                'buy_threshold': round(adj_buy, 2),
                'sell_threshold': round(adj_sell, 2)
            },
            '50w': {
                'signal': sig_50w,
                'signal_text': 'BUY' if sig_50w == 1 else ('SELL' if sig_50w == -1 else 'NEUTRAL'),
                'weighted': round(sig_50w * w['w_50w'], 2),
                'weight': round(w['w_50w'], 2),
                'ratio': round(latest_row['ma50w_ratio'], 3),
                'regime': latest_row['regime'].upper()
            },
            'fg': {
                'signal': sig_fg,
                'signal_text': 'BUY' if sig_fg == 1 else ('SELL' if sig_fg == -1 else 'NEUTRAL'),
                'weighted': round(sig_fg * w['w_fg'], 2),
                'weight': round(w['w_fg'], 2),
                'value': round(latest_row['fg_ema'], 1),
                'buy_threshold': round(p['fg']['buy'], 1),
                'sell_threshold': round(p['fg']['sell'], 1)
            },
            'rsi': {
                'signal': sig_rsi,
                'signal_text': 'BUY' if sig_rsi == 1 else ('SELL' if sig_rsi == -1 else 'NEUTRAL'),
                'weighted': round(sig_rsi * w['w_rsi'], 2),
                'weight': round(w['w_rsi'], 2),
                'value': round(latest_row['rsi'], 1),
                'buy_threshold': round(p['rsi']['buy'], 1),
                'sell_threshold': round(p['rsi']['sell'], 1)
            },
            'combined': round(combined, 2),
            'action': action,
            'action_class': action_class
        }
    except Exception as e:
        print(f"[ERROR] Signal calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_all_data():
    """Fetch and prepare all indicator data"""
    print("\n[INFO] Fetching data...")
    start_time = time.time()

    # Fetch raw data
    daily_df = fetch_daily_btc_data()
    if daily_df is None:
        return None

    fg_df = fetch_fear_greed()
    if fg_df is None:
        print("[WARNING] No F&G data available")
        return None

    # Calculate 200W SMA
    daily_df = calculate_200w_sma(daily_df)
    if daily_df is None:
        return None

    # Calculate F&G EMA
    if CONFIG is not None:
        fg_period = CONFIG['parameters']['fg']['ema_period']
        fg_df = fg_df.sort_values('time').copy()
        fg_df['fg_ema'] = fg_df['value'].ewm(span=fg_period, adjust=False).mean()

    # Calculate RSI
    if CONFIG is not None:
        rsi_period = CONFIG['parameters']['rsi']['period']
        daily_df['rsi'] = calculate_rsi(daily_df['close'], rsi_period)

    # Calculate 50W MA with regime
    ma_df = calculate_50w_ma_with_regime(daily_df)
    if ma_df is None:
        return None

    # Merge all data
    daily_df = daily_df.set_index('time')
    daily_df = daily_df.merge(ma_df[['ma_50w', 'regime', 'ma50w_ratio']], left_index=True, right_index=True, how='left')
    daily_df = daily_df.merge(fg_df[['time', 'value', 'fg_ema']].set_index('time'), left_index=True, right_index=True, how='left')
    daily_df = daily_df.rename(columns={'value': 'fg_index'})

    # Forward fill missing values
    for col in ['sma_200w', 'sma_ratio', 'ma_50w', 'regime', 'ma50w_ratio', 'fg_index', 'fg_ema', 'rsi']:
        if col in daily_df.columns:
            daily_df[col] = daily_df[col].ffill()

    daily_df = daily_df.reset_index()

    # Get current values
    current_price = fetch_current_price()
    if current_price is None and len(daily_df) > 0:
        current_price = float(daily_df['close'].iloc[-1])

    print(f"[OK] Data prepared in {time.time() - start_time:.2f}s")

    return {
        'daily': daily_df,
        'current_price': current_price
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/service-worker.js')
def service_worker():
    """Serve the service worker file"""
    try:
        return send_file('service-worker.js', mimetype='application/javascript')
    except:
        return '', 404

@app.route('/api/data')
def api_data():
    global DATA_CACHE
    now = time.time()

    # Check cache
    if (DATA_CACHE['chart_data'] and DATA_CACHE['last_update'] and
        now - DATA_CACHE['last_update'] < DATA_CACHE['cache_duration']):
        print("[INFO] Serving from cache")
        response = jsonify(DATA_CACHE['chart_data'])
        response.headers['Cache-Control'] = 'public, max-age=180'
        return response

    print("[INFO] Fetching fresh data")
    data = prepare_all_data()
    if data is None:
        return jsonify({'error': 'Failed to fetch data'}), 500

    # Get latest row for signal calculation
    latest_row = data['daily'].iloc[-1]

    # Calculate current signals
    signals = get_signals(latest_row, CONFIG)

    # Generate historical buy/sell signals
    buy_signals = {'dates': [], 'prices': []}
    sell_signals = {'dates': [], 'prices': []}

    # Calculate signals for all historical data
    for idx, row in data['daily'].iterrows():
        if pd.isna(row['sma_ratio']) or pd.isna(row['ma50w_ratio']) or pd.isna(row['fg_ema']) or pd.isna(row['rsi']):
            continue

        hist_signals = get_signals(row, CONFIG)
        if hist_signals:
            if hist_signals['action_class'] == 'buy':
                buy_signals['dates'].append(row['time'].strftime('%Y-%m-%d'))
                buy_signals['prices'].append(float(row['close']))
            elif hist_signals['action_class'] == 'sell':
                sell_signals['dates'].append(row['time'].strftime('%Y-%m-%d'))
                sell_signals['prices'].append(float(row['close']))

    # Prepare daily data for charts
    daily_data = []
    for idx, row in data['daily'].iterrows():
        daily_data.append({
            'time': row['time'].strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
            'sma_200w': float(row['sma_200w']) if not pd.isna(row['sma_200w']) else None,
            'ma_50w': float(row['ma_50w']) if not pd.isna(row['ma_50w']) else None,
            'fg_index': int(row['fg_index']) if not pd.isna(row['fg_index']) else None,
            'fg_ema': float(row['fg_ema']) if not pd.isna(row['fg_ema']) else None,
            'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else None
        })

    # Extract current values for compatibility with old frontend structure
    current_sma = float(latest_row['sma_200w']) if not pd.isna(latest_row['sma_200w']) else 0
    current_ma_50w = float(latest_row['ma_50w']) if not pd.isna(latest_row['ma_50w']) else 0
    current_fg = int(latest_row['fg_index']) if not pd.isna(latest_row['fg_index']) else 0
    current_fg_ema = float(latest_row['fg_ema']) if not pd.isna(latest_row['fg_ema']) else 0
    current_rsi = float(latest_row['rsi']) if not pd.isna(latest_row['rsi']) else 0
    current_ratio = float(latest_row['sma_ratio']) if not pd.isna(latest_row['sma_ratio']) else 0
    current_ma50w_ratio = float(latest_row['ma50w_ratio']) if not pd.isna(latest_row['ma50w_ratio']) else 0
    current_regime = latest_row['regime'] if 'regime' in latest_row else 'bull'

    # Create signal objects in the old format
    if signals:
        overall_signal = {
            'value': 2 if signals['action_class'] == 'buy' else (0 if signals['action_class'] == 'sell' else 1),
            'text': signals['action']
        }
        sig_200w = {
            'value': signals['200w']['signal'] + 2,  # Convert -1,0,1 to 1,2,3
            'text': signals['200w']['signal_text']
        }
        sig_50w = {
            'value': signals['50w']['signal'] + 2,
            'text': signals['50w']['signal_text']
        }
        sig_fg = {
            'value': signals['fg']['signal'] + 2,
            'text': signals['fg']['signal_text']
        }
        sig_rsi = {
            'value': signals['rsi']['signal'] + 2,
            'text': signals['rsi']['signal_text']
        }
    else:
        overall_signal = {'value': 1, 'text': 'HOLD'}
        sig_200w = {'value': 1, 'text': 'NEUTRAL'}
        sig_50w = {'value': 1, 'text': 'NEUTRAL'}
        sig_fg = {'value': 1, 'text': 'NEUTRAL'}
        sig_rsi = {'value': 1, 'text': 'NEUTRAL'}

    # Prepare response with both old and new formats
    response = {
        # Old format fields for compatibility
        'current_price': float(data['current_price']) if data['current_price'] else 0,
        'current_sma': current_sma,
        'current_ma_50w': current_ma_50w,
        'current_fg': current_fg,
        'current_fg_ema': current_fg_ema,
        'current_rsi': current_rsi,
        'current_ratio': current_ratio,
        'current_ma50w_ratio': current_ma50w_ratio,
        'current_regime': current_regime,
        'timestamp': latest_row['time'].strftime('%Y-%m-%d %H:%M UTC'),
        'current_overall_signal': overall_signal,
        'current_200w_signal': sig_200w,
        'current_50w_signal': sig_50w,
        'current_fg_signal': sig_fg,
        'current_rsi_signal': sig_rsi,
        'daily': daily_data,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'fgsma_params': CONFIG,  # Keep name for compatibility

        # New format for detailed signal info
        'signals': signals
    }

    DATA_CACHE['chart_data'] = response
    DATA_CACHE['last_update'] = now

    json_response = jsonify(response)
    json_response.headers['Cache-Control'] = 'public, max-age=180'
    return json_response

# ============================================================================
# FRONTEND
# ============================================================================

@app.route('/')
def index():
    print("[INFO] Main dashboard route / called")
    html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <title>Bitcoin 4-Indicator Strategy</title>

    <!-- PWA Meta Tags -->
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#000000">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="BTC 4-Ind">
    <link rel="apple-touch-icon" href="/static/icon-192.png">
    <meta name="description" content="4-Indicator trading signals for Bitcoin: 200W SMA, 50W MA, F&G, RSI">

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
        .time-range-selector {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 12px;
            justify-content: center;
        }
        .time-range-btn {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e5e7eb;
            background: #ffffff;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 700;
            color: #6b7280;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: uppercase;
        }
        .time-range-btn:hover {
            background: #f3f4f6;
            border-color: #10b981;
        }
        .time-range-btn.active {
            background: #10b981;
            border-color: #10b981;
            color: #ffffff;
        }

        .visibility-toggles {
            display: flex;
            gap: 16px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 12px;
            margin-bottom: 16px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .visibility-toggles label {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            font-weight: 600;
            color: #374151;
            cursor: pointer;
        }
        .visibility-toggles input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
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
        .data-value.orange { color: #000000; }

        /* Price flash animations */
        @keyframes flash-green-anim {
            0%, 100% { color: #000; }
            50% { color: #10b981; font-weight: 700; }
        }
        @keyframes flash-red-anim {
            0%, 100% { color: #000; }
            50% { color: #ef4444; font-weight: 700; }
        }
        .data-value.flash-green {
            animation: flash-green-anim 600ms ease-in-out;
        }
        .data-value.flash-red {
            animation: flash-red-anim 600ms ease-in-out;
        }

        .signal-card {
            border: none;
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
        }
        .signal-card.buy {
            background: #d1fae5;
            color: #065f46;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
        }
        .signal-card.sell {
            background: #fecdd3;
            color: #881337;
            box-shadow: 0 10px 30px rgba(225, 29, 72, 0.2);
        }
        .signal-card.hold {
            background: #e5e7eb;
            color: #374151;
            box-shadow: 0 10px 30px rgba(107, 114, 128, 0.2);
        }
        .signal-date {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            opacity: 0.9;
        }
        .signal-main {
            text-align: center;
            margin: 20px 0;
        }
        .signal-text {
            font-size: 48px;
            font-weight: 900;
            margin-bottom: 8px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .signal-score {
            font-size: 16px;
            font-weight: 600;
            opacity: 0.8;
        }
        .subsignals {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 16px;
        }
        .subsignal {
            background: rgba(255,255,255,0.3);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .subsignal-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.8;
            margin-bottom: 6px;
        }
        .subsignal-value {
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
        .faq-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 40px 32px;
            margin-bottom: 32px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        .faq-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: heroGlow 10s ease-in-out infinite;
        }
        @keyframes heroGlow {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(-20px, -20px) scale(1.1); }
        }
        .faq-hero-title {
            font-size: 32px;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 16px 0;
            text-align: center;
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .faq-hero-text {
            font-size: 16px;
            line-height: 1.8;
            color: rgba(255,255,255,0.95);
            margin: 0;
            text-align: center;
            position: relative;
            z-index: 1;
            max-width: 900px;
            margin: 0 auto;
        }
        .faq-hero-text strong {
            color: #ffffff;
            font-weight: 700;
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
        .faq-section-content {
            color: #374151;
            line-height: 1.6;
        }
        .faq-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 12px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .faq-table th {
            text-align: left;
            padding: 14px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            font-weight: 600;
            color: #ffffff;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .faq-table th:first-child {
            border-top-left-radius: 8px;
        }
        .faq-table th:last-child {
            border-top-right-radius: 8px;
        }
        .faq-table td {
            padding: 12px 16px;
            border: 1px solid #e5e7eb;
            border-left: none;
            border-right: none;
            color: #374151;
            background: #ffffff;
            transition: background 0.2s ease;
        }
        .faq-table tr:hover td {
            background: #f8f9fa;
        }
        .faq-table tbody tr:last-child td {
            border-bottom: none;
        }
        .faq-table tbody tr:last-child td:first-child {
            border-bottom-left-radius: 8px;
        }
        .faq-table tbody tr:last-child td:last-child {
            border-bottom-right-radius: 8px;
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

        /* MOBILE RESPONSIVE STYLES */
        @media (max-width: 768px) {
            body {
                overflow-x: hidden;
                overflow-y: auto;
            }

            .container {
                flex-direction: column;
                min-height: 100vh;
                height: auto;
                display: flex;
            }

            .data-panel {
                width: 100%;
                padding: 12px;
                border-left: none;
                border-bottom: 1px solid #e5e7eb;
                flex-shrink: 0;
                order: -1;
            }

            .section-title {
                font-size: 15px;
                margin-bottom: 10px;
                font-weight: 700;
            }

            .signal-card {
                padding: 16px;
                margin-bottom: 12px;
                border-radius: 12px;
            }

            .signal-date {
                font-size: 12px;
                margin-bottom: 8px;
                opacity: 0.8;
            }

            .signal-text {
                font-size: 32px;
                font-weight: 800;
                margin-bottom: 12px;
                text-align: center;
            }

            .subsignals {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }

            .subsignal {
                padding: 12px;
                border-radius: 8px;
                text-align: center;
            }

            .subsignal-label {
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                margin-bottom: 4px;
            }

            .subsignal-value {
                font-size: 18px;
                font-weight: 700;
            }

            .data-card {
                padding: 12px;
                margin-bottom: 8px;
                border-radius: 8px;
                background: #f9fafb;
                display: inline-block;
                width: calc(50% - 6px);
                margin-right: 6px;
                vertical-align: top;
            }

            .data-card:nth-child(2n) {
                margin-right: 0;
            }

            .data-label {
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                opacity: 0.6;
                margin-bottom: 4px;
            }

            .data-value {
                font-size: 20px;
                font-weight: 800;
            }

            .chart-section {
                width: 100%;
                flex: 1;
                display: flex;
                flex-direction: column;
                padding: 8px;
                overflow: hidden;
            }

            .chart-title {
                display: none;
            }

            .chart-tabs {
                display: flex;
                gap: 4px;
                margin-bottom: 6px;
                flex-shrink: 0;
            }

            .chart-tab {
                flex: 1;
                padding: 8px 4px;
                font-size: 12px;
                text-align: center;
                font-weight: 600;
                border-radius: 6px;
            }

            .time-range-selector {
                display: flex;
                gap: 6px;
                padding: 8px;
                margin-bottom: 8px;
                flex-shrink: 0;
            }

            .time-range-btn {
                flex: 1;
                padding: 10px 8px;
                font-size: 13px;
                font-weight: 700;
            }

            #chart, #signal-chart {
                flex: 1 !important;
                height: auto !important;
                min-height: 600px !important;
                max-height: none !important;
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
            <h1 class="chart-title">Bitcoin 4-Indicator Strategy</h1>

            <div class="chart-tabs">
                <button class="chart-tab active" onclick="switchChart('price')">Price Chart</button>
                <button class="chart-tab" onclick="switchChart('signals')">Historical Signals</button>
                <button class="chart-tab" onclick="switchChart('faq')">FAQ</button>
            </div>

            <div class="time-range-selector">
                <button class="time-range-btn" onclick="setTimeRange('1M')">1M</button>
                <button class="time-range-btn" onclick="setTimeRange('3M')">3M</button>
                <button class="time-range-btn" onclick="setTimeRange('1Y')">1Y</button>
                <button class="time-range-btn active" onclick="setTimeRange('ALL')">All Time</button>
            </div>
            <div class="visibility-toggles">
                <label><input type="checkbox" id="vis-price" checked onchange="toggleVisibility('price')"> Price</label>
                <label><input type="checkbox" id="vis-sma" checked onchange="toggleVisibility('sma')"> 200W SMA</label>
                <label><input type="checkbox" id="vis-50w" checked onchange="toggleVisibility('50w')"> 50W MA</label>
                <label><input type="checkbox" id="vis-fg" onchange="toggleVisibility('fg')"> F&G</label>
                <label><input type="checkbox" id="vis-rsi" onchange="toggleVisibility('rsi')"> RSI</label>
                <label><input type="checkbox" id="vis-volume" checked onchange="toggleVisibility('volume')"> Volume</label>
            </div>

            <div class="chart-wrapper" id="chart"></div>
            <div class="chart-wrapper hidden" id="signal-chart"></div>
            <div class="chart-wrapper hidden" id="faq-content"></div>
        </div>

        <div class="data-panel">
            <div>
                <div class="section-title">4-Indicator Signal</div>
                <div class="signal-card" id="signal-card">
                    <div class="signal-date" id="signal-date">--</div>
                    <div class="signal-main">
                        <div class="signal-text" id="signal-text">--</div>
                        <div class="signal-score" id="signal-score">--</div>
                    </div>
                    <div class="subsignals">
                        <div class="subsignal">
                            <div class="subsignal-label">200W SMA</div>
                            <div class="subsignal-value" id="sig-200w">--</div>
                            <div style="font-size: 11px; opacity: 0.8; margin-top: 2px;" id="sig-200w-val">--</div>
                        </div>
                        <div class="subsignal">
                            <div class="subsignal-label">50W MA</div>
                            <div class="subsignal-value" id="sig-50w">--</div>
                            <div style="font-size: 11px; opacity: 0.8; margin-top: 2px;" id="sig-50w-val">--</div>
                        </div>
                        <div class="subsignal">
                            <div class="subsignal-label">F&G EMA</div>
                            <div class="subsignal-value" id="sig-fg">--</div>
                            <div style="font-size: 11px; opacity: 0.8; margin-top: 2px;" id="sig-fg-val">--</div>
                        </div>
                        <div class="subsignal">
                            <div class="subsignal-label">RSI</div>
                            <div class="subsignal-value" id="sig-rsi">--</div>
                            <div style="font-size: 11px; opacity: 0.8; margin-top: 2px;" id="sig-rsi-val">--</div>
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
                    <div class="data-value orange" id="sma-200w">$--,---</div>
                </div>
                <div class="data-card">
                    <div class="data-label">50-Week MA</div>
                    <div class="data-value orange" id="ma-50w">$--,---</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Fear & Greed Index</div>
                    <div class="data-value" id="fg">--</div>
                </div>
                <div class="data-card">
                    <div class="data-label">RSI (25-day)</div>
                    <div class="data-value" id="rsi">--</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let chartData = null;
        let currentLayout = null;
        let currentSignalLayout = null;
        let currentChartView = 'price';
        let currentTimeRange = 'ALL';
        let showPrice = true;
        let showSMA = true;
        let show50W = true;
        let showFG = false;
        let showRSI = false;
        let showVolume = true;
        let previousPrice = null;

        function setTimeRange(range) {
            currentTimeRange = range;
            document.querySelectorAll('.time-range-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');

            if (chartData) {
                if (currentChartView === 'price') {
                    renderChart(chartData);
                } else if (currentChartView === 'signals') {
                    renderSignalChart(chartData);
                }
            }
        }

        function toggleVisibility(element) {
            if (element === 'price') showPrice = document.getElementById('vis-price').checked;
            if (element === 'sma') showSMA = document.getElementById('vis-sma').checked;
            if (element === '50w') show50W = document.getElementById('vis-50w').checked;
            if (element === 'fg') showFG = document.getElementById('vis-fg').checked;
            if (element === 'rsi') showRSI = document.getElementById('vis-rsi').checked;
            if (element === 'volume') showVolume = document.getElementById('vis-volume').checked;

            if (chartData) {
                if (currentChartView === 'price') {
                    renderChart(chartData);
                } else if (currentChartView === 'signals') {
                    renderSignalChart(chartData);
                }
            }
        }

        function getCutoffDate() {
            if (currentTimeRange === 'ALL') return null;
            const now = new Date();
            const cutoff = new Date(now);
            switch(currentTimeRange) {
                case '1M':
                    cutoff.setMonth(cutoff.getMonth() - 1);
                    break;
                case '3M':
                    cutoff.setMonth(cutoff.getMonth() - 3);
                    break;
                case '1Y':
                    cutoff.setFullYear(cutoff.getFullYear() - 1);
                    break;
            }
            return cutoff;
        }

        function switchChart(view) {
            currentChartView = view;
            const priceChart = document.getElementById('chart');
            const signalChart = document.getElementById('signal-chart');
            const faqContent = document.getElementById('faq-content');
            const timeRangeSelector = document.querySelector('.time-range-selector');
            const tabs = document.querySelectorAll('.chart-tab');

            tabs.forEach(tab => tab.classList.remove('active'));

            if (view === 'price') {
                priceChart.classList.remove('hidden');
                signalChart.classList.add('hidden');
                faqContent.classList.add('hidden');
                if (timeRangeSelector) timeRangeSelector.style.display = 'flex';
                tabs[0].classList.add('active');
                if (chartData) {
                    renderChart(chartData);
                    setTimeout(() => Plotly.Plots.resize('chart'), 100);
                }
            } else if (view === 'signals') {
                priceChart.classList.add('hidden');
                signalChart.classList.remove('hidden');
                faqContent.classList.add('hidden');
                if (timeRangeSelector) timeRangeSelector.style.display = 'flex';
                tabs[1].classList.add('active');
                if (chartData) {
                    renderSignalChart(chartData);
                    setTimeout(() => Plotly.Plots.resize('signal-chart'), 100);
                }
            } else if (view === 'faq') {
                priceChart.classList.add('hidden');
                signalChart.classList.add('hidden');
                faqContent.classList.remove('hidden');
                if (timeRangeSelector) timeRangeSelector.style.display = 'none';
                tabs[2].classList.add('active');
                if (chartData) {
                    renderFAQ(chartData);
                }
            }
        }

        async function loadData() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();

                if (data.error) {
                    console.error('API Error:', data.error);
                    return;
                }

                chartData = data;
                updateLiveData(data);
                updateSignalCard(data);

                if (currentChartView === 'price') {
                    renderChart(data);
                } else if (currentChartView === 'signals') {
                    renderSignalChart(data);
                }

                document.getElementById('loading').classList.add('hidden');
            } catch (err) {
                console.error('Error loading data:', err);
                document.getElementById('loading').classList.add('hidden');
            }
        }

        function renderChart(data) {
            try {
                if (!data.daily || data.daily.length === 0) return;

                // Apply time range filter
                const cutoffDate = getCutoffDate();
                let filteredDaily = data.daily;
                if (cutoffDate) {
                    filteredDaily = data.daily.filter(d => new Date(d.time) >= cutoffDate);
                }

                const traces = [];

                if (showPrice) {
                    traces.push({
                        type: 'candlestick',
                        x: filteredDaily.map(d => d.time),
                        open: filteredDaily.map(d => d.open),
                        high: filteredDaily.map(d => d.high),
                        low: filteredDaily.map(d => d.low),
                        close: filteredDaily.map(d => d.close),
                        name: 'BTC/USD',
                        increasing: {line: {color: '#10b981'}},
                        decreasing: {line: {color: '#ef4444'}},
                        yaxis: 'y',
                        showlegend: true
                    });
                }

                if (showVolume) {
                    traces.push({
                        type: 'bar',
                        x: filteredDaily.map(d => d.time),
                        y: filteredDaily.map(d => d.volume),
                        name: 'Volume',
                        marker: {color: filteredDaily.map(d => d.close >= d.open ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)')},
                        yaxis: 'y2',
                        showlegend: true
                    });
                }

                if (showSMA) {
                    const smaData = filteredDaily.filter(d => d.sma_200w !== null);
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

                if (show50W) {
                    const ma50Data = filteredDaily.filter(d => d.ma_50w !== null);
                    traces.push({
                        type: 'scatter',
                        mode: 'lines',
                        x: ma50Data.map(d => d.time),
                        y: ma50Data.map(d => d.ma_50w),
                        name: '50W MA',
                        line: {color: '#3b82f6', width: 2},
                        yaxis: 'y',
                        showlegend: true
                    });
                }

                if (showFG) {
                    const fgData = filteredDaily.filter(d => d.fg_index !== null);
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

                if (showRSI) {
                    const rsiData = filteredDaily.filter(d => d.rsi !== null);
                    traces.push({
                        type: 'scatter',
                        mode: 'lines',
                        x: rsiData.map(d => d.time),
                        y: rsiData.map(d => d.rsi),
                        name: 'RSI',
                        line: {color: '#ec4899', width: 2},
                        yaxis: 'y3',
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
                        itemclick: false,
                        itemdoubleclick: false,
                        orientation: 'h',
                        x: 0,
                        y: 1.08,
                        xanchor: 'left',
                        yanchor: 'top'
                    },
                    hovermode: 'x unified',
                    dragmode: false,
                    xaxis: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        fixedrange: true,
                        autorange: true,
                        rangeslider: {visible: false}
                    },
                    yaxis: {
                        title: {text: 'Price (USD)', font: {size: 12, color: '#6b7280'}},
                        gridcolor: '#e5e7eb',
                        side: 'left',
                        domain: [0.25, 1],
                        fixedrange: true,
                        autorange: true,
                        type: 'linear'
                    },
                    yaxis2: {
                        gridcolor: 'transparent',
                        showticklabels: false,
                        domain: [0, 0.2],
                        fixedrange: true,
                        autorange: true
                    },
                    yaxis3: {
                        title: {text: 'F&G / RSI', font: {size: 12, color: '#6b7280'}},
                        gridcolor: 'transparent',
                        side: 'right',
                        overlaying: 'y',
                        fixedrange: true,
                        autorange: true
                    },
                    margin: {l: 60, r: 80, t: 60, b: 60}
                };

                const config = {
                    responsive: true,
                    displayModeBar: false,
                    displaylogo: false,
                    staticPlot: true
                };

                Plotly.newPlot('chart', traces, layout, config);
                currentLayout = layout;
            } catch (err) {
                console.error('Error rendering chart:', err);
            }
        }

        function renderSignalChart(data) {
            try {
                if (!data.daily || data.daily.length === 0) return;

                // Apply time range filter
                const cutoffDate = getCutoffDate();
                let filteredDaily = data.daily;
                let filteredBuys = data.buy_signals;
                let filteredSells = data.sell_signals;

                if (cutoffDate) {
                    filteredDaily = data.daily.filter(d => new Date(d.time) >= cutoffDate);

                    const buyDates = [];
                    const buyPrices = [];
                    for (let i = 0; i < data.buy_signals.dates.length; i++) {
                        if (new Date(data.buy_signals.dates[i]) >= cutoffDate) {
                            buyDates.push(data.buy_signals.dates[i]);
                            buyPrices.push(data.buy_signals.prices[i]);
                        }
                    }
                    filteredBuys = {dates: buyDates, prices: buyPrices};

                    const sellDates = [];
                    const sellPrices = [];
                    for (let i = 0; i < data.sell_signals.dates.length; i++) {
                        if (new Date(data.sell_signals.dates[i]) >= cutoffDate) {
                            sellDates.push(data.sell_signals.dates[i]);
                            sellPrices.push(data.sell_signals.prices[i]);
                        }
                    }
                    filteredSells = {dates: sellDates, prices: sellPrices};
                }

                const traces = [];

                // BTC price line
                traces.push({
                    type: 'scatter',
                    mode: 'lines',
                    x: filteredDaily.map(d => d.time),
                    y: filteredDaily.map(d => d.close),
                    name: 'BTC Price',
                    line: {color: '#1f2937', width: 1.5},
                    showlegend: true
                });

                // Buy signals
                if (filteredBuys.dates.length > 0) {
                    traces.push({
                        type: 'scatter',
                        mode: 'markers',
                        x: filteredBuys.dates,
                        y: filteredBuys.prices,
                        name: `Buy (${filteredBuys.dates.length})`,
                        marker: {color: '#10b981', size: 8, symbol: 'triangle-up', opacity: 0.7},
                        showlegend: true
                    });
                }

                // Sell signals
                if (filteredSells.dates.length > 0) {
                    traces.push({
                        type: 'scatter',
                        mode: 'markers',
                        x: filteredSells.dates,
                        y: filteredSells.prices,
                        name: `Sell (${filteredSells.dates.length})`,
                        marker: {color: '#ef4444', size: 8, symbol: 'triangle-down', opacity: 0.7},
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
                    dragmode: false,
                    xaxis: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        fixedrange: true,
                        autorange: true
                    },
                    yaxis: {
                        title: {text: 'Price (USD)', font: {size: 12, color: '#6b7280'}},
                        gridcolor: '#e5e7eb',
                        fixedrange: true,
                        autorange: true,
                        type: 'linear'
                    },
                    margin: {l: 60, r: 40, t: 60, b: 60}
                };

                const config = {
                    responsive: true,
                    displayModeBar: false,
                    displaylogo: false,
                    staticPlot: true
                };

                Plotly.newPlot('signal-chart', traces, layout, config);
                currentSignalLayout = layout;
            } catch (err) {
                console.error('Error rendering signal chart:', err);
            }
        }

        function renderFAQ(data) {
            const faqDiv = document.getElementById('faq-content');

            if (!data.signals || !data.fgsma_params) {
                faqDiv.innerHTML = '<div class="faq-content"><p>Loading...</p></div>';
                return;
            }

            const signals = data.signals;
            const config = data.fgsma_params;
            const p = config.parameters;

            faqDiv.innerHTML = `
                <div class="faq-content">
                    <div class="faq-hero">
                        <h2 class="faq-hero-title">How the 4-Indicator Strategy Works</h2>
                        <p class="faq-hero-text">
                            This optimized strategy combines <strong>200-week SMA</strong> (with decay),
                            <strong>50-week MA</strong> (regime-based), <strong>Fear & Greed EMA</strong>,
                            and <strong>RSI</strong> to generate high-conviction signals. Each indicator provides
                            a +1 (BUY), 0 (NEUTRAL), or -1 (SELL) signal, weighted by their optimized importance.
                            Combined score ≥2 = BUY, ≤-2 = SELL, otherwise HOLD.
                        </p>
                    </div>

                    <div class="faq-section">
                        <div class="faq-section-title">📊 Indicator Thresholds</div>
                        <div class="faq-section-content">
                            <table class="faq-table">
                                <thead>
                                    <tr>
                                        <th>Indicator</th>
                                        <th>Buy Threshold</th>
                                        <th>Sell Threshold</th>
                                        <th>Weight</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><strong>200W SMA Ratio</strong></td>
                                        <td>&lt; ${signals['200w'].buy_threshold}</td>
                                        <td>&gt; ${signals['200w'].sell_threshold}</td>
                                        <td>${signals['200w'].weight}x</td>
                                    </tr>
                                    <tr>
                                        <td><strong>50W MA (Bull)</strong></td>
                                        <td>&lt; ${p.ma_50w.bull_supp.toFixed(2)}</td>
                                        <td>&gt; ${p.ma_50w.bull_ext.toFixed(2)}</td>
                                        <td>${signals['50w'].weight}x</td>
                                    </tr>
                                    <tr>
                                        <td><strong>50W MA (Bear)</strong></td>
                                        <td>&lt; ${p.ma_50w.bear_deep.toFixed(2)}</td>
                                        <td>&gt; ${p.ma_50w.bear_res.toFixed(2)}</td>
                                        <td>${signals['50w'].weight}x</td>
                                    </tr>
                                    <tr>
                                        <td><strong>F&G EMA (${p.fg.ema_period}d)</strong></td>
                                        <td>&lt; ${signals.fg.buy_threshold}</td>
                                        <td>&gt; ${signals.fg.sell_threshold}</td>
                                        <td>${signals.fg.weight}x</td>
                                    </tr>
                                    <tr>
                                        <td><strong>RSI (${p.rsi.period}d)</strong></td>
                                        <td>&lt; ${signals.rsi.buy_threshold}</td>
                                        <td>&gt; ${signals.rsi.sell_threshold}</td>
                                        <td>${signals.rsi.weight}x</td>
                                    </tr>
                                </tbody>
                            </table>

                            <div style="margin-top: 20px;">
                                <p><strong>Note:</strong> 200W SMA thresholds decay exponentially at ${(p.sma_200w.decay * 100).toFixed(1)}% per year to account for Bitcoin's maturing market.</p>
                                <p><strong>Performance:</strong> ${config.total_return.toFixed(0)}% total return in backtesting.</p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        function updateLiveData(data) {
            const priceElement = document.getElementById('price');
            const newPrice = data.current_price;

            const formattedPrice = '$' + newPrice.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
            priceElement.textContent = formattedPrice;

            // Flash animation
            if (previousPrice !== null && previousPrice !== newPrice) {
                priceElement.classList.remove('flash-green', 'flash-red');
                void priceElement.offsetWidth;

                if (newPrice > previousPrice) {
                    priceElement.classList.add('flash-green');
                } else if (newPrice < previousPrice) {
                    priceElement.classList.add('flash-red');
                }

                setTimeout(() => {
                    priceElement.classList.remove('flash-green', 'flash-red');
                }, 600);
            }
            previousPrice = newPrice;

            // Update other data cards with actual values from API
            if (data.current_sma) {
                document.getElementById('sma-200w').textContent = '$' + data.current_sma.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
            }

            if (data.current_ma_50w) {
                document.getElementById('ma-50w').textContent = '$' + data.current_ma_50w.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
            }

            if (data.current_fg_ema) {
                document.getElementById('fg').textContent = Math.round(data.current_fg_ema);
            }

            if (data.current_rsi) {
                document.getElementById('rsi').textContent = Math.round(data.current_rsi);
            }
        }

        function updateSignalCard(data) {
            const dateEl = document.getElementById('signal-date');
            const textEl = document.getElementById('signal-text');
            const scoreEl = document.getElementById('signal-score');
            const cardEl = document.getElementById('signal-card');

            const now = new Date();
            const dateStr = now.toLocaleDateString('en-US', {year: 'numeric', month: 'long', day: 'numeric'});
            dateEl.textContent = dateStr;

            if (data.current_overall_signal && data.signals) {
                // Main signal display
                textEl.textContent = data.current_overall_signal.text;
                scoreEl.textContent = `Combined Signal: ${data.signals.combined >= 0 ? '+' : ''}${data.signals.combined}`;
                cardEl.className = 'signal-card ' + data.signals.action_class;

                // Update subsignals with proper formatting
                const s = data.signals;

                // 200W SMA Signal
                document.getElementById('sig-200w').textContent = data.current_200w_signal.text;
                document.getElementById('sig-200w-val').textContent = `${s['200w'].ratio} (${s['200w'].weighted >= 0 ? '+' : ''}${s['200w'].weighted})`;

                // 50W MA Signal
                document.getElementById('sig-50w').textContent = data.current_50w_signal.text;
                document.getElementById('sig-50w-val').textContent = `${s['50w'].ratio} ${s['50w'].regime} (${s['50w'].weighted >= 0 ? '+' : ''}${s['50w'].weighted})`;

                // F&G Signal
                document.getElementById('sig-fg').textContent = data.current_fg_signal.text;
                document.getElementById('sig-fg-val').textContent = `${s.fg.value} (${s.fg.weighted >= 0 ? '+' : ''}${s.fg.weighted})`;

                // RSI Signal
                document.getElementById('sig-rsi').textContent = data.current_rsi_signal.text;
                document.getElementById('sig-rsi-val').textContent = `${s.rsi.value} (${s.rsi.weighted >= 0 ? '+' : ''}${s.rsi.weighted})`;
            } else {
                textEl.textContent = 'HOLD';
                scoreEl.textContent = 'Loading...';
                cardEl.className = 'signal-card hold';
            }
        }

        loadData();
        setInterval(loadData, 300000); // Refresh every 5 minutes

        // Handle window resize
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
    </script>
</body>
</html>
    '''
    return render_template_string(html)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("   Bitcoin 4-Indicator Optimized Dashboard")
    print("="*70)

    load_config()

    if CONFIG is None:
        print("\n[ERROR] Failed to load configuration. Please check final_5stage_config.json exists.")
        sys.exit(1)

    print("\n[INFO] Starting Flask server...")
    port = int(os.environ.get('PORT', 5000))

    # Open browser after a short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f'http://127.0.0.1:{port}')

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n[OK] Dashboard running at: http://127.0.0.1:{port}")
    print("[INFO] Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=port, debug=False)
