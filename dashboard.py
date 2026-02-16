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
from flask_compress import Compress
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
Compress(app)  # Enable gzip compression

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

    # Prepare response
    response = {
        'current_price': float(data['current_price']) if data['current_price'] else 0,
        'timestamp': latest_row['time'].strftime('%Y-%m-%d %H:%M UTC'),
        'signals': signals,
        'config': CONFIG
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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#1a1a2e">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="BTC Strategy">
    <link rel="manifest" href="/static/manifest.json">
    <title>BTC 4-Indicator Strategy</title>

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            padding: 1rem;
            min-height: 100vh;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .price {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }

        .timestamp {
            color: #888;
            font-size: 0.9rem;
        }

        .loading {
            text-align: center;
            padding: 2rem;
            color: #888;
            font-size: 1.2rem;
        }

        .error {
            background: rgba(239, 68, 68, 0.1);
            border: 2px solid rgba(239, 68, 68, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            color: #ef4444;
        }

        .action-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            text-align: center;
            border: 2px solid rgba(255, 255, 255, 0.1);
        }

        .action {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .action.buy { color: #10b981; }
        .action.sell { color: #ef4444; }
        .action.hold { color: #f59e0b; }

        .combined-score {
            font-size: 1.5rem;
            color: #888;
        }

        .indicators {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .indicator {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem 1rem;
            border: 2px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }

        .indicator:hover {
            transform: translateY(-5px);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .indicator-name {
            font-size: 0.85rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }

        .indicator-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .signal-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .signal-buy {
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
        }

        .signal-sell {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }

        .signal-neutral {
            background: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
        }

        .weighted-signal {
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: #888;
        }

        .config {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .config-title {
            font-size: 1rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1rem;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .config-item {
            font-size: 0.85rem;
        }

        .config-label {
            color: #888;
            margin-bottom: 0.25rem;
        }

        .config-value {
            color: #fff;
            font-weight: 600;
        }

        .footer {
            text-align: center;
            color: #666;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding: 1rem;
        }

        @media (max-width: 600px) {
            .price {
                font-size: 2.5rem;
            }

            .action {
                font-size: 2rem;
            }

            .indicators {
                grid-template-columns: repeat(2, 1fr);
            }

            .action-card {
                padding: 1.5rem;
            }

            .config-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="price" id="price">--</div>
            <div class="timestamp" id="timestamp">Loading...</div>
        </div>

        <div id="content">
            <div class="loading">Loading data...</div>
        </div>
    </div>

    <script>
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();

                if (data.error) {
                    document.getElementById('content').innerHTML = `
                        <div class="error">
                            Failed to load data: ${data.error}
                        </div>
                    `;
                    return;
                }

                // Update price and timestamp
                document.getElementById('price').textContent = `$${data.current_price.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0})}`;
                document.getElementById('timestamp').textContent = `Last updated: ${data.timestamp}`;

                const signals = data.signals;
                const config = data.config;

                // Build indicators HTML
                let indicatorsHTML = '';

                // 200W SMA
                indicatorsHTML += `
                    <div class="indicator">
                        <div class="indicator-name">200W SMA</div>
                        <div class="indicator-value">${signals['200w'].ratio}</div>
                        <span class="signal-badge signal-${signals['200w'].signal_text === 'BUY' ? 'buy' : (signals['200w'].signal_text === 'SELL' ? 'sell' : 'neutral')}">
                            ${signals['200w'].signal_text}
                        </span>
                        <div class="weighted-signal">
                            Weight: ${signals['200w'].weight}x → ${signals['200w'].weighted >= 0 ? '+' : ''}${signals['200w'].weighted}
                        </div>
                    </div>
                `;

                // 50W MA
                indicatorsHTML += `
                    <div class="indicator">
                        <div class="indicator-name">50W MA (${signals['50w'].regime})</div>
                        <div class="indicator-value">${signals['50w'].ratio}</div>
                        <span class="signal-badge signal-${signals['50w'].signal_text === 'BUY' ? 'buy' : (signals['50w'].signal_text === 'SELL' ? 'sell' : 'neutral')}">
                            ${signals['50w'].signal_text}
                        </span>
                        <div class="weighted-signal">
                            Weight: ${signals['50w'].weight}x → ${signals['50w'].weighted >= 0 ? '+' : ''}${signals['50w'].weighted}
                        </div>
                    </div>
                `;

                // F&G
                indicatorsHTML += `
                    <div class="indicator">
                        <div class="indicator-name">Fear & Greed</div>
                        <div class="indicator-value">${signals.fg.value}</div>
                        <span class="signal-badge signal-${signals.fg.signal_text === 'BUY' ? 'buy' : (signals.fg.signal_text === 'SELL' ? 'sell' : 'neutral')}">
                            ${signals.fg.signal_text}
                        </span>
                        <div class="weighted-signal">
                            Weight: ${signals.fg.weight}x → ${signals.fg.weighted >= 0 ? '+' : ''}${signals.fg.weighted}
                        </div>
                    </div>
                `;

                // RSI
                indicatorsHTML += `
                    <div class="indicator">
                        <div class="indicator-name">RSI</div>
                        <div class="indicator-value">${signals.rsi.value}</div>
                        <span class="signal-badge signal-${signals.rsi.signal_text === 'BUY' ? 'buy' : (signals.rsi.signal_text === 'SELL' ? 'sell' : 'neutral')}">
                            ${signals.rsi.signal_text}
                        </span>
                        <div class="weighted-signal">
                            Weight: ${signals.rsi.weight}x → ${signals.rsi.weighted >= 0 ? '+' : ''}${signals.rsi.weighted}
                        </div>
                    </div>
                `;

                // Build config HTML
                const p = config.parameters;
                const configHTML = `
                    <div class="config">
                        <div class="config-title">Strategy Configuration</div>
                        <div class="config-grid">
                            <div class="config-item">
                                <div class="config-label">200W Thresholds</div>
                                <div class="config-value">Buy: ${signals['200w'].buy_threshold} | Sell: ${signals['200w'].sell_threshold}</div>
                            </div>
                            <div class="config-item">
                                <div class="config-label">50W Bull</div>
                                <div class="config-value">Ext: ${p.ma_50w.bull_ext.toFixed(2)} | Supp: ${p.ma_50w.bull_supp.toFixed(2)}</div>
                            </div>
                            <div class="config-item">
                                <div class="config-label">50W Bear</div>
                                <div class="config-value">Res: ${p.ma_50w.bear_res.toFixed(2)} | Deep: ${p.ma_50w.bear_deep.toFixed(2)}</div>
                            </div>
                            <div class="config-item">
                                <div class="config-label">F&G EMA</div>
                                <div class="config-value">${p.fg.ema_period}d | Buy: <${signals.fg.buy_threshold} | Sell: >${signals.fg.sell_threshold}</div>
                            </div>
                            <div class="config-item">
                                <div class="config-label">RSI</div>
                                <div class="config-value">${p.rsi.period}d | Buy: <${signals.rsi.buy_threshold} | Sell: >${signals.rsi.sell_threshold}</div>
                            </div>
                            <div class="config-item">
                                <div class="config-label">Performance</div>
                                <div class="config-value">${config.total_return.toFixed(0)}% Return</div>
                            </div>
                        </div>
                    </div>
                `;

                // Build main content
                document.getElementById('content').innerHTML = `
                    <div class="action-card">
                        <div class="action ${signals.action_class}">
                            ${signals.action}
                        </div>
                        <div class="combined-score">Combined Signal: ${signals.combined >= 0 ? '+' : ''}${signals.combined}</div>
                    </div>

                    <div class="indicators">
                        ${indicatorsHTML}
                    </div>

                    ${configHTML}

                    <div class="footer">
                        5-Stage Optimized Strategy | Auto-refresh every 5 minutes
                    </div>
                `;

            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('content').innerHTML = `
                    <div class="error">
                        Error loading data. Please try again.
                    </div>
                `;
            }
        }

        // Load data on page load
        loadData();

        // Auto-refresh every 5 minutes
        setInterval(loadData, 300000);

        // Register service worker for PWA
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/service-worker.js').catch(err => {
                console.log('Service worker registration failed:', err);
            });
        }
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
