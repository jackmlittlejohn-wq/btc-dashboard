#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin Pocojuan Model Dashboard
Professional, minimalistic trading signals dashboard
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
        json_path = os.path.join(script_dir, 'tuner_defaults_config.json')

        print(f"[INFO] Looking for config at: {json_path}")

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                CONFIG = json.load(f)
            print("[INFO] ✓ Loaded Constrained Pocojuan Model configuration")
            print(f"[INFO] Total Return: {CONFIG.get('total_return', 0):.2f}%")
            print(f"[INFO] Balanced Weights: 200W={CONFIG['weights']['w_200w']:.2f}, 50W={CONFIG['weights']['w_50w']:.2f}, F&G={CONFIG['weights']['w_fg']:.2f}, RSI={CONFIG['weights']['w_rsi']:.2f}")
        else:
            print(f"[ERROR] tuner_defaults_config.json not found at {json_path}")
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
    """Calculate 50-week MA with regime detection using DAILY consecutive closes"""
    try:
        # Calculate 50-week MA (350 days) on daily data
        df_daily = df.set_index('time').copy()
        ma_50w_daily = df_daily['close'].rolling(window=350, min_periods=1).mean()

        # Detect regime using 3 consecutive DAILY closes
        regimes = []
        for i in range(len(df_daily)):
            if i < 2:
                regimes.append('bull')
                continue

            last_3_closes = df_daily['close'].iloc[max(0, i-2):i+1]
            last_3_ma = ma_50w_daily.iloc[max(0, i-2):i+1]
            above_count = (last_3_closes > last_3_ma).sum()

            if above_count == 3:
                regimes.append('bull')
            elif above_count == 0:
                regimes.append('bear')
            else:
                regimes.append(regimes[-1] if regimes else 'bull')

        df_daily['regime'] = regimes
        df_daily['ma_50w'] = ma_50w_daily
        df_daily['ma50w_ratio'] = df_daily['close'] / ma_50w_daily

        return df_daily[['ma_50w', 'regime', 'ma50w_ratio']]
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

        # 1. 200W SMA Signal (with dual decay support)
        decay_buy = p['sma_200w'].get('decay_buy', p['sma_200w'].get('decay', 1.0)) ** years
        decay_sell = p['sma_200w'].get('decay_sell', p['sma_200w'].get('decay', 1.0)) ** years
        adj_buy = p['sma_200w']['buy'] * decay_buy
        adj_sell = p['sma_200w']['sell'] * decay_sell
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
            action = 'BUY'
            action_class = 'buy'
        elif combined <= -2:
            action = 'SELL'
            action_class = 'sell'
        else:
            action = 'HOLD'
            action_class = 'hold'

        return {
            '200w': {
                'signal': sig_200w,
                'signal_text': 'BUY' if sig_200w == 1 else ('SELL' if sig_200w == -1 else 'HOLD'),
                'weighted': round(sig_200w * w['w_200w'], 2),
                'weight': round(w['w_200w'], 2),
                'ratio': round(latest_row['sma_ratio'], 3),
                'buy_threshold': round(adj_buy, 2),
                'sell_threshold': round(adj_sell, 2)
            },
            '50w': {
                'signal': sig_50w,
                'signal_text': 'BUY' if sig_50w == 1 else ('SELL' if sig_50w == -1 else 'HOLD'),
                'weighted': round(sig_50w * w['w_50w'], 2),
                'weight': round(w['w_50w'], 2),
                'ratio': round(latest_row['ma50w_ratio'], 3),
                'regime': latest_row['regime'].upper()
            },
            'fg': {
                'signal': sig_fg,
                'signal_text': 'BUY' if sig_fg == 1 else ('SELL' if sig_fg == -1 else 'HOLD'),
                'weighted': round(sig_fg * w['w_fg'], 2),
                'weight': round(w['w_fg'], 2),
                'value': round(latest_row['fg_ema'], 1),
                'buy_threshold': round(p['fg']['buy'], 1),
                'sell_threshold': round(p['fg']['sell'], 1)
            },
            'rsi': {
                'signal': sig_rsi,
                'signal_text': 'BUY' if sig_rsi == 1 else ('SELL' if sig_rsi == -1 else 'HOLD'),
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

    # Calculate RSI
    if CONFIG is not None:
        rsi_period = CONFIG['parameters']['rsi']['period']
        daily_df['rsi'] = calculate_rsi(daily_df['close'], rsi_period)

    # Calculate 50W MA with regime
    ma_df = calculate_50w_ma_with_regime(daily_df)
    if ma_df is None:
        return None

    # CRITICAL FIX: Backfill F&G data BEFORE calculating EMA
    # F&G API only has data from Feb 2018+, need to fill pre-2018 dates with neutral value (50)
    if CONFIG is not None:
        fg_period = CONFIG['parameters']['fg']['ema_period']

        # Create complete date range matching daily_df
        complete_dates = pd.DataFrame({'time': daily_df['time']})

        # Merge F&G data with complete dates, filling missing with neutral value 50
        fg_df_sorted = fg_df.sort_values('time').copy()
        fg_complete = complete_dates.merge(fg_df_sorted[['time', 'value']], on='time', how='left')
        fg_complete['value'] = fg_complete['value'].fillna(50)

        # NOW calculate EMA on the COMPLETE series (with backfilled values)
        fg_complete['fg_ema'] = fg_complete['value'].ewm(span=fg_period, adjust=False).mean()

        # Use the complete F&G dataframe for merging
        fg_df = fg_complete

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
# PERFORMANCE CALCULATIONS
# ============================================================================

def calculate_performance_stats(data, config):
    """Calculate performance statistics for the model"""
    try:
        trades = []
        position = None

        for idx, row in data['daily'].iterrows():
            if pd.isna(row['sma_ratio']) or pd.isna(row['ma50w_ratio']) or pd.isna(row['fg_ema']) or pd.isna(row['rsi']):
                continue

            signals = get_signals(row, config)
            if not signals:
                continue

            if signals['action_class'] == 'buy' and position is None:
                position = {'entry_price': row['close'], 'entry_date': row['time']}
            elif signals['action_class'] == 'sell' and position is not None:
                trade_return = (row['close'] - position['entry_price']) / position['entry_price']
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': row['time'],
                    'return': trade_return,
                    'win': trade_return > 0
                })
                position = None

        if len(trades) == 0:
            return {'total_trades': 0, 'win_rate': 0, 'total_return': 0}

        win_count = sum(1 for t in trades if t['win'])
        win_rate = (win_count / len(trades)) * 100

        return {
            'total_trades': len(trades),
            'win_rate': round(win_rate, 1),
            'total_return': round(config.get('total_return', 0), 1)
        }
    except Exception as e:
        print(f"[ERROR] Performance calculation: {e}")
        return {'total_trades': 0, 'win_rate': 0, 'total_return': 0}

def calculate_dca_comparison(data, config):
    """Calculate Pocojuan Model vs DCA - apples to apples comparison from Jan 2021"""
    try:
        # Filter data from Jan 2021 onwards
        df = data['daily'].copy()
        df = df[df['time'] >= pd.Timestamp('2021-01-01')].reset_index(drop=True)

        if len(df) == 0:
            print("[ERROR] No data for 2021+")
            return None

        # Calculate DCA strategy: $100 per week
        weekly_investment = 100
        num_weeks = len(df) // 7
        total_dca_deployed = weekly_investment * num_weeks

        # DCA simulation
        dca_btc = 0
        dca_invested = 0

        for idx, row in df.iterrows():
            if idx % 7 == 0 and dca_invested < total_dca_deployed:  # Weekly investment
                dca_btc += weekly_investment / row['close']
                dca_invested += weekly_investment

        dca_final_value = dca_btc * df.iloc[-1]['close']
        dca_return = ((dca_final_value - dca_invested) / dca_invested) * 100

        # Pocojuan Model: Start with SAME total amount as DCA will deploy
        model_cash = float(total_dca_deployed)
        model_btc = 0.0
        trades = 0
        START_YEAR = 2017

        p = config['parameters']
        w = config['weights']

        for idx, row in df.iterrows():
            # Skip rows with missing data
            if pd.isna(row.get('sma_ratio')) or pd.isna(row.get('ma50w_ratio')) or pd.isna(row.get('fg_ema')) or pd.isna(row.get('rsi')):
                continue

            years = row['time'].year - START_YEAR

            # Calculate signals directly (same as get_signals)
            # 200W SMA
            decay_buy = p['sma_200w'].get('decay_buy', p['sma_200w'].get('decay', 1.0)) ** years
            decay_sell = p['sma_200w'].get('decay_sell', p['sma_200w'].get('decay', 1.0)) ** years
            adj_buy = p['sma_200w']['buy'] * decay_buy
            adj_sell = p['sma_200w']['sell'] * decay_sell
            sig_200w = 1 if row['sma_ratio'] < adj_buy else (-1 if row['sma_ratio'] > adj_sell else 0)

            # 50W MA
            if row['regime'] == 'bull':
                sig_50w = -1 if row['ma50w_ratio'] > p['ma_50w']['bull_ext'] else (1 if row['ma50w_ratio'] < p['ma_50w']['bull_supp'] else 0)
            else:
                sig_50w = -1 if row['ma50w_ratio'] > p['ma_50w']['bear_res'] else (1 if row['ma50w_ratio'] < p['ma_50w']['bear_deep'] else 0)

            # F&G
            sig_fg = 1 if row['fg_ema'] < p['fg']['buy'] else (-1 if row['fg_ema'] > p['fg']['sell'] else 0)

            # RSI
            sig_rsi = 1 if row['rsi'] < p['rsi']['buy'] else (-1 if row['rsi'] > p['rsi']['sell'] else 0)

            # Combined signal
            combined = (sig_200w * w['w_200w'] + sig_50w * w['w_50w'] + sig_fg * w['w_fg'] + sig_rsi * w['w_rsi'])

            # Execute trades (threshold = 2.0 for 4 indicators)
            if combined >= 2.0 and model_cash > 0:
                buy_amount = model_cash * 0.01
                model_btc += buy_amount / row['close']
                model_cash -= buy_amount
                trades += 1
            elif combined <= -2.0 and model_btc > 0:
                btc_to_sell = model_btc * 0.01
                model_cash += btc_to_sell * row['close']
                model_btc -= btc_to_sell
                trades += 1

        model_final_value = model_cash + (model_btc * df.iloc[-1]['close'])
        model_return = ((model_final_value - total_dca_deployed) / total_dca_deployed) * 100
        outperformance = model_return - dca_return

        result = {
            'model_final_value': round(model_final_value, 2),
            'model_return': round(model_return, 2),
            'dca_final_value': round(dca_final_value, 2),
            'dca_return': round(dca_return, 2),
            'outperformance': round(outperformance, 2),
            'trades': trades,
            'starting_amount': round(total_dca_deployed, 2),
            'start_date': df.iloc[0]['time'].strftime('%Y-%m-%d'),
            'end_date': df.iloc[-1]['time'].strftime('%Y-%m-%d')
        }

        print(f"[INFO] DCA comparison calculated: Model ${model_final_value:,.0f} vs DCA ${dca_final_value:,.0f}")
        return result
    except Exception as e:
        print(f"[ERROR] DCA comparison calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

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

@app.route('/api/price')
def api_price():
    """Quick endpoint for live price updates"""
    try:
        price = fetch_current_price()
        if price:
            return jsonify({'price': price, 'timestamp': datetime.now().isoformat()})
        return jsonify({'error': 'Failed to fetch price'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

    # Calculate historical signals for all rows
    historical_signals = []
    signal_start_idx = None
    for idx, row in data['daily'].iterrows():
        if pd.isna(row['sma_ratio']) or pd.isna(row['ma50w_ratio']) or pd.isna(row['fg_ema']) or pd.isna(row['rsi']):
            historical_signals.append(None)
            continue

        hist_sig = get_signals(row, CONFIG)
        historical_signals.append(hist_sig)

        # Track first valid signal for debugging
        if signal_start_idx is None and hist_sig is not None:
            signal_start_idx = idx
            print(f"[INFO] First valid signal at index {idx}, date: {row['time'].strftime('%Y-%m-%d')}")

    # Print signal summary
    valid_signals = sum(1 for sig in historical_signals if sig is not None)
    print(f"[INFO] Generated {valid_signals} valid historical signals out of {len(historical_signals)} total days")

    # Generate buy/sell signals
    buy_signals = {'dates': [], 'prices': []}
    sell_signals = {'dates': [], 'prices': []}

    for idx, (_, row) in enumerate(data['daily'].iterrows()):
        # Only include signals from 2021+ onwards
        if row['time'] >= pd.Timestamp('2021-01-01'):
            if historical_signals[idx] and historical_signals[idx]['action_class'] == 'buy':
                buy_signals['dates'].append(row['time'].strftime('%Y-%m-%d'))
                buy_signals['prices'].append(float(row['close']))
            elif historical_signals[idx] and historical_signals[idx]['action_class'] == 'sell':
                sell_signals['dates'].append(row['time'].strftime('%Y-%m-%d'))
                sell_signals['prices'].append(float(row['close']))

    print(f"[INFO] Buy signals: {len(buy_signals['dates'])}, Sell signals: {len(sell_signals['dates'])}")
    if len(buy_signals['dates']) > 0:
        print(f"[INFO] First buy signal: {buy_signals['dates'][0]}")
    if len(sell_signals['dates']) > 0:
        print(f"[INFO] First sell signal: {sell_signals['dates'][0]}")

    # Prepare daily data for charts (filter to 2021+ only)
    daily_filtered = data['daily'][data['daily']['time'] >= pd.Timestamp('2021-01-01')].reset_index(drop=True)

    daily_data = []
    for idx, row in daily_filtered.iterrows():
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
            'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else None,
            'regime': row['regime'] if 'regime' in row and not pd.isna(row['regime']) else None
        })

    # Prepare historical signal data for visualization (filter to 2021+ only)
    historical_signal_data = []
    for idx, sig in enumerate(historical_signals):
        if sig and data['daily'].iloc[idx]['time'] >= pd.Timestamp('2021-01-01'):
            historical_signal_data.append({
                'time': data['daily'].iloc[idx]['time'].strftime('%Y-%m-%d'),
                'sig_200w': sig['200w']['signal'],
                'sig_50w': sig['50w']['signal'],
                'sig_fg': sig['fg']['signal'],
                'sig_rsi': sig['rsi']['signal'],
                'weighted_200w': sig['200w']['weighted'],
                'weighted_50w': sig['50w']['weighted'],
                'weighted_fg': sig['fg']['weighted'],
                'weighted_rsi': sig['rsi']['weighted'],
                'combined': sig['combined'],
                'regime': sig['50w']['regime']
            })

    # Extract current values
    current_sma = float(latest_row['sma_200w']) if not pd.isna(latest_row['sma_200w']) else 0
    current_ma_50w = float(latest_row['ma_50w']) if not pd.isna(latest_row['ma_50w']) else 0
    current_fg = int(latest_row['fg_index']) if not pd.isna(latest_row['fg_index']) else 0
    current_fg_ema = float(latest_row['fg_ema']) if not pd.isna(latest_row['fg_ema']) else 0
    current_rsi = float(latest_row['rsi']) if not pd.isna(latest_row['rsi']) else 0
    current_ratio = float(latest_row['sma_ratio']) if not pd.isna(latest_row['sma_ratio']) else 0
    current_ma50w_ratio = float(latest_row['ma50w_ratio']) if not pd.isna(latest_row['ma50w_ratio']) else 0
    current_regime = latest_row['regime'] if 'regime' in latest_row else 'bull'

    # Create signal objects
    if signals:
        overall_signal = {
            'value': 2 if signals['action_class'] == 'buy' else (0 if signals['action_class'] == 'sell' else 1),
            'text': signals['action']
        }
        sig_200w = {
            'value': signals['200w']['signal'] + 2,
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
        sig_200w = {'value': 1, 'text': 'HOLD'}
        sig_50w = {'value': 1, 'text': 'HOLD'}
        sig_fg = {'value': 1, 'text': 'HOLD'}
        sig_rsi = {'value': 1, 'text': 'HOLD'}

    # Calculate Portfolio Growth
    dca_comparison = calculate_dca_comparison(data, CONFIG)

    # Prepare response
    response_data = {
        # Current data
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

        # Signals
        'current_overall_signal': overall_signal,
        'current_200w_signal': sig_200w,
        'current_50w_signal': sig_50w,
        'current_fg_signal': sig_fg,
        'current_rsi_signal': sig_rsi,
        'signals': signals,

        # Chart data
        'daily': daily_data,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'historical_signals': historical_signal_data,

        # Config & stats
        'config': CONFIG,
        'dca_comparison': dca_comparison
    }

    DATA_CACHE['chart_data'] = response_data
    DATA_CACHE['last_update'] = now

    json_response = jsonify(response_data)
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
    <title>BTC Pocojuan</title>

    <!-- PWA Meta Tags -->
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#000000">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="BTC Pocojuan">
    <link rel="apple-touch-icon" href="/static/icon-192.png">
    <meta name="description" content="Bitcoin Pocojuan Model - Professional trading signals">

    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: #fafafa;
            color: #1a1a1a;
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .container {
            display: flex;
            height: 100vh;
        }

        .chart-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px 24px;
            background: #ffffff;
            overflow-y: auto;
        }

        .chart-title {
            font-size: 24px;
            font-weight: 600;
            color: #000;
            margin-bottom: 20px;
            letter-spacing: -0.3px;
        }

        .chart-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            border-bottom: 1px solid #e5e7eb;
            position: sticky;
            top: 0;
            background: white;
            z-index: 10;
            padding-bottom: 0;
        }

        .chart-tab {
            padding: 10px 20px;
            background: none;
            border: none;
            font-size: 14px;
            font-weight: 500;
            color: #6b7280;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            transition: all 0.15s ease;
        }

        .chart-tab:hover { color: #000; }
        .chart-tab.active {
            color: #10b981;
            border-bottom-color: #10b981;
            font-weight: 600;
        }

        .time-range-selector {
            display: flex;
            gap: 6px;
            margin-bottom: 12px;
            padding: 6px;
            background: #f9fafb;
            border-radius: 10px;
            justify-content: center;
        }

        .time-range-btn {
            flex: 1;
            padding: 10px 14px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .time-range-btn:hover {
            background: #f3f4f6;
            border-color: #d1d5db;
        }

        .time-range-btn.active {
            background: #10b981;
            border-color: #10b981;
            color: #ffffff;
        }

        .visibility-toggles {
            display: flex;
            gap: 12px;
            padding: 10px;
            background: #f9fafb;
            border-radius: 10px;
            margin-bottom: 12px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .visibility-toggles label {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 13px;
            font-weight: 500;
            color: #374151;
            cursor: pointer;
        }

        .visibility-toggles input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }

        .chart-wrapper {
            border-radius: 12px;
            margin-bottom: 16px;
        }

        .chart-wrapper.hidden { display: none; }

        #chart { height: 600px; }
        #historical-signals-chart { height: 800px; }

        .data-panel {
            width: 340px;
            padding: 20px;
            background: #ffffff;
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow-y: auto;
            border-left: 1px solid #e5e7eb;
        }

        .section-title {
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .data-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 14px;
            transition: all 0.15s ease;
        }

        .data-card:hover {
            background: #f3f4f6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .data-label {
            font-size: 11px;
            color: #6b7280;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.3px;
            margin-bottom: 4px;
        }

        .data-value {
            font-size: 22px;
            font-weight: 700;
            color: #000;
            letter-spacing: -0.5px;
        }

        .data-value.green { color: #059669; }
        .data-value.red { color: #dc2626; }

        @keyframes flash-green {
            0%, 100% { background: #f9fafb; }
            50% { background: #d1fae5; }
        }

        @keyframes flash-red {
            0%, 100% { background: #f9fafb; }
            50% { background: #fee2e2; }
        }

        .data-card.flash-green { animation: flash-green 400ms ease-out; }
        .data-card.flash-red { animation: flash-red 400ms ease-out; }

        .signal-card {
            border-radius: 12px;
            padding: 20px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }

        .signal-card.buy {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            color: #065f46;
        }

        .signal-card.sell {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            color: #991b1b;
        }

        .signal-card.hold {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: #92400e;
        }

        .signal-date {
            font-size: 11px;
            font-weight: 600;
            margin-bottom: 12px;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .signal-main {
            text-align: center;
            margin: 12px 0;
        }

        .signal-text {
            font-size: 40px;
            font-weight: 800;
            margin-bottom: 4px;
            letter-spacing: -1px;
        }

        .signal-score {
            font-size: 13px;
            font-weight: 600;
            opacity: 0.7;
        }

        .subsignals {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 12px;
        }

        .subsignal {
            background: rgba(255,255,255,0.4);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }

        .subsignal-label {
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            opacity: 0.7;
            margin-bottom: 4px;
            font-weight: 600;
        }

        .subsignal-value {
            font-size: 15px;
            font-weight: 700;
        }

        .subsignal-detail {
            font-size: 10px;
            opacity: 0.7;
            margin-top: 2px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
        }

        .stat-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }

        .stat-label {
            font-size: 9px;
            color: #6b7280;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.3px;
            margin-bottom: 4px;
        }

        .stat-value {
            font-size: 18px;
            font-weight: 700;
            color: #000;
        }

        .faq-content {
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }

        .faq-hero {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 12px;
            padding: 32px 28px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.2);
        }

        .faq-hero-title {
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 12px 0;
            letter-spacing: -0.5px;
        }

        .faq-hero-text {
            font-size: 15px;
            line-height: 1.6;
            color: rgba(255,255,255,0.95);
            margin: 0;
        }

        .faq-section {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .faq-section-title {
            font-size: 18px;
            font-weight: 600;
            color: #000;
            margin-bottom: 12px;
        }

        .faq-section-content {
            color: #374151;
            line-height: 1.6;
            font-size: 14px;
        }

        .faq-section-content p {
            margin-bottom: 12px;
        }

        .faq-section-content strong {
            color: #000;
            font-weight: 600;
        }

        .indicator-list {
            margin: 16px 0;
        }

        .indicator-item {
            margin-bottom: 16px;
            padding-left: 16px;
            border-left: 3px solid #10b981;
        }

        .indicator-item h4 {
            font-size: 15px;
            font-weight: 600;
            color: #000;
            margin-bottom: 6px;
        }

        .indicator-item p {
            font-size: 14px;
            color: #6b7280;
            margin: 0;
            line-height: 1.5;
        }

        .comparison-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 16px;
            margin: 16px 0;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 12px;
        }

        .comparison-item {
            text-align: center;
        }

        .comparison-label {
            font-size: 12px;
            color: #6b7280;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .comparison-value {
            font-size: 24px;
            font-weight: 700;
            color: #10b981;
        }

        .loading {
            position: fixed;
            inset: 0;
            background: rgba(255, 255, 255, 0.98);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            flex-direction: column;
            gap: 16px;
        }

        .loading.hidden { display: none; }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #e5e7eb;
            border-top-color: #10b981;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .loading-text {
            color: #6b7280;
            font-size: 13px;
            font-weight: 500;
        }

        .last-update {
            font-size: 11px;
            color: #6b7280;
            text-align: center;
            margin-top: 8px;
        }

        /* MOBILE RESPONSIVE */
        @media (max-width: 768px) {
            body { overflow-y: auto; }

            .container {
                flex-direction: column;
                min-height: 100vh;
                height: auto;
            }

            .data-panel {
                width: 100%;
                padding: 16px;
                border-left: none;
                border-bottom: 1px solid #e5e7eb;
                order: -1;
            }

            .signal-card { padding: 16px; }
            .signal-text { font-size: 32px; }

            .subsignals {
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }

            /* RSI centering fix */
            .subsignals .subsignal:nth-child(5):last-child {
                grid-column: 1 / -1;
                max-width: 50%;
                margin: 0 auto;
            }

            .chart-section {
                width: 100%;
                padding: 12px;
            }

            .chart-title { display: none; }

            #chart { height: 500px !important; }
            #historical-signals-chart { height: 700px !important; }

            .faq-content { padding: 16px; }
        }
    </style>
</head>
<body>
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div class="loading-text">Loading Pocojuan Model...</div>
    </div>

    <div class="container">
        <div class="chart-section">
            <h1 class="chart-title">Bitcoin Pocojuan Model</h1>

            <div class="chart-tabs">
                <button class="chart-tab active" onclick="switchChart('price')">Price Chart</button>
                <button class="chart-tab" onclick="switchChart('signals')">Historical Signals</button>
                <button class="chart-tab" onclick="switchChart('faq')">How It Works</button>
            </div>

            <div class="time-range-selector" id="time-range-selector">
                <button class="time-range-btn" onclick="setTimeRange('1M')">1M</button>
                <button class="time-range-btn" onclick="setTimeRange('3M')">3M</button>
                <button class="time-range-btn" onclick="setTimeRange('1Y')">1Y</button>
                <button class="time-range-btn active" onclick="setTimeRange('ALL')">All</button>
            </div>

            <div class="visibility-toggles" id="visibility-toggles">
                <label><input type="checkbox" id="vis-price" checked onchange="toggleVisibility('price')"> Price</label>
                <label><input type="checkbox" id="vis-sma" checked onchange="toggleVisibility('sma')"> 200W SMA</label>
                <label><input type="checkbox" id="vis-50w" checked onchange="toggleVisibility('50w')"> 50W MA</label>
                <label><input type="checkbox" id="vis-fg" onchange="toggleVisibility('fg')"> F&G</label>
                <label><input type="checkbox" id="vis-rsi" onchange="toggleVisibility('rsi')"> RSI</label>
                <label><input type="checkbox" id="vis-volume" checked onchange="toggleVisibility('volume')"> Volume</label>
            </div>

            <div class="chart-wrapper" id="chart"></div>
            <div class="chart-wrapper hidden" id="historical-signals-chart"></div>
            <div class="chart-wrapper hidden" id="faq-content"></div>
        </div>

        <div class="data-panel">
            <div>
                <div class="section-title">Current Signal</div>
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
                            <div class="subsignal-detail" id="sig-200w-val">--</div>
                        </div>
                        <div class="subsignal">
                            <div class="subsignal-label">50W MA</div>
                            <div class="subsignal-value" id="sig-50w">--</div>
                            <div class="subsignal-detail" id="sig-50w-val">--</div>
                        </div>
                        <div class="subsignal">
                            <div class="subsignal-label">F&G</div>
                            <div class="subsignal-value" id="sig-fg">--</div>
                            <div class="subsignal-detail" id="sig-fg-val">--</div>
                        </div>
                        <div class="subsignal">
                            <div class="subsignal-label">RSI</div>
                            <div class="subsignal-value" id="sig-rsi">--</div>
                            <div class="subsignal-detail" id="sig-rsi-val">--</div>
                        </div>
                    </div>
                </div>
            </div>


            <div>
                <div class="section-title">Live Market Data</div>
                <div class="data-card" id="price-card">
                    <div class="data-label">Bitcoin Price</div>
                    <div class="data-value" id="price">$--,---</div>
                </div>
                <div class="data-card">
                    <div class="data-label">200-Week SMA</div>
                    <div class="data-value" id="sma-200w">$--,---</div>
                </div>
                <div class="data-card">
                    <div class="data-label">50-Week MA</div>
                    <div class="data-value" id="ma-50w">$--,---</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Fear & Greed</div>
                    <div class="data-value" id="fg">--</div>
                </div>
                <div class="data-card">
                    <div class="data-label">RSI (21d)</div>
                    <div class="data-value" id="rsi">--</div>
                </div>
                <div class="last-update" id="last-update">Last update: --</div>
            </div>
        </div>
    </div>

    <script>
        let chartData = null;
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
            document.querySelectorAll('.time-range-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            if (chartData) {
                if (currentChartView === 'price') {
                    renderChart(chartData);
                } else if (currentChartView === 'signals') {
                    renderHistoricalSignals(chartData);
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

            if (chartData && currentChartView === 'price') renderChart(chartData);
        }

        function getCutoffDate() {
            if (currentTimeRange === 'ALL') return null;
            const now = new Date();
            const cutoff = new Date(now);
            switch(currentTimeRange) {
                case '1M': cutoff.setMonth(cutoff.getMonth() - 1); break;
                case '3M': cutoff.setMonth(cutoff.getMonth() - 3); break;
                case '1Y': cutoff.setFullYear(cutoff.getFullYear() - 1); break;
            }
            return cutoff;
        }

        function switchChart(view) {
            currentChartView = view;
            const priceChart = document.getElementById('chart');
            const signalChart = document.getElementById('historical-signals-chart');
            const faqContent = document.getElementById('faq-content');
            const timeRangeSelector = document.getElementById('time-range-selector');
            const visibilityToggles = document.getElementById('visibility-toggles');
            const tabs = document.querySelectorAll('.chart-tab');

            tabs.forEach(tab => tab.classList.remove('active'));

            if (view === 'price') {
                priceChart.classList.remove('hidden');
                signalChart.classList.add('hidden');
                faqContent.classList.add('hidden');
                timeRangeSelector.style.display = 'flex';
                visibilityToggles.style.display = 'flex';
                tabs[0].classList.add('active');
                if (chartData) {
                    renderChart(chartData);
                    setTimeout(() => Plotly.Plots.resize('chart'), 100);
                }
            } else if (view === 'signals') {
                priceChart.classList.add('hidden');
                signalChart.classList.remove('hidden');
                faqContent.classList.add('hidden');
                timeRangeSelector.style.display = 'flex';
                visibilityToggles.style.display = 'none';
                tabs[1].classList.add('active');
                if (chartData) {
                    renderHistoricalSignals(chartData);
                    setTimeout(() => Plotly.Plots.resize('historical-signals-chart'), 100);
                }
            } else if (view === 'faq') {
                priceChart.classList.add('hidden');
                signalChart.classList.add('hidden');
                faqContent.classList.remove('hidden');
                timeRangeSelector.style.display = 'none';
                visibilityToggles.style.display = 'none';
                tabs[2].classList.add('active');
                if (chartData) renderFAQ(chartData);
            }
        }

        async function updateLivePrice() {
            try {
                const res = await fetch('/api/price');
                const data = await res.json();
                if (data.price && !data.error) {
                    const priceCard = document.getElementById('price-card');
                    const priceElement = document.getElementById('price');
                    const newPrice = data.price;
                    const formattedPrice = '$' + newPrice.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});

                    priceElement.textContent = formattedPrice;

                    if (previousPrice !== null && previousPrice !== newPrice) {
                        priceCard.classList.remove('flash-green', 'flash-red');
                        void priceCard.offsetWidth;

                        if (newPrice > previousPrice) {
                            priceCard.classList.add('flash-green');
                        } else if (newPrice < previousPrice) {
                            priceCard.classList.add('flash-red');
                        }

                        setTimeout(() => {
                            priceCard.classList.remove('flash-green', 'flash-red');
                        }, 400);
                    }
                    previousPrice = newPrice;
                }
            } catch (err) {
                console.error('Error updating live price:', err);
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
                updateDataPanel(data);
                updateSignalCard(data);

                if (currentChartView === 'price') renderChart(data);
                else if (currentChartView === 'signals') renderHistoricalSignals(data);
                else if (currentChartView === 'faq') renderFAQ(data);

                document.getElementById('loading').classList.add('hidden');

                const now = new Date();
                document.getElementById('last-update').textContent = `Last update: ${now.toLocaleTimeString()}`;
            } catch (err) {
                console.error('Error loading data:', err);
                document.getElementById('loading').classList.add('hidden');
            }
        }

        function renderChart(data) {
            try {
                if (!data.daily || data.daily.length === 0) return;

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
                        name: 'BTC',
                        increasing: {line: {color: '#10b981', width: 1}},
                        decreasing: {line: {color: '#ef4444', width: 1}},
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
                        marker: {color: filteredDaily.map(d => d.close >= d.open ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)')},
                        yaxis: 'y2',
                        showlegend: false
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
                        line: {color: '#f59e0b', width: 1.5},
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
                        line: {color: '#3b82f6', width: 1.5},
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
                        name: 'F&G',
                        line: {color: '#8b5cf6', width: 1.5},
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
                        line: {color: '#ec4899', width: 1.5},
                        yaxis: 'y3',
                        showlegend: true
                    });
                }

                const layout = {
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#fafafa',
                    font: {color: '#374151', family: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif', size: 11},
                    showlegend: true,
                    legend: {
                        orientation: 'h',
                        x: 0,
                        y: 1.05,
                        xanchor: 'left',
                        yanchor: 'bottom',
                        bgcolor: 'rgba(255,255,255,0.9)',
                        bordercolor: '#e5e7eb',
                        borderwidth: 1
                    },
                    hovermode: 'x unified',
                    dragmode: false,
                    xaxis: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        fixedrange: true,
                        rangeslider: {visible: false}
                    },
                    yaxis: {
                        title: {text: 'Price (USD)', font: {size: 11, color: '#6b7280'}},
                        gridcolor: '#e5e7eb',
                        side: 'left',
                        domain: [0.2, 1],
                        fixedrange: true,
                        type: 'log'
                    },
                    yaxis2: {
                        gridcolor: 'transparent',
                        showticklabels: false,
                        domain: [0, 0.15],
                        fixedrange: true
                    },
                    yaxis3: {
                        title: {text: 'Indicators', font: {size: 11, color: '#6b7280'}},
                        gridcolor: '#f3f4f6',
                        side: 'right',
                        overlaying: 'y',
                        fixedrange: true
                    },
                    margin: {l: 60, r: 60, t: 40, b: 60}
                };

                Plotly.newPlot('chart', traces, layout, {responsive: true, displayModeBar: false, staticPlot: true});
            } catch (err) {
                console.error('Error rendering chart:', err);
            }
        }

        function renderHistoricalSignals(data) {
            try {
                if (!data.historical_signals || data.historical_signals.length === 0) return;

                const cutoffDate = getCutoffDate();
                let filtered = data.historical_signals;
                let filteredDaily = data.daily;
                let filteredBuySignals = {dates: data.buy_signals.dates, prices: data.buy_signals.prices};
                let filteredSellSignals = {dates: data.sell_signals.dates, prices: data.sell_signals.prices};

                if (cutoffDate) {
                    filtered = data.historical_signals.filter(d => new Date(d.time) >= cutoffDate);
                    filteredDaily = data.daily.filter(d => new Date(d.time) >= cutoffDate);

                    // Filter buy/sell signals based on time range
                    const buyIndices = [];
                    const sellIndices = [];
                    for (let i = 0; i < data.buy_signals.dates.length; i++) {
                        if (new Date(data.buy_signals.dates[i]) >= cutoffDate) {
                            buyIndices.push(i);
                        }
                    }
                    for (let i = 0; i < data.sell_signals.dates.length; i++) {
                        if (new Date(data.sell_signals.dates[i]) >= cutoffDate) {
                            sellIndices.push(i);
                        }
                    }
                    filteredBuySignals = {
                        dates: buyIndices.map(i => data.buy_signals.dates[i]),
                        prices: buyIndices.map(i => data.buy_signals.prices[i])
                    };
                    filteredSellSignals = {
                        dates: sellIndices.map(i => data.sell_signals.dates[i]),
                        prices: sellIndices.map(i => data.sell_signals.prices[i])
                    };
                }

                const times = filtered.map(d => d.time);

                // === MAIN CHART: Price with Buy/Sell Signals ===
                const priceTrace = {
                    type: 'scatter',
                    mode: 'lines',
                    x: filteredDaily.map(d => d.time),
                    y: filteredDaily.map(d => d.close),
                    name: 'BTC Price',
                    line: {color: '#000000', width: 1.5},
                    yaxis: 'y',
                    xaxis: 'x',
                    showlegend: true
                };

                // Buy signals (green triangles up) - SEMI-TRANSPARENT
                const buyTrace = {
                    type: 'scatter',
                    mode: 'markers',
                    x: filteredBuySignals.dates,
                    y: filteredBuySignals.prices,
                    name: 'Buy Signal',
                    marker: {
                        symbol: 'triangle-up',
                        size: 8,
                        color: 'rgba(16,185,129,0.6)',
                        line: {color: 'rgba(6,95,70,0.8)', width: 1}
                    },
                    yaxis: 'y',
                    xaxis: 'x',
                    showlegend: true
                };

                // Sell signals (red triangles down) - SEMI-TRANSPARENT
                const sellTrace = {
                    type: 'scatter',
                    mode: 'markers',
                    x: filteredSellSignals.dates,
                    y: filteredSellSignals.prices,
                    name: 'Sell Signal',
                    marker: {
                        symbol: 'triangle-down',
                        size: 8,
                        color: 'rgba(239,68,68,0.6)',
                        line: {color: 'rgba(153,27,27,0.8)', width: 1}
                    },
                    yaxis: 'y',
                    xaxis: 'x',
                    showlegend: true
                };

                // Individual weighted signals (STEPPED) - Separate stacked mini-charts
                const trace200w = {
                    type: 'scatter',
                    mode: 'lines',
                    x: times,
                    y: filtered.map(d => d.weighted_200w),
                    name: '200W SMA',
                    line: {color: '#f59e0b', width: 1.5, shape: 'hv'},
                    yaxis: 'y4',
                    xaxis: 'x4',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(245,158,11,0.1)'
                };

                const trace50w = {
                    type: 'scatter',
                    mode: 'lines',
                    x: times,
                    y: filtered.map(d => d.weighted_50w),
                    name: '50W MA',
                    line: {color: '#3b82f6', width: 1.5, shape: 'hv'},
                    yaxis: 'y5',
                    xaxis: 'x5',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(59,130,246,0.1)'
                };

                const traceFG = {
                    type: 'scatter',
                    mode: 'lines',
                    x: times,
                    y: filtered.map(d => d.weighted_fg),
                    name: 'F&G',
                    line: {color: '#8b5cf6', width: 1.5, shape: 'hv'},
                    yaxis: 'y6',
                    xaxis: 'x6',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(139,92,246,0.1)'
                };

                const traceRSI = {
                    type: 'scatter',
                    mode: 'lines',
                    x: times,
                    y: filtered.map(d => d.weighted_rsi),
                    name: 'RSI',
                    line: {color: '#ec4899', width: 1.5, shape: 'hv'},
                    yaxis: 'y7',
                    xaxis: 'x7',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(236,72,153,0.1)'
                };

                // Combined signal (STEPPED)
                const traceCombined = {
                    type: 'scatter',
                    mode: 'lines',
                    x: times,
                    y: filtered.map(d => d.combined),
                    name: 'Combined',
                    line: {color: '#1f2937', width: 2, shape: 'hv'},
                    yaxis: 'y2',
                    xaxis: 'x2',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(107,114,128,0.1)'
                };

                // Buy/Sell threshold lines
                const buyThreshold = {
                    type: 'scatter',
                    mode: 'lines',
                    x: [times[0], times[times.length - 1]],
                    y: [2, 2],
                    name: 'Buy Threshold',
                    line: {color: '#10b981', width: 1.5, dash: 'dash'},
                    yaxis: 'y2',
                    xaxis: 'x2',
                    showlegend: false
                };

                const sellThreshold = {
                    type: 'scatter',
                    mode: 'lines',
                    x: [times[0], times[times.length - 1]],
                    y: [-2, -2],
                    name: 'Sell Threshold',
                    line: {color: '#ef4444', width: 1.5, dash: 'dash'},
                    yaxis: 'y2',
                    xaxis: 'x2',
                    showlegend: false
                };

                // Market regime - thin colored stripe at bottom
                const regimeTrace = {
                    type: 'bar',
                    x: times,
                    y: Array(times.length).fill(1),
                    name: 'Regime',
                    marker: {
                        color: filtered.map(d => d.regime === 'BULL' ? '#10b981' : '#ef4444')
                    },
                    yaxis: 'y8',
                    xaxis: 'x8',
                    showlegend: false,
                    hovertemplate: '%{text}<extra></extra>',
                    text: filtered.map(d => d.regime === 'BULL' ? 'BULL MARKET' : 'BEAR MARKET')
                };

                const layout = {
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#fafafa',
                    font: {color: '#374151', family: '-apple-system, BlinkMacSystemFont, sans-serif', size: 11},
                    showlegend: true,
                    legend: {
                        orientation: 'h',
                        x: 0,
                        y: 1.02,
                        xanchor: 'left',
                        yanchor: 'bottom'
                    },
                    hovermode: 'x unified',
                    dragmode: false,

                    // Price chart (top, largest) - More space allocated
                    xaxis: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        domain: [0, 1],
                        anchor: 'y',
                        fixedrange: true
                    },
                    yaxis: {
                        title: {text: 'BTC Price (USD)', font: {size: 11, color: '#6b7280'}},
                        gridcolor: '#e5e7eb',
                        domain: [0.58, 1],
                        fixedrange: true,
                        type: 'log'
                    },

                    // Combined signal chart - Increased spacing
                    xaxis2: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        domain: [0, 1],
                        anchor: 'y2',
                        fixedrange: true
                    },
                    yaxis2: {
                        title: {text: 'Combined Signal', font: {size: 11, color: '#6b7280'}},
                        gridcolor: '#e5e7eb',
                        domain: [0.46, 0.55],
                        fixedrange: true
                    },

                    // 200W SMA weighted signal - More breathing room
                    xaxis4: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        domain: [0, 1],
                        anchor: 'y4',
                        fixedrange: true,
                        showticklabels: false
                    },
                    yaxis4: {
                        title: {text: '200W', font: {size: 10, color: '#f59e0b'}},
                        gridcolor: '#f3f4f6',
                        domain: [0.36, 0.43],
                        fixedrange: true
                    },

                    // 50W MA weighted signal - More breathing room
                    xaxis5: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        domain: [0, 1],
                        anchor: 'y5',
                        fixedrange: true,
                        showticklabels: false
                    },
                    yaxis5: {
                        title: {text: '50W', font: {size: 10, color: '#3b82f6'}},
                        gridcolor: '#f3f4f6',
                        domain: [0.26, 0.33],
                        fixedrange: true
                    },

                    // F&G weighted signal - More breathing room
                    xaxis6: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        domain: [0, 1],
                        anchor: 'y6',
                        fixedrange: true,
                        showticklabels: false
                    },
                    yaxis6: {
                        title: {text: 'F&G', font: {size: 10, color: '#8b5cf6'}},
                        gridcolor: '#f3f4f6',
                        domain: [0.16, 0.23],
                        fixedrange: true
                    },

                    // RSI weighted signal - More breathing room
                    xaxis7: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        domain: [0, 1],
                        anchor: 'y7',
                        fixedrange: true,
                        showticklabels: false
                    },
                    yaxis7: {
                        title: {text: 'RSI', font: {size: 10, color: '#ec4899'}},
                        gridcolor: '#f3f4f6',
                        domain: [0.06, 0.13],
                        fixedrange: true
                    },

                    // Regime stripe (very bottom)
                    xaxis8: {
                        type: 'date',
                        gridcolor: '#e5e7eb',
                        domain: [0, 1],
                        anchor: 'y8',
                        fixedrange: true
                    },
                    yaxis8: {
                        gridcolor: 'transparent',
                        domain: [0, 0.04],
                        fixedrange: true,
                        showticklabels: false,
                        zeroline: false,
                        showgrid: false
                    },

                    margin: {l: 60, r: 40, t: 60, b: 60}
                };

                const traces = [priceTrace, buyTrace, sellTrace, trace200w, trace50w, traceFG, traceRSI, traceCombined, buyThreshold, sellThreshold, regimeTrace];
                Plotly.newPlot('historical-signals-chart', traces, layout, {responsive: true, displayModeBar: false, staticPlot: true});
            } catch (err) {
                console.error('Error rendering historical signals:', err);
            }
        }

        function renderFAQ(data) {
            const faqDiv = document.getElementById('faq-content');

            if (!data.signals || !data.config) {
                faqDiv.innerHTML = '<div class="faq-content"><p>Loading...</p></div>';
                return;
            }

            const signals = data.signals;
            const config = data.config;
            const p = config.parameters;
            const dca = data.dca_comparison || {};

            faqDiv.innerHTML = `
                <div class="faq-content">
                    <div class="faq-hero">
                        <h2 class="faq-hero-title">Bitcoin Pocojuan Model</h2>
                        <p class="faq-hero-text">
                            A professional trading system combining four optimized technical indicators
                            to generate high-conviction buy and sell signals for Bitcoin.
                        </p>
                    </div>

                    <div class="faq-section">
                        <div class="faq-section-title">How It Works</div>
                        <div class="faq-section-content">
                            <p>The Pocojuan Model combines four technical indicators, each providing a signal of +1 (BUY), 0 (HOLD), or -1 (SELL). These signals are weighted by their optimized importance and combined into a single score.</p>

                            <p><strong>Signal Rules:</strong></p>
                            <ul style="margin: 12px 0; padding-left: 20px;">
                                <li>Combined Score ≥ +2.0 → <strong>BUY</strong></li>
                                <li>Combined Score ≤ -2.0 → <strong>SELL</strong></li>
                                <li>Otherwise → <strong>HOLD</strong></li>
                            </ul>

                            <div style="margin-top: 24px; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px; border: 1px solid #dee2e6;">
                                <h4 style="margin: 0 0 16px 0; font-size: 14px; color: #495057; font-weight: 600;">US National Debt & Bitcoin</h4>
                                <p style="margin: 0 0 16px 0; font-size: 13px; color: #6c757d; line-height: 1.5;">
                                    As government debt increases and currency is debased through monetary expansion, Bitcoin's scarcity (21M fixed supply) makes it increasingly valuable as a store of value and hedge against inflation.
                                </p>

                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
                                    <div style="background: white; padding: 16px; border-radius: 6px; border: 1px solid #dee2e6;">
                                        <div style="font-size: 11px; color: #6c757d; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px;">Total US Debt</div>
                                        <div id="debt-counter" style="font-size: 20px; font-weight: 700; color: #dc3545; font-variant-numeric: tabular-nums;">Loading...</div>
                                        <div id="debt-subtitle" style="font-size: 10px; color: #868e96; margin-top: 4px;">Fetching from Treasury API...</div>
                                    </div>

                                    <div style="background: white; padding: 16px; border-radius: 6px; border: 1px solid #dee2e6;">
                                        <div style="font-size: 11px; color: #6c757d; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px;">2026 Deficit (YTD)</div>
                                        <div id="deficit-counter" style="font-size: 20px; font-weight: 700; color: #e67700; font-variant-numeric: tabular-nums;">Loading...</div>
                                        <div id="deficit-subtitle" style="font-size: 10px; color: #868e96; margin-top: 4px;">Year-to-date change</div>
                                    </div>
                                </div>

                                <div id="debt-updated" style="font-size: 11px; color: #868e96; text-align: center;">
                                    Real-time data from US Treasury API
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="faq-section">
                        <div class="faq-section-title">The Four Indicators</div>
                        <div class="faq-section-content">
                            <div class="indicator-list">
                                <div class="indicator-item">
                                    <h4>200-Week SMA (Weight: ${signals['200w'].weight}x)</h4>
                                    <p>A long-term moving average that identifies major support levels. The thresholds decay over time as Bitcoin matures. Signals BUY when price is significantly below the SMA.</p>
                                </div>

                                <div class="indicator-item">
                                    <h4>50-Week MA Regime (Weight: ${signals['50w'].weight}x)</h4>
                                    <p>Detects bull/bear market regimes and adjusts thresholds accordingly. In bull markets, it identifies overextension. In bear markets, it finds deep value opportunities.</p>
                                </div>

                                <div class="indicator-item">
                                    <h4>Fear & Greed EMA (Weight: ${signals.fg.weight}x)</h4>
                                    <p>A ${p.fg.ema_period}-day exponential moving average of market sentiment. Extreme fear often presents buying opportunities, while extreme greed signals potential tops.</p>
                                </div>

                                <div class="indicator-item">
                                    <h4>RSI ${p.rsi.period}-day (Weight: ${signals.rsi.weight}x)</h4>
                                    <p>Relative Strength Index identifies overbought and oversold conditions. Extremely low RSI suggests strong buying opportunities, while very high RSI indicates potential selling zones.</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="faq-section">
                        <div class="faq-section-title">Performance vs. DCA</div>
                        <div class="faq-section-content">
                            <p>Starting from ${dca.start_date || '2017'}, comparing the Pocojuan Model against a simple $100/week Dollar-Cost Averaging strategy. Both strategies start with the same total capital ($${dca.starting_amount ? dca.starting_amount.toLocaleString() : '--'}).</p>

                            <div class="comparison-card">
                                <div class="comparison-grid">
                                    <div class="comparison-item">
                                        <div class="comparison-label">Pocojuan Model</div>
                                        <div class="comparison-value">$${dca.model_final_value ? dca.model_final_value.toLocaleString() : '--'}</div>
                                        <div style="font-size: 12px; color: #10b981; font-weight: 600; margin-top: 4px;">
                                            +${dca.model_return ? dca.model_return.toFixed(0) : '--'}%
                                        </div>
                                    </div>

                                    <div class="comparison-item">
                                        <div class="comparison-label">Simple DCA</div>
                                        <div class="comparison-value" style="color: #6b7280;">$${dca.dca_final_value ? dca.dca_final_value.toLocaleString() : '--'}</div>
                                        <div style="font-size: 12px; color: #6b7280; font-weight: 600; margin-top: 4px;">
                                            +${dca.dca_return ? dca.dca_return.toFixed(0) : '--'}%
                                        </div>
                                    </div>
                                </div>

                                <div style="text-align: center; margin-top: 12px; padding-top: 12px; border-top: 1px solid #e5e7eb;">
                                    <div style="font-size: 12px; color: #6b7280; margin-bottom: 4px;">Outperformance</div>
                                    <div style="font-size: 20px; font-weight: 700; color: #10b981;">
                                        +${dca.outperformance ? dca.outperformance.toFixed(0) : '--'}%
                                    </div>
                                    <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
                                        ${dca.trades || '--'} total trades executed
                                    </div>
                                </div>
                            </div>

                            <p style="margin-top: 16px;">The Pocojuan Model's weighted indicator system outperforms dollar-cost averaging by making strategic buys during market fear and sells during extreme greed. Both strategies had equal capital deployed over the same time period for a fair comparison.</p>
                        </div>
                    </div>

                    <div class="faq-section">
                        <div class="faq-section-title">Risk Disclaimer</div>
                        <div class="faq-section-content">
                            <p>This model is for educational and informational purposes only. Past performance does not guarantee future results. Cryptocurrency trading carries significant risk. Always conduct your own research and never invest more than you can afford to lose.</p>
                        </div>
                    </div>
                </div>
            `;

            // Initialize debt counters after FAQ is rendered
            initializeDebtCounters();
        }

        // National Debt Counter Logic - API Powered
        let debtCounterInterval = null;
        let debtApiData = null;

        async function fetchDebtData() {
            try {
                // Fetch latest debt data
                const response = await fetch('https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny?sort=-record_date&page[size]=1');
                if (!response.ok) throw new Error('API request failed');
                const data = await response.json();

                if (data.data && data.data.length > 0) {
                    const latestDebt = parseFloat(data.data[0].tot_pub_debt_out_amt);
                    const debtDate = data.data[0].record_date;

                    // Fetch year-start debt for deficit calculation
                    const yearStartResponse = await fetch('https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny?filter=record_date:gte:2026-01-01,record_date:lte:2026-01-05&sort=record_date');
                    const yearStartData = await yearStartResponse.json();
                    const yearStartDebt = yearStartData.data && yearStartData.data.length > 0
                        ? parseFloat(yearStartData.data[0].tot_pub_debt_out_amt)
                        : latestDebt;

                    // Calculate daily increase rate from historical data (estimate $8-10B/day)
                    const daysSinceYearStart = Math.max(1, Math.floor((new Date(debtDate) - new Date('2026-01-02')) / (1000 * 60 * 60 * 24)));
                    const totalIncrease = latestDebt - yearStartDebt;
                    const dailyIncreaseRate = daysSinceYearStart > 0 ? totalIncrease / daysSinceYearStart : 8.5e9;

                    debtApiData = {
                        latestDebt: latestDebt,
                        yearStartDebt: yearStartDebt,
                        debtDate: debtDate,
                        dailyIncreaseRate: dailyIncreaseRate,
                        fetchTime: Date.now()
                    };

                    updateDebtDisplayFromApi();
                    updateDebtUpdatedTime();
                }
            } catch (error) {
                console.error('Failed to fetch debt data:', error);
                // Fallback to estimated values
                debtApiData = {
                    latestDebt: 38.65e12,
                    yearStartDebt: 38.42e12,
                    debtDate: new Date().toISOString().split('T')[0],
                    dailyIncreaseRate: 9e9,
                    fetchTime: Date.now(),
                    isEstimate: true
                };
                updateDebtDisplayFromApi();
                document.getElementById('debt-updated').textContent = 'Using estimated values (API unavailable)';
            }
        }

        function initializeDebtCounters() {
            // Clear any existing intervals
            if (debtCounterInterval) clearInterval(debtCounterInterval);

            // Fetch initial data
            fetchDebtData();

            // Update display every 100ms for smooth ticking
            debtCounterInterval = setInterval(updateDebtDisplayFromApi, 100);

            // Refresh from API every 5 minutes
            setInterval(fetchDebtData, 300000);
        }

        function updateDebtDisplayFromApi() {
            if (!debtApiData) return;

            const debtElement = document.getElementById('debt-counter');
            const deficitElement = document.getElementById('deficit-counter');
            const debtSubtitle = document.getElementById('debt-subtitle');
            const deficitSubtitle = document.getElementById('deficit-subtitle');

            if (!debtElement || !deficitElement) return;

            // Calculate elapsed time since last API fetch (in days)
            const elapsedMs = Date.now() - debtApiData.fetchTime;
            const elapsedDays = elapsedMs / (1000 * 60 * 60 * 24);

            // Calculate current debt with linear interpolation
            const currentDebt = debtApiData.latestDebt + (debtApiData.dailyIncreaseRate * elapsedDays);
            const currentDeficit = currentDebt - debtApiData.yearStartDebt;

            // Format and display
            debtElement.textContent = formatDebt(currentDebt);
            deficitElement.textContent = formatDebt(currentDeficit);

            // Update subtitles with rate information
            const ratePerSecond = debtApiData.dailyIncreaseRate / 86400;
            if (ratePerSecond >= 1e6) {
                debtSubtitle.textContent = '+$' + (ratePerSecond / 1e6).toFixed(1) + 'M per second';
            } else if (ratePerSecond >= 1e3) {
                debtSubtitle.textContent = '+$' + (ratePerSecond / 1e3).toFixed(0) + 'K per second';
            }

            deficitSubtitle.textContent = 'Since Jan 2, 2026';
        }

        function updateDebtUpdatedTime() {
            if (!debtApiData) return;
            const updateElement = document.getElementById('debt-updated');
            if (!updateElement) return;

            const debtDate = new Date(debtApiData.debtDate);
            const dateStr = debtDate.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'});
            const now = new Date();
            const timeStr = now.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});

            if (debtApiData.isEstimate) {
                updateElement.textContent = 'Using estimated values (API unavailable)';
            } else {
                updateElement.textContent = `Real-time data from US Treasury (as of ${dateStr}, updated ${timeStr})`;
            }
        }

        function formatDebt(value) {
            if (value >= 1e12) {
                return '$' + (value / 1e12).toFixed(3) + 'T';
            } else if (value >= 1e9) {
                return '$' + (value / 1e9).toFixed(2) + 'B';
            } else if (value >= 1e6) {
                return '$' + (value / 1e6).toFixed(2) + 'M';
            }
            return '$' + value.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
        }

        function updateDataPanel(data) {
            const priceElement = document.getElementById('price');
            priceElement.textContent = '$' + data.current_price.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
            previousPrice = data.current_price;

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
            const dateStr = now.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'});
            dateEl.textContent = dateStr;

            if (data.signals) {
                const s = data.signals;
                textEl.textContent = s.action;
                scoreEl.textContent = `Score: ${s.combined >= 0 ? '+' : ''}${s.combined}`;
                cardEl.className = 'signal-card ' + s.action_class;

                document.getElementById('sig-200w').textContent = s['200w'].signal_text;
                document.getElementById('sig-200w-val').textContent = `${s['200w'].weighted >= 0 ? '+' : ''}${s['200w'].weighted}`;

                document.getElementById('sig-50w').textContent = s['50w'].signal_text;
                document.getElementById('sig-50w-val').textContent = `${s['50w'].weighted >= 0 ? '+' : ''}${s['50w'].weighted}`;

                document.getElementById('sig-fg').textContent = s.fg.signal_text;
                document.getElementById('sig-fg-val').textContent = `${s.fg.weighted >= 0 ? '+' : ''}${s.fg.weighted}`;

                document.getElementById('sig-rsi').textContent = s.rsi.signal_text;
                document.getElementById('sig-rsi-val').textContent = `${s.rsi.weighted >= 0 ? '+' : ''}${s.rsi.weighted}`;
            }
        }

        loadData();
        setInterval(loadData, 300000); // Refresh every 5 minutes
        setInterval(updateLivePrice, 2000); // Update price every 2 seconds

        window.addEventListener('resize', function() {
            setTimeout(function() {
                if (chartData) {
                    if (currentChartView === 'price') Plotly.Plots.resize('chart');
                    else if (currentChartView === 'signals') Plotly.Plots.resize('historical-signals-chart');
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
    print("   Bitcoin Pocojuan Model Dashboard")
    print("="*70)

    load_config()

    if CONFIG is None:
        print("\n[ERROR] Failed to load configuration. Please check constrained_5stage_config.json exists.")
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
