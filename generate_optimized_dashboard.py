#!/usr/bin/env python3
"""
Generate clean, minimalistic web dashboard for optimized 4-indicator strategy
Mobile-responsive PWA
"""
import ccxt
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime

def fetch_btc_data():
    exchange = ccxt.bitstamp()
    since = exchange.parse8601('2018-01-01T00:00:00Z')
    all_ohlcv = []
    current_since = since
    while True:
        ohlcv = exchange.fetch_ohlcv('BTC/USD', '1d', since=current_since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        current_since = ohlcv[-1][0] + 86400000
        if current_since > datetime.now().timestamp() * 1000:
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

def fetch_fg_data():
    url = "https://api.alternative.me/fng/?limit=0"
    data = requests.get(url).json()['data']
    fg_df = pd.DataFrame(data)
    fg_df['time'] = pd.to_datetime(fg_df['timestamp'].astype(int), unit='s')
    fg_df['value'] = fg_df['value'].astype(int)
    return fg_df[['time', 'value']].rename(columns={'value': 'fg_index'})

def calculate_all_indicators(df, fg_df, config):
    """Calculate all indicators"""
    p = config['parameters']

    # 200W SMA
    df_weekly_200 = df.set_index('time').resample('W-SUN')['close'].last().dropna()
    sma_200w = df_weekly_200.rolling(window=200, min_periods=1).mean()
    sma_df = pd.DataFrame({'sma_200w': sma_200w, 'sma_ratio': df_weekly_200 / sma_200w})

    # 50W MA with regime
    df_weekly_50 = df.set_index('time').resample('W-SUN')['close'].last().dropna()
    ma_50w = df_weekly_50.rolling(window=50, min_periods=1).mean()
    ma_df = pd.DataFrame({'weekly_close': df_weekly_50, 'ma_50w': ma_50w})

    regimes = []
    for i in range(len(ma_df)):
        if i < 2:
            regimes.append('bull')
            continue
        last_3 = ma_df.iloc[max(0, i-2):i+1]
        above_count = (last_3['weekly_close'] > last_3['ma_50w']).sum()
        regimes.append('bull' if above_count == 3 else ('bear' if above_count == 0 else (regimes[-1] if regimes else 'bull')))
    ma_df['regime'] = regimes
    ma_df['ma50w_ratio'] = ma_df['weekly_close'] / ma_df['ma_50w']

    # F&G
    fg_sorted = fg_df.sort_values('time').copy()
    fg_sorted['fg_ema'] = fg_sorted['fg_index'].ewm(span=p['fg']['ema_period'], adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=p['rsi']['period'], min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=p['rsi']['period'], min_periods=1).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    df['rsi'] = rsi

    # Merge
    df = df.set_index('time')
    df = df.merge(sma_df, left_index=True, right_index=True, how='left')
    df = df.merge(ma_df[['ma_50w', 'regime', 'ma50w_ratio']], left_index=True, right_index=True, how='left')
    df = df.merge(fg_sorted[['time', 'fg_index', 'fg_ema']].set_index('time'), left_index=True, right_index=True, how='left')

    for col in ['sma_200w', 'sma_ratio', 'ma_50w', 'regime', 'ma50w_ratio', 'fg_ema', 'rsi']:
        df[col] = df[col].ffill()

    return df.reset_index()

def get_current_signals(latest_row, config, start_year):
    """Get current signal status"""
    p = config['parameters']
    w = config['weights']

    years = latest_row['time'].year - start_year

    # 200W SMA - DUAL DECAY
    decay_buy = p['sma_200w'].get('decay_buy', p['sma_200w']['decay']) ** years
    decay_sell = p['sma_200w'].get('decay_sell', p['sma_200w']['decay']) ** years
    adj_buy = p['sma_200w']['buy'] * decay_buy
    adj_sell = p['sma_200w']['sell'] * decay_sell
    sig_sma = 1 if latest_row['sma_ratio'] < adj_buy else (-1 if latest_row['sma_ratio'] > adj_sell else 0)

    # 50W MA
    if latest_row['regime'] == 'bull':
        sig_ma50w = -1 if latest_row['ma50w_ratio'] > p['ma_50w']['bull_ext'] else (1 if latest_row['ma50w_ratio'] < p['ma_50w']['bull_supp'] else 0)
    else:
        sig_ma50w = -1 if latest_row['ma50w_ratio'] > p['ma_50w']['bear_res'] else (1 if latest_row['ma50w_ratio'] < p['ma_50w']['bear_deep'] else 0)

    # F&G
    sig_fg = 1 if latest_row['fg_ema'] < p['fg']['buy'] else (-1 if latest_row['fg_ema'] > p['fg']['sell'] else 0)

    # RSI
    sig_rsi = 1 if latest_row['rsi'] < p['rsi']['buy'] else (-1 if latest_row['rsi'] > p['rsi']['sell'] else 0)

    # Weighted combined
    combined = (sig_sma * w['w_200w']) + (sig_ma50w * w['w_50w']) + (sig_fg * w['w_fg']) + (sig_rsi * w['w_rsi'])

    return {
        '200w': {'signal': sig_sma, 'weighted': sig_sma * w['w_200w'], 'ratio': latest_row['sma_ratio']},
        '50w': {'signal': sig_ma50w, 'weighted': sig_ma50w * w['w_50w'], 'ratio': latest_row['ma50w_ratio'], 'regime': latest_row['regime']},
        'fg': {'signal': sig_fg, 'weighted': sig_fg * w['w_fg'], 'value': latest_row['fg_ema']},
        'rsi': {'signal': sig_rsi, 'weighted': sig_rsi * w['w_rsi'], 'value': latest_row['rsi']},
        'combined': combined,
        'action': 'STRONG BUY' if combined >= 2 else ('STRONG SELL' if combined <= -2 else 'HOLD')
    }

def generate_html(df, config, signals):
    """Generate clean HTML dashboard"""
    latest = df.iloc[-1]
    p = config['parameters']
    w = config['weights']

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#1a1a2e">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <link rel="manifest" href="manifest.json">
    <title>BTC Strategy Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            padding: 1rem;
            min-height: 100vh;
        }}

        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 2rem;
        }}

        .price {{
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .timestamp {{
            color: #888;
            font-size: 0.9rem;
        }}

        .action-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            text-align: center;
            border: 2px solid rgba(255, 255, 255, 0.1);
        }}

        .action {{
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }}

        .action.buy {{ color: #10b981; }}
        .action.sell {{ color: #ef4444; }}
        .action.hold {{ color: #f59e0b; }}

        .combined-score {{
            font-size: 1.5rem;
            color: #888;
        }}

        .indicators {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .indicator {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem 1rem;
            border: 2px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }}

        .indicator:hover {{
            transform: translateY(-5px);
            border-color: rgba(255, 255, 255, 0.2);
        }}

        .indicator-name {{
            font-size: 0.85rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }}

        .indicator-value {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .signal-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .signal-buy {{ background: #10b981; color: #000; }}
        .signal-sell {{ background: #ef4444; color: #fff; }}
        .signal-neutral {{ background: #6b7280; color: #fff; }}

        .config {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .config-title {{
            font-size: 0.9rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1rem;
        }}

        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
        }}

        .config-item {{
            font-size: 0.85rem;
        }}

        .config-label {{
            color: #888;
            margin-bottom: 0.25rem;
        }}

        .config-value {{
            color: #fff;
            font-weight: 600;
        }}

        .footer {{
            text-align: center;
            color: #666;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding: 1rem;
        }}

        @media (max-width: 600px) {{
            .price {{
                font-size: 2.5rem;
            }}

            .action {{
                font-size: 2rem;
            }}

            .indicators {{
                grid-template-columns: repeat(2, 1fr);
            }}

            .action-card {{
                padding: 1.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="price">${latest['close']:,.0f}</div>
            <div class="timestamp">Last updated: {latest['time'].strftime('%Y-%m-%d %H:%M UTC')}</div>
        </div>

        <div class="action-card">
            <div class="action {signals['action'].lower().split()[1] if len(signals['action'].split()) > 1 else signals['action'].lower()}">
                {signals['action']}
            </div>
            <div class="combined-score">Combined Signal: {signals['combined']:.2f}</div>
        </div>

        <div class="indicators">
            <div class="indicator">
                <div class="indicator-name">200W SMA</div>
                <div class="indicator-value">{signals['200w']['ratio']:.3f}</div>
                <span class="signal-badge signal-{'buy' if signals['200w']['signal'] == 1 else 'sell' if signals['200w']['signal'] == -1 else 'neutral'}">
                    {'BUY' if signals['200w']['signal'] == 1 else 'SELL' if signals['200w']['signal'] == -1 else 'NEUTRAL'}
                </span>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #888;">
                    Weight: {w['w_200w']:.2f}x → {signals['200w']['weighted']:.2f}
                </div>
            </div>

            <div class="indicator">
                <div class="indicator-name">50W MA ({signals['50w']['regime'].upper()})</div>
                <div class="indicator-value">{signals['50w']['ratio']:.3f}</div>
                <span class="signal-badge signal-{'buy' if signals['50w']['signal'] == 1 else 'sell' if signals['50w']['signal'] == -1 else 'neutral'}">
                    {'BUY' if signals['50w']['signal'] == 1 else 'SELL' if signals['50w']['signal'] == -1 else 'NEUTRAL'}
                </span>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #888;">
                    Weight: {w['w_50w']:.2f}x → {signals['50w']['weighted']:.2f}
                </div>
            </div>

            <div class="indicator">
                <div class="indicator-name">Fear & Greed</div>
                <div class="indicator-value">{signals['fg']['value']:.0f}</div>
                <span class="signal-badge signal-{'buy' if signals['fg']['signal'] == 1 else 'sell' if signals['fg']['signal'] == -1 else 'neutral'}">
                    {'BUY' if signals['fg']['signal'] == 1 else 'SELL' if signals['fg']['signal'] == -1 else 'NEUTRAL'}
                </span>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #888;">
                    Weight: {w['w_fg']:.2f}x → {signals['fg']['weighted']:.2f}
                </div>
            </div>

            <div class="indicator">
                <div class="indicator-name">RSI</div>
                <div class="indicator-value">{signals['rsi']['value']:.1f}</div>
                <span class="signal-badge signal-{'buy' if signals['rsi']['signal'] == 1 else 'sell' if signals['rsi']['signal'] == -1 else 'neutral'}">
                    {'BUY' if signals['rsi']['signal'] == 1 else 'SELL' if signals['rsi']['signal'] == -1 else 'NEUTRAL'}
                </span>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #888;">
                    Weight: {w['w_rsi']:.2f}x → {signals['rsi']['weighted']:.2f}
                </div>
            </div>
        </div>

        <div class="config">
            <div class="config-title">Strategy Configuration</div>
            <div class="config-grid">
                <div class="config-item">
                    <div class="config-label">200W Thresholds</div>
                    <div class="config-value">Buy: {p['sma_200w']['buy']:.2f} | Sell: {p['sma_200w']['sell']:.2f}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">200W Decay Rates</div>
                    <div class="config-value">Buy: {p['sma_200w'].get('decay_buy', p['sma_200w']['decay']):.4f} | Sell: {p['sma_200w'].get('decay_sell', p['sma_200w']['decay']):.4f}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">50W Bull</div>
                    <div class="config-value">Ext: {p['ma_50w']['bull_ext']:.2f} | Supp: {p['ma_50w']['bull_supp']:.2f}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">50W Bear</div>
                    <div class="config-value">Res: {p['ma_50w']['bear_res']:.2f} | Deep: {p['ma_50w']['bear_deep']:.2f}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">F&G EMA</div>
                    <div class="config-value">{p['fg']['ema_period']}d | Buy: <{p['fg']['buy']:.0f}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">RSI</div>
                    <div class="config-value">{p['rsi']['period']}d | Buy: <{p['rsi']['buy']:.0f}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Performance</div>
                    <div class="config-value">{((config.get('final_value', 49545) - 1000) / 10):.0f}% Return</div>
                </div>
            </div>
        </div>

        <div class="footer">
            5-Stage Dual Decay Constrained Strategy (Balanced Weights ≤2.0) | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>

    <script>
        if ('serviceWorker' in navigator) {{
            navigator.serviceWorker.register('service-worker.js');
        }}
    </script>
</body>
</html>'''

    return html

def main():
    print("\n[INFO] Generating optimized dashboard...")

    # Load config (using constrained model with balanced weights)
    with open('constrained_5stage_config.json') as f:
        config = json.load(f)

    print("[INFO] Fetching data...")
    df = fetch_btc_data()
    fg_df = fetch_fg_data()

    print("[INFO] Calculating indicators...")
    df = calculate_all_indicators(df, fg_df, config)

    print("[INFO] Getting current signals...")
    start_year = df['time'].iloc[0].year
    signals = get_current_signals(df.iloc[-1], config, start_year)

    print("[INFO] Generating HTML...")
    html = generate_html(df, config, signals)

    with open('btc_optimized_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\\n[OK] Dashboard generated: btc_optimized_dashboard.html")
    print(f"[INFO] Current signal: {signals['action']} ({signals['combined']:.2f})")
    print(f"[INFO] Price: ${df.iloc[-1]['close']:,.0f}")
    print()

if __name__ == '__main__':
    main()
