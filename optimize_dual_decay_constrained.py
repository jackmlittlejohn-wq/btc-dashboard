#!/usr/bin/env python3
"""
5-Stage Multi-Stage Gradient Optimization - DUAL DECAY CONSTRAINED
18 parameters (17 original + split decay into buy/sell)
CONSTRAINT: No single indicator weight may exceed threshold magnitude (2.0)
Testing hypothesis: Buy and Sell decay at different rates with balanced weights
"""
import ccxt
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

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

def calculate_all_indicators(df_base, fg_df, params):
    """
    Calculate all indicators with given parameters
    params: [sma_buy, sma_sell, sma_decay_buy, sma_decay_sell,
             bull_ext, bull_supp, bear_res, bear_deep,
             fg_period, fg_buy, fg_sell,
             rsi_period, rsi_buy, rsi_sell,
             w_200w, w_50w, w_fg, w_rsi]
    """
    df = df_base.copy()

    # Unpack parameters (18 params now with dual decay)
    sma_buy, sma_sell, sma_decay_buy, sma_decay_sell = params[0:4]
    bull_ext, bull_supp, bear_res, bear_deep = params[4:8]
    fg_period, fg_buy, fg_sell = params[8:11]
    rsi_period, rsi_buy, rsi_sell = params[11:14]
    w_200w, w_50w, w_fg, w_rsi = params[14:18]

    # 200W SMA
    df_weekly_200 = df.set_index('time').resample('W-SUN')['close'].last().dropna()
    sma_200w = df_weekly_200.rolling(window=200, min_periods=1).mean()
    sma_df = pd.DataFrame({'sma_200w': sma_200w, 'sma_ratio': df_weekly_200 / sma_200w})

    # 50W MA
    df_weekly_50 = df.set_index('time').resample('W-SUN')['close'].last().dropna()
    ma_50w = df_weekly_50.rolling(window=50, min_periods=1).mean()
    ma_df = pd.DataFrame({'weekly_close': df_weekly_50, 'ma_50w': ma_50w})

    # Regime detection
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

    # F&G EMA
    fg_sorted = fg_df.sort_values('time').copy()
    fg_sorted['fg_ema'] = fg_sorted['fg_index'].ewm(span=int(fg_period), adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=int(rsi_period), min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=int(rsi_period), min_periods=1).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    df['rsi'] = rsi

    # Merge all
    df = df.set_index('time')
    df = df.merge(sma_df, left_index=True, right_index=True, how='left')
    df = df.merge(ma_df[['ma_50w', 'regime', 'ma50w_ratio']], left_index=True, right_index=True, how='left')
    df = df.merge(fg_sorted[['time', 'fg_index', 'fg_ema']].set_index('time'), left_index=True, right_index=True, how='left')

    for col in ['sma_200w', 'sma_ratio', 'ma_50w', 'regime', 'ma50w_ratio', 'fg_ema', 'rsi']:
        df[col] = df[col].ffill()

    return df.reset_index()

def objective_function(params, df_base, fg_df, start_year):
    """Objective function to maximize"""
    # Unpack (18 params with dual decay)
    sma_buy, sma_sell, sma_decay_buy, sma_decay_sell = params[0:4]
    bull_ext, bull_supp, bear_res, bear_deep = params[4:8]
    fg_period, fg_buy, fg_sell = params[8:11]
    rsi_period, rsi_buy, rsi_sell = params[11:14]
    w_200w, w_50w, w_fg, w_rsi = params[14:18]

    # Validate constraints
    if sma_buy >= sma_sell:
        return -1000000
    if bull_supp >= bull_ext:
        return -1000000
    if bear_deep >= bear_res:
        return -1000000
    if fg_buy >= fg_sell:
        return -1000000
    if rsi_buy >= rsi_sell:
        return -1000000

    # NEW CONSTRAINT: No single weight can exceed threshold magnitude (2.0)
    THRESHOLD_MAGNITUDE = 2.0
    if w_200w > THRESHOLD_MAGNITUDE or w_50w > THRESHOLD_MAGNITUDE or w_fg > THRESHOLD_MAGNITUDE or w_rsi > THRESHOLD_MAGNITUDE:
        return -1000000

    # Calculate indicators
    df = calculate_all_indicators(df_base, fg_df, params)

    # Simulate
    cash, btc, trades = 1000.0, 0.0, []

    for _, row in df.iterrows():
        if pd.isna(row['sma_ratio']) or pd.isna(row['fg_ema']) or pd.isna(row['rsi']) or pd.isna(row['ma50w_ratio']):
            continue

        years = row['time'].year - start_year

        # Get binary signals with DUAL DECAY
        decay_buy = sma_decay_buy ** years
        decay_sell = sma_decay_sell ** years
        adj_buy = sma_buy * decay_buy
        adj_sell = sma_sell * decay_sell
        sig_sma = 1 if row['sma_ratio'] < adj_buy else (-1 if row['sma_ratio'] > adj_sell else 0)

        if row['regime'] == 'bull':
            sig_ma50w = -1 if row['ma50w_ratio'] > bull_ext else (1 if row['ma50w_ratio'] < bull_supp else 0)
        else:
            sig_ma50w = -1 if row['ma50w_ratio'] > bear_res else (1 if row['ma50w_ratio'] < bear_deep else 0)

        sig_fg = 1 if row['fg_ema'] < fg_buy else (-1 if row['fg_ema'] > fg_sell else 0)
        sig_rsi = 1 if row['rsi'] < rsi_buy else (-1 if row['rsi'] > rsi_sell else 0)

        # Apply weights
        combined = (sig_sma * w_200w) + (sig_ma50w * w_50w) + (sig_fg * w_fg) + (sig_rsi * w_rsi)

        if combined >= 2:
            amt = cash * 0.01
            if amt > 0:
                btc += amt / row['close']
                cash -= amt
                trades.append(('buy', row['time'], row['close']))
        elif combined <= -2:
            amt = btc * 0.01
            if amt > 0:
                cash += amt * row['close']
                btc -= amt
                trades.append(('sell', row['time'], row['close']))

    value = cash + (btc * df.iloc[-1]['close'])

    # Penalties for key periods
    penalty = 0
    april_buys = [t for t in trades if t[0] == 'buy' and t[1].year == 2025 and t[1].month == 4]
    if len(april_buys) < 5:
        penalty += 5000

    feb_buys = [t for t in trades if t[0] == 'buy' and t[1] >= pd.Timestamp('2026-02-01')]
    if len(feb_buys) < 3:
        penalty += 5000

    peak_sells = [t for t in trades if t[0] == 'sell' and 2024 <= t[1].year <= 2025]
    if len(peak_sells) < 50:
        penalty += 2000

    return value - penalty

def run_stage(df, fg_df, start_year, stage_num, bounds, initial_guess=None):
    """Run a single optimization stage"""
    print(f"\n{'='*70}")
    print(f"STAGE {stage_num}")
    print(f"{'='*70}")

    def de_obj(params):
        return -objective_function(params, df, fg_df, start_year)

    result = differential_evolution(
        de_obj, bounds, x0=initial_guess,
        maxiter=15, popsize=12,
        tol=0.01, atol=0.01, seed=42+stage_num, disp=True, workers=1
    )

    best_value = -result.fun
    best_params = result.x

    print(f"\n[STAGE {stage_num} COMPLETE] Best value: ${best_value:,.0f}")
    return best_params, best_value

def create_dashboard(df, signals, trades, params_dict, value, filename):
    """Create visualization"""
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.25)

    weights = params_dict['weights']
    weight_str = f"200W={weights['w_200w']:.2f}, 50W={weights['w_50w']:.2f}, F&G={weights['w_fg']:.2f}, RSI={weights['w_rsi']:.2f}"
    fig.suptitle(f'5-Stage Optimized: ${value:,.0f} ({((value-1000)/1000)*100:.0f}% Return) | {weight_str}',
                 fontsize=18, fontweight='bold')

    # Panel 1
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['time'], df['close'], label='BTC Price', color='#2962FF', linewidth=1.5, alpha=0.8)
    ax1.plot(df['time'], df['sma_200w'], label='200W SMA', color='#FF6D00', linewidth=2, linestyle='--', alpha=0.6)
    ax1.plot(df['time'], df['ma_50w'], label='50W MA', color='#00897B', linewidth=2, linestyle='--', alpha=0.6)

    buy_trades = [t for t in trades if t[0] == 'buy']
    sell_trades = [t for t in trades if t[0] == 'sell']

    if buy_trades:
        ax1.scatter([t[1] for t in buy_trades], [t[2] for t in buy_trades],
                   color='green', s=60, marker='^', label=f'Buy ({len(buy_trades)})', zorder=5, alpha=0.8)
    if sell_trades:
        ax1.scatter([t[1] for t in sell_trades], [t[2] for t in sell_trades],
                   color='red', s=60, marker='v', label=f'Sell ({len(sell_trades)})', zorder=5, alpha=0.8)

    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('5-Stage Optimized: 18 Parameters (Weight Constraint: Max 2.0)', fontsize=14, pad=10)

    # Panel 2
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(df['time'], signals['sma'], label=f"200W (w={weights['w_200w']:.2f})", linewidth=1.5, alpha=0.7, color='#FF6D00')
    ax2.plot(df['time'], signals['ma50w'], label=f"50W (w={weights['w_50w']:.2f})", linewidth=1.5, alpha=0.7, color='#00897B')
    ax2.plot(df['time'], signals['fg'], label=f"F&G (w={weights['w_fg']:.2f})", linewidth=1.5, alpha=0.7, color='#9C27B0')
    ax2.plot(df['time'], signals['rsi'], label=f"RSI (w={weights['w_rsi']:.2f})", linewidth=1.5, alpha=0.7, color='#D32F2F')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Weighted Signal', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Individual Weighted Signals', fontsize=12, pad=8)

    # Panel 3
    ax3 = fig.add_subplot(gs[2])
    colors = ['red' if s <= -2 else 'green' if s >= 2 else 'gray' for s in signals['combined']]
    ax3.scatter(df['time'], signals['combined'], c=colors, s=10, alpha=0.6)
    ax3.plot(df['time'], signals['combined'], color='#9C27B0', linewidth=2, alpha=0.7)
    ax3.axhline(y=2, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Buy Threshold')
    ax3.axhline(y=-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Sell Threshold')
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Combined', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Combined Weighted Signal', fontsize=12, pad=8)

    # Panel 4
    ax4 = fig.add_subplot(gs[3])
    for i in range(len(df)-1):
        color = 'green' if df['regime'].iloc[i] == 'bull' else 'red'
        ax4.axvspan(df['time'].iloc[i], df['time'].iloc[i+1], alpha=0.3, color=color)
    ax4.plot(df['time'], (df['regime'] == 'bull').astype(int), color='black', linewidth=2)
    ax4.set_ylabel('Regime', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Bear', 'Bull'])
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_title('Market Regime', fontsize=12, pad=8)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Results text
    p = params_dict
    textstr = f'5-STAGE OPTIMIZED:\\n'
    textstr += f'Value: ${value:,.0f}\\n'
    textstr += f'Return: {((value-1000)/1000)*100:.0f}%\\n'
    textstr += f'Trades: {len(buy_trades)}/{len(sell_trades)}\\n\\n'
    textstr += f'WEIGHTS:\\n'
    textstr += f"200W: {weights['w_200w']:.2f}\\n"
    textstr += f"50W: {weights['w_50w']:.2f}\\n"
    textstr += f"F&G: {weights['w_fg']:.2f}\\n"
    textstr += f"RSI: {weights['w_rsi']:.2f}"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.95)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("\n" + "="*70)
    print("5-STAGE MULTI-STAGE GRADIENT OPTIMIZATION - DUAL DECAY CONSTRAINED")
    print("18 parameters with WEIGHT CONSTRAINT: No weight > 2.0 (threshold magnitude)")
    print("="*70 + "\n")

    df = fetch_btc_data()
    fg_df = fetch_fg_data()
    start_year = df['time'].iloc[0].year
    print(f"[INFO] Loaded {len(df)} days\n")

    # Define initial wide bounds
    # [sma_buy, sma_sell, sma_decay_buy, sma_decay_sell, bull_ext, bull_supp, bear_res, bear_deep,
    #  fg_period, fg_buy, fg_sell, rsi_period, rsi_buy, rsi_sell,
    #  w_200w, w_50w, w_fg, w_rsi]

    initial_bounds = [
        (0.8, 2.0),    # sma_buy
        (2.0, 4.0),    # sma_sell
        (0.92, 0.99),  # sma_decay_buy (slower decay hypothesis)
        (0.88, 0.96),  # sma_decay_sell (faster decay hypothesis)
        (1.3, 2.5),    # bull_ext
        (0.9, 1.4),    # bull_supp
        (0.6, 1.3),    # bear_res
        (0.4, 0.9),    # bear_deep
        (7, 50),       # fg_period
        (15, 50),      # fg_buy
        (50, 80),      # fg_sell
        (7, 30),       # rsi_period
        (15, 70),      # rsi_buy
        (60, 110),     # rsi_sell
        (0.5, 2.0),    # w_200w - CONSTRAINED to max 2.0
        (0.5, 2.0),    # w_50w - CONSTRAINED to max 2.0
        (0.5, 2.0),    # w_fg - CONSTRAINED to max 2.0
        (0.5, 2.0),    # w_rsi - CONSTRAINED to max 2.0
    ]

    # Stage 1: Broad exploration
    best_params, best_value = run_stage(df, fg_df, start_year, 1, initial_bounds)

    # Stages 2-5: Progressive narrowing
    for stage in range(2, 6):
        # Narrow bounds around best params
        narrowing_factor = 0.5 ** (stage - 1)  # 0.5, 0.25, 0.125, 0.0625

        new_bounds = []
        for i, (param, (low, high)) in enumerate(zip(best_params, initial_bounds)):
            param_range = high - low
            new_range = param_range * narrowing_factor
            new_low = max(low, param - new_range/2)
            new_high = min(high, param + new_range/2)
            new_bounds.append((new_low, new_high))

        best_params, best_value = run_stage(df, fg_df, start_year, stage, new_bounds, best_params)

    # Final simulation with best parameters
    print(f"\n{'='*70}")
    print("FINAL SIMULATION")
    print(f"{'='*70}")

    df_final = calculate_all_indicators(df, fg_df, best_params)

    # Extract signals for visualization
    start_year = df_final['time'].iloc[0].year
    sma_buy, sma_sell, sma_decay_buy, sma_decay_sell = best_params[0:4]
    bull_ext, bull_supp, bear_res, bear_deep = best_params[4:8]
    fg_buy, fg_sell = best_params[9:11]
    rsi_buy, rsi_sell = best_params[12:14]
    w_200w, w_50w, w_fg, w_rsi = best_params[14:18]

    cash, btc, trades = 1000.0, 0.0, []
    signals_list = []

    for _, row in df_final.iterrows():
        if pd.isna(row['sma_ratio']) or pd.isna(row['fg_ema']) or pd.isna(row['rsi']) or pd.isna(row['ma50w_ratio']):
            signals_list.append({'sma': 0, 'ma50w': 0, 'fg': 0, 'rsi': 0, 'combined': 0})
            continue

        years = row['time'].year - start_year
        decay_buy = sma_decay_buy ** years
        decay_sell = sma_decay_sell ** years
        adj_buy = sma_buy * decay_buy
        adj_sell = sma_sell * decay_sell
        sig_sma = 1 if row['sma_ratio'] < adj_buy else (-1 if row['sma_ratio'] > adj_sell else 0)

        if row['regime'] == 'bull':
            sig_ma50w = -1 if row['ma50w_ratio'] > bull_ext else (1 if row['ma50w_ratio'] < bull_supp else 0)
        else:
            sig_ma50w = -1 if row['ma50w_ratio'] > bear_res else (1 if row['ma50w_ratio'] < bear_deep else 0)

        sig_fg = 1 if row['fg_ema'] < fg_buy else (-1 if row['fg_ema'] > fg_sell else 0)
        sig_rsi = 1 if row['rsi'] < rsi_buy else (-1 if row['rsi'] > rsi_sell else 0)

        combined = (sig_sma * w_200w) + (sig_ma50w * w_50w) + (sig_fg * w_fg) + (sig_rsi * w_rsi)

        signals_list.append({
            'sma': sig_sma * w_200w,
            'ma50w': sig_ma50w * w_50w,
            'fg': sig_fg * w_fg,
            'rsi': sig_rsi * w_rsi,
            'combined': combined
        })

        if combined >= 2:
            amt = cash * 0.01
            if amt > 0:
                btc += amt / row['close']
                cash -= amt
                trades.append(('buy', row['time'], row['close']))
        elif combined <= -2:
            amt = btc * 0.01
            if amt > 0:
                cash += amt * row['close']
                btc -= amt
                trades.append(('sell', row['time'], row['close']))

    final_value = cash + (btc * df_final.iloc[-1]['close'])

    buy_trades = [t for t in trades if t[0] == 'buy']
    sell_trades = [t for t in trades if t[0] == 'sell']
    april_buys = [t for t in buy_trades if t[1].year == 2025 and t[1].month == 4]
    feb_buys = [t for t in buy_trades if t[1] >= pd.Timestamp('2026-02-01')]
    peak_sells = [t for t in sell_trades if 2024 <= t[1].year <= 2025]

    print(f"\nFinal Value: ${final_value:,.0f}")
    print(f"Total Return: {((final_value-1000)/1000)*100:.0f}%")
    print(f"Total Trades: {len(trades)} ({len(buy_trades)} buys / {len(sell_trades)} sells)")
    print(f"\nKey Periods:")
    print(f"  April 2025 buys: {len(april_buys)}")
    print(f"  Feb 2026 buys: {len(feb_buys)}")
    print(f"  2024-2025 sells: {len(peak_sells)}")

    print(f"\nOptimized Parameters:")
    print(f"  200W SMA: buy={best_params[0]:.3f}, sell={best_params[1]:.3f}, decay_buy={best_params[2]:.4f}, decay_sell={best_params[3]:.4f}")
    print(f"  50W MA: bull_ext={best_params[4]:.3f}, bull_supp={best_params[5]:.3f}, bear_res={best_params[6]:.3f}, bear_deep={best_params[7]:.3f}")
    print(f"  F&G: period={int(best_params[8])}, buy={best_params[9]:.1f}, sell={best_params[10]:.1f}")
    print(f"  RSI: period={int(best_params[11])}, buy={best_params[12]:.1f}, sell={best_params[13]:.1f}")
    print(f"  Weights: 200W={best_params[14]:.2f}, 50W={best_params[15]:.2f}, F&G={best_params[16]:.2f}, RSI={best_params[17]:.2f}")

    # Save configuration
    params_dict = {
        'parameters': {
            'sma_200w': {'buy': best_params[0], 'sell': best_params[1],
                        'decay_buy': best_params[2], 'decay_sell': best_params[3],
                        'decay': best_params[3]},  # Keep decay for backward compatibility
            'ma_50w': {'bull_ext': best_params[4], 'bull_supp': best_params[5],
                      'bear_res': best_params[6], 'bear_deep': best_params[7]},
            'fg': {'ema_period': int(best_params[8]), 'buy': best_params[9], 'sell': best_params[10]},
            'rsi': {'period': int(best_params[11]), 'buy': best_params[12], 'sell': best_params[13]}
        },
        'weights': {
            'w_200w': best_params[14],
            'w_50w': best_params[15],
            'w_fg': best_params[16],
            'w_rsi': best_params[17]
        },
        'final_value': final_value,
        'total_return': ((final_value - 1000) / 1000) * 100,
        'system': '5-stage optimization with DUAL DECAY (18 params, weight constraint: max 2.0)'
    }

    with open('constrained_5stage_config.json', 'w') as f:
        json.dump(params_dict, f, indent=2)

    print("\n[INFO] Creating dashboard...")
    signals_df = pd.DataFrame(signals_list)
    create_dashboard(df_final, signals_df, trades, params_dict, final_value, 'constrained_5stage_dashboard.png')
    print("[OK] Saved to constrained_5stage_dashboard.png")
    print("[OK] Saved configuration to constrained_5stage_config.json")

    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()
