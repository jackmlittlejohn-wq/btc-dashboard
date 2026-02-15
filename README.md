# BTC/USD Advanced Dashboard 🚀

Professional Bitcoin price visualization with real-time updates, technical indicators, and market sentiment analysis.

## Features

### 📊 Chart Options
- **Multiple Timeframes**: 1Hr, 1D, 1W, 1M, 3M, 1Y, 5Y, All (back to 2012)
- **Chart Types**: Toggle between candlestick and line chart views
- **Granular Data**: Candle intervals automatically match timeframe for optimal visualization

### 📈 Technical Analysis
- **Configurable SMA Overlay**: Simple Moving Average with adjustable period (default 200)
- **Toggle Controls**: Show/hide SMA on the fly
- **Left Axis Integration**: SMA plotted on the same scale as price data

### 😱 Market Sentiment
- **Fear & Greed Index**: Live data from alternative.me
- **Separate Right Axis**: 0-100 scale for easy reading
- **Aggregation Options**: Daily, weekly, or monthly averaging
- **Toggle Controls**: Show/hide Fear & Greed overlay

### 🔄 Live Updates
- **Auto-Refresh**: Chart updates every 30 seconds automatically
- **No Page Reload**: Seamless data updates without browser refresh
- **Real-Time Status**: See last update time in the interface

### 🎯 User Experience
- **Interactive Controls**: All features controllable via UI buttons and toggles
- **Zoom & Pan**: Full Plotly interactivity built-in
- **Dark Theme**: Easy on the eyes for long analysis sessions
- **Responsive Design**: Adapts to different screen sizes

## Installation

1. Install required dependencies:
```bash
cd btc-dashboard
pip install -r requirements.txt
```

## Usage

### Option 1: Desktop Launcher (Easiest)
Simply double-click **"Launch BTC Dashboard.bat"** on your desktop.

### Option 2: Command Line
```bash
cd ~/Desktop/btc-dashboard
python btc_dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:5000`

## Controls Guide

### Timeframe Selection
Click any timeframe button to instantly switch views:
- **1Hr**: Last hour (1-minute candles)
- **1D**: Last day (30-minute candles)
- **1W**: Last week (3-hour candles)
- **1M**: Last month (6-hour candles)
- **3M**: Last 3 months (daily candles)
- **1Y**: Last year (3-day candles)
- **5Y**: Last 5 years (weekly candles)
- **All**: Complete history back to 2012 (weekly candles)

### Chart Type Toggle
- **Candlestick**: Full OHLC visualization with green/red candles
- **Line**: Simplified close price line chart

### SMA (Simple Moving Average)
- **Toggle Switch**: Turn SMA overlay on/off
- **Period Input**: Adjust SMA calculation period (1-1000)
- **Default**: 200-period SMA (popular for long-term trends)

### Fear & Greed Index
- **Toggle Switch**: Show/hide sentiment indicator
- **Period Dropdown**: Choose daily, weekly, or monthly averaging
- **Scale**: 0-100 (Extreme Fear to Extreme Greed)

## Data Sources

- **Price Data**: CoinGecko API (OHLC data back to 2012)
- **Fear & Greed**: Alternative.me Crypto Fear & Greed Index
- **Update Frequency**: Auto-refresh every 30 seconds

## Technical Details

### Candle Granularity by Timeframe
| Timeframe | Candle Interval | Data Points |
|-----------|----------------|-------------|
| 1Hr       | 1 minute       | ~60         |
| 1D        | 30 minutes     | ~48         |
| 1W        | 3 hours        | ~56         |
| 1M        | 6 hours        | ~120        |
| 3M        | 1 day          | ~90         |
| 1Y        | 3 days         | ~122        |
| 5Y        | 1 week         | ~260        |
| All       | 1 week         | ~650+       |

### Historical Coverage
- **2012-Present**: Complete Bitcoin price history
- **Fear & Greed**: Available data from alternative.me (typically last ~2 years)

## Requirements

- Python 3.7+
- Internet connection for API access
- Web browser (Chrome, Firefox, Edge, etc.)

## Keyboard Shortcuts

While the server is running in the terminal:
- `Ctrl+C`: Stop the dashboard server
- Browser refresh: Reload the page (maintains server state)

## Troubleshooting

**Server won't start?**
- Check if port 5000 is already in use
- Make sure all dependencies are installed

**No data showing?**
- Check your internet connection
- APIs may have rate limits - wait a moment and try again

**SMA not appearing?**
- Ensure enough data points exist for the selected period
- Try reducing the SMA period for shorter timeframes

## Advanced Usage

The dashboard runs a Flask web server locally. You can:
- Keep it running in the background
- Access from other devices on your network using your PC's IP
- Multiple browser tabs will share the same data source

---

Built with ❤️ using Python, Flask, Plotly, and multiple financial data APIs.
