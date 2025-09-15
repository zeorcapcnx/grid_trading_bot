import ccxt
import pandas as pd
from datetime import datetime

# Create exchange instance
exchange = ccxt.binance()

# Fetch OHLCV data for the same period as config3
since = exchange.parse8601('2023-09-01T09:00:00Z')
until = exchange.parse8601('2024-03-01T09:00:00Z')

print('Fetching SOL/USDT price data for 2023-09-01 to 2024-03-01...')
data = []
current = since
while current < until:
    try:
        ohlcv = exchange.fetch_ohlcv('SOL/USDT', '1d', since=current, limit=500)
        if not ohlcv:
            break
        data.extend(ohlcv)
        current = ohlcv[-1][0] + 86400000  # Add 24 hours in ms
        if len(ohlcv) < 500:
            break
    except Exception as e:
        print(f'Error: {e}')
        break

if data:
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Filter for the exact period
    mask = (df['date'] >= '2023-09-01') & (df['date'] <= '2024-03-01')
    period_data = df[mask]

    print(f'Period: {period_data.date.min()} to {period_data.date.max()}')
    print(f'Starting price: ${period_data.open.iloc[0]:.2f}')
    print(f'Ending price: ${period_data.close.iloc[-1]:.2f}')
    print(f'Highest price: ${period_data.high.max():.2f}')
    print(f'Lowest price: ${period_data.low.min():.2f}')
    print(f'')
    print(f'Crypto_zero calculation for starting price ${period_data.open.iloc[0]:.2f}:')
    start_price = period_data.open.iloc[0]
    bottom = start_price / 5
    top = start_price + (start_price - bottom)
    print(f'  Bottom: ${bottom:.2f}')
    print(f'  Top (TP threshold): ${top:.2f}')
    print(f'')
    print(f'Did price exceed TP threshold?')
    print(f'  Max price: ${period_data.high.max():.2f}')
    print(f'  TP threshold: ${top:.2f}')
    print(f'  Exceeded: {period_data.high.max() > top}')
else:
    print('No data retrieved')