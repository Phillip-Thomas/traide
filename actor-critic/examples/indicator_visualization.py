"""Example of indicator visualization with AAPL data."""

import pandas as pd
from datetime import datetime, timedelta

from marketdata.providers import YFinanceProvider
from marketdata.visualization.components import (
    ChartDimensions,
    ChartScales,
    CandlestickChart,
    CandlestickStyle
)
from marketdata.visualization.utils import CairoRenderer
from marketdata.indicators import (
    BollingerBands,
    RelativeStrengthIndex,
    MovingAverageConvergenceDivergence,
    SimpleMovingAverage
)

# Set up data provider and fetch data
provider = YFinanceProvider()
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
data = provider.fetch_historical_data(
    symbol='AAPL',
    start_date=start_date,
    end_date=end_date,
    timeframe='15m'
)

# Create chart
dimensions = ChartDimensions(
    width=1200,
    height=800,
    margin_top=20,
    margin_right=50,
    margin_bottom=30,
    margin_left=50
)

scales = ChartScales(
    x_scale_type='time',
    y_scale_type='linear',
    price_format='.2f',
    time_format='%Y-%m-%d %H:%M',
    show_grid=True,
    show_crosshair=True
)

style = CandlestickStyle(
    up_color=(0.0, 0.8, 0.0),  # Green
    down_color=(0.8, 0.0, 0.0),  # Red
    wick_width=1.0,
    body_width=8.0,
    hollow_up=False,
    hollow_down=False
)

chart = CandlestickChart(dimensions, scales, style)
chart.set_data(data)

# Calculate indicators
# 1. Bollinger Bands
bbands = BollingerBands()
bbands_result = bbands.calculate(data)

# 2. RSI
rsi = RelativeStrengthIndex()
rsi_result = rsi.calculate(data)

# 3. MACD
macd = MovingAverageConvergenceDivergence()
macd_result = macd.calculate(data)

# 4. 20-period SMA
sma = SimpleMovingAverage()
sma_result = sma.calculate(data)

# Create renderer and render chart
renderer = CairoRenderer(dimensions.width, dimensions.height)

# Draw main chart
chart.render(renderer)

# Draw indicators
# TODO: Add indicator overlay rendering once implemented

# Save chart
renderer.save('aapl_chart.png')

print("Chart saved as 'aapl_chart.png'") 