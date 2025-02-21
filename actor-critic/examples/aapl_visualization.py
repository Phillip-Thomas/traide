"""Example script demonstrating technical analysis visualization with AAPL data."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cairo

from marketdata.visualization.components import (
    BaseChart,
    ChartDimensions,
    ChartScales,
    CandlestickChart,
    CandlestickStyle,
    BarChart,
    BarStyle
)
from marketdata.visualization.overlays import (
    BandOverlay,
    BandStyle,
    OscillatorOverlay,
    OscillatorStyle,
    OscillatorLevel,
    LineOverlay,
    LineStyle,
    HistogramOverlay,
    HistogramStyle,
    OverlayManager
)
from marketdata.visualization.utils import CairoRenderer
from marketdata.indicators import (
    BollingerBands,
    RelativeStrengthIndex,
    MovingAverageConvergenceDivergence
)


def fetch_aapl_data() -> pd.DataFrame:
    """Create mock AAPL data for testing.
    
    Returns:
        DataFrame with OHLCV data
    """
    print("Fetching AAPL data...")
    
    # Create date range for the last 30 days with 15-minute intervals
    end_date = pd.Timestamp.now().floor('15min')
    start_date = end_date - pd.Timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='15min')
    
    # Generate mock price data
    base_price = 150.0
    trend = np.linspace(-10, 10, len(dates))
    noise = np.random.randn(len(dates)) * 2
    
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(len(dates))) * 2
    low = close - np.abs(np.random.randn(len(dates))) * 2
    open_ = close + np.random.randn(len(dates)) * 2
    
    # Ensure OHLC relationships are valid
    high = np.maximum.reduce([high, close, open_])
    low = np.minimum.reduce([low, close, open_])
    
    # Generate mock volume data
    volume = np.random.uniform(1000000, 2000000, len(dates))
    volume = volume * (1 + trend/20)  # Volume tends to be higher in uptrends
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Set timestamp as index
    data.set_index('timestamp', inplace=True)
    
    print("Data fetched successfully")
    return data


def create_chart(data: pd.DataFrame) -> CandlestickChart:
    """Create and configure the main chart.
    
    Args:
        data: OHLCV data
    
    Returns:
        Configured candlestick chart
    """
    # Reset index to make timestamp a column
    data = data.reset_index()
    
    # Create chart dimensions
    dimensions = ChartDimensions(
        width=1200,
        height=800,
        margin_top=20,
        margin_right=50,
        margin_bottom=30,
        margin_left=50
    )
    
    # Configure scales
    scales = ChartScales(
        x_scale_type='time',
        y_scale_type='linear',
        price_format='.2f',
        time_format='%Y-%m-%d %H:%M',
        show_grid=True,
        show_crosshair=True
    )
    
    # Configure candlestick style
    style = CandlestickStyle(
        up_color=(0.0, 0.8, 0.0),  # Green
        down_color=(0.8, 0.0, 0.0),  # Red
        wick_width=1.0,
        body_width=8.0,
        hollow_up=False,
        hollow_down=False
    )
    
    # Create and initialize chart
    chart = CandlestickChart(dimensions, scales, style)
    chart.set_data(data)
    
    return chart


def add_bollinger_bands(chart: BaseChart, data: pd.DataFrame) -> None:
    """Add Bollinger Bands overlay to the chart.
    
    Args:
        chart: The chart to add to
        data: OHLCV data
    """
    # Calculate Bollinger Bands
    bbands = BollingerBands()
    bbands_result = bbands.calculate(data)
    
    # Configure band style
    style = BandStyle(
        color=(0.2, 0.6, 1.0),  # Blue
        line_width=1.0,
        line_style='dashed',
        fill_opacity=0.1,
        smooth=True,
        label='Bollinger Bands'
    )
    
    # Create and add overlay
    overlay = BandOverlay(bbands, style)
    chart.overlay_manager.add_overlay(overlay)


def add_rsi(chart: BaseChart, data: pd.DataFrame) -> None:
    """Add RSI overlay to the chart.
    
    Args:
        chart: The chart to add to
        data: OHLCV data
    """
    # Calculate RSI
    rsi = RelativeStrengthIndex()
    rsi_result = rsi.calculate(data)
    
    # Configure oscillator style
    style = OscillatorStyle(
        color=(0.8, 0.4, 0.0),  # Orange
        line_width=1.5,
        smooth=True,
        fill_opacity=0.1,
        label='RSI (14)',
        levels=[
            OscillatorLevel(70, (0.8, 0.0, 0.0), label='Overbought'),
            OscillatorLevel(30, (0.0, 0.8, 0.0), label='Oversold')
        ]
    )
    
    # Create and add overlay
    overlay = OscillatorOverlay(rsi, style)
    chart.overlay_manager.add_overlay(overlay)


def add_macd(chart: BaseChart, data: pd.DataFrame) -> None:
    """Add MACD overlays to the chart.
    
    Args:
        chart: The chart to add to
        data: OHLCV data
    """
    # Calculate MACD
    macd = MovingAverageConvergenceDivergence()
    macd_result = macd.calculate(data)
    
    # Configure MACD line style
    macd_style = LineStyle(
        color=(0.2, 0.6, 1.0),  # Blue
        line_width=1.5,
        smooth=True,
        label='MACD',
        value_column='macd'
    )
    
    # Configure signal line style
    signal_style = LineStyle(
        color=(1.0, 0.4, 0.0),  # Orange
        line_width=1.5,
        smooth=True,
        label='Signal',
        value_column='signal'
    )
    
    # Configure histogram style
    histogram_style = HistogramStyle(
        positive_color=(0.0, 0.8, 0.0),  # Green
        negative_color=(0.8, 0.0, 0.0),  # Red
        bar_width=0.8,
        label='MACD Histogram',
        value_column='histogram'
    )
    
    # Create and add overlays
    macd_overlay = LineOverlay(macd, macd_style)
    signal_overlay = LineOverlay(macd, signal_style)
    histogram_overlay = HistogramOverlay(macd, histogram_style)
    
    chart.overlay_manager.add_overlay(macd_overlay)
    chart.overlay_manager.add_overlay(signal_overlay)
    chart.overlay_manager.add_overlay(histogram_overlay)


def add_volume(chart: BaseChart, data: pd.DataFrame) -> None:
    """Add volume bars to the chart.
    
    Args:
        chart: The chart to add to
        data: OHLCV data
    """
    # Reset index to make timestamp a column
    data = data.reset_index()
    
    # Configure volume bar style
    style = BarStyle(
        up_color=(0.0, 0.8, 0.0, 0.6),  # Green with alpha
        down_color=(0.8, 0.0, 0.0, 0.6),  # Red with alpha
        neutral_color=(0.5, 0.5, 0.5, 0.6),  # Gray with alpha
        bar_width=0.8,
        bar_spacing=0.2
    )
    
    # Create volume chart
    volume_chart = BarChart(chart.dimensions, chart.scales, style)
    volume_chart.set_data(data)
    
    # Add to main chart
    chart.add_subchart(volume_chart, height_ratio=0.2)


def main():
    """Run the AAPL visualization example."""
    # Fetch data
    print("Fetching AAPL data...")
    data = fetch_aapl_data()
    
    # Create main chart
    print("Creating chart...")
    chart = create_chart(data)
    
    # Add indicators
    print("Adding indicators...")
    add_bollinger_bands(chart, data)
    add_rsi(chart, data)
    add_macd(chart, data)
    add_volume(chart, data)
    
    # Create surface and renderer
    print("Rendering chart...")
    surface = cairo.ImageSurface(
        cairo.FORMAT_ARGB32,
        int(chart.dimensions.width),
        int(chart.dimensions.height + sum(
            subchart.dimensions.height + subchart.dimensions.margin_bottom
            for subchart, _ in chart._subcharts
        ))
    )
    renderer = CairoRenderer(surface)
    
    # Render chart
    chart.render(renderer)
    
    # Save to file
    print("Saving chart...")
    surface.write_to_png('aapl_chart.png')
    print("Chart saved as 'aapl_chart.png'")


if __name__ == '__main__':
    main() 