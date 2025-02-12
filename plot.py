# plot_signals.py
import mplfinance as mpf
import pandas as pd
import numpy as np

def plot_candlestick_signals(df_plot, title="Candlestick Chart with Buy/Sell Signals"):
    """
    Plots a candlestick chart of df_plot with buy/sell markers overlaid.
    
    df_plot must have:
      - A DateTime index
      - Columns: Open, High, Low, Close, Volume (optionally)
      - A 'Signal' column with +1 (buy), -1 (sell), 0 (none)
    """

    # Ensure df_plot has a DatetimeIndex
    df_plot = df_plot.copy()
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        df_plot.index = pd.to_datetime(df_plot.index)

    # Create columns for buy/sell booleans
    df_plot['Buy'] = (df_plot['Signal'] == 1)
    df_plot['Sell'] = (df_plot['Signal'] == -1)

    buy_locs = df_plot.index[df_plot['Buy']]
    sell_locs = df_plot.index[df_plot['Sell']]

    buy_prices = df_plot.loc[buy_locs, 'Close']
    sell_prices = df_plot.loc[sell_locs, 'Close']

    df_plot['BuyPrices'] = np.nan
    df_plot.loc[df_plot['Signal'] == 1, 'BuyPrices'] = df_plot.loc[df_plot['Signal'] == 1, 'Close']

    df_plot['SellPrices'] = np.nan
    df_plot.loc[df_plot['Signal'] == -1, 'SellPrices'] = df_plot.loc[df_plot['Signal'] == -1, 'Close']


    buy_addplot = mpf.make_addplot(
        df_plot['BuyPrices'],
        type='scatter',
        marker='^',
        color='green'
        )

    sell_addplot = mpf.make_addplot(
        df_plot['SellPrices'],
        type='scatter',
        marker='v',
        color='red'
    )

    mpf.plot(
        df_plot,
        type='line',
        style='yahoo',
        addplot=[buy_addplot, sell_addplot],
        volume=True,
        title="Candlestick with Buy/Sell Overlays"
    )
