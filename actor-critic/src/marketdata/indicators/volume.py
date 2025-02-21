"""Volume-based technical indicators."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .base import (
    BaseIndicator,
    IndicatorParams,
    IndicatorResult,
    register_indicator
)


@dataclass
class OBVParams(IndicatorParams):
    """On-Balance Volume indicator parameters."""
    source: str = 'close'
    signal_period: int = 20
    min_periods: int = None


@register_indicator('obv')
class OnBalanceVolume(BaseIndicator):
    """On-Balance Volume (OBV) indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return OBVParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate OBV values.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: OBV values
        """
        params = self.params
        min_periods = params.min_periods if params.min_periods is not None else params.signal_period
        
        # Calculate price changes
        price_changes = data[params.source].diff()
        
        # Calculate OBV
        obv = pd.Series(0, index=data.index)
        obv[1:] = np.where(
            price_changes[1:] > 0,
            data['volume'][1:],
            np.where(
                price_changes[1:] < 0,
                -data['volume'][1:],
                0
            )
        ).cumsum()
        
        # Calculate signal line (SMA of OBV)
        signal = obv.rolling(
            window=params.signal_period,
            min_periods=min_periods
        ).mean()
        
        # Calculate histogram
        histogram = obv - signal
        
        # Set NaN values for initial periods
        if min_periods > 1:
            signal.iloc[:min_periods-1] = np.nan
            histogram.iloc[:min_periods-1] = np.nan
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'obv': obv,
            'signal': signal,
            'histogram': histogram
        })
        
        return IndicatorResult(
            values=result_df,
            metadata={
                'source': params.source,
                'signal_period': params.signal_period,
                'min_periods': min_periods
            }
        )


@dataclass
class VolumeProfileParams(IndicatorParams):
    """Volume Profile indicator parameters."""
    price_source: str = 'close'
    n_bins: int = 24
    window: int = None  # None for entire data range
    min_periods: int = None


@register_indicator('volume_profile')
class VolumeProfile(BaseIndicator):
    """Volume Profile indicator."""
    
    def _default_params(self) -> IndicatorParams:
        """Get default parameters."""
        return VolumeProfileParams()
    
    def _calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate Volume Profile.
        
        Args:
            data: Input data with OHLCV columns
            
        Returns:
            IndicatorResult: Volume Profile values
        """
        params = self.params
        min_periods = params.min_periods if params.min_periods is not None else 1
        
        # Initialize result DataFrame with same index as input
        result_df = pd.DataFrame(index=data.index)
        
        # Calculate profile for each point using rolling window if specified
        if params.window is not None:
            for i in range(len(data)):
                start_idx = max(0, i - params.window + 1)
                window_data = data.iloc[start_idx:i+1]
                
                if len(window_data) >= min_periods:
                    # Calculate price bins
                    price_range = (window_data[params.price_source].max(),
                                window_data[params.price_source].min())
                    bins = np.linspace(price_range[1], price_range[0], params.n_bins + 1)
                    
                    # Calculate volume distribution
                    hist, bin_edges = np.histogram(
                        window_data[params.price_source],
                        bins=bins,
                        weights=window_data['volume']
                    )
                    
                    # Calculate POC (Point of Control)
                    poc_idx = np.argmax(hist)
                    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
                    
                    # Calculate Value Area
                    total_volume = np.sum(hist)
                    value_area_volume = 0
                    value_area_bins = []
                    center_idx = poc_idx
                    
                    while value_area_volume < 0.68 * total_volume and len(value_area_bins) < len(hist):
                        if len(value_area_bins) == 0:
                            value_area_bins.append(center_idx)
                            value_area_volume += hist[center_idx]
                        else:
                            # Look above and below current value area
                            above_idx = max(value_area_bins) + 1
                            below_idx = min(value_area_bins) - 1
                            
                            above_vol = hist[above_idx] if above_idx < len(hist) else 0
                            below_vol = hist[below_idx] if below_idx >= 0 else 0
                            
                            if above_vol >= below_vol and above_idx < len(hist):
                                value_area_bins.append(above_idx)
                                value_area_volume += above_vol
                            elif below_idx >= 0:
                                value_area_bins.append(below_idx)
                                value_area_volume += below_vol
                    
                    # Calculate Value Area High/Low
                    vah = bin_edges[max(value_area_bins) + 1]
                    val = bin_edges[min(value_area_bins)]
                    
                    # Store results
                    result_df.loc[data.index[i], 'poc'] = poc_price
                    result_df.loc[data.index[i], 'vah'] = vah
                    result_df.loc[data.index[i], 'val'] = val
                    result_df.loc[data.index[i], 'volume_total'] = total_volume
                    
                    # Store volume distribution
                    for j, (vol, price) in enumerate(zip(hist, bin_edges[:-1])):
                        result_df.loc[data.index[i], f'vol_bin_{j}'] = vol
                        result_df.loc[data.index[i], f'price_bin_{j}'] = (price + bin_edges[j+1]) / 2
                else:
                    # Fill with NaN if not enough data
                    result_df.loc[data.index[i], :] = np.nan
        else:
            # Calculate for entire dataset
            price_range = (data[params.price_source].max(),
                         data[params.price_source].min())
            bins = np.linspace(price_range[1], price_range[0], params.n_bins + 1)
            
            hist, bin_edges = np.histogram(
                data[params.price_source],
                bins=bins,
                weights=data['volume']
            )
            
            # Calculate POC
            poc_idx = np.argmax(hist)
            poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
            
            # Calculate Value Area
            total_volume = np.sum(hist)
            value_area_volume = 0
            value_area_bins = []
            center_idx = poc_idx
            
            while value_area_volume < 0.68 * total_volume and len(value_area_bins) < len(hist):
                if len(value_area_bins) == 0:
                    value_area_bins.append(center_idx)
                    value_area_volume += hist[center_idx]
                else:
                    # Look above and below current value area
                    above_idx = max(value_area_bins) + 1
                    below_idx = min(value_area_bins) - 1
                    
                    above_vol = hist[above_idx] if above_idx < len(hist) else 0
                    below_vol = hist[below_idx] if below_idx >= 0 else 0
                    
                    if above_vol >= below_vol and above_idx < len(hist):
                        value_area_bins.append(above_idx)
                        value_area_volume += above_vol
                    elif below_idx >= 0:
                        value_area_bins.append(below_idx)
                        value_area_volume += below_vol
            
            # Calculate Value Area High/Low
            vah = bin_edges[max(value_area_bins) + 1]
            val = bin_edges[min(value_area_bins)]
            
            # Fill results for all points
            result_df['poc'] = poc_price
            result_df['vah'] = vah
            result_df['val'] = val
            result_df['volume_total'] = total_volume
            
            # Store volume distribution
            for i, (vol, price) in enumerate(zip(hist, bin_edges[:-1])):
                result_df[f'vol_bin_{i}'] = vol
                result_df[f'price_bin_{i}'] = (price + bin_edges[i+1]) / 2
        
        return IndicatorResult(
            values=result_df,
            metadata={
                'price_source': params.price_source,
                'n_bins': params.n_bins,
                'window': params.window,
                'min_periods': min_periods
            }
        ) 