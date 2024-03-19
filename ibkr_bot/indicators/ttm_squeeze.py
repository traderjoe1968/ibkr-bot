
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Final




col_grow_above = 'rgba(34,158,131,0.8)'
col_fall_above = 'rgba(104,113,105,1.0)'
col_grow_below = 'rgba(201,96,100,0.8)'
col_fall_below = 'rgba(251,116,110,1.0)'

col_HiSqz  = 'rgba(253,173,107,1.0)'
col_MidSqz = 'rgba(251,116,110,1.0)'
col_LowSqz = 'rgba(34,158,131,0.8)'
col_NoSqz  = '#9e9e9e'


def _color(x, x_1):
    if np.isnan(x):
        return col_fall_above
    if x >= 0:
        if x > x_1:
            return col_grow_above
        else:
            return col_fall_above
    else:
        if x > x_1:
            return col_grow_below
        else:
            return col_fall_below


def ttm_squeeze(df:pd.DataFrame, window_size:int=10) -> pd.DataFrame:
    """
    Calculate the TTM Squeeze for a given ticker.
    """
    if len(df) < window_size:
        raise ValueError(f"Not enough data to calculate TTM Squeeze.  Need at least {window_size} data points.")
    ttm = pd.DataFrame(index=df.index)

    # BOLLINGER BANDS
    BBmult: Final[float] = 2.0
    ttm['ttm_sma'] = ta.sma(df['Close'],length=window_size)

    bb = ta.bbands(df['Close'], length=window_size, std=2.0)
    ttm['ttm_lower_band'] = bb[f"BBL_{window_size}_{BBmult}"]
    ttm['ttm_upper_band'] = bb[f"BBU_{window_size}_{BBmult}"]
    # KELTNER CHANNELS
    KC_mult_high: Final[float] = 1.0
    KC_mult_mid: Final[float] = 1.5
    KC_mult_low: Final[float] = 2.0
    # ttm['tr']  = ta.true_range(df['High'],df['Low'],df['Close'])
    ttm['ttm_atr'] = ta.atr(df['High'],df['Low'],df['Close'],lenght=window_size)

    ttm['ttm_KC_upper_high'] = ttm['ttm_sma'] + ttm['ttm_atr'] * KC_mult_high
    ttm['ttm_KC_lower_high'] = ttm['ttm_sma'] - ttm['ttm_atr'] * KC_mult_high
    ttm['ttm_KC_upper_mid']  = ttm['ttm_sma'] + ttm['ttm_atr'] * KC_mult_mid
    ttm['ttm_KC_lower_mid']  = ttm['ttm_sma'] - ttm['ttm_atr'] * KC_mult_mid
    ttm['ttm_KC_upper_low']  = ttm['ttm_sma'] + ttm['ttm_atr'] * KC_mult_low
    ttm['ttm_KC_lower_low']  = ttm['ttm_sma'] - ttm['ttm_atr'] * KC_mult_low
    # SQUEEZE CONDITIONS
    NoSqz   = (ttm['ttm_lower_band'] <  ttm['ttm_KC_lower_low'])  | (ttm['ttm_upper_band'] >  ttm['ttm_KC_upper_low'])  # NO SQUEEZE: GREEN
    LowSqz  = (ttm['ttm_lower_band'] >= ttm['ttm_KC_lower_low'])  | (ttm['ttm_upper_band'] <= ttm['ttm_KC_upper_low'])  # LOW COMPRESSION: BLACK
    MidSqz  = (ttm['ttm_lower_band'] >= ttm['ttm_KC_lower_mid'])  | (ttm['ttm_upper_band'] <= ttm['ttm_KC_upper_mid'])  # MID COMPRESSION: RED
    HighSqz = (ttm['ttm_lower_band'] >= ttm['ttm_KC_lower_high']) | (ttm['ttm_upper_band'] <= ttm['ttm_KC_upper_high']) # HIGH COMPRESSION: ORANGE
    # MOMENTUM OSCILLATOR
    ttm['ttm_mom'] = ta.mom(ttm['ttm_sma'], length=window_size)

    # MOMENTUM HISTOGRAM COLOR
    ttm['ttm_mom_1'] = ttm['ttm_mom'].shift(1, fill_value=0)
    ttm['ttm_mom_color'] = ttm.apply(lambda x:_color(x['ttm_mom'], x['ttm_mom_1']), axis=1)

    # SQUEEZE DOTS COLOR
    conditions = [
        (HighSqz),
        (MidSqz),
        (LowSqz),
        (NoSqz),
    ]
    choices = [col_HiSqz, col_MidSqz, col_LowSqz, col_NoSqz]
    ttm['ttm_sq_line'] = 0
    ttm['ttm_sq_color'] = np.select(conditions, choices, default=col_NoSqz)

    return ttm