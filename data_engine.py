import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return None
        
        # Handle multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def add_technical_indicators(df):
    """Add 15+ technical indicators to the dataframe."""
    # Ensure we have enough data
    if len(df) < 50:
        return df

    # Trend Indicators
    # Ensure inputs are 1D Series
    close_ser = df['Close'].iloc[:, 0] if len(df['Close'].shape) > 1 else df['Close']
    high_ser = df['High'].iloc[:, 0] if len(df['High'].shape) > 1 else df['High']
    low_ser = df['Low'].iloc[:, 0] if len(df['Low'].shape) > 1 else df['Low']
    vol_ser = df['Volume'].iloc[:, 0] if len(df['Volume'].shape) > 1 else df['Volume']
    
    # Convert to Series to be safe
    close_ser = pd.Series(close_ser).astype(float)
    high_ser = pd.Series(high_ser).astype(float)
    low_ser = pd.Series(low_ser).astype(float)
    vol_ser = pd.Series(vol_ser).astype(float)

    # Trend Indicators
    df['SMA_20'] = SMAIndicator(close=close_ser, window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=close_ser, window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(close=close_ser, window=20).ema_indicator()
    
    macd = MACD(close=close_ser)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    adx = ADXIndicator(high=high_ser, low=low_ser, close=close_ser)
    df['ADX'] = adx.adx()
    df['ADX_Pos'] = adx.adx_pos()
    df['ADX_Neg'] = adx.adx_neg()

    # Momentum Indicators
    df['RSI'] = RSIIndicator(close=close_ser).rsi()
    
    stoch = StochasticOscillator(high=high_ser, low=low_ser, close=close_ser)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # Volatility Indicators
    bb = BollingerBands(close=close_ser)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    df['ATR'] = AverageTrueRange(high=high_ser, low=low_ser, close=close_ser).average_true_range()

    # Volume Indicators
    df['OBV'] = OnBalanceVolumeIndicator(close=close_ser, volume=vol_ser).on_balance_volume()

    return df

def get_company_info(ticker):
    """Fetch company fundamental information."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        print(f"Error fetching info: {e}")
        return {}
