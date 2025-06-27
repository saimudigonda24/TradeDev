import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import talib

class FeatureEngineering:
    def __init__(self):
        self.scalers = {}
        
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features"""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_change'] = df['Close'] - df['Open']
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_to_sma_{period}'] = df['Close'] / df[f'sma_{period}']
        
        # Volatility features
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_30'] = df['returns'].rolling(window=30).std()
        
        # Volume features
        df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
        df['price_volume'] = df['Close'] * df['Volume']
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['Close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['Close'])
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['Close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Support and Resistance
        df['support'], df['resistance'] = self._calculate_support_resistance(df)
        
        return df
    
    def create_fundamental_features(self, price_data: pd.DataFrame, 
                                  fundamental_data: Dict) -> pd.DataFrame:
        """Add fundamental analysis features"""
        df = price_data.copy()
        
        if fundamental_data:
            # P/E ratio impact
            pe_ratio = fundamental_data.get('trailingPE', 0)
            df['pe_signal'] = 1 if pe_ratio < 15 else (-1 if pe_ratio > 25 else 0)
            
            # Market cap category
            market_cap = fundamental_data.get('marketCap', 0)
            if market_cap > 200e9:
                df['cap_category'] = 'large'
            elif market_cap > 10e9:
                df['cap_category'] = 'mid'
            else:
                df['cap_category'] = 'small'
        
        return df
    
    def create_market_regime_features(self, data: pd.DataFrame, 
                                    market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create market regime and correlation features"""
        df = data.copy()
        
        # Market correlation
        if 'SPY' in market_data:
            spy_returns = market_data['SPY']['Close'].pct_change()
            stock_returns = df['returns']
            
            # Rolling correlation with market
            df['market_correlation'] = stock_returns.rolling(window=30).corr(spy_returns)
            
            # Beta calculation
            covariance = stock_returns.rolling(window=60).cov(spy_returns)
            market_variance = spy_returns.rolling(window=60).var()
            df['beta'] = covariance / market_variance
        
        # VIX-like volatility regime
        df['volatility_regime'] = np.where(df['volatility_30'] > df['volatility_30'].rolling(window=252).quantile(0.8), 
                                          'high', 'normal')
        
        return df
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = data.copy()
        
        # Day of week effect
        df['day_of_week'] = df.index.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Month effect
        df['month'] = df.index.month
        df['is_january'] = (df['month'] == 1).astype(int)
        df['is_december'] = (df['month'] == 12).astype(int)
        
        # Quarter effect
        df['quarter'] = df.index.quarter
        
        # Days to earnings (if available)
        # This would require earnings calendar data
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, 
                           columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lagged features"""
        df = data.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, data: pd.DataFrame, 
                              columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Create rolling statistical features"""
        df = data.copy()
        
        for col in columns:
            for window in windows:
                df[f'{col}_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
                df[f'{col}_skew_{window}'] = df[col].rolling(window=window).skew()
        
        return df
    
    def scale_features(self, data: pd.DataFrame, 
                      method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        df = data.copy()
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        self.scalers[method] = scaler
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_stochastic(self, data: pd.DataFrame, 
                             k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                  period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_support_resistance(self, data: pd.DataFrame, 
                                    window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate support and resistance levels"""
        support = data['Low'].rolling(window=window).min()
        resistance = data['High'].rolling(window=window).max()
        return support, resistance
