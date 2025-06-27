import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import requests
import time
from dataclasses import dataclass

@dataclass
class MarketData:
    symbol: str
    prices: pd.DataFrame
    volume: pd.Series
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

class DataIngestion:
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.cache = {}
        
    def get_stock_data(self, symbol: str, period: str = "2y") -> MarketData:
        """Fetch stock data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period=period)
            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Get company info
            info = ticker.info
            
            return MarketData(
                symbol=symbol,
                prices=hist[['Open', 'High', 'Low', 'Close', 'Adj Close']],
                volume=hist['Volume'],
                market_cap=info.get('marketCap'),
                sector=info.get('sector'),
                industry=info.get('industry')
            )
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "2y") -> Dict[str, MarketData]:
        """Fetch data for multiple stocks"""
        results = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.get_stock_data(symbol, period)
            if data:
                results[symbol] = data
            time.sleep(0.1)  # Rate limiting
            
        return results
    
    def get_etf_data(self, etf_symbol: str, period: str = "2y") -> MarketData:
        """Fetch ETF data and holdings information"""
        return self.get_stock_data(etf_symbol, period)
    
    def get_options_data(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain data"""
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            
            if not options_dates:
                return pd.DataFrame()
            
            # Get options for the nearest expiration
            options = ticker.option_chain(options_dates[0])
            
            # Combine calls and puts
            calls = options.calls.copy()
            calls['type'] = 'call'
            puts = options.puts.copy()
            puts['type'] = 'put'
            
            return pd.concat([calls, puts], ignore_index=True)
            
        except Exception as e:
            print(f"Error fetching options data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_indices(self) -> Dict[str, MarketData]:
        """Fetch major market indices"""
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'IWM': 'Russell 2000',
            'VTI': 'Total Stock Market',
            'VEA': 'Developed Markets',
            'VWO': 'Emerging Markets'
        }
        
        return self.get_multiple_stocks(list(indices.keys()))
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate daily returns"""
        return prices.pct_change().dropna()
    
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns"""
        return np.log(prices / prices.shift(1)).dropna()
    
    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate (10-year Treasury)"""
        try:
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="5d")
            return hist['Close'].iloc[-1] / 100  # Convert percentage to decimal
        except:
            return 0.02  # Default 2%
    
    def get_sector_etfs(self) -> Dict[str, str]:
        """Get sector ETF mappings"""
        return {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }
