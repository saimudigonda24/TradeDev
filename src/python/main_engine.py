import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
import json
from datetime import datetime, timedelta
import logging

from data_ingestion import DataIngestion, MarketData
from feature_engineering import FeatureEngineering
from ml_models import MLModels, ModelResult
from trade_generator import TradeGenerator, TradeRecommendation, OptionTrade
from risk_management import RiskManager, RiskMetrics
import quant_cpp

class QuantTradingEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.data_ingestion = DataIngestion()
        self.feature_engineering = FeatureEngineering()
        self.ml_models = MLModels()
        self.trade_generator = TradeGenerator()
        self.risk_manager = RiskManager()
        self.statistical_models = quant_cpp.StatisticalModels()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def run_full_analysis(self, symbols: List[str], 
                         timeframes: List[str] = ['1d', '1w', '1m', '3m', '6m', '1y']) -> Dict:
        """Run complete quantitative analysis"""
        self.logger.info(f"Starting analysis for {len(symbols)} symbols")
        
        # Step 1: Data Ingestion
        market_data = self._ingest_data(symbols)
        
        # Step 2: Feature Engineering
        engineered_data = self._engineer_features(market_data)
        
        # Step 3: Generate Forecasts
        forecasts = self._generate_forecasts(engineered_data, timeframes)
        
        # Step 4: Generate Trade Recommendations
        trade_recommendations = self._generate_trade_recommendations(
            forecasts, market_data
        )
        
        # Step 5: Risk Analysis
        risk_analysis = self._perform_risk_analysis(
            trade_recommendations, market_data
        )
        
        # Step 6: Generate Report
        report = self._generate_report(
            forecasts, trade_recommendations, risk_analysis
        )
        
        self.logger.info("Analysis completed successfully")
        return report
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'models': {
                'monte_carlo': {
                    'num_simulations': 10000,
                    'confidence_level': 0.95
                },
                'ml_models': ['random_forest', 'xgboost', 'ensemble'],
                'forecast_horizons': [1, 5, 21, 63, 126, 252]  # days
            },
            'risk': {
                'max_position_size': 0.05,
                'max_portfolio_var': 0.02,
                'max_drawdown': 0.15,
                'risk_free_rate': 0.02
            },
            'trading': {
                'min_confidence': 0.6,
                'min_expected_return': 0.05,
                'max_correlation': 0.8
            }
        }
    
    def _ingest_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Ingest market data for all symbols"""
        self.logger.info("Ingesting market data...")
        
        market_data = self.data_ingestion.get_multiple_stocks(symbols, period="2y")
        
        # Add market indices for correlation analysis
        indices = self.data_ingestion.get_market_indices()
        market_data.update(indices)
        
        self.logger.info(f"Successfully ingested data for {len(market_data)} symbols")
        return market_data
    
    def _engineer_features(self, market_data: Dict[str, MarketData]) -> Dict[str, pd.DataFrame]:
        """Engineer features for all symbols"""
        self.logger.info("Engineering features...")
        
        engineered_data = {}
        
        for symbol, data in market_data.items():
            try:
                # Create technical features
                df = self.feature_engineering.create_technical_features(data.prices)
                
                # Add time-based features
                df = self.feature_engineering.create_time_features(df)
                
                # Add market regime features
                df = self.feature_engineering.create_market_regime_features(
                    df, {k: v.prices for k, v in market_data.items()}
                )
                
                # Create lag features
                df = self.feature_engineering.create_lag_features(
                    df, ['returns', 'volatility_10'], [1, 2, 3, 5]
                )
                
                # Create rolling features
                df = self.feature_engineering.create_rolling_features(
                    df, ['returns', 'volume_ratio'], [5, 10, 20]
                )
                
                engineered_data[symbol] = df.dropna()
                
            except Exception as e:
                self.logger.error(f"Error engineering features for {symbol}: {e}")
        
        self.logger.info(f"Feature engineering completed for {len(engineered_data)} symbols")
        return engineered_data
    
    def _generate_forecasts(self, engineered_data: Dict[str, pd.DataFrame],
                          timeframes: List[str]) -> Dict[str, Dict[str, Dict]]:
        """Generate forecasts using multiple models"""
        self.logger.info("Generating forecasts...")
        
        forecasts = {}
        
        for symbol, data in engineered_data.items():
            if len(data) < 100:  # Need sufficient data
                continue
                
            symbol_forecasts = {}
            current_price = data['Close'].iloc[-1]
            
            # Calculate volatility for Monte Carlo
            returns = data['returns'].dropna()
            volatility = np.std(returns) * np.sqrt(252)
            drift = np.mean(returns) * 252
            
            for timeframe in timeframes:
                try:
                    # Convert timeframe to days
                    days = self._timeframe_to_days(timeframe)
                    
                    # Monte Carlo forecast
                    mc_params = quant_cpp.MonteCarloParams()
                    mc_params.num_simulations = self.config['models']['monte_carlo']['num_simulations']
                    mc_params.drift = drift
                    mc_params.volatility = volatility
                    mc_params.time_horizon = days
                    
                    mc_forecast = self.statistical_models.monte_carlo_forecast(
                        current_price, mc_params
                    )
                    
                    # ARIMA forecast
                    prices = data['Close'].values.tolist()
                    arima_forecast = self.statistical_models.arima_forecast(
                        prices, days
                    )
                    
                    # ML forecast (if enough data)
                    ml_forecast = self._generate_ml_forecast(data, days)
                    
                    # Combine forecasts
                    combined_forecast = self._combine_forecasts(
                        mc_forecast, arima_forecast, ml_forecast
                    )
                    
                    symbol_forecasts[timeframe] = combined_forecast
                    
                except Exception as e:
                    self.logger.error(f"Error forecasting {symbol} for {timeframe}: {e}")
            
            if symbol_forecasts:
                forecasts[symbol] = symbol_forecasts
        
        self.logger.info(f"Forecasts generated for {len(forecasts)} symbols")
        return forecasts
    
    def _generate_ml_forecast(self, data: pd.DataFrame, forecast_days: int) -> Optional[Dict]:
        """Generate ML-based forecast"""
        try:
            # Prepare data for ML
            X, y, feature_names = self.ml_models.prepare_data(
                data, 'Close', forecast_days
            )
            
            if len(X) < 50:  # Need sufficient training data
                return None
            
            # Split data for training
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train ensemble model
            self.ml_models.train_ensemble(X_train, y_train, feature_names)
            
            # Make prediction on latest data
            latest_features = X[-1:] if len(X) > 0 else None
            if latest_features is not None:
                result = self.ml_models.predict_with_confidence('ensemble', latest_features)
                
                return {
                    'predicted_price': result.predictions[0],
                    'lower_bound': result.confidence_intervals[0][0],
                    'upper_bound': result.confidence_intervals[1][0],
                    'confidence': 0.8,  # Default confidence for ML
                    'classification': self._classify_prediction(
                        data['Close'].iloc[-1], result.predictions[0]
                    )
                }
        
        except Exception as e:
            self.logger.error(f"ML forecast error: {e}")
            return None
    
    def _combine_forecasts(self, mc_forecast, arima_forecast, ml_forecast=None) -> Dict:
        """Combine multiple forecasts using weighted average"""
        forecasts = [mc_forecast, arima_forecast]
        weights = [0.4, 0.4]  # Monte Carlo and ARIMA
        
        if ml_forecast:
            forecasts.append(ml_forecast)
            weights = [0.3, 0.3, 0.4]  # Add ML with higher weight
        
        # Weighted average of predictions
        predicted_price = sum(f['predicted_price'] * w for f, w in zip(forecasts, weights))
        
        # Conservative confidence (minimum of all forecasts)
        confidence = min(f['confidence'] for f in forecasts)
        
        # Widest bounds for conservative estimate
        lower_bound = min(f['lower_bound'] for f in forecasts)
        upper_bound = max(f['upper_bound'] for f in forecasts)
        
        # Classification based on combined prediction
        classification = forecasts[0]['classification']  # Use Monte Carlo classification
        
        return {
            'predicted_price': predicted_price,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': confidence,
            'classification': classification
        }
    
    def _generate_trade_recommendations(self, forecasts: Dict[str, Dict[str, Dict]],
                                      market_data: Dict[str, MarketData]) -> Dict:
        """Generate comprehensive trade recommendations"""
        self.logger.info("Generating trade recommendations...")
        
        current_prices = {symbol: data.prices['Close'].iloc[-1] 
                         for symbol, data in market_data.items()}
        
        # Stock trades
        stock_trades = self.trade_generator.generate_stock_trades(
            forecasts, current_prices, 
            risk_tolerance=self.config['risk']['max_position_size']
        )
        
        # Short candidates
        short_candidates = self.trade_generator.generate_short_candidates(
            forecasts, current_prices
        )
        
        # Option trades (for top stock picks)
        option_trades = []
        top_stocks = stock_trades[:5]  # Top 5 stock recommendations
        
        for trade in top_stocks:
            symbol = trade.symbol
            if symbol in market_data:
                # Get options data
                options_data = self.data_ingestion.get_options_data(symbol)
                
                # Calculate volatility
                returns = market_data[symbol].prices['Close'].pct_change().dropna()
                volatility = np.std(returns) * np.sqrt(252)
                
                # Generate option trades
                symbol_forecasts = forecasts.get(symbol, {})
                if '1m' in symbol_forecasts:  # Use 1-month forecast for options
                    symbol_options = self.trade_generator.generate_option_trades(
                        symbol, current_prices[symbol], 
                        symbol_forecasts['1m'], options_data, volatility
                    )
                    option_trades.extend(symbol_options)
        
        # ETF trades
        etf_symbols = ['SPY', 'QQQ', 'IWM', 'VTI']
        etf_forecasts = {symbol: forecasts[symbol] for symbol in etf_symbols 
                        if symbol in forecasts}
        
        etf_trades = self.trade_generator.generate_etf_trades(
            etf_forecasts, {}  # Sector correlations would be calculated here
        )
        
        return {
            'stock_trades': stock_trades,
            'short_candidates': short_candidates,
            'option_trades': option_trades,
            'etf_trades': etf_trades
        }
    
    def _perform_risk_analysis(self, trade_recommendations: Dict,
                             market_data: Dict[str, MarketData]) -> Dict:
        """Perform comprehensive risk analysis"""
        self.logger.info("Performing risk analysis...")
        
        # Create sample portfolio from top recommendations
        stock_trades = trade_recommendations['stock_trades'][:10]  # Top 10
        portfolio_positions = {trade.symbol: trade.position_size 
                             for trade in stock_trades}
        
        # Get returns data
        returns_data = {}
        for symbol, data in market_data.items():
            if symbol in portfolio_positions:
                returns_data[symbol] = data.prices['Close'].pct_change().dropna()
        
        # Market returns (SPY)
        market_returns = market_data.get('SPY', market_data[list(market_data.keys())[0]]).prices['Close'].pct_change().dropna()
        
        # Calculate portfolio risk metrics
        risk_metrics = self.risk_manager.calculate_portfolio_risk(
            portfolio_positions, returns_data, market_returns
        )
        
        # Stress testing scenarios
        stress_scenarios = {
            'market_crash': {symbol: -0.20 for symbol in portfolio_positions.keys()},
            'sector_rotation': {symbol: -0.10 if i % 2 == 0 else 0.05 
                              for i, symbol in enumerate(portfolio_positions.keys())},
            'volatility_spike': {symbol: -0.15 for symbol in portfolio_positions.keys()}
        }
        
        current_prices = {symbol: data.prices['Close'].iloc[-1] 
                         for symbol, data in market_data.items()}
        
        stress_results = self.risk_manager.stress_test_portfolio(
            portfolio_positions, current_prices, stress_scenarios
        )
        
        return {
            'risk_metrics': risk_metrics,
            'stress_test_results': stress_results,
            'portfolio_positions': portfolio_positions
        }
    
    def _generate_report(self, forecasts: Dict, trade_recommendations: Dict,
                        risk_analysis: Dict) -> Dict:
        """Generate comprehensive analysis report"""
        self.logger.info("Generating final report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_symbols_analyzed': len(forecasts),
                'stock_recommendations': len(trade_recommendations['stock_trades']),
                'short_candidates': len(trade_recommendations['short_candidates']),
                'option_strategies': len(trade_recommendations['option_trades']),
                'etf_recommendations': len(trade_recommendations['etf_trades'])
            },
            'forecasts': forecasts,
            'trade_recommendations': {
                'top_longs': [self._trade_to_dict(trade) for trade in trade_recommendations['stock_trades'][:10]],
                'top_shorts': [self._trade_to_dict(trade) for trade in trade_recommendations['short_candidates'][:5]],
                'best_options': [self._option_to_dict(option) for option in trade_recommendations['option_trades'][:10]],
                'etf_plays': [self._trade_to_dict(trade) for trade in trade_recommendations['etf_trades']]
            },
            'risk_analysis': {
                'portfolio_var': risk_analysis['risk_metrics'].portfolio_var,
                'max_drawdown': risk_analysis['risk_metrics'].max_drawdown,
                'sharpe_ratio': risk_analysis['risk_metrics'].sharpe_ratio,
                'concentration_risk': risk_analysis['risk_metrics'].concentration_risk,
                'stress_test_results': risk_analysis['stress_test_results']
            },
            'market_outlook': self._generate_market_outlook(forecasts)
        }
        
        return report
    
    def _timeframe_to_days(self, timeframe: str) -> int:
        """Convert timeframe string to number of days"""
        mapping = {
            '1d': 1, '1w': 7, '1m': 21, '3m': 63,
            '6m': 126, '1y': 252, '5y': 1260
        }
        return mapping.get(timeframe, 21)
    
    def _classify_prediction(self, current_price: float, predicted_price: float) -> str:
        """Classify prediction as BUY, SELL, or HOLD"""
        change_pct = (predicted_price - current_price) / current_price
        
        if change_pct > 0.05:
            return 'BUY'
        elif change_pct < -0.05:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _trade_to_dict(self, trade: TradeRecommendation) -> Dict:
        """Convert trade recommendation to dictionary"""
        return {
            'symbol': trade.symbol,
            'trade_type': trade.trade_type.value,
            'entry_price': trade.entry_price,
            'target_price': trade.target_price,
            'stop_loss': trade.stop_loss,
            'confidence': trade.confidence,
            'risk_reward_ratio': trade.risk_reward_ratio,
            'expected_return': trade.expected_return,
            'reasoning': trade.reasoning
        }
    
    def _option_to_dict(self, option: OptionTrade) -> Dict:
        """Convert option trade to dictionary"""
        return {
            'symbol': option.symbol,
            'option_type': option.option_type,
            'strike_price': option.strike_price,
            'premium': option.premium,
            'max_profit': option.max_profit,
            'max_loss': option.max_loss,
            'probability_of_profit': option.probability_of_profit
        }
    
    def _generate_market_outlook(self, forecasts: Dict) -> Dict:
        """Generate overall market outlook"""
        total_symbols = len(forecasts)
        if total_symbols == 0:
            return {'outlook': 'NEUTRAL', 'confidence': 0.5}
        
        # Count classifications across all timeframes
        buy_count = sell_count = hold_count = 0
        
        for symbol_forecasts in forecasts.values():
            for forecast in symbol_forecasts.values():
                classification = forecast.get('classification', 'HOLD')
                if classification == 'BUY':
                    buy_count += 1
                elif classification == 'SELL':
                    sell_count += 1
                else:
                    hold_count += 1
        
        total_forecasts = buy_count + sell_count + hold_count
        
        if total_forecasts == 0:
            return {'outlook': 'NEUTRAL', 'confidence': 0.5}
        
        buy_pct = buy_count / total_forecasts
        sell_pct = sell_count / total_forecasts
        
        if buy_pct > 0.6:
            outlook = 'BULLISH'
            confidence = buy_pct
        elif sell_pct > 0.6:
            outlook = 'BEARISH'
            confidence = sell_pct
        else:
            outlook = 'NEUTRAL'
            confidence = max(buy_pct, sell_pct)
        
        return {
            'outlook': outlook,
            'confidence': confidence,
            'buy_percentage': buy_pct,
            'sell_percentage': sell_pct,
            'hold_percentage': hold_count / total_forecasts
        }
    
    def export_results(self, report: Dict, format: str = 'json', 
                      filepath: str = None) -> str:
        """Export results to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/trading_analysis_{timestamp}.{format}"
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'csv':
            # Export key data to CSV
            trades_df = pd.DataFrame(report['trade_recommendations']['top_longs'])
            trades_df.to_csv(filepath, index=False)
        
        self.logger.info(f"Results exported to {filepath}")
        return filepath

# Example usage
if __name__ == "__main__":
    # Initialize the engine
    engine = QuantTradingEngine()
    
    # Define symbols to analyze
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'META', 'NFLX']
    
    # Run full analysis
    results = engine.run_full_analysis(symbols)
    
    # Export results
    engine.export_results(results, 'json')
    
    # Print summary
    print("=== QUANTITATIVE TRADING ANALYSIS SUMMARY ===")
    print(f"Symbols Analyzed: {results['summary']['total_symbols_analyzed']}")
    print(f"Stock Recommendations: {results['summary']['stock_recommendations']}")
    print(f"Short Candidates: {results['summary']['short_candidates']}")
    print(f"Option Strategies: {results['summary']['option_strategies']}")
    print(f"Market Outlook: {results['market_outlook']['outlook']} ({results['market_outlook']['confidence']:.1%} confidence)")
