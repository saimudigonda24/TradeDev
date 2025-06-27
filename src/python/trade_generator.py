import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import quant_cpp

class TradeType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    CALL = "CALL"
    PUT = "PUT"
    SPREAD = "SPREAD"

@dataclass
class TradeRecommendation:
    symbol: str
    trade_type: TradeType
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    risk_reward_ratio: float
    position_size: float
    expected_return: float
    max_loss: float
    probability_of_success: float
    reasoning: str

@dataclass
class OptionTrade:
    symbol: str
    option_type: str  # 'call' or 'put'
    strike_price: float
    expiration_date: str
    premium: float
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float
    max_profit: float
    max_loss: float
    breakeven_price: float
    probability_of_profit: float

class TradeGenerator:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.statistical_models = quant_cpp.StatisticalModels()
        
    def generate_stock_trades(self, forecasts: Dict[str, Dict], 
                            current_prices: Dict[str, float],
                            risk_tolerance: float = 0.02) -> List[TradeRecommendation]:
        """Generate stock trade recommendations based on forecasts"""
        trades = []
        
        for symbol, forecast_data in forecasts.items():
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            for timeframe, forecast in forecast_data.items():
                trade = self._create_stock_trade(
                    symbol, current_price, forecast, timeframe, risk_tolerance
                )
                if trade:
                    trades.append(trade)
        
        # Sort by risk-reward ratio and confidence
        trades.sort(key=lambda x: x.risk_reward_ratio * x.confidence, reverse=True)
        
        return trades
    
    def generate_option_trades(self, symbol: str, current_price: float,
                             forecast: Dict, options_data: pd.DataFrame,
                             volatility: float) -> List[OptionTrade]:
        """Generate option trade recommendations"""
        option_trades = []
        
        if options_data.empty:
            return option_trades
        
        predicted_price = forecast.get('predicted_price', current_price)
        confidence = forecast.get('confidence', 0.5)
        classification = forecast.get('classification', 'HOLD')
        
        # Filter options by expiration (30-60 days out)
        # This is simplified - in practice you'd parse actual expiration dates
        
        if classification == 'BUY':
            # Generate call options
            call_options = options_data[options_data['type'] == 'call']
            for _, option in call_options.iterrows():
                option_trade = self._create_call_trade(
                    symbol, current_price, predicted_price, option, volatility, confidence
                )
                if option_trade:
                    option_trades.append(option_trade)
        
        elif classification == 'SELL':
            # Generate put options
            put_options = options_data[options_data['type'] == 'put']
            for _, option in put_options.iterrows():
                option_trade = self._create_put_trade(
                    symbol, current_price, predicted_price, option, volatility, confidence
                )
                if option_trade:
                    option_trades.append(option_trade)
        
        # Generate spread strategies
        spread_trades = self._generate_spread_strategies(
            symbol, current_price, predicted_price, options_data, volatility
        )
        option_trades.extend(spread_trades)
        
        return option_trades
    
    def generate_short_candidates(self, forecasts: Dict[str, Dict],
                                current_prices: Dict[str, float]) -> List[TradeRecommendation]:
        """Generate short selling candidates"""
        short_candidates = []
        
        for symbol, forecast_data in forecasts.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Look for strong bearish signals across multiple timeframes
            bearish_signals = 0
            total_expected_decline = 0
            
            for timeframe, forecast in forecast_data.items():
                if forecast.get('classification') == 'SELL':
                    bearish_signals += 1
                    predicted_price = forecast.get('predicted_price', current_price)
                    expected_decline = (current_price - predicted_price) / current_price
                    total_expected_decline += expected_decline
            
            # Only consider stocks with multiple bearish signals
            if bearish_signals >= 2:
                avg_expected_decline = total_expected_decline / bearish_signals
                
                if avg_expected_decline > 0.1:  # At least 10% expected decline
                    trade = TradeRecommendation(
                        symbol=symbol,
                        trade_type=TradeType.SHORT,
                        entry_price=current_price,
                        target_price=current_price * (1 - avg_expected_decline),
                        stop_loss=current_price * 1.05,  # 5% stop loss
                        confidence=min(bearish_signals / 3, 1.0),
                        risk_reward_ratio=avg_expected_decline / 0.05,
                        position_size=0.02,  # 2% of portfolio
                        expected_return=avg_expected_decline,
                        max_loss=0.05,
                        probability_of_success=0.6,
                        reasoning=f"Multiple bearish signals across {bearish_signals} timeframes"
                    )
                    short_candidates.append(trade)
        
        return sorted(short_candidates, key=lambda x: x.expected_return, reverse=True)
    
    def generate_etf_trades(self, etf_forecasts: Dict[str, Dict],
                           sector_correlations: Dict[str, float]) -> List[TradeRecommendation]:
        """Generate ETF trading recommendations"""
        etf_trades = []
        
        for etf_symbol, forecast_data in etf_forecasts.items():
            # Analyze ETF based on sector correlations and forecasts
            for timeframe, forecast in forecast_data.items():
                if forecast.get('confidence', 0) > 0.7:
                    trade = self._create_etf_trade(etf_symbol, forecast, sector_correlations)
                    if trade:
                        etf_trades.append(trade)
        
        return etf_trades
    
    def calculate_position_sizing(self, trade: TradeRecommendation,
                                portfolio_value: float,
                                max_risk_per_trade: float = 0.02) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        # Kelly Criterion: f = (bp - q) / b
        # where f = fraction of capital to wager
        # b = odds received (risk/reward ratio)
        # p = probability of winning
        # q = probability of losing (1 - p)
        
        p = trade.probability_of_success
        q = 1 - p
        b = trade.risk_reward_ratio
        
        if b > 0 and p > q/b:
            kelly_fraction = (b * p - q) / b
            # Cap Kelly fraction to prevent over-leveraging
            kelly_fraction = min(kelly_fraction, max_risk_per_trade * 2)
        else:
            kelly_fraction = 0
        
        # Calculate position size
        risk_amount = portfolio_value * max_risk_per_trade
        position_value = risk_amount / trade.max_loss if trade.max_loss > 0 else 0
        
        # Use the more conservative of Kelly or risk-based sizing
        kelly_position_value = portfolio_value * kelly_fraction
        final_position_value = min(position_value, kelly_position_value)
        
        return max(final_position_value, 0)
    
    def _create_stock_trade(self, symbol: str, current_price: float,
                           forecast: Dict, timeframe: str,
                           risk_tolerance: float) -> Optional[TradeRecommendation]:
        """Create a stock trade recommendation"""
        predicted_price = forecast.get('predicted_price', current_price)
        confidence = forecast.get('confidence', 0.5)
        classification = forecast.get('classification', 'HOLD')
        
        if classification == 'HOLD' or confidence < 0.6:
            return None
        
        expected_return = (predicted_price - current_price) / current_price
        
        if abs(expected_return) < 0.05:  # Less than 5% expected move
            return None
        
        if classification == 'BUY':
            trade_type = TradeType.LONG
            target_price = predicted_price
            stop_loss = current_price * (1 - risk_tolerance)
            max_loss = risk_tolerance
        else:  # SELL
            trade_type = TradeType.SHORT
            target_price = predicted_price
            stop_loss = current_price * (1 + risk_tolerance)
            max_loss = risk_tolerance
            expected_return = -expected_return
        
        risk_reward_ratio = abs(expected_return) / max_loss
        
        return TradeRecommendation(
            symbol=symbol,
            trade_type=trade_type,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            risk_reward_ratio=risk_reward_ratio,
            position_size=0.02,  # Default 2% position
            expected_return=expected_return,
            max_loss=max_loss,
            probability_of_success=confidence,
            reasoning=f"{timeframe} forecast: {classification} with {confidence:.1%} confidence"
        )
    
    def _create_call_trade(self, symbol: str, current_price: float,
                          predicted_price: float, option_data: pd.Series,
                          volatility: float, confidence: float) -> Optional[OptionTrade]:
        """Create a call option trade"""
        strike_price = option_data.get('strike', 0)
        premium = option_data.get('lastPrice', 0)
        
        if strike_price <= 0 or premium <= 0:
            return None
        
        # Calculate option Greeks using C++ module
        option_params = quant_cpp.OptionParams()
        option_params.spot_price = current_price
        option_params.strike_price = strike_price
        option_params.time_to_expiry = 30/365  # Assume 30 days
        option_params.risk_free_rate = self.risk_free_rate
        option_params.volatility = volatility
        
        option_result = quant_cpp.OptionPricing.black_scholes(option_params)
        
        # Calculate profit/loss scenarios
        max_profit = max(0, predicted_price - strike_price - premium)
        max_loss = premium
        breakeven_price = strike_price + premium
        
        # Probability of profit (simplified)
        prob_profit = confidence if predicted_price > breakeven_price else 1 - confidence
        
        return OptionTrade(
            symbol=symbol,
            option_type='call',
            strike_price=strike_price,
            expiration_date="30 days",  # Simplified
            premium=premium,
            delta=option_result.call_delta,
            gamma=option_result.gamma,
            theta=option_result.theta,
            vega=option_result.vega,
            implied_volatility=volatility,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_price=breakeven_price,
            probability_of_profit=prob_profit
        )
    
    def _create_put_trade(self, symbol: str, current_price: float,
                         predicted_price: float, option_data: pd.Series,
                         volatility: float, confidence: float) -> Optional[OptionTrade]:
        """Create a put option trade"""
        strike_price = option_data.get('strike', 0)
        premium = option_data.get('lastPrice', 0)
        
        if strike_price <= 0 or premium <= 0:
            return None
        
        # Calculate option Greeks
        option_params = quant_cpp.OptionParams()
        option_params.spot_price = current_price
        option_params.strike_price = strike_price
        option_params.time_to_expiry = 30/365
        option_params.risk_free_rate = self.risk_free_rate
        option_params.volatility = volatility
        
        option_result = quant_cpp.OptionPricing.black_scholes(option_params)
        
        max_profit = max(0, strike_price - predicted_price - premium)
        max_loss = premium
        breakeven_price = strike_price - premium
        
        prob_profit = confidence if predicted_price < breakeven_price else 1 - confidence
        
        return OptionTrade(
            symbol=symbol,
            option_type='put',
            strike_price=strike_price,
            expiration_date="30 days",
            premium=premium,
            delta=option_result.put_delta,
            gamma=option_result.gamma,
            theta=option_result.theta,
            vega=option_result.vega,
            implied_volatility=volatility,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_price=breakeven_price,
            probability_of_profit=prob_profit
        )
    
    def _generate_spread_strategies(self, symbol: str, current_price: float,
                                  predicted_price: float, options_data: pd.DataFrame,
                                  volatility: float) -> List[OptionTrade]:
        """Generate option spread strategies"""
        spreads = []
        
        # Bull Call Spread
        if predicted_price > current_price * 1.05:
            call_options = options_data[options_data['type'] == 'call'].sort_values('strike')
            if len(call_options) >= 2:
                lower_strike = call_options.iloc[0]
                upper_strike = call_options.iloc[1]
                
                spread_trade = self._create_bull_call_spread(
                    symbol, current_price, lower_strike, upper_strike
                )
                if spread_trade:
                    spreads.append(spread_trade)
        
        return spreads
    
    def _create_bull_call_spread(self, symbol: str, current_price: float,
                                lower_strike: pd.Series, upper_strike: pd.Series) -> Optional[OptionTrade]:
        """Create a bull call spread"""
        lower_premium = lower_strike.get('lastPrice', 0)
        upper_premium = upper_strike.get('lastPrice', 0)
        
        if lower_premium <= 0 or upper_premium <= 0:
            return None
        
        net_debit = lower_premium - upper_premium
        max_profit = (upper_strike['strike'] - lower_strike['strike']) - net_debit
        max_loss = net_debit
        breakeven = lower_strike['strike'] + net_debit
        
        return OptionTrade(
            symbol=symbol,
            option_type='spread',
            strike_price=lower_strike['strike'],
            expiration_date="30 days",
            premium=net_debit,
            delta=0,  # Would need to calculate
            gamma=0,
            theta=0,
            vega=0,
            implied_volatility=0,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_price=breakeven,
            probability_of_profit=0.6  # Simplified
        )
    
    def _create_etf_trade(self, etf_symbol: str, forecast: Dict,
                         sector_correlations: Dict[str, float]) -> Optional[TradeRecommendation]:
        """Create ETF trade recommendation"""
        classification = forecast.get('classification', 'HOLD')
        confidence = forecast.get('confidence', 0.5)
        
        if classification == 'HOLD' or confidence < 0.7:
            return None
        
        # ETF trades typically have lower risk/reward but higher probability
        expected_return = 0.08 if classification == 'BUY' else -0.08
        
        return TradeRecommendation(
            symbol=etf_symbol,
            trade_type=TradeType.LONG if classification == 'BUY' else TradeType.SHORT,
            entry_price=0,  # Would be filled with current price
            target_price=0,  # Would be calculated
            stop_loss=0,    # Would be calculated
            confidence=confidence,
            risk_reward_ratio=4.0,  # ETFs typically have good risk/reward
            position_size=0.05,     # Larger position for ETFs
            expected_return=expected_return,
            max_loss=0.02,
            probability_of_success=confidence,
            reasoning=f"ETF sector play based on correlation analysis"
        )
