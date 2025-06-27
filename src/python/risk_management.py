import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import quant_cpp

@dataclass
class RiskMetrics:
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_with_market: float
    concentration_risk: float

@dataclass
class PositionRisk:
    symbol: str
    position_size: float
    var_contribution: float
    correlation_risk: float
    liquidity_risk: float
    sector_concentration: float

class RiskManager:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.statistical_models = quant_cpp.StatisticalModels()
        
    def calculate_portfolio_risk(self, positions: Dict[str, float],
                               returns_data: Dict[str, pd.Series],
                               market_returns: pd.Series) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        # Create portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
        
        # Value at Risk (95% confidence)
        portfolio_var = self.statistical_models.calculate_var(
            portfolio_returns.values.tolist(), 0.95
        )
        
        # Conditional Value at Risk (Expected Shortfall)
        portfolio_cvar = self._calculate_cvar(portfolio_returns, 0.95)
        
        # Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        # Sharpe Ratio
        excess_returns = portfolio_returns - self.risk_free_rate/252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Sortino Ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino_ratio = float('inf')
        
        # Beta and correlation with market
        if len(market_returns) > 0:
            beta = self._calculate_beta(portfolio_returns, market_returns)
            correlation = self.statistical_models.correlation(
                portfolio_returns.values.tolist(),
                market_returns.values.tolist()
            )
        else:
            beta = 1.0
            correlation = 0.0
        
        # Concentration risk (Herfindahl Index)
        concentration_risk = sum(weight**2 for weight in positions.values())
        
        return RiskMetrics(
            portfolio_var=portfolio_var,
            portfolio_cvar=portfolio_cvar,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            beta=beta,
            correlation_with_market=correlation,
            concentration_risk=concentration_risk
        )
    
    def calculate_position_risks(self, positions: Dict[str, float],
                               returns_data: Dict[str, pd.Series],
                               sector_mapping: Dict[str, str]) -> List[PositionRisk]:
        """Calculate risk metrics for individual positions"""
        position_risks = []
        
        # Calculate correlation matrix
        symbols = list(positions.keys())
        correlation_matrix = self._calculate_correlation_matrix(symbols, returns_data)
        
        for symbol, weight in positions.items():
            if symbol not in returns_data:
                continue
                
            returns = returns_data[symbol]
            
            # VaR contribution
            var_contribution = self._calculate_var_contribution(
                symbol, weight, returns, positions, returns_data
            )
            
            # Correlation risk (average correlation with other positions)
            correlation_risk = self._calculate_correlation_risk(
                symbol, symbols, correlation_matrix, positions
            )
            
            # Liquidity risk (simplified - based on volatility)
            liquidity_risk = np.std(returns) * np.sqrt(252)
            
            # Sector concentration
            sector = sector_mapping.get(symbol, 'Unknown')
            sector_concentration = sum(
                w for s, w in positions.items() 
                if sector_mapping.get(s) == sector
            )
            
            position_risks.append(PositionRisk(
                symbol=symbol,
                position_size=weight,
                var_contribution=var_contribution,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                sector_concentration=sector_concentration
            ))
        
        return position_risks
    
    def stress_test_portfolio(self, positions: Dict[str, float],
                            current_prices: Dict[str, float],
                            scenarios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Perform stress testing on portfolio"""
        stress_results = {}
        
        for scenario_name, price_changes in scenarios.items():
            portfolio_change = 0.0
            
            for symbol, weight in positions.items():
                if symbol in price_changes and symbol in current_prices:
                    price_change = price_changes[symbol]
                    position_value = weight * current_prices[symbol]
                    position_change = position_value * price_change
                    portfolio_change += position_change
            
            stress_results[scenario_name] = portfolio_change
        
        return stress_results
    
    def optimize_portfolio_risk(self, expected_returns: Dict[str, float],
                              returns_data: Dict[str, pd.Series],
                              target_return: float = 0.12) -> Dict[str, float]:
        """Optimize portfolio using mean-variance optimization"""
        symbols = list(expected_returns.keys())
        
        # Calculate covariance matrix
        returns_df = pd.DataFrame(returns_data)[symbols].dropna()
        cov_matrix = returns_df.cov().values * 252  # Annualized
        
        # Expected returns vector
        mu = np.array([expected_returns[symbol] for symbol in symbols])
        
        # Use C++ optimization (simplified Markowitz)
        optimized_weights = self.statistical_models.optimize_portfolio(
            {symbol: returns_data[symbol].values.tolist() for symbol in symbols},
            self.risk_free_rate
        )
        
        return optimized_weights
    
    def calculate_hedge_ratios(self, portfolio_positions: Dict[str, float],
                             hedge_instruments: List[str],
                             returns_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate optimal hedge ratios"""
        hedge_ratios = {}
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(portfolio_positions, returns_data)
        
        for hedge_instrument in hedge_instruments:
            if hedge_instrument in returns_data:
                hedge_returns = returns_data[hedge_instrument]
                
                # Calculate hedge ratio using regression
                hedge_ratio = self._calculate_hedge_ratio(portfolio_returns, hedge_returns)
                hedge_ratios[hedge_instrument] = hedge_ratio
        
        return hedge_ratios
    
    def monitor_risk_limits(self, current_metrics: RiskMetrics,
                           risk_limits: Dict[str, float]) -> Dict[str, bool]:
        """Monitor if portfolio exceeds risk limits"""
        violations = {}
        
        violations['var_limit'] = current_metrics.portfolio_var > risk_limits.get('max_var', 0.05)
        violations['drawdown_limit'] = current_metrics.max_drawdown > risk_limits.get('max_drawdown', 0.20)
        violations['concentration_limit'] = current_metrics.concentration_risk > risk_limits.get('max_concentration', 0.25)
        violations['beta_limit'] = abs(current_metrics.beta) > risk_limits.get('max_beta', 1.5)
        
        return violations
    
    def _calculate_portfolio_returns(self, positions: Dict[str, float],
                                   returns_data: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio returns from positions and individual returns"""
        portfolio_returns = None
        
        for symbol, weight in positions.items():
            if symbol in returns_data:
                weighted_returns = returns_data[symbol] * weight
                
                if portfolio_returns is None:
                    portfolio_returns = weighted_returns
                else:
                    portfolio_returns += weighted_returns
        
        return portfolio_returns if portfolio_returns is not None else pd.Series()
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var_threshold = returns.quantile(1 - confidence_level)
        tail_losses = returns[returns <= var_threshold]
        return -tail_losses.mean() if len(tail_losses) > 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return abs(drawdown.min())
    
    def _calculate_beta(self, portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate portfolio beta"""
        aligned_data = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 1.0
        
        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data.iloc[:, 1].var()
        
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def _calculate_correlation_matrix(self, symbols: List[str],
                                    returns_data: Dict[str, pd.Series]) -> np.ndarray:
        """Calculate correlation matrix for symbols"""
        returns_df = pd.DataFrame({symbol: returns_data[symbol] for symbol in symbols if symbol in returns_data})
        return returns_df.corr().values
    
    def _calculate_var_contribution(self, symbol: str, weight: float,
                                  returns: pd.Series, positions: Dict[str, float],
                                  returns_data: Dict[str, pd.Series]) -> float:
        """Calculate VaR contribution of a position"""
        # Simplified marginal VaR calculation
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
        
        if len(portfolio_returns) == 0:
            return 0.0
        
        # Calculate correlation between asset and portfolio
        correlation = self.statistical_models.correlation(
            returns.values.tolist(),
            portfolio_returns.values.tolist()
        )
        
        asset_vol = np.std(returns) * np.sqrt(252)
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        
        marginal_var = correlation * asset_vol / portfolio_vol if portfolio_vol != 0 else 0
        return weight * marginal_var
    
    def _calculate_correlation_risk(self, symbol: str, symbols: List[str],
                                  correlation_matrix: np.ndarray,
                                  positions: Dict[str, float]) -> float:
        """Calculate correlation risk for a position"""
        if symbol not in symbols:
            return 0.0
        
        symbol_idx = symbols.index(symbol)
        correlations = correlation_matrix[symbol_idx, :]
        
        # Weight correlations by position sizes
        weighted_correlation = 0.0
        total_weight = 0.0
        
        for i, other_symbol in enumerate(symbols):
            if other_symbol != symbol and other_symbol in positions:
                weight = positions[other_symbol]
                weighted_correlation += abs(correlations[i]) * weight
                total_weight += weight
        
        return weighted_correlation / total_weight if total_weight > 0 else 0.0
    
    def _calculate_hedge_ratio(self, portfolio_returns: pd.Series,
                             hedge_returns: pd.Series) -> float:
        """Calculate optimal hedge ratio using regression"""
        aligned_data = pd.concat([portfolio_returns, hedge_returns], axis=1).dropna()
        
        if len(aligned_data) < 10:
            return 0.0
        
        # Simple linear regression: portfolio_returns = alpha + beta * hedge_returns
        x = aligned_data.iloc[:, 1].values  # hedge returns
        y = aligned_data.iloc[:, 0].values  # portfolio returns
        
        # Calculate beta (hedge ratio)
        covariance = np.cov(x, y)[0, 1]
        variance = np.var(x)
        
        return -covariance / variance if variance != 0 else 0.0  # Negative for hedge
