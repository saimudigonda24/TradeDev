#include "option_pricing.hpp"
#include <random>

namespace QuantTrading {

double OptionPricing::normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double OptionPricing::normal_pdf(double x) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

OptionResult OptionPricing::black_scholes(const OptionParams& params) {
    double S = params.spot_price;
    double K = params.strike_price;
    double T = params.time_to_expiry;
    double r = params.risk_free_rate;
    double sigma = params.volatility;
    
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    
    OptionResult result;
    
    // Option prices
    result.call_price = S * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
    result.put_price = K * std::exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1);
    
    // Greeks
    result.call_delta = normal_cdf(d1);
    result.put_delta = normal_cdf(d1) - 1.0;
    result.gamma = normal_pdf(d1) / (S * sigma * std::sqrt(T));
    result.theta = -(S * normal_pdf(d1) * sigma) / (2.0 * std::sqrt(T)) - 
                   r * K * std::exp(-r * T) * normal_cdf(d2);
    result.vega = S * normal_pdf(d1) * std::sqrt(T);
    
    return result;
}

OptionResult OptionPricing::monte_carlo_option(const OptionParams& params, int num_simulations) {
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    double S = params.spot_price;
    double K = params.strike_price;
    double T = params.time_to_expiry;
    double r = params.risk_free_rate;
    double sigma = params.volatility;
    
    double call_payoff_sum = 0.0;
    double put_payoff_sum = 0.0;
    
    for (int i = 0; i < num_simulations; ++i) {
        double z = normal_dist(rng);
        double ST = S * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
        
        call_payoff_sum += std::max(ST - K, 0.0);
        put_payoff_sum += std::max(K - ST, 0.0);
    }
    
    OptionResult result;
    result.call_price = std::exp(-r * T) * call_payoff_sum / num_simulations;
    result.put_price = std::exp(-r * T) * put_payoff_sum / num_simulations;
    
    // Greeks would require additional calculations (finite differences)
    result.call_delta = result.put_delta = result.gamma = result.theta = result.vega = 0.0;
    
    return result;
}

SpreadStrategy OptionStrategies::bull_call_spread(
    double spot_price,
    double lower_strike,
    double upper_strike,
    double call_lower_price,
    double call_upper_price) {
    
    SpreadStrategy strategy;
    strategy.strategy_name = "Bull Call Spread";
    
    double net_debit = call_lower_price - call_upper_price;
    strategy.max_profit = (upper_strike - lower_strike) - net_debit;
    strategy.max_loss = net_debit;
    
    strategy.breakeven_points[0] = lower_strike + net_debit;
    strategy.breakeven_points[1] = 0; // Only one breakeven point
    
    // Simplified probability calculation
    strategy.probability_of_profit = spot_price > strategy.breakeven_points[0] ? 0.6 : 0.4;
    
    return strategy;
}

}
