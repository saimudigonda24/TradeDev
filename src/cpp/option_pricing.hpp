#pragma once
#include <cmath>

namespace QuantTrading {

struct OptionParams {
    double spot_price;
    double strike_price;
    double time_to_expiry; // in years
    double risk_free_rate;
    double volatility;
};

struct OptionResult {
    double call_price;
    double put_price;
    double call_delta;
    double put_delta;
    double gamma;
    double theta;
    double vega;
};

class OptionPricing {
public:
    // Black-Scholes option pricing
    static OptionResult black_scholes(const OptionParams& params);
    
    // Monte Carlo option pricing
    static OptionResult monte_carlo_option(const OptionParams& params, int num_simulations = 100000);
    
    // Implied volatility calculation
    static double implied_volatility(double market_price, const OptionParams& params, bool is_call = true);
    
private:
    static double normal_cdf(double x);
    static double normal_pdf(double x);
};

struct SpreadStrategy {
    std::string strategy_name;
    double max_profit;
    double max_loss;
    double breakeven_points[2];
    double probability_of_profit;
};

class OptionStrategies {
public:
    // Bull Call Spread
    static SpreadStrategy bull_call_spread(
        double spot_price,
        double lower_strike,
        double upper_strike,
        double call_lower_price,
        double call_upper_price
    );
    
    // Iron Condor
    static SpreadStrategy iron_condor(
        double spot_price,
        double put_strike_low,
        double put_strike_high,
        double call_strike_low,
        double call_strike_high,
        double put_low_price,
        double put_high_price,
        double call_low_price,
        double call_high_price
    );
};

}
