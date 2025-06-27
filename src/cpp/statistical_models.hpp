#pragma once
#include <vector>
#include <random>
#include <map>

namespace QuantTrading {

struct ForecastResult {
    double predicted_price;
    double lower_bound;
    double upper_bound;
    double confidence;
    std::string classification; // "BUY", "SELL", "HOLD"
};

struct MonteCarloParams {
    int num_simulations = 10000;
    double drift;
    double volatility;
    int time_horizon; // days
};

class StatisticalModels {
private:
    std::mt19937 rng;
    
public:
    StatisticalModels();
    
    // Monte Carlo simulation for price forecasting
    ForecastResult monte_carlo_forecast(
        double current_price,
        const MonteCarloParams& params,
        double confidence_level = 0.95
    );
    
    // ARIMA-like autoregressive model
    ForecastResult arima_forecast(
        const std::vector<double>& prices,
        int forecast_days,
        double confidence_level = 0.95
    );
    
    // Mean reversion model
    ForecastResult mean_reversion_forecast(
        const std::vector<double>& prices,
        int forecast_days,
        double confidence_level = 0.95
    );
    
    // Calculate correlation between two price series
    double correlation(const std::vector<double>& series1, const std::vector<double>& series2);
    
    // Calculate Value at Risk (VaR)
    double calculate_var(const std::vector<double>& returns, double confidence_level = 0.95);
    
    // Portfolio optimization (simplified Markowitz)
    std::map<std::string, double> optimize_portfolio(
        const std::map<std::string, std::vector<double>>& asset_returns,
        double risk_free_rate = 0.02
    );
};

}
