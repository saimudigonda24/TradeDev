#include "statistical_models.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace QuantTrading {

StatisticalModels::StatisticalModels() : rng(std::random_device{}()) {}

ForecastResult StatisticalModels::monte_carlo_forecast(
    double current_price,
    const MonteCarloParams& params,
    double confidence_level) {
    
    std::vector<double> final_prices;
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    double dt = 1.0 / 252.0; // Daily time step
    
    for (int sim = 0; sim < params.num_simulations; ++sim) {
        double price = current_price;
        
        for (int day = 0; day < params.time_horizon; ++day) {
            double random_shock = normal_dist(rng);
            double drift_component = params.drift * dt;
            double volatility_component = params.volatility * std::sqrt(dt) * random_shock;
            
            price *= std::exp(drift_component + volatility_component);
        }
        
        final_prices.push_back(price);
    }
    
    // Sort results for percentile calculation
    std::sort(final_prices.begin(), final_prices.end());
    
    ForecastResult result;
    result.predicted_price = std::accumulate(final_prices.begin(), final_prices.end(), 0.0) / final_prices.size();
    
    double lower_percentile = (1.0 - confidence_level) / 2.0;
    double upper_percentile = 1.0 - lower_percentile;
    
    size_t lower_idx = static_cast<size_t>(lower_percentile * final_prices.size());
    size_t upper_idx = static_cast<size_t>(upper_percentile * final_prices.size());
    
    result.lower_bound = final_prices[lower_idx];
    result.upper_bound = final_prices[upper_idx];
    result.confidence = confidence_level;
    
    // Classification based on expected return
    double expected_return = (result.predicted_price - current_price) / current_price;
    if (expected_return > 0.05) {
        result.classification = "BUY";
    } else if (expected_return < -0.05) {
        result.classification = "SELL";
    } else {
        result.classification = "HOLD";
    }
    
    return result;
}

ForecastResult StatisticalModels::arima_forecast(
    const std::vector<double>& prices,
    int forecast_days,
    double confidence_level) {
    
    if (prices.size() < 10) {
        throw std::invalid_argument("Insufficient data for ARIMA forecast");
    }
    
    // Simplified AR(1) model
    std::vector<double> returns;
    for (size_t i = 1; i < prices.size(); ++i) {
        returns.push_back(std::log(prices[i] / prices[i-1]));
    }
    
    // Calculate AR(1) coefficient
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    
    double numerator = 0.0, denominator = 0.0;
    for (size_t i = 1; i < returns.size(); ++i) {
        numerator += (returns[i-1] - mean_return) * (returns[i] - mean_return);
        denominator += (returns[i-1] - mean_return) * (returns[i-1] - mean_return);
    }
    
    double ar_coeff = denominator != 0 ? numerator / denominator : 0.0;
    
    // Calculate residual variance
    double residual_var = 0.0;
    for (size_t i = 1; i < returns.size(); ++i) {
        double predicted = mean_return + ar_coeff * (returns[i-1] - mean_return);
        double residual = returns[i] - predicted;
        residual_var += residual * residual;
    }
    residual_var /= (returns.size() - 2);
    
    // Forecast
    double current_price = prices.back();
    double last_return = returns.back();
    
    double forecasted_return = mean_return;
    for (int day = 0; day < forecast_days; ++day) {
        forecasted_return = mean_return + ar_coeff * (forecasted_return - mean_return);
    }
    
    ForecastResult result;
    result.predicted_price = current_price * std::exp(forecasted_return * forecast_days);
    
    // Confidence intervals
    double forecast_var = residual_var * forecast_days;
    double std_error = std::sqrt(forecast_var);
    double z_score = 1.96; // 95% confidence
    
    result.lower_bound = current_price * std::exp((forecasted_return * forecast_days) - (z_score * std_error));
    result.upper_bound = current_price * std::exp((forecasted_return * forecast_days) + (z_score * std_error));
    result.confidence = confidence_level;
    
    // Classification
    double expected_return = (result.predicted_price - current_price) / current_price;
    if (expected_return > 0.03) {
        result.classification = "BUY";
    } else if (expected_return < -0.03) {
        result.classification = "SELL";
    } else {
        result.classification = "HOLD";
    }
    
    return result;
}

double StatisticalModels::correlation(const std::vector<double>& series1, const std::vector<double>& series2) {
    if (series1.size() != series2.size() || series1.empty()) {
        return 0.0;
    }
    
    double mean1 = std::accumulate(series1.begin(), series1.end(), 0.0) / series1.size();
    double mean2 = std::accumulate(series2.begin(), series2.end(), 0.0) / series2.size();
    
    double numerator = 0.0, sum_sq1 = 0.0, sum_sq2 = 0.0;
    
    for (size_t i = 0; i < series1.size(); ++i) {
        double diff1 = series1[i] - mean1;
        double diff2 = series2[i] - mean2;
        
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }
    
    double denominator = std::sqrt(sum_sq1 * sum_sq2);
    return denominator != 0 ? numerator / denominator : 0.0;
}

double StatisticalModels::calculate_var(const std::vector<double>& returns, double confidence_level) {
    if (returns.empty()) return 0.0;
    
    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    size_t index = static_cast<size_t>((1.0 - confidence_level) * sorted_returns.size());
    return -sorted_returns[index]; // VaR is typically reported as positive
}

}
