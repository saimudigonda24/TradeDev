#include "technical_indicators.hpp"

namespace QuantTrading {

std::vector<double> TechnicalIndicators::sma(const std::vector<double>& prices, int period) {
    std::vector<double> result;
    if (prices.size() < period) return result;
    
    for (size_t i = period - 1; i < prices.size(); ++i) {
        double sum = std::accumulate(prices.begin() + i - period + 1, prices.begin() + i + 1, 0.0);
        result.push_back(sum / period);
    }
    return result;
}

std::vector<double> TechnicalIndicators::ema(const std::vector<double>& prices, int period) {
    std::vector<double> result;
    if (prices.empty()) return result;
    
    double multiplier = 2.0 / (period + 1);
    result.push_back(prices[0]);
    
    for (size_t i = 1; i < prices.size(); ++i) {
        double ema_value = (prices[i] * multiplier) + (result.back() * (1 - multiplier));
        result.push_back(ema_value);
    }
    return result;
}

TechnicalIndicators::BollingerBands TechnicalIndicators::bollinger_bands(
    const std::vector<double>& prices, int period, double std_dev) {
    
    BollingerBands bands;
    auto sma_values = sma(prices, period);
    
    for (size_t i = period - 1; i < prices.size(); ++i) {
        // Calculate standard deviation for the period
        double mean = sma_values[i - period + 1];
        double variance = 0.0;
        
        for (int j = 0; j < period; ++j) {
            double diff = prices[i - j] - mean;
            variance += diff * diff;
        }
        variance /= period;
        double std = std::sqrt(variance);
        
        bands.middle.push_back(mean);
        bands.upper.push_back(mean + (std_dev * std));
        bands.lower.push_back(mean - (std_dev * std));
    }
    
    return bands;
}

std::vector<double> TechnicalIndicators::rsi(const std::vector<double>& prices, int period) {
    std::vector<double> result;
    if (prices.size() <= period) return result;
    
    std::vector<double> gains, losses;
    
    // Calculate price changes
    for (size_t i = 1; i < prices.size(); ++i) {
        double change = prices[i] - prices[i-1];
        gains.push_back(change > 0 ? change : 0);
        losses.push_back(change < 0 ? -change : 0);
    }
    
    // Calculate initial averages
    double avg_gain = std::accumulate(gains.begin(), gains.begin() + period, 0.0) / period;
    double avg_loss = std::accumulate(losses.begin(), losses.begin() + period, 0.0) / period;
    
    for (size_t i = period; i < gains.size(); ++i) {
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period;
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period;
        
        double rs = avg_loss == 0 ? 100 : avg_gain / avg_loss;
        double rsi_value = 100 - (100 / (1 + rs));
        result.push_back(rsi_value);
    }
    
    return result;
}

std::vector<double> TechnicalIndicators::volatility(const std::vector<double>& prices, int period) {
    std::vector<double> result;
    if (prices.size() < period) return result;
    
    // Calculate returns
    std::vector<double> returns;
    for (size_t i = 1; i < prices.size(); ++i) {
        returns.push_back(std::log(prices[i] / prices[i-1]));
    }
    
    // Calculate rolling volatility
    for (size_t i = period - 1; i < returns.size(); ++i) {
        double mean = std::accumulate(returns.begin() + i - period + 1, returns.begin() + i + 1, 0.0) / period;
        
        double variance = 0.0;
        for (int j = 0; j < period; ++j) {
            double diff = returns[i - j] - mean;
            variance += diff * diff;
        }
        variance /= (period - 1);
        
        result.push_back(std::sqrt(variance * 252)); // Annualized volatility
    }
    
    return result;
}

}
