#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace QuantTrading {
    
class TechnicalIndicators {
public:
    // Simple Moving Average
    static std::vector<double> sma(const std::vector<double>& prices, int period);
    
    // Exponential Moving Average
    static std::vector<double> ema(const std::vector<double>& prices, int period);
    
    // Bollinger Bands
    struct BollingerBands {
        std::vector<double> upper;
        std::vector<double> middle;
        std::vector<double> lower;
    };
    static BollingerBands bollinger_bands(const std::vector<double>& prices, int period, double std_dev = 2.0);
    
    // RSI (Relative Strength Index)
    static std::vector<double> rsi(const std::vector<double>& prices, int period = 14);
    
    // MACD
    struct MACD {
        std::vector<double> macd_line;
        std::vector<double> signal_line;
        std::vector<double> histogram;
    };
    static MACD macd(const std::vector<double>& prices, int fast_period = 12, int slow_period = 26, int signal_period = 9);
    
    // Volatility (rolling standard deviation)
    static std::vector<double> volatility(const std::vector<double>& prices, int period);
    
    // Support and Resistance levels
    static std::vector<double> support_resistance(const std::vector<double>& prices, int window = 20);
};

}
