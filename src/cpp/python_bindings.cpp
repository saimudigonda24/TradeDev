#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "technical_indicators.hpp"
#include "statistical_models.hpp"
#include "option_pricing.hpp"

namespace py = pybind11;
using namespace QuantTrading;

PYBIND11_MODULE(quant_cpp, m) {
    m.doc() = "Quantitative Trading C++ Module";
    
    // Technical Indicators
    py::class_<TechnicalIndicators>(m, "TechnicalIndicators")
        .def_static("sma", &TechnicalIndicators::sma)
        .def_static("ema", &TechnicalIndicators::ema)
        .def_static("rsi", &TechnicalIndicators::rsi)
        .def_static("volatility", &TechnicalIndicators::volatility);
    
    py::class_<TechnicalIndicators::BollingerBands>(m, "BollingerBands")
        .def_readwrite("upper", &TechnicalIndicators::BollingerBands::upper)
        .def_readwrite("middle", &TechnicalIndicators::BollingerBands::middle)
        .def_readwrite("lower", &TechnicalIndicators::BollingerBands::lower);
    
    // Statistical Models
    py::class_<ForecastResult>(m, "ForecastResult")
        .def_readwrite("predicted_price", &ForecastResult::predicted_price)
        .def_readwrite("lower_bound", &ForecastResult::lower_bound)
        .def_readwrite("upper_bound", &ForecastResult::upper_bound)
        .def_readwrite("confidence", &ForecastResult::confidence)
        .def_readwrite("classification", &ForecastResult::classification);
    
    py::class_<MonteCarloParams>(m, "MonteCarloParams")
        .def(py::init<>())
        .def_readwrite("num_simulations", &MonteCarloParams::num_simulations)
        .def_readwrite("drift", &MonteCarloParams::drift)
        .def_readwrite("volatility", &MonteCarloParams::volatility)
        .def_readwrite("time_horizon", &MonteCarloParams::time_horizon);
    
    py::class_<StatisticalModels>(m, "StatisticalModels")
        .def(py::init<>())
        .def("monte_carlo_forecast", &StatisticalModels::monte_carlo_forecast)
        .def("arima_forecast", &StatisticalModels::arima_forecast)
        .def("correlation", &StatisticalModels::correlation)
        .def("calculate_var", &StatisticalModels::calculate_var);
    
    // Option Pricing
    py::class_<OptionParams>(m, "OptionParams")
        .def(py::init<>())
        .def_readwrite("spot_price", &OptionParams::spot_price)
        .def_readwrite("strike_price", &OptionParams::strike_price)
        .def_readwrite("time_to_expiry", &OptionParams::time_to_expiry)
        .def_readwrite("risk_free_rate", &OptionParams::risk_free_rate)
        .def_readwrite("volatility", &OptionParams::volatility);
    
    py::class_<OptionResult>(m, "OptionResult")
        .def_readwrite("call_price", &OptionResult::call_price)
        .def_readwrite("put_price", &OptionResult::put_price)
        .def_readwrite("call_delta", &OptionResult::call_delta)
        .def_readwrite("put_delta", &OptionResult::put_delta)
        .def_readwrite("gamma", &OptionResult::gamma)
        .def_readwrite("theta", &OptionResult::theta)
        .def_readwrite("vega", &OptionResult::vega);
    
    py::class_<OptionPricing>(m, "OptionPricing")
        .def_static("black_scholes", &OptionPricing::black_scholes)
        .def_static("monte_carlo_option", &OptionPricing::monte_carlo_option);
}
