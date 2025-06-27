"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Play, Settings, TrendingUp, Brain, Target, Shield } from "lucide-react"

interface DashboardProps {
  onSubmit: (data: any) => void
  isLoading?: boolean
}

const Dashboard: React.FC<DashboardProps> = ({ onSubmit, isLoading = false }) => {
  const [symbols, setSymbols] = useState("AAPL,GOOGL,MSFT,TSLA,NVDA,AMD,META,NFLX")
  const [timeframes, setTimeframes] = useState(["1m", "3m", "6m", "1y"])
  const [models, setModels] = useState(["monte_carlo", "arima", "ml_ensemble"])
  const [riskTolerance, setRiskTolerance] = useState([5])
  const [includeOptions, setIncludeOptions] = useState(true)
  const [includeShorts, setIncludeShorts] = useState(true)
  const [includeETFs, setIncludeETFs] = useState(true)

  const timeframeOptions = [
    { id: "1d", label: "1 Day", description: "Short-term momentum" },
    { id: "1w", label: "1 Week", description: "Weekly trends" },
    { id: "1m", label: "1 Month", description: "Monthly patterns" },
    { id: "3m", label: "3 Months", description: "Quarterly outlook" },
    { id: "6m", label: "6 Months", description: "Medium-term trends" },
    { id: "1y", label: "1 Year", description: "Long-term analysis" },
    { id: "5y", label: "5 Years", description: "Strategic outlook" },
  ]

  const modelOptions = [
    { id: "monte_carlo", label: "Monte Carlo", description: "Probabilistic simulation" },
    { id: "arima", label: "ARIMA", description: "Time series analysis" },
    { id: "ml_ensemble", label: "ML Ensemble", description: "Machine learning models" },
    { id: "technical", label: "Technical Analysis", description: "Chart patterns & indicators" },
  ]

  const popularSymbols = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "TSLA",
    "NVDA",
    "AMD",
    "META",
    "NFLX",
    "AMZN",
    "SPY",
    "QQQ",
    "IWM",
    "VTI",
    "BTC-USD",
    "ETH-USD",
  ]

  const handleTimeframeChange = (timeframeId: string, checked: boolean) => {
    if (checked) {
      setTimeframes([...timeframes, timeframeId])
    } else {
      setTimeframes(timeframes.filter((t) => t !== timeframeId))
    }
  }

  const handleModelChange = (modelId: string, checked: boolean) => {
    if (checked) {
      setModels([...models, modelId])
    } else {
      setModels(models.filter((m) => m !== modelId))
    }
  }

  const addSymbol = (symbol: string) => {
    const currentSymbols = symbols
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s)
    if (!currentSymbols.includes(symbol)) {
      setSymbols([...currentSymbols, symbol].join(","))
    }
  }

  const removeSymbol = (symbolToRemove: string) => {
    const currentSymbols = symbols
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s)
    setSymbols(currentSymbols.filter((s) => s !== symbolToRemove).join(","))
  }

  const handleSubmit = () => {
    const symbolList = symbols
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s)

    if (symbolList.length === 0) {
      alert("Please enter at least one symbol")
      return
    }

    if (timeframes.length === 0) {
      alert("Please select at least one timeframe")
      return
    }

    if (models.length === 0) {
      alert("Please select at least one model")
      return
    }

    onSubmit({
      symbols: symbolList,
      timeframes,
      models,
      riskTolerance: riskTolerance[0],
      includeOptions,
      includeShorts,
      includeETFs,
    })
  }

  const currentSymbols = symbols
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s)

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Configuration Panel */}
      <div className="lg:col-span-2 space-y-6">
        {/* Symbols Input */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100 flex items-center gap-2">
              <Target className="h-5 w-5 text-blue-400" />
              Stock Symbols
            </CardTitle>
            <CardDescription className="text-slate-400">
              Enter stock symbols to analyze (comma-separated)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="symbols" className="text-slate-200">
                Symbols
              </Label>
              <Input
                id="symbols"
                value={symbols}
                onChange={(e) => setSymbols(e.target.value)}
                placeholder="AAPL,GOOGL,MSFT,TSLA..."
                className="bg-slate-700 border-slate-600 text-slate-100 placeholder:text-slate-400 mt-2"
              />
            </div>

            {/* Current Symbols */}
            {currentSymbols.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {currentSymbols.map((symbol) => (
                  <Badge
                    key={symbol}
                    variant="secondary"
                    className="bg-blue-600/20 text-blue-300 border-blue-500/30 cursor-pointer hover:bg-red-600/20 hover:text-red-300"
                    onClick={() => removeSymbol(symbol)}
                  >
                    {symbol} ×
                  </Badge>
                ))}
              </div>
            )}

            {/* Popular Symbols */}
            <div>
              <Label className="text-slate-100 text-sm">Popular Symbols</Label>
              <div className="flex flex-wrap gap-2 mt-2">
                {popularSymbols.map((symbol) => (
                  <Badge
                    key={symbol}
                    variant="outline"
                    className="border-slate-600 text-slate-300 cursor-pointer hover:bg-slate-700"
                    onClick={() => addSymbol(symbol)}
                  >
                    + {symbol}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Timeframes */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100 flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-green-400" />
              Analysis Timeframes
            </CardTitle>
            <CardDescription className="text-slate-400">Select prediction timeframes for analysis</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {timeframeOptions.map((timeframe) => (
                <div key={timeframe.id} className="flex items-start space-x-2">
                  <Checkbox
                    id={timeframe.id}
                    checked={timeframes.includes(timeframe.id)}
                    onCheckedChange={(checked) => handleTimeframeChange(timeframe.id, checked as boolean)}
                    className="border-slate-600 data-[state=checked]:bg-blue-600"
                  />
                  <div className="grid gap-1.5 leading-none">
                    <label htmlFor={timeframe.id} className="text-sm font-medium text-slate-100 cursor-pointer">
                      {timeframe.label}
                    </label>
                    <p className="text-xs text-slate-400">{timeframe.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Models */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100 flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-400" />
              Prediction Models
            </CardTitle>
            <CardDescription className="text-slate-400">Choose forecasting models to combine</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {modelOptions.map((model) => (
                <div key={model.id} className="flex items-start space-x-2">
                  <Checkbox
                    id={model.id}
                    checked={models.includes(model.id)}
                    onCheckedChange={(checked) => handleModelChange(model.id, checked as boolean)}
                    className="border-slate-600 data-[state=checked]:bg-purple-600"
                  />
                  <div className="grid gap-1.5 leading-none">
                    <label htmlFor={model.id} className="text-sm font-medium text-slate-100 cursor-pointer">
                      {model.label}
                    </label>
                    <p className="text-xs text-slate-400">{model.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Settings Panel */}
      <div className="space-y-6">
        {/* Risk Settings */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100 flex items-center gap-2">
              <Shield className="h-5 w-5 text-orange-400" />
              Risk Settings
            </CardTitle>
            <CardDescription className="text-slate-400">Configure risk tolerance and preferences</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <Label className="text-slate-200">Risk Tolerance: {riskTolerance[0]}%</Label>
              <Slider
                value={riskTolerance}
                onValueChange={setRiskTolerance}
                max={20}
                min={1}
                step={1}
                className="mt-2"
              />
              <div className="flex justify-between text-xs text-slate-400 mt-1">
                <span>Conservative</span>
                <span>Aggressive</span>
              </div>
            </div>

            <Separator className="bg-slate-700" />

            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="options"
                  checked={includeOptions}
                  onCheckedChange={setIncludeOptions}
                  className="border-slate-600 data-[state=checked]:bg-blue-600"
                />
                <label htmlFor="options" className="text-sm text-slate-100 cursor-pointer">
                  Include Options Strategies
                </label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="shorts"
                  checked={includeShorts}
                  onCheckedChange={setIncludeShorts}
                  className="border-slate-600 data-[state=checked]:bg-blue-600"
                />
                <label htmlFor="shorts" className="text-sm text-slate-100 cursor-pointer">
                  Include Short Selling
                </label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="etfs"
                  checked={includeETFs}
                  onCheckedChange={setIncludeETFs}
                  className="border-slate-600 data-[state=checked]:bg-blue-600"
                />
                <label htmlFor="etfs" className="text-sm text-slate-100 cursor-pointer">
                  Include ETF Analysis
                </label>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Analysis Summary */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100">Analysis Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Symbols:</span>
              <span className="text-slate-200">{currentSymbols.length}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Timeframes:</span>
              <span className="text-slate-200">{timeframes.length}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Models:</span>
              <span className="text-slate-200">{models.length}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Risk Level:</span>
              <span className="text-slate-200">{riskTolerance[0]}%</span>
            </div>
          </CardContent>
        </Card>

        {/* Run Analysis Button */}
        <Button
          onClick={handleSubmit}
          disabled={isLoading || currentSymbols.length === 0}
          className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-medium py-3"
        >
          {isLoading ? (
            <>
              <Settings className="h-4 w-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Run Analysis
            </>
          )}
        </Button>
      </div>
    </div>
  )
}

export default Dashboard
