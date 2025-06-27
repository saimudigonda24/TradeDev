"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Dashboard from "@/components/dashboard"
import ResultsView from "@/components/results-view"
import TradeRecommendations from "@/components/trade-recommendations"
import RiskAnalysis from "@/components/risk-analysis"
import { BarChart3, TrendingUp, Target, Download, Shield } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface AnalysisRequest {
  symbols: string[]
  timeframes: string[]
  models: string[]
  riskTolerance: number
}

interface AnalysisResults {
  timestamp: string
  summary: {
    total_symbols_analyzed: number
    stock_recommendations: number
    short_candidates: number
    option_strategies: number
    etf_recommendations: number
  }
  forecasts: Record<string, any>
  trade_recommendations: {
    top_longs: any[]
    top_shorts: any[]
    best_options: any[]
    etf_plays: any[]
  }
  risk_analysis: {
    portfolio_var: number
    max_drawdown: number
    sharpe_ratio: number
    concentration_risk: number
    stress_test_results: Record<string, number>
  }
  market_outlook: {
    outlook: string
    confidence: number
    buy_percentage: number
    sell_percentage: number
    hold_percentage: number
  }
}

const Index = () => {
  const [activeTab, setActiveTab] = useState("dashboard")
  const [hasResults, setHasResults] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null)
  const { toast } = useToast()

  // Mock data for demonstration (keeping your original data)
  const mockPredictions = [
    {
      ticker: "AAPL",
      currentPrice: 185.2,
      predictedHigh: 205.5,
      predictedLow: 175.8,
      movement: 8.5,
      confidence: 0.87,
      classification: "up" as const,
      volatility: 22.3,
      riskScore: 4,
    },
    {
      ticker: "GOOGL",
      currentPrice: 142.35,
      predictedHigh: 158.9,
      predictedLow: 138.2,
      movement: 5.2,
      confidence: 0.79,
      classification: "up" as const,
      volatility: 28.7,
      riskScore: 5,
    },
    {
      ticker: "TSLA",
      currentPrice: 248.7,
      predictedHigh: 235.4,
      predictedLow: 220.15,
      movement: -7.8,
      confidence: 0.82,
      classification: "down" as const,
      volatility: 45.2,
      riskScore: 8,
    },
    {
      ticker: "MSFT",
      currentPrice: 378.85,
      predictedHigh: 385.2,
      predictedLow: 372.1,
      movement: 1.2,
      confidence: 0.73,
      classification: "neutral" as const,
      volatility: 18.9,
      riskScore: 3,
    },
  ]

  const mockChartData = [
    { date: "2024-01", current: 180, high: 185, low: 175 },
    { date: "2024-02", current: 182, high: 188, low: 177 },
    { date: "2024-03", current: 185, high: 192, low: 180 },
    { date: "2024-04", current: 190, high: 198, low: 185 },
    { date: "2024-05", current: 195, high: 205, low: 188 },
    { date: "2024-06", current: 200, high: 210, low: 195 },
  ]

  const mockTrades = [
    {
      id: "1",
      ticker: "AAPL",
      type: "long" as const,
      strategy: "Momentum Breakout",
      entryPrice: 185.2,
      targetPrice: 205.5,
      stopLoss: 175.8,
      expectedReturn: 10.96,
      riskScore: 4,
      timeframe: "3M",
      sector: "Technology",
      confidence: 0.87,
    },
    {
      id: "2",
      ticker: "TSLA",
      type: "short" as const,
      strategy: "Reversal Pattern",
      entryPrice: 248.7,
      targetPrice: 220.15,
      stopLoss: 265.0,
      expectedReturn: 11.48,
      riskScore: 8,
      timeframe: "2M",
      sector: "Automotive",
      confidence: 0.82,
    },
    {
      id: "3",
      ticker: "GOOGL",
      type: "call" as const,
      strategy: "Earnings Play Call",
      entryPrice: 5.2,
      targetPrice: 12.5,
      stopLoss: 2.0,
      expectedReturn: 140.38,
      riskScore: 7,
      timeframe: "1M",
      sector: "Technology",
      confidence: 0.79,
    },
    {
      id: "4",
      ticker: "SPY",
      type: "long" as const,
      strategy: "Index Trend Following",
      entryPrice: 455.8,
      targetPrice: 475.2,
      stopLoss: 445.5,
      expectedReturn: 4.26,
      riskScore: 2,
      timeframe: "6M",
      sector: "Index",
      confidence: 0.91,
    },
    {
      id: "5",
      ticker: "NVDA",
      type: "put" as const,
      strategy: "Volatility Hedge",
      entryPrice: 8.4,
      targetPrice: 18.75,
      stopLoss: 4.2,
      expectedReturn: 123.21,
      riskScore: 9,
      timeframe: "1M",
      sector: "Technology",
      confidence: 0.68,
    },
    {
      id: "6",
      ticker: "QQQ",
      type: "long" as const,
      strategy: "Tech Sector Rotation",
      entryPrice: 385.9,
      targetPrice: 410.3,
      stopLoss: 375.2,
      expectedReturn: 6.32,
      riskScore: 3,
      timeframe: "4M",
      sector: "Technology",
      confidence: 0.85,
    },
  ]

  const handleSubmit = async (data: AnalysisRequest) => {
    setIsLoading(true)

    try {
      // Call the backend API
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        throw new Error("Analysis failed")
      }

      const results: AnalysisResults = await response.json()
      setAnalysisResults(results)
      setHasResults(true)
      setActiveTab("results")

      toast({
        title: "Analysis Complete",
        description: `Successfully analyzed ${results.summary.total_symbols_analyzed} symbols with ${results.summary.stock_recommendations} recommendations.`,
      })
    } catch (error) {
      console.error("Analysis error:", error)

      // For demo purposes, use mock data
      const mockResults: AnalysisResults = {
        timestamp: new Date().toISOString(),
        summary: {
          total_symbols_analyzed: data.symbols.length,
          stock_recommendations: 8,
          short_candidates: 3,
          option_strategies: 12,
          etf_recommendations: 4,
        },
        forecasts: {},
        trade_recommendations: {
          top_longs: mockTrades.filter((t) => t.type === "long"),
          top_shorts: mockTrades.filter((t) => t.type === "short"),
          best_options: mockTrades.filter((t) => t.type === "call" || t.type === "put"),
          etf_plays: mockTrades.filter((t) => t.ticker.includes("SPY") || t.ticker.includes("QQQ")),
        },
        risk_analysis: {
          portfolio_var: 0.023,
          max_drawdown: 0.156,
          sharpe_ratio: 1.34,
          concentration_risk: 0.18,
          stress_test_results: {
            market_crash: -0.28,
            sector_rotation: -0.12,
            volatility_spike: -0.19,
          },
        },
        market_outlook: {
          outlook: "BULLISH",
          confidence: 0.73,
          buy_percentage: 0.62,
          sell_percentage: 0.23,
          hold_percentage: 0.15,
        },
      }

      setAnalysisResults(mockResults)
      setHasResults(true)
      setActiveTab("results")

      toast({
        title: "Demo Mode",
        description: "Using mock data for demonstration. Backend integration in progress.",
        variant: "default",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleExport = async () => {
    if (!analysisResults) return

    try {
      const response = await fetch("/api/export", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          results: analysisResults,
          format: "json",
        }),
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `trading_analysis_${new Date().toISOString().split("T")[0]}.json`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)

        toast({
          title: "Export Successful",
          description: "Analysis results have been downloaded.",
        })
      }
    } catch (error) {
      // Fallback export
      const dataStr = JSON.stringify(analysisResults, null, 2)
      const dataBlob = new Blob([dataStr], { type: "application/json" })
      const url = URL.createObjectURL(dataBlob)
      const link = document.createElement("a")
      link.href = url
      link.download = `trading_analysis_${new Date().toISOString().split("T")[0]}.json`
      link.click()
      URL.revokeObjectURL(url)

      toast({
        title: "Export Complete",
        description: "Analysis results downloaded as JSON file.",
      })
    }
  }

  const getMarketOutlookColor = (outlook: string) => {
    switch (outlook) {
      case "BULLISH":
        return "text-green-400"
      case "BEARISH":
        return "text-red-400"
      default:
        return "text-yellow-400"
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                <BarChart3 className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-100">QuantTrade Pro</h1>
                <p className="text-sm text-slate-400">AI-Powered Trading Intelligence</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Market Outlook Indicator */}
              {analysisResults && (
                <div className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-lg border border-slate-700">
                  <div
                    className={`h-2 w-2 rounded-full ${
                      analysisResults.market_outlook.outlook === "BULLISH"
                        ? "bg-green-400"
                        : analysisResults.market_outlook.outlook === "BEARISH"
                          ? "bg-red-400"
                          : "bg-yellow-400"
                    } animate-pulse`}
                  ></div>
                  <span
                    className={`text-sm font-medium ${getMarketOutlookColor(analysisResults.market_outlook.outlook)}`}
                  >
                    {analysisResults.market_outlook.outlook}
                  </span>
                  <span className="text-xs text-slate-400">
                    {(analysisResults.market_outlook.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              )}

              <Button
                variant="outline"
                className="border-slate-600 text-slate-200 hover:bg-slate-800 bg-transparent"
                onClick={handleExport}
                disabled={!hasResults}
              >
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>

              <div className="h-8 w-8 bg-slate-700 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-slate-200">U</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-slate-800/50 border border-slate-700">
            <TabsTrigger value="dashboard" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
              <BarChart3 className="h-4 w-4 mr-2" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger
              value="results"
              className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
              disabled={!hasResults}
            >
              <TrendingUp className="h-4 w-4 mr-2" />
              Results
            </TabsTrigger>
            <TabsTrigger
              value="trades"
              className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
              disabled={!hasResults}
            >
              <Target className="h-4 w-4 mr-2" />
              Trade Ideas
            </TabsTrigger>
            <TabsTrigger
              value="risk"
              className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
              disabled={!hasResults}
            >
              <Shield className="h-4 w-4 mr-2" />
              Risk Analysis
            </TabsTrigger>
          </TabsList>

          <div className="mt-8">
            <TabsContent value="dashboard" className="space-y-6">
              <Dashboard onSubmit={handleSubmit} isLoading={isLoading} />
            </TabsContent>

            <TabsContent value="results" className="space-y-6">
              <ResultsView predictions={mockPredictions} chartData={mockChartData} analysisResults={analysisResults} />
            </TabsContent>

            <TabsContent value="trades" className="space-y-6">
              <TradeRecommendations trades={mockTrades} analysisResults={analysisResults} />
            </TabsContent>

            <TabsContent value="risk" className="space-y-6">
              <RiskAnalysis analysisResults={analysisResults} />
            </TabsContent>
          </div>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/30 mt-16">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between text-sm text-slate-400">
            <div>© 2024 QuantTrade Pro. All rights reserved.</div>
            <div className="flex items-center gap-4">
              <span>Powered by C++ & Python AI Engine</span>
              <div className="flex items-center gap-1">
                <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live Data</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default Index
