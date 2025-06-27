"use client"

import type React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Shield, TrendingDown, AlertTriangle, Target, BarChart3, Activity, Zap } from "lucide-react"

interface RiskAnalysisProps {
  analysisResults: any
}

const RiskAnalysis: React.FC<RiskAnalysisProps> = ({ analysisResults }) => {
  if (!analysisResults) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Shield className="h-12 w-12 text-slate-400 mx-auto mb-4" />
          <p className="text-slate-400">No risk analysis data available</p>
        </div>
      </div>
    )
  }

  const { risk_analysis } = analysisResults

  const getRiskLevel = (value: number, thresholds: number[]) => {
    if (value <= thresholds[0]) return { level: "Low", color: "text-green-400", bg: "bg-green-400/20" }
    if (value <= thresholds[1]) return { level: "Medium", color: "text-yellow-400", bg: "bg-yellow-400/20" }
    return { level: "High", color: "text-red-400", bg: "bg-red-400/20" }
  }

  const varRisk = getRiskLevel(risk_analysis.portfolio_var * 100, [2, 5])
  const drawdownRisk = getRiskLevel(risk_analysis.max_drawdown * 100, [10, 20])
  const concentrationRisk = getRiskLevel(risk_analysis.concentration_risk * 100, [20, 40])

  const getSharpeRating = (sharpe: number) => {
    if (sharpe > 2) return { rating: "Excellent", color: "text-green-400" }
    if (sharpe > 1) return { rating: "Good", color: "text-blue-400" }
    if (sharpe > 0.5) return { rating: "Fair", color: "text-yellow-400" }
    return { rating: "Poor", color: "text-red-400" }
  }

  const sharpeRating = getSharpeRating(risk_analysis.sharpe_ratio)

  return (
    <div className="space-y-6">
      {/* Risk Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">Portfolio VaR (95%)</p>
                <p className="text-2xl font-bold text-slate-100">{(risk_analysis.portfolio_var * 100).toFixed(2)}%</p>
              </div>
              <div className={`p-2 rounded-lg ${varRisk.bg}`}>
                <TrendingDown className={`h-5 w-5 ${varRisk.color}`} />
              </div>
            </div>
            <Badge variant="outline" className={`mt-2 ${varRisk.color} border-current`}>
              {varRisk.level} Risk
            </Badge>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">Max Drawdown</p>
                <p className="text-2xl font-bold text-slate-100">{(risk_analysis.max_drawdown * 100).toFixed(1)}%</p>
              </div>
              <div className={`p-2 rounded-lg ${drawdownRisk.bg}`}>
                <BarChart3 className={`h-5 w-5 ${drawdownRisk.color}`} />
              </div>
            </div>
            <Badge variant="outline" className={`mt-2 ${drawdownRisk.color} border-current`}>
              {drawdownRisk.level} Risk
            </Badge>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">Sharpe Ratio</p>
                <p className="text-2xl font-bold text-slate-100">{risk_analysis.sharpe_ratio.toFixed(2)}</p>
              </div>
              <div className="p-2 rounded-lg bg-blue-400/20">
                <Activity className="h-5 w-5 text-blue-400" />
              </div>
            </div>
            <Badge variant="outline" className={`mt-2 ${sharpeRating.color} border-current`}>
              {sharpeRating.rating}
            </Badge>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">Concentration</p>
                <p className="text-2xl font-bold text-slate-100">
                  {(risk_analysis.concentration_risk * 100).toFixed(1)}%
                </p>
              </div>
              <div className={`p-2 rounded-lg ${concentrationRisk.bg}`}>
                <Target className={`h-5 w-5 ${concentrationRisk.color}`} />
              </div>
            </div>
            <Badge variant="outline" className={`mt-2 ${concentrationRisk.color} border-current`}>
              {concentrationRisk.level} Risk
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Stress Test Results */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-slate-100 flex items-center gap-2">
            <Zap className="h-5 w-5 text-orange-400" />
            Stress Test Results
          </CardTitle>
          <CardDescription className="text-slate-400">
            Portfolio performance under adverse market conditions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {Object.entries(risk_analysis.stress_test_results).map(([scenario, impact]) => {
            const impactPercent = (impact as number) * 100
            const severity = Math.abs(impactPercent)
            const severityColor = severity > 25 ? "text-red-400" : severity > 15 ? "text-yellow-400" : "text-green-400"

            return (
              <div key={scenario} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-slate-200 capitalize">{scenario.replace("_", " ")}</span>
                  <span className={`font-medium ${severityColor}`}>
                    {impactPercent > 0 ? "+" : ""}
                    {impactPercent.toFixed(1)}%
                  </span>
                </div>
                <Progress value={Math.min(Math.abs(impactPercent), 50)} className="h-2" />
              </div>
            )
          })}
        </CardContent>
      </Card>

      {/* Risk Alerts */}
      <div className="space-y-4">
        {risk_analysis.portfolio_var > 0.05 && (
          <Alert className="border-red-500/50 bg-red-500/10">
            <AlertTriangle className="h-4 w-4 text-red-400" />
            <AlertDescription className="text-red-300">
              <strong>High VaR Warning:</strong> Portfolio VaR exceeds 5% threshold. Consider reducing position sizes or
              diversifying holdings.
            </AlertDescription>
          </Alert>
        )}

        {risk_analysis.max_drawdown > 0.2 && (
          <Alert className="border-orange-500/50 bg-orange-500/10">
            <AlertTriangle className="h-4 w-4 text-orange-400" />
            <AlertDescription className="text-orange-300">
              <strong>Drawdown Risk:</strong> Maximum drawdown exceeds 20%. Review stop-loss strategies and position
              sizing.
            </AlertDescription>
          </Alert>
        )}

        {risk_analysis.concentration_risk > 0.4 && (
          <Alert className="border-yellow-500/50 bg-yellow-500/10">
            <AlertTriangle className="h-4 w-4 text-yellow-400" />
            <AlertDescription className="text-yellow-300">
              <strong>Concentration Risk:</strong> Portfolio is highly concentrated. Consider diversifying across more
              assets or sectors.
            </AlertDescription>
          </Alert>
        )}

        {risk_analysis.sharpe_ratio < 0.5 && (
          <Alert className="border-blue-500/50 bg-blue-500/10">
            <AlertTriangle className="h-4 w-4 text-blue-400" />
            <AlertDescription className="text-blue-300">
              <strong>Low Risk-Adjusted Returns:</strong> Sharpe ratio is below optimal levels. Review strategy
              performance and risk management.
            </AlertDescription>
          </Alert>
        )}
      </div>

      {/* Risk Metrics Details */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100">Risk Metrics</CardTitle>
            <CardDescription className="text-slate-400">Detailed portfolio risk measurements</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between">
              <span className="text-slate-400">Value at Risk (95%)</span>
              <span className="text-slate-200">{(risk_analysis.portfolio_var * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Conditional VaR</span>
              <span className="text-slate-200">{(risk_analysis.portfolio_cvar * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Maximum Drawdown</span>
              <span className="text-slate-200">{(risk_analysis.max_drawdown * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Sharpe Ratio</span>
              <span className="text-slate-200">{risk_analysis.sharpe_ratio.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Sortino Ratio</span>
              <span className="text-slate-200">{risk_analysis.sortino_ratio.toFixed(2)}</span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100">Market Exposure</CardTitle>
            <CardDescription className="text-slate-400">Portfolio correlation and beta analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between">
              <span className="text-slate-400">Portfolio Beta</span>
              <span className="text-slate-200">{risk_analysis.beta.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Market Correlation</span>
              <span className="text-slate-200">{(risk_analysis.correlation_with_market * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Concentration Risk</span>
              <span className="text-slate-200">{(risk_analysis.concentration_risk * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Diversification Score</span>
              <span className="text-slate-200">{((1 - risk_analysis.concentration_risk) * 100).toFixed(1)}%</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default RiskAnalysis
