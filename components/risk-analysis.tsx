"use client"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"

interface Props {
  analysisResults: any | null
}

export default function RiskAnalysis({ analysisResults }: Props) {
  if (!analysisResults) {
    return (
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-slate-100">Risk Analysis</CardTitle>
          <CardDescription className="text-slate-400">Run an analysis first to view portfolio risk.</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  const r = analysisResults.risk_analysis
  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-slate-100">Risk Analysis</CardTitle>
        <CardDescription className="text-slate-400">Portfolio-level risk metrics</CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div>
          <p className="text-2xl font-bold text-slate-100">{(r.portfolio_var * 100).toFixed(2)}%</p>
          <span className="text-sm text-slate-400">VaR</span>
        </div>
        <div>
          <p className="text-2xl font-bold text-slate-100">{(r.max_drawdown * 100).toFixed(1)}%</p>
          <span className="text-sm text-slate-400">Max Drawdown</span>
        </div>
        <div>
          <p className="text-2xl font-bold text-slate-100">{r.sharpe_ratio.toFixed(2)}</p>
          <span className="text-sm text-slate-400">Sharpe Ratio</span>
        </div>
        <div>
          <p className="text-2xl font-bold text-slate-100">{(r.concentration_risk * 100).toFixed(0)}%</p>
          <span className="text-sm text-slate-400">Concentration</span>
        </div>
      </CardContent>
    </Card>
  )
}
