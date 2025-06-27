"use client"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

interface Prediction {
  ticker: string
  currentPrice: number
  predictedHigh: number
  predictedLow: number
  movement: number
  confidence: number
  classification: "up" | "down" | "neutral"
  volatility: number
  riskScore: number
}
interface ChartPoint {
  date: string
  current: number
  high: number
  low: number
}
interface ResultsViewProps {
  predictions: Prediction[]
  chartData: ChartPoint[]
  analysisResults: any | null
}

export default function ResultsView({ predictions, chartData, analysisResults }: ResultsViewProps) {
  return (
    <div className="space-y-6">
      {/* Summary */}
      {analysisResults && (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100">Summary</CardTitle>
            <CardDescription className="text-slate-400">
              {analysisResults.summary.total_symbols_analyzed} symbols analysed –{" "}
              {analysisResults.summary.stock_recommendations} stock recommendations
            </CardDescription>
          </CardHeader>
          <CardContent className="text-slate-300 grid grid-cols-2 gap-4 md:grid-cols-4">
            <div>
              <p className="text-xl font-bold text-blue-400">{analysisResults.summary.stock_recommendations}</p>
              <span className="text-xs text-slate-400">Stock Ideas</span>
            </div>
            <div>
              <p className="text-xl font-bold text-red-400">{analysisResults.summary.short_candidates}</p>
              <span className="text-xs text-slate-400">Shorts</span>
            </div>
            <div>
              <p className="text-xl font-bold text-purple-400">{analysisResults.summary.option_strategies}</p>
              <span className="text-xs text-slate-400">Option Plays</span>
            </div>
            <div>
              <p className="text-xl font-bold text-emerald-400">{analysisResults.summary.etf_recommendations}</p>
              <span className="text-xs text-slate-400">ETF Picks</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Predictions list */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-slate-100">Forecasts</CardTitle>
          <CardDescription className="text-slate-400">Model-blended price ranges &amp; confidence</CardDescription>
        </CardHeader>
        <Separator className="bg-slate-700" />
        <CardContent className="overflow-auto">
          <table className="w-full text-sm">
            <thead className="text-slate-400">
              <tr className="text-left">
                <th className="p-2">Ticker</th>
                <th className="p-2">Now</th>
                <th className="p-2">High</th>
                <th className="p-2">Low</th>
                <th className="p-2">Move %</th>
                <th className="p-2">Conf.</th>
                <th className="p-2">Class</th>
              </tr>
            </thead>
            <tbody className="text-slate-200">
              {predictions.map((p) => (
                <tr key={p.ticker} className="border-t border-slate-700">
                  <td className="p-2 font-medium text-slate-100">{p.ticker}</td>
                  <td className="p-2 text-slate-200">${p.currentPrice.toFixed(2)}</td>
                  <td className="p-2 text-slate-200">${p.predictedHigh.toFixed(2)}</td>
                  <td className="p-2 text-slate-200">${p.predictedLow.toFixed(2)}</td>
                  <td
                    className={`p-2 font-medium ${
                      p.movement > 0 ? "text-green-400" : p.movement < 0 ? "text-red-400" : "text-yellow-400"
                    }`}
                  >
                    {p.movement.toFixed(1)}%
                  </td>
                  <td className="p-2 text-slate-200">{(p.confidence * 100).toFixed(0)}%</td>
                  <td className="p-2 capitalize text-slate-200">{p.classification}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* TODO: replace with a proper line/area chart later */}
      <pre className="text-xs text-slate-300 bg-slate-800/30 p-4 rounded-lg border border-slate-700">
        {JSON.stringify(chartData.slice(0, 3), null, 2)} …
      </pre>
    </div>
  )
}
