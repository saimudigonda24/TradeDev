"use client"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

interface Trade {
  id: string
  ticker: string
  type: string
  strategy: string
  entryPrice: number
  targetPrice: number
  stopLoss: number
  expectedReturn: number
  riskScore: number
  timeframe: string
  sector: string
  confidence: number
}
interface Props {
  trades: Trade[]
  analysisResults: any | null
}

export default function TradeRecommendations({ trades }: Props) {
  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-slate-100">Trade Ideas</CardTitle>
        <CardDescription className="text-slate-400">Highest risk-adjusted opportunities</CardDescription>
      </CardHeader>
      <Separator className="bg-slate-700" />
      <CardContent className="overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-slate-400">
            <tr className="text-left">
              <th className="p-2">Ticker</th>
              <th className="p-2">Type</th>
              <th className="p-2">Strategy</th>
              <th className="p-2">Entry</th>
              <th className="p-2">Target</th>
              <th className="p-2">Stop</th>
              <th className="p-2">Return %</th>
              <th className="p-2">Risk</th>
            </tr>
          </thead>
          <tbody className="text-slate-200">
            {trades.map((t) => (
              <tr key={t.id} className="border-t border-slate-700">
                <td className="p-2 font-medium text-slate-100">{t.ticker}</td>
                <td className="p-2 capitalize text-slate-200">{t.type}</td>
                <td className="p-2 text-slate-200">{t.strategy}</td>
                <td className="p-2 text-slate-200">${t.entryPrice}</td>
                <td className="p-2 text-slate-200">${t.targetPrice}</td>
                <td className="p-2 text-slate-200">${t.stopLoss}</td>
                <td className={`p-2 font-medium ${t.expectedReturn >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {t.expectedReturn.toFixed(1)}%
                </td>
                <td className="p-2 text-slate-200">{t.riskScore}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}
