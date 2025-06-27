import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { results, format = "json" } = body

    if (format === "json") {
      const jsonData = JSON.stringify(results, null, 2)

      return new NextResponse(jsonData, {
        headers: {
          "Content-Type": "application/json",
          "Content-Disposition": `attachment; filename="trading_analysis_${new Date().toISOString().split("T")[0]}.json"`,
        },
      })
    }

    if (format === "csv") {
      // Convert results to CSV format
      const trades = results.trade_recommendations?.top_longs || []
      const csvHeader = "Symbol,Type,Entry Price,Target Price,Stop Loss,Expected Return,Risk Score,Confidence\n"
      const csvData = trades
        .map(
          (trade: any) =>
            `${trade.symbol},${trade.trade_type},${trade.entry_price},${trade.target_price},${trade.stop_loss},${trade.expected_return},${trade.risk_score || "N/A"},${trade.confidence}`,
        )
        .join("\n")

      return new NextResponse(csvHeader + csvData, {
        headers: {
          "Content-Type": "text/csv",
          "Content-Disposition": `attachment; filename="trading_analysis_${new Date().toISOString().split("T")[0]}.csv"`,
        },
      })
    }

    return NextResponse.json({ error: "Unsupported format" }, { status: 400 })
  } catch (error) {
    console.error("Export API error:", error)
    return NextResponse.json({ error: "Export failed" }, { status: 500 })
  }
}
