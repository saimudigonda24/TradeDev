from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from main_engine import QuantTradingEngine

app = FastAPI(title="QuantTrade Pro API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the trading engine
trading_engine = QuantTradingEngine()

class AnalysisRequest(BaseModel):
    symbols: List[str]
    timeframes: List[str]
    models: List[str]
    riskTolerance: float
    includeOptions: bool = True
    includeShorts: bool = True
    includeETFs: bool = True

class ExportRequest(BaseModel):
    results: Dict
    format: str = "json"

@app.post("/api/analyze")
async def analyze_stocks(request: AnalysisRequest):
    """
    Run comprehensive quantitative analysis on the provided symbols
    """
    try:
        # Convert timeframe format if needed
        timeframe_mapping = {
            '1d': '1d', '1w': '1w', '1m': '1m', 
            '3m': '3m', '6m': '6m', '1y': '1y', '5y': '5y'
        }
        
        mapped_timeframes = [timeframe_mapping.get(tf, tf) for tf in request.timeframes]
        
        # Run the analysis
        results = trading_engine.run_full_analysis(
            symbols=request.symbols,
            timeframes=mapped_timeframes
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/export")
async def export_results(request: ExportRequest):
    """
    Export analysis results in the specified format
    """
    try:
        filepath = trading_engine.export_results(
            report=request.results,
            format=request.format
        )
        
        return {"message": "Export successful", "filepath": filepath}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "QuantTrade Pro API is running"}

@app.get("/api/models")
async def get_available_models():
    """
    Get list of available prediction models
    """
    return {
        "models": [
            {"id": "monte_carlo", "name": "Monte Carlo", "description": "Probabilistic simulation"},
            {"id": "arima", "name": "ARIMA", "description": "Time series analysis"},
            {"id": "ml_ensemble", "name": "ML Ensemble", "description": "Machine learning models"},
            {"id": "technical", "name": "Technical Analysis", "description": "Chart patterns & indicators"}
        ]
    }

@app.get("/api/timeframes")
async def get_available_timeframes():
    """
    Get list of available analysis timeframes
    """
    return {
        "timeframes": [
            {"id": "1d", "name": "1 Day", "description": "Short-term momentum"},
            {"id": "1w", "name": "1 Week", "description": "Weekly trends"},
            {"id": "1m", "name": "1 Month", "description": "Monthly patterns"},
            {"id": "3m", "name": "3 Months", "description": "Quarterly outlook"},
            {"id": "6m", "name": "6 Months", "description": "Medium-term trends"},
            {"id": "1y", "name": "1 Year", "description": "Long-term analysis"},
            {"id": "5y", "name": "5 Years", "description": "Strategic outlook"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
