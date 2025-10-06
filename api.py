from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
import pandas as pd
import json
import os
from datetime import datetime

app = FastAPI(
    title="Inventory Forecasting API",
    description="API for accessing inventory, inflows, and outflows datasets",
    version="1.0.0"
)

DATA_DIR = "api_data"

def load_dataset(dataset_type: str, period: str) -> Optional[pd.DataFrame]:
    """
    Load dataset from JSON file
    
    Args:
        dataset_type: Type of dataset (inventory, inflows, outflows)
        period: Period of dataset (train, tune, test)
    
    Returns:
        DataFrame or None if not found
    """
    filename = f"{period}_{dataset_type}.json"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Inventory Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "datasets": "/datasets",
            "inventory": "/inventory/{period}",
            "inflows": "/inflows/{period}",
            "outflows": "/outflows/{period}",
            "summary": "/summary",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/datasets")
async def list_datasets():
    """List all available datasets"""
    if not os.path.exists(DATA_DIR):
        return {"datasets": [], "message": "No datasets available"}
    
    available_datasets = []
    files = os.listdir(DATA_DIR)
    
    for file in files:
        if file.endswith('.json'):
            parts = file.replace('.json', '').split('_', 1)
            if len(parts) == 2:
                period, dataset_type = parts
                filepath = os.path.join(DATA_DIR, file)
                file_size = os.path.getsize(filepath)
                
                df = load_dataset(dataset_type, period)
                row_count = len(df) if df is not None else 0
                
                available_datasets.append({
                    "period": period,
                    "type": dataset_type,
                    "filename": file,
                    "size_bytes": file_size,
                    "row_count": row_count
                })
    
    return {
        "datasets": available_datasets,
        "count": len(available_datasets)
    }

@app.get("/inventory/{period}")
async def get_inventory(
    period: str,
    limit: Optional[int] = Query(None, description="Limit number of records"),
    offset: Optional[int] = Query(0, description="Offset for pagination"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)")
):
    """
    Get inventory data for specified period
    
    Args:
        period: train, tune, or test
        limit: Maximum number of records to return
        offset: Number of records to skip
        date_from: Start date filter
        date_to: End date filter
    """
    df = load_dataset('inventory', period)
    
    if df is None:
        raise HTTPException(status_code=404, detail=f"Inventory data for {period} period not found")
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
        if date_from:
            df = df[df['Date'] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df['Date'] <= pd.to_datetime(date_to)]
        
        df = df.sort_values('Date')
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    total_records = len(df)
    
    if offset:
        df = df.iloc[offset:]
    if limit:
        df = df.head(limit)
    
    return {
        "period": period,
        "type": "inventory",
        "total_records": total_records,
        "returned_records": len(df),
        "data": df.to_dict(orient='records')
    }

@app.get("/inflows/{period}")
async def get_inflows(
    period: str,
    limit: Optional[int] = Query(None, description="Limit number of records"),
    offset: Optional[int] = Query(0, description="Offset for pagination"),
    warehouse: Optional[str] = Query(None, description="Filter by warehouse"),
    category: Optional[str] = Query(None, description="Filter by category"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)")
):
    """
    Get inflows data for specified period
    
    Args:
        period: train, tune, or test
        limit: Maximum number of records to return
        offset: Number of records to skip
        warehouse: Filter by warehouse name
        category: Filter by category
        date_from: Start date filter
        date_to: End date filter
    """
    df = load_dataset('inflows', period)
    
    if df is None:
        raise HTTPException(status_code=404, detail=f"Inflows data for {period} period not found")
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
        if date_from:
            df = df[df['Date'] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df['Date'] <= pd.to_datetime(date_to)]
    
    if warehouse and 'Warehouse' in df.columns:
        df = df[df['Warehouse'] == warehouse]
    
    if category and 'Category' in df.columns:
        df = df[df['Category'] == category]
    
    if 'Date' in df.columns:
        df = df.sort_values('Date')
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    total_records = len(df)
    
    if offset:
        df = df.iloc[offset:]
    if limit:
        df = df.head(limit)
    
    return {
        "period": period,
        "type": "inflows",
        "total_records": total_records,
        "returned_records": len(df),
        "filters": {
            "warehouse": warehouse,
            "category": category,
            "date_from": date_from,
            "date_to": date_to
        },
        "data": df.to_dict(orient='records')
    }

@app.get("/outflows/{period}")
async def get_outflows(
    period: str,
    limit: Optional[int] = Query(None, description="Limit number of records"),
    offset: Optional[int] = Query(0, description="Offset for pagination"),
    warehouse: Optional[str] = Query(None, description="Filter by warehouse"),
    category: Optional[str] = Query(None, description="Filter by category"),
    partner: Optional[str] = Query(None, description="Filter by partner"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)")
):
    """
    Get outflows data for specified period
    
    Args:
        period: train, tune, or test
        limit: Maximum number of records to return
        offset: Number of records to skip
        warehouse: Filter by warehouse name
        category: Filter by category
        partner: Filter by partner
        date_from: Start date filter
        date_to: End date filter
    """
    df = load_dataset('outflows', period)
    
    if df is None:
        raise HTTPException(status_code=404, detail=f"Outflows data for {period} period not found")
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
        if date_from:
            df = df[df['Date'] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df['Date'] <= pd.to_datetime(date_to)]
    
    if warehouse and 'Warehouse' in df.columns:
        df = df[df['Warehouse'] == warehouse]
    
    if category and 'Category' in df.columns:
        df = df[df['Category'] == category]
    
    if partner and 'Partner' in df.columns:
        df = df[df['Partner'] == partner]
    
    if 'Date' in df.columns:
        df = df.sort_values('Date')
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    total_records = len(df)
    
    if offset:
        df = df.iloc[offset:]
    if limit:
        df = df.head(limit)
    
    return {
        "period": period,
        "type": "outflows",
        "total_records": total_records,
        "returned_records": len(df),
        "filters": {
            "warehouse": warehouse,
            "category": category,
            "partner": partner,
            "date_from": date_from,
            "date_to": date_to
        },
        "data": df.to_dict(orient='records')
    }

@app.get("/summary")
async def get_summary():
    """Get summary statistics for all datasets"""
    summary = {
        "inventory": {},
        "inflows": {},
        "outflows": {}
    }
    
    for period in ['train', 'tune', 'test']:
        inv_df = load_dataset('inventory', period)
        if inv_df is not None and 'Inventory_Level' in inv_df.columns:
            summary['inventory'][period] = {
                "record_count": len(inv_df),
                "avg_inventory": float(inv_df['Inventory_Level'].mean()),
                "min_inventory": float(inv_df['Inventory_Level'].min()),
                "max_inventory": float(inv_df['Inventory_Level'].max())
            }
        
        in_df = load_dataset('inflows', period)
        if in_df is not None:
            inflow_summary = {
                "record_count": len(in_df)
            }
            if 'Quantity' in in_df.columns:
                inflow_summary["total_quantity"] = float(in_df['Quantity'].sum())
                inflow_summary["avg_quantity"] = float(in_df['Quantity'].mean())
            if 'Total_GIK' in in_df.columns:
                inflow_summary["total_gik"] = float(in_df['Total_GIK'].sum())
            summary['inflows'][period] = inflow_summary
        
        out_df = load_dataset('outflows', period)
        if out_df is not None:
            outflow_summary = {
                "record_count": len(out_df)
            }
            if 'Quantity' in out_df.columns:
                outflow_summary["total_quantity"] = float(out_df['Quantity'].sum())
                outflow_summary["avg_quantity"] = float(out_df['Quantity'].mean())
            if 'Total_GIK' in out_df.columns:
                outflow_summary["total_gik"] = float(out_df['Total_GIK'].sum())
            summary['outflows'][period] = outflow_summary
    
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
