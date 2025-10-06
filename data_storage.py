import pandas as pd
import json
import os
from typing import Dict, Optional

DATA_DIR = "api_data"

def ensure_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def save_dataset(df: pd.DataFrame, dataset_type: str, period: str) -> bool:
    """
    Save dataset to JSON file
    
    Args:
        df: DataFrame to save
        dataset_type: Type of dataset (inventory, inflows, outflows)
        period: Period of dataset (train, tune, test)
    
    Returns:
        True if successful, False otherwise
    """
    ensure_data_directory()
    
    filename = f"{period}_{dataset_type}.json"
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        df_copy = df.copy()
        
        if 'Date' in df_copy.columns:
            df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.strftime('%Y-%m-%d')
        
        data = df_copy.to_dict(orient='records')
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Saved {len(df)} records to {filepath}")
        return True
        
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False

def save_all_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
    """
    Save all datasets from dictionary
    
    Args:
        datasets: Dictionary of datasets with keys like 'train_inventory', 'tune_inflows', etc.
    
    Returns:
        Dictionary with save status for each dataset
    """
    results = {}
    
    dataset_mapping = {
        'train_inventory': ('inventory', 'train'),
        'tune_inventory': ('inventory', 'tune'),
        'test_inventory': ('inventory', 'test'),
        'train_inflows': ('inflows', 'train'),
        'tune_inflows': ('inflows', 'tune'),
        'test_inflows': ('inflows', 'test'),
        'train_outflows': ('outflows', 'train'),
        'tune_outflows': ('outflows', 'tune'),
        'test_outflows': ('outflows', 'test'),
    }
    
    for key, (dataset_type, period) in dataset_mapping.items():
        if key in datasets and datasets[key] is not None:
            results[key] = save_dataset(datasets[key], dataset_type, period)
        else:
            results[key] = False
    
    return results

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

def get_dataset_info() -> Dict:
    """
    Get information about all saved datasets
    
    Returns:
        Dictionary with dataset information
    """
    if not os.path.exists(DATA_DIR):
        return {"datasets": [], "count": 0}
    
    datasets = []
    files = os.listdir(DATA_DIR)
    
    for file in files:
        if file.endswith('.json'):
            parts = file.replace('.json', '').split('_', 1)
            if len(parts) == 2:
                period, dataset_type = parts
                filepath = os.path.join(DATA_DIR, file)
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    datasets.append({
                        "period": period,
                        "type": dataset_type,
                        "filename": file,
                        "size_bytes": os.path.getsize(filepath),
                        "row_count": len(data)
                    })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return {
        "datasets": datasets,
        "count": len(datasets)
    }
