import pandas as pd
import numpy as np
import io
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

class DataProcessor:
    def __init__(self):
        self.inventory_data = None
        self.inflows_data = None
        self.outflows_data = None
        self.processed = False
        
    def process_uploads(self, inventory_file, inflows_file, outflows_file) -> Dict[str, Any]:
        """Process uploaded files and validate schema"""
        try:
            # Read files based on extension
            inventory_data = self._read_file(inventory_file)
            inflows_data = self._read_file(inflows_file)
            outflows_data = self._read_file(outflows_file)
            
            # Validate schemas
            validation_result = self._validate_schemas(inventory_data, inflows_data, outflows_data)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}
            
            # Clean and preprocess data
            inventory_data = self._clean_inventory_data(inventory_data)
            inflows_data = self._clean_inflows_data(inflows_data)
            outflows_data = self._clean_outflows_data(outflows_data)
            
            # Store processed data
            self.inventory_data = inventory_data
            self.inflows_data = inflows_data
            self.outflows_data = outflows_data
            self.processed = True
            
            return {
                'success': True,
                'inventory_data': inventory_data,
                'inflows_data': inflows_data,
                'outflows_data': outflows_data
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _read_file(self, file) -> pd.DataFrame:
        """Read CSV or Excel file"""
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        else:
            raise ValueError(f"Unsupported file format: {file.name}")
    
    def _validate_schemas(self, inventory_df, inflows_df, outflows_df) -> Dict[str, Any]:
        """Validate data schemas"""
        # Expected columns
        inventory_cols = ['Date', 'Inventory_Level']
        inflows_cols = ['Date', 'Vendor', 'Quantity', 'Product_Type', 'Category', 'Brand', 'Grade', 'Size', 'Warehouse', 'GIK_Per_Unit', 'Total_GIK']
        outflows_cols = ['Date', 'Quantity', 'Product_Type', 'Category', 'Brand', 'Grade', 'Size', 'Warehouse', 'GIK_Per_Unit', 'Total_GIK', 'Partner']
        
        # Check inventory schema
        missing_inventory = [col for col in inventory_cols if col not in inventory_df.columns]
        if missing_inventory:
            return {'valid': False, 'error': f"Missing inventory columns: {missing_inventory}"}
        
        # Check inflows schema
        missing_inflows = [col for col in inflows_cols if col not in inflows_df.columns]
        if missing_inflows:
            return {'valid': False, 'error': f"Missing inflows columns: {missing_inflows}"}
        
        # Check outflows schema
        missing_outflows = [col for col in outflows_cols if col not in outflows_df.columns]
        if missing_outflows:
            return {'valid': False, 'error': f"Missing outflows columns: {missing_outflows}"}
        
        return {'valid': True}
    
    def _clean_inventory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess inventory data"""
        df = df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        
        # Remove BOM and clean numeric columns
        df['Inventory_Level'] = pd.to_numeric(df['Inventory_Level'], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Date', 'Inventory_Level'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def _clean_inflows_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess inflows data"""
        df = df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        
        # Clean numeric columns
        numeric_cols = ['Quantity', 'GIK_Per_Unit', 'Total_GIK']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df = df.dropna(subset=['Date', 'Quantity'])
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Fill missing categorical values
        categorical_cols = ['Vendor', 'Product_Type', 'Category', 'Brand', 'Grade', 'Size', 'Warehouse']
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        # Apply winsorization to extreme GIK values
        df = self._winsorize_extreme_values(df, 'Total_GIK')
        
        return df
    
    def _clean_outflows_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess outflows data"""
        df = df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        
        # Clean numeric columns
        numeric_cols = ['Quantity', 'GIK_Per_Unit', 'Total_GIK']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df = df.dropna(subset=['Date', 'Quantity'])
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill missing categorical values
        categorical_cols = ['Product_Type', 'Category', 'Brand', 'Grade', 'Size', 'Warehouse', 'Partner']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Apply winsorization to extreme GIK values
        if 'Total_GIK' in df.columns:
            df = self._winsorize_extreme_values(df, 'Total_GIK')
        
        return df
    
    def _winsorize_extreme_values(self, df: pd.DataFrame, column: str, lower=0.01, upper=0.99) -> pd.DataFrame:
        """Apply winsorization to handle extreme outliers"""
        if column in df.columns:
            lower_bound = df[column].quantile(lower)
            upper_bound = df[column].quantile(upper)
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df
    
    def get_processed_data(self) -> Dict[str, pd.DataFrame]:
        """Return processed data"""
        if not self.processed:
            raise ValueError("Data not processed yet")
        
        return {
            'inventory': self.inventory_data,
            'inflows': self.inflows_data,
            'outflows': self.outflows_data
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Generate data summary statistics"""
        if not self.processed:
            return {}
        
        return {
            'inventory_records': len(self.inventory_data),
            'inflows_records': len(self.inflows_data),
            'outflows_records': len(self.outflows_data),
            'date_range': {
                'start': min(self.inventory_data['Date'].min(), 
                           self.inflows_data['Date'].min(), 
                           self.outflows_data['Date'].min()),
                'end': max(self.inventory_data['Date'].max(), 
                         self.inflows_data['Date'].max(), 
                         self.outflows_data['Date'].max())
            },
            'inventory_stats': {
                'mean': self.inventory_data['Inventory_Level'].mean(),
                'std': self.inventory_data['Inventory_Level'].std(),
                'min': self.inventory_data['Inventory_Level'].min(),
                'max': self.inventory_data['Inventory_Level'].max()
            }
        }
