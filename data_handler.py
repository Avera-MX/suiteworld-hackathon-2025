import pandas as pd
import numpy as np
from datetime import datetime
import openpyxl
from typing import Dict, Any, Optional, Tuple
import logging

class DataHandler:
    def __init__(self):
        self.schema_validation_rules = {
            'inventory': {
                'required_columns': ['Date', 'Inventory_Level'],
                'date_columns': ['Date'],
                'numeric_columns': ['Inventory_Level']
            },
            'inflows': {
                'required_columns': ['Date', 'Quantity'],
                'date_columns': ['Date'],
                'numeric_columns': ['Quantity', 'Total_GIK', 'GIK_Per_Unit']
            },
            'outflows': {
                'required_columns': ['Date', 'Quantity'],
                'date_columns': ['Date'],
                'numeric_columns': ['Quantity', 'Total_GIK', 'GIK_Per_Unit', 'Total_EI', 'EI_Per_Unit']
            }
        }
    
    def process_datasets(self, train_inventory: pd.DataFrame, train_inflows: pd.DataFrame, 
                        train_outflows: Optional[pd.DataFrame], tune_inventory: pd.DataFrame,
                        tune_inflows: pd.DataFrame, tune_outflows: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Process and validate all datasets
        """
        try:
            datasets = {}
            
            # Process training data
            if train_inventory is not None:
                result = self._validate_and_clean_dataset(train_inventory, 'inventory', 'train_inventory')
                if not result['success']:
                    return {'success': False, 'error': f"Training inventory validation failed: {result['error']}"}
                datasets['train_inventory'] = result['data']
            
            if train_inflows is not None:
                result = self._validate_and_clean_dataset(train_inflows, 'inflows', 'train_inflows')
                if not result['success']:
                    return {'success': False, 'error': f"Training inflows validation failed: {result['error']}"}
                datasets['train_inflows'] = result['data']
            
            if train_outflows is not None:
                result = self._validate_and_clean_dataset(train_outflows, 'outflows', 'train_outflows')
                if not result['success']:
                    return {'success': False, 'error': f"Training outflows validation failed: {result['error']}"}
                datasets['train_outflows'] = result['data']
            
            # Process tuning data
            if tune_inventory is not None:
                result = self._validate_and_clean_dataset(tune_inventory, 'inventory', 'tune_inventory')
                if not result['success']:
                    return {'success': False, 'error': f"Tuning inventory validation failed: {result['error']}"}
                datasets['tune_inventory'] = result['data']
            
            if tune_inflows is not None:
                result = self._validate_and_clean_dataset(tune_inflows, 'inflows', 'tune_inflows')
                if not result['success']:
                    return {'success': False, 'error': f"Tuning inflows validation failed: {result['error']}"}
                datasets['tune_inflows'] = result['data']
            
            if tune_outflows is not None:
                result = self._validate_and_clean_dataset(tune_outflows, 'outflows', 'tune_outflows')
                if not result['success']:
                    return {'success': False, 'error': f"Tuning outflows validation failed: {result['error']}"}
                datasets['tune_outflows'] = result['data']
            
            # Extract warehouse features from inflows/outflows
            if train_inflows is not None and 'train_inflows' in datasets:
                datasets['train_warehouse_features'] = self._extract_warehouse_features(
                    datasets['train_inflows'], 
                    datasets.get('train_outflows')
                )
            
            if tune_inflows is not None and 'tune_inflows' in datasets:
                datasets['tune_warehouse_features'] = self._extract_warehouse_features(
                    datasets['tune_inflows'], 
                    datasets.get('tune_outflows')
                )
            
            # Extract category features from inflows/outflows
            if train_inflows is not None and 'train_inflows' in datasets:
                datasets['train_category_features'] = self._extract_category_features(
                    datasets['train_inflows'], 
                    datasets.get('train_outflows')
                )
            
            if tune_inflows is not None and 'tune_inflows' in datasets:
                datasets['tune_category_features'] = self._extract_category_features(
                    datasets['tune_inflows'], 
                    datasets.get('tune_outflows')
                )
            
            return {'success': True, 'datasets': datasets}
            
        except Exception as e:
            return {'success': False, 'error': f"Dataset processing error: {str(e)}"}
    
    def _validate_and_clean_dataset(self, df: pd.DataFrame, dataset_type: str, dataset_name: str) -> Dict[str, Any]:
        """
        Validate and clean individual dataset
        """
        try:
            # Make a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Get validation rules
            rules = self.schema_validation_rules.get(dataset_type, {})
            
            # Check required columns
            required_cols = rules.get('required_columns', [])
            missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
            if missing_cols:
                return {'success': False, 'error': f"Missing required columns: {missing_cols}"}
            
            # Clean and validate date columns
            date_cols = rules.get('date_columns', [])
            for col in date_cols:
                if col in cleaned_df.columns:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                    # Remove rows with invalid dates
                    invalid_dates = cleaned_df[col].isna()
                    if invalid_dates.sum() > 0:
                        print(f"Warning: Removed {invalid_dates.sum()} rows with invalid dates from {dataset_name}")
                        cleaned_df = cleaned_df[~invalid_dates]
            
            # Clean and validate numeric columns
            numeric_cols = rules.get('numeric_columns', [])
            for col in numeric_cols:
                if col in cleaned_df.columns:
                    # Convert to numeric, handling errors
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    
                    # Handle extreme outliers using winsorization
                    if col in ['Total_GIK', 'GIK_Per_Unit', 'Total_EI', 'EI_Per_Unit']:
                        cleaned_df = self._handle_extreme_values(cleaned_df, col)
            
            # Remove rows with all NaN values in critical columns
            critical_cols = required_cols
            cleaned_df = cleaned_df.dropna(subset=critical_cols, how='all')
            
            # Sort by date if date column exists
            if 'Date' in cleaned_df.columns:
                cleaned_df = cleaned_df.sort_values('Date').reset_index(drop=True)
            
            return {'success': True, 'data': cleaned_df}
            
        except Exception as e:
            return {'success': False, 'error': f"Dataset validation error: {str(e)}"}
    
    def _handle_extreme_values(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Handle extreme values using winsorization
        """
        if column not in df.columns:
            return df
        
        # Calculate percentiles for winsorization
        values = df[column].dropna()
        if len(values) == 0:
            return df
        
        # Use 1st and 99th percentiles for winsorization
        lower_bound = values.quantile(0.01)
        upper_bound = values.quantile(0.99)
        
        # Apply winsorization
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def get_data_quality_report(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate data quality report
        """
        report = {}
        
        for dataset_name, df in datasets.items():
            if df is not None:
                quality_metrics = {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'missing_values': df.isnull().sum().to_dict(),
                    'data_types': df.dtypes.to_dict(),
                    'date_range': None
                }
                
                # Add date range if Date column exists
                if 'Date' in df.columns:
                    quality_metrics['date_range'] = {
                        'start': df['Date'].min(),
                        'end': df['Date'].max(),
                        'total_days': (df['Date'].max() - df['Date'].min()).days
                    }
                
                # Add numeric column statistics
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    quality_metrics['numeric_stats'] = df[numeric_cols].describe().to_dict()
                
                report[dataset_name] = quality_metrics
        
        return report
    
    def detect_scale_changes(self, train_df: pd.DataFrame, tune_df: pd.DataFrame, 
                           column: str = 'Inventory_Level') -> Dict[str, float]:
        """
        Detect scale changes between training and tuning periods
        """
        if column not in train_df.columns or column not in tune_df.columns:
            return {}
        
        train_mean = train_df[column].mean()
        tune_mean = tune_df[column].mean()
        
        train_std = train_df[column].std()
        tune_std = tune_df[column].std()
        
        train_median = train_df[column].median()
        tune_median = tune_df[column].median()
        
        return {
            'mean_ratio': tune_mean / train_mean if train_mean != 0 else 0,
            'std_ratio': tune_std / train_std if train_std != 0 else 0,
            'median_ratio': tune_median / train_median if train_median != 0 else 0,
            'range_ratio': (tune_df[column].max() - tune_df[column].min()) / 
                          (train_df[column].max() - train_df[column].min()) if (train_df[column].max() - train_df[column].min()) != 0 else 0
        }
    
    def prepare_time_series_data(self, df: pd.DataFrame, date_col: str = 'Date', 
                               value_col: str = 'Inventory_Level') -> pd.DataFrame:
        """
        Prepare data for time series forecasting
        """
        if date_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"Required columns {date_col} or {value_col} not found")
        
        # Create time series dataframe
        ts_df = df[[date_col, value_col]].copy()
        ts_df = ts_df.rename(columns={date_col: 'ds', value_col: 'y'})
        
        # Ensure proper sorting
        ts_df = ts_df.sort_values('ds').reset_index(drop=True)
        
        # Remove duplicates, keeping last value
        ts_df = ts_df.drop_duplicates(subset=['ds'], keep='last')
        
        # Fill missing dates if needed
        ts_df = self._fill_missing_dates(ts_df)
        
        return ts_df
    
    def _fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing dates in time series
        """
        # Create date range
        date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
        
        # Create full dataframe
        full_df = pd.DataFrame({'ds': date_range})
        
        # Merge with existing data
        result_df = full_df.merge(df, on='ds', how='left')
        
        # Forward fill missing values
        result_df['y'] = result_df['y'].fillna(method='ffill')
        
        # Backward fill any remaining missing values
        result_df['y'] = result_df['y'].fillna(method='bfill')
        
        return result_df
    
    def _extract_warehouse_features(self, inflows_df: pd.DataFrame, 
                                   outflows_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract warehouse-based features from inflows and outflows data
        Aggregates warehouse activity by date to create features for forecasting
        """
        try:
            warehouse_features = pd.DataFrame()
            
            # Check if Warehouse column exists in inflows
            if 'Warehouse' not in inflows_df.columns:
                return warehouse_features
            
            # Aggregate inflows by date and warehouse
            inflows_agg = inflows_df.groupby(['Date', 'Warehouse']).agg({
                'Quantity': 'sum'
            }).reset_index()
            inflows_agg.rename(columns={'Quantity': 'Inflow_Quantity'}, inplace=True)
            
            # Aggregate outflows if available
            if outflows_df is not None and 'Warehouse' in outflows_df.columns:
                outflows_agg = outflows_df.groupby(['Date', 'Warehouse']).agg({
                    'Quantity': 'sum'
                }).reset_index()
                outflows_agg.rename(columns={'Quantity': 'Outflow_Quantity'}, inplace=True)
                
                # Merge inflows and outflows
                warehouse_features = pd.merge(
                    inflows_agg, outflows_agg, 
                    on=['Date', 'Warehouse'], 
                    how='outer'
                ).fillna(0)
            else:
                warehouse_features = inflows_agg.copy()
                warehouse_features['Outflow_Quantity'] = 0
            
            # Create aggregated features by date (across all warehouses)
            daily_features = warehouse_features.groupby('Date').agg({
                'Inflow_Quantity': 'sum',
                'Outflow_Quantity': 'sum',
                'Warehouse': 'nunique'
            }).reset_index()
            
            daily_features.rename(columns={
                'Inflow_Quantity': 'Total_Daily_Inflows',
                'Outflow_Quantity': 'Total_Daily_Outflows',
                'Warehouse': 'Active_Warehouses'
            }, inplace=True)
            
            # Add warehouse activity concentration (measure of warehouse diversity)
            warehouse_counts = warehouse_features.groupby('Date')['Warehouse'].value_counts()
            warehouse_diversity = warehouse_counts.groupby(level=0).apply(
                lambda x: 1 - (x ** 2).sum() / (x.sum() ** 2) if x.sum() > 0 else 0
            ).reset_index(name='Warehouse_Diversity')
            
            daily_features = pd.merge(daily_features, warehouse_diversity, on='Date', how='left')
            daily_features['Warehouse_Diversity'] = daily_features['Warehouse_Diversity'].fillna(0)
            
            return daily_features
            
        except Exception as e:
            print(f"Error extracting warehouse features: {str(e)}")
            return pd.DataFrame()
    
    def _extract_category_features(self, inflows_df: pd.DataFrame, 
                                   outflows_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract category-based features from inflows and outflows data
        Aggregates category activity by date to create features for forecasting
        """
        try:
            category_features = pd.DataFrame()
            
            # Check if Category column exists in inflows
            if 'Category' not in inflows_df.columns:
                return category_features
            
            # Aggregate inflows by date and category
            inflows_agg = inflows_df.groupby(['Date', 'Category']).agg({
                'Quantity': 'sum'
            }).reset_index()
            inflows_agg.rename(columns={'Quantity': 'Inflow_Quantity'}, inplace=True)
            
            # Aggregate outflows if available
            if outflows_df is not None and 'Category' in outflows_df.columns:
                outflows_agg = outflows_df.groupby(['Date', 'Category']).agg({
                    'Quantity': 'sum'
                }).reset_index()
                outflows_agg.rename(columns={'Quantity': 'Outflow_Quantity'}, inplace=True)
                
                # Merge inflows and outflows
                category_features = pd.merge(
                    inflows_agg, outflows_agg, 
                    on=['Date', 'Category'], 
                    how='outer'
                ).fillna(0)
            else:
                category_features = inflows_agg.copy()
                category_features['Outflow_Quantity'] = 0
            
            # Create aggregated features by date (across all categories)
            daily_features = category_features.groupby('Date').agg({
                'Inflow_Quantity': 'sum',
                'Outflow_Quantity': 'sum',
                'Category': 'nunique'
            }).reset_index()
            
            daily_features.rename(columns={
                'Inflow_Quantity': 'Total_Category_Inflows',
                'Outflow_Quantity': 'Total_Category_Outflows',
                'Category': 'Active_Categories'
            }, inplace=True)
            
            # Add category activity concentration (measure of category diversity)
            category_counts = category_features.groupby('Date')['Category'].value_counts()
            category_diversity = category_counts.groupby(level=0).apply(
                lambda x: 1 - (x ** 2).sum() / (x.sum() ** 2) if x.sum() > 0 else 0
            ).reset_index(name='Category_Diversity')
            
            daily_features = pd.merge(daily_features, category_diversity, on='Date', how='left')
            daily_features['Category_Diversity'] = daily_features['Category_Diversity'].fillna(0)
            
            return daily_features
            
        except Exception as e:
            print(f"Error extracting category features: {str(e)}")
            return pd.DataFrame()
