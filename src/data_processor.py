import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Handles data preprocessing, cleaning, and normalization for inventory forecasting.
    Includes outlier detection, missing value imputation, and scale normalization.
    """
    
    def __init__(self):
        self.scalers = {}
        self.outlier_bounds = {}
        self.imputation_values = {}
    
    def process_data(self, data_dict, period_name):
        """
        Main data processing pipeline
        """
        processed_data = {}
        
        for data_type, df in data_dict.items():
            if df is not None:
                processed_df = self._process_single_dataset(df.copy(), data_type, period_name)
                processed_data[data_type] = processed_df
        
        return processed_data
    
    def _process_single_dataset(self, df, data_type, period_name):
        """
        Process a single dataset with appropriate cleaning and preprocessing
        """
        # Clean column names
        df.columns = df.columns.str.strip().str.replace('﻿', '')
        
        # Convert date columns
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date')
        
        # Handle missing values
        df = self._handle_missing_values(df, data_type)
        
        # Remove duplicates
        if 'Date' in df.columns:
            df = df.drop_duplicates(subset=['Date'], keep='last')
        
        # Handle outliers
        df = self._handle_outliers(df, data_type, period_name)
        
        # Data type conversions
        df = self._convert_data_types(df, data_type)
        
        # Feature engineering
        df = self._engineer_features(df, data_type)
        
        # Normalize scales if needed
        df = self._normalize_scales(df, data_type, period_name)
        
        return df
    
    def _handle_missing_values(self, df, data_type):
        """
        Handle missing values with appropriate strategies
        """
        # For inventory data, interpolate missing levels
        if data_type == 'inventory' and 'Inventory_Level' in df.columns:
            df['Inventory_Level'] = df['Inventory_Level'].interpolate(method='time')
            df['Inventory_Level'] = df['Inventory_Level'].fillna(df['Inventory_Level'].median())
        
        # For inflows/outflows, handle numeric and categorical separately
        elif data_type in ['inflows', 'outflows']:
            # Numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in ['Quantity', 'GIK_Per_Unit', 'Total_GIK', 'EI_Per_Unit', 'Total_EI']:
                    # Use median for quantity and monetary values
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)
            
            # Categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != 'Date':
                    df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _handle_outliers(self, df, data_type, period_name):
        """
        Handle outliers using winsorization and statistical methods
        """
        if data_type == 'inventory' and 'Inventory_Level' in df.columns:
            # Use IQR method for inventory levels
            Q1 = df['Inventory_Level'].quantile(0.25)
            Q3 = df['Inventory_Level'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Store bounds for later reference
            self.outlier_bounds[f'{period_name}_inventory'] = {
                'lower': lower_bound, 'upper': upper_bound
            }
            
            # Winsorize extreme values
            df['Inventory_Level'] = np.where(
                df['Inventory_Level'] > upper_bound,
                upper_bound,
                df['Inventory_Level']
            )
            df['Inventory_Level'] = np.where(
                df['Inventory_Level'] < lower_bound,
                lower_bound,
                df['Inventory_Level']
            )
        
        elif data_type in ['inflows', 'outflows']:
            # Handle extreme GIK values
            if 'GIK_Per_Unit' in df.columns:
                # Cap extremely high GIK values (likely data entry errors)
                gik_99 = df['GIK_Per_Unit'].quantile(0.99)
                df['GIK_Per_Unit'] = np.where(
                    df['GIK_Per_Unit'] > gik_99 * 10,  # 10x the 99th percentile
                    gik_99,
                    df['GIK_Per_Unit']
                )
            
            # Handle extreme quantities
            if 'Quantity' in df.columns:
                qty_99 = df['Quantity'].quantile(0.99)
                df['Quantity'] = np.where(
                    df['Quantity'] > qty_99 * 5,  # 5x the 99th percentile
                    qty_99,
                    df['Quantity']
                )
        
        return df
    
    def _convert_data_types(self, df, data_type):
        """
        Ensure proper data types
        """
        # Ensure numeric columns are properly typed
        numeric_columns = ['Inventory_Level', 'Quantity', 'GIK_Per_Unit', 'Total_GIK', 'EI_Per_Unit', 'Total_EI']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        # Ensure categorical columns are strings
        categorical_columns = ['Vendor', 'Product_Type', 'Category', 'Brand', 'Grade', 'Size', 'Warehouse', 'Partner', 'Shipment_Type', 'Program']
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df
    
    def _engineer_features(self, df, data_type):
        """
        Create additional features for analysis
        """
        if 'Date' in df.columns:
            # Extract date components
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['DayOfYear'] = df['Date'].dt.dayofyear
            
            # Create lag features for inventory
            if data_type == 'inventory' and 'Inventory_Level' in df.columns:
                df['Inventory_Lag1'] = df['Inventory_Level'].shift(1)
                df['Inventory_Lag7'] = df['Inventory_Level'].shift(7)
                df['Inventory_MA7'] = df['Inventory_Level'].rolling(window=7).mean()
                df['Inventory_MA30'] = df['Inventory_Level'].rolling(window=30).mean()
                
                # Calculate day-over-day changes
                df['Inventory_Change'] = df['Inventory_Level'].diff()
                df['Inventory_PctChange'] = df['Inventory_Level'].pct_change()
        
        # Create value-based features for inflows/outflows
        if data_type in ['inflows', 'outflows']:
            if 'Quantity' in df.columns and 'GIK_Per_Unit' in df.columns:
                df['Total_Value'] = df['Quantity'] * df['GIK_Per_Unit']
                
            # Create category-based features
            if 'Product_Type' in df.columns:
                df['Product_Type_Encoded'] = pd.Categorical(df['Product_Type']).codes
            
            if 'Category' in df.columns:
                df['Category_Encoded'] = pd.Categorical(df['Category']).codes
        
        return df
    
    def _normalize_scales(self, df, data_type, period_name):
        """
        Normalize scales to handle different inventory levels between periods
        """
        if data_type == 'inventory' and 'Inventory_Level' in df.columns:
            scaler_key = f'{data_type}_{period_name}'
            
            if scaler_key not in self.scalers:
                # Use RobustScaler to handle outliers better
                self.scalers[scaler_key] = RobustScaler()
                
            # Don't actually scale the main data, just store the scaler
            # We'll use this for model input later if needed
            inventory_values = df[['Inventory_Level']].values
            self.scalers[scaler_key].fit(inventory_values)
            
            # Store scaling parameters for reference
            df['Inventory_Level_Scaled'] = self.scalers[scaler_key].transform(inventory_values).flatten()
        
        return df
    
    def detect_scale_differences(self, train_data, tune_data):
        """
        Detect and quantify scale differences between training and tuning periods
        """
        scale_analysis = {}
        
        if 'inventory' in train_data and 'inventory' in tune_data:
            train_inv = train_data['inventory']['Inventory_Level']
            tune_inv = tune_data['inventory']['Inventory_Level']
            
            scale_analysis['inventory'] = {
                'train_mean': train_inv.mean(),
                'train_std': train_inv.std(),
                'tune_mean': tune_inv.mean(),
                'tune_std': tune_inv.std(),
                'scale_ratio': tune_inv.mean() / train_inv.mean(),
                'variance_ratio': tune_inv.var() / train_inv.var()
            }
        
        # Analyze quantity scales for inflows/outflows
        for data_type in ['inflows', 'outflows']:
            if data_type in train_data or data_type in tune_data:
                train_qty = train_data.get(data_type, {}).get('Quantity', pd.Series())
                tune_qty = tune_data.get(data_type, {}).get('Quantity', pd.Series())
                
                if not train_qty.empty and not tune_qty.empty:
                    scale_analysis[data_type] = {
                        'train_mean': train_qty.mean(),
                        'tune_mean': tune_qty.mean(),
                        'scale_ratio': tune_qty.mean() / train_qty.mean()
                    }
        
        return scale_analysis
    
    def create_time_series(self, df, date_col='Date', value_col='Inventory_Level'):
        """
        Create a proper time series with regular intervals
        """
        if date_col not in df.columns or value_col not in df.columns:
            return df
        
        # Create date range with daily frequency
        start_date = df[date_col].min()
        end_date = df[date_col].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create complete time series
        ts_df = pd.DataFrame({date_col: date_range})
        ts_df = ts_df.merge(df[[date_col, value_col]], on=date_col, how='left')
        
        # Forward fill missing values
        ts_df[value_col] = ts_df[value_col].fillna(method='ffill')
        ts_df[value_col] = ts_df[value_col].fillna(method='bfill')
        
        return ts_df
    
    def aggregate_by_period(self, df, period='D', date_col='Date', agg_cols=None):
        """
        Aggregate data by specified time period
        """
        if agg_cols is None:
            agg_cols = ['Quantity', 'Total_GIK', 'Total_EI']
        
        # Filter for available columns
        available_agg_cols = [col for col in agg_cols if col in df.columns]
        
        if not available_agg_cols or date_col not in df.columns:
            return df
        
        # Set date as index for resampling
        df_agg = df.set_index(date_col)
        
        # Define aggregation functions
        agg_funcs = {}
        for col in available_agg_cols:
            if col in ['Quantity']:
                agg_funcs[col] = 'sum'
            elif 'GIK' in col or 'EI' in col:
                agg_funcs[col] = 'sum'
            else:
                agg_funcs[col] = 'mean'
        
        # Resample and aggregate
        df_resampled = df_agg.resample(period).agg(agg_funcs).reset_index()
        
        return df_resampled
