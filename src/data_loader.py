import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import io

class DataLoader:
    """
    Handles data loading and schema validation for inventory, inflows, and outflows data.
    Supports CSV and Excel formats with automatic schema detection.
    """
    
    def __init__(self):
        self.expected_schemas = {
            'inventory': ['Date', 'Inventory_Level'],
            'inflows': ['Date', 'Vendor', 'Quantity', 'Product_Type', 'Category', 'Brand', 'Grade', 'Size', 'Warehouse', 'Vendor_Address', 'GIK_Per_Unit', 'Total_GIK'],
            'outflows': ['Date', 'Quantity', 'Product_Type', 'Category', 'Brand', 'Grade', 'Size', 'Warehouse', 'GIK_Per_Unit', 'Total_GIK', 'Partner', 'EI_Per_Unit', 'Total_EI', 'Partner_Address', 'Shipment_Type', 'Program']
        }
    
    def load_file(self, file):
        """
        Load data from uploaded file (CSV or Excel)
        """
        try:
            if file.name.endswith('.csv'):
                # Handle CSV files
                df = pd.read_csv(file, encoding='utf-8')
            elif file.name.endswith(('.xlsx', '.xls')):
                # Handle Excel files
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
            
            # Clean column names (remove BOM and extra whitespace)
            df.columns = df.columns.str.strip().str.replace('﻿', '')
            
            return df
            
        except Exception as e:
            st.error(f"Error loading file {file.name}: {str(e)}")
            return None
    
    def validate_schemas(self, train_data, tune_data):
        """
        Validate data schemas against expected formats
        """
        validation_results = {
            'valid': True,
            'errors': []
        }
        
        # Validate training data
        for data_type, df in train_data.items():
            if df is not None:
                errors = self._validate_single_schema(df, data_type, 'train')
                validation_results['errors'].extend(errors)
        
        # Validate tuning data
        for data_type, df in tune_data.items():
            if df is not None:
                errors = self._validate_single_schema(df, data_type, 'tune')
                validation_results['errors'].extend(errors)
        
        # Check if we have minimum required data
        if 'inventory' not in train_data or train_data['inventory'] is None:
            validation_results['errors'].append("Training inventory data is required")
        
        if 'inventory' not in tune_data or tune_data['inventory'] is None:
            validation_results['errors'].append("Tuning inventory data is required")
        
        validation_results['valid'] = len(validation_results['errors']) == 0
        
        return validation_results
    
    def _validate_single_schema(self, df, data_type, period):
        """
        Validate a single dataset's schema
        """
        errors = []
        
        if data_type not in self.expected_schemas:
            errors.append(f"Unknown data type: {data_type}")
            return errors
        
        expected_columns = self.expected_schemas[data_type]
        actual_columns = list(df.columns)
        
        # Check for missing required columns
        missing_columns = set(expected_columns) - set(actual_columns)
        if missing_columns:
            errors.append(f"{period} {data_type} missing columns: {', '.join(missing_columns)}")
        
        # Validate data types and ranges
        try:
            # Date validation
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                null_dates = df['Date'].isnull().sum()
                if null_dates > 0:
                    errors.append(f"{period} {data_type}: {null_dates} invalid dates found")
            
            # Numeric validations
            if 'Inventory_Level' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['Inventory_Level']):
                    errors.append(f"{period} {data_type}: Inventory_Level must be numeric")
                elif (df['Inventory_Level'] < 0).any():
                    errors.append(f"{period} {data_type}: Negative inventory levels found")
            
            if 'Quantity' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['Quantity']):
                    errors.append(f"{period} {data_type}: Quantity must be numeric")
                elif (df['Quantity'] <= 0).any():
                    errors.append(f"{period} {data_type}: Non-positive quantities found")
            
            # GIK validation (check for extreme values)
            if 'GIK_Per_Unit' in df.columns:
                gik_values = pd.to_numeric(df['GIK_Per_Unit'], errors='coerce')
                extreme_gik = (gik_values > 10000).sum()  # Flag values > $10,000 per unit
                if extreme_gik > 0:
                    errors.append(f"{period} {data_type}: {extreme_gik} extreme GIK values (>$10,000) detected")
        
        except Exception as e:
            errors.append(f"{period} {data_type}: Data validation error - {str(e)}")
        
        return errors
    
    def detect_data_quality_issues(self, df, data_type):
        """
        Detect data quality issues beyond schema validation
        """
        issues = []
        
        # Check for duplicates
        if 'Date' in df.columns:
            duplicates = df.duplicated(subset=['Date']).sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate date entries")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        critical_missing = missing_counts[missing_counts > 0]
        if len(critical_missing) > 0:
            issues.append(f"Missing values found: {critical_missing.to_dict()}")
        
        # Check for outliers in key metrics
        if 'Inventory_Level' in df.columns:
            q99 = df['Inventory_Level'].quantile(0.99)
            q1 = df['Inventory_Level'].quantile(0.01)
            outliers = ((df['Inventory_Level'] > q99) | (df['Inventory_Level'] < q1)).sum()
            if outliers > len(df) * 0.05:  # More than 5% outliers
                issues.append(f"High number of inventory outliers detected: {outliers}")
        
        return issues
    
    def preview_data(self, df, data_type):
        """
        Generate data preview with key statistics
        """
        preview = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head().to_dict('records')
        }
        
        # Add specific statistics based on data type
        if data_type == 'inventory' and 'Inventory_Level' in df.columns:
            preview['inventory_stats'] = {
                'mean': df['Inventory_Level'].mean(),
                'median': df['Inventory_Level'].median(),
                'min': df['Inventory_Level'].min(),
                'max': df['Inventory_Level'].max(),
                'std': df['Inventory_Level'].std()
            }
        
        elif data_type in ['inflows', 'outflows'] and 'Quantity' in df.columns:
            preview['quantity_stats'] = {
                'total': df['Quantity'].sum(),
                'mean': df['Quantity'].mean(),
                'median': df['Quantity'].median(),
                'min': df['Quantity'].min(),
                'max': df['Quantity'].max()
            }
            
            if 'Product_Type' in df.columns:
                preview['product_distribution'] = df['Product_Type'].value_counts().to_dict()
            
            if 'Warehouse' in df.columns:
                preview['warehouse_distribution'] = df['Warehouse'].value_counts().to_dict()
        
        return preview
