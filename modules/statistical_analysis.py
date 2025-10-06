import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    def __init__(self):
        pass
    
    def analyze_data(self, data_dict):
        """Comprehensive statistical analysis of inventory data"""
        try:
            inventory_data = data_dict['inventory']
            inflows_data = data_dict['inflows']
            outflows_data = data_dict['outflows']
            
            analysis_results = {
                'inventory_trends': self._analyze_inventory_trends(inventory_data),
                'turnover_analysis': self._calculate_turnover_metrics(inventory_data, inflows_data, outflows_data),
                'distribution_analysis': self._analyze_distributions(inflows_data, outflows_data),
                'temporal_patterns': self._analyze_temporal_patterns(inventory_data),
                'category_performance': self._analyze_category_performance(inflows_data, outflows_data),
                'warehouse_analysis': self._analyze_warehouse_performance(inflows_data, outflows_data),
                'yoy_comparison': self._analyze_year_over_year(inventory_data)
            }
            
            # Add key metrics
            analysis_results.update(self._calculate_key_metrics(analysis_results))
            
            return analysis_results
            
        except Exception as e:
            raise Exception(f"Statistical analysis failed: {str(e)}")
    
    def _analyze_inventory_trends(self, inventory_data):
        """Analyze inventory level trends and patterns"""
        trends = {}
        
        # Basic trend statistics
        trends['mean_level'] = inventory_data['Inventory_Level'].mean()
        trends['std_level'] = inventory_data['Inventory_Level'].std()
        trends['min_level'] = inventory_data['Inventory_Level'].min()
        trends['max_level'] = inventory_data['Inventory_Level'].max()
        trends['median_level'] = inventory_data['Inventory_Level'].median()
        
        # Trend analysis using linear regression
        inventory_data_sorted = inventory_data.sort_values('Date')
        x = np.arange(len(inventory_data_sorted))
        y = inventory_data_sorted['Inventory_Level'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        trends['linear_trend'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing'
        }
        
        # Volatility analysis
        inventory_data_sorted['Daily_Change'] = inventory_data_sorted['Inventory_Level'].pct_change()
        trends['volatility'] = {
            'daily_volatility': inventory_data_sorted['Daily_Change'].std(),
            'max_daily_change': inventory_data_sorted['Daily_Change'].abs().max(),
            'avg_daily_change': inventory_data_sorted['Daily_Change'].mean()
        }
        
        # Seasonal patterns
        inventory_data_sorted['Month'] = inventory_data_sorted['Date'].dt.month
        monthly_avg = inventory_data_sorted.groupby('Month')['Inventory_Level'].mean()
        trends['seasonal_patterns'] = {
            'monthly_averages': monthly_avg.to_dict(),
            'seasonal_coefficient_variation': monthly_avg.std() / monthly_avg.mean()
        }
        
        return trends
    
    def _calculate_turnover_metrics(self, inventory_data, inflows_data, outflows_data):
        """Calculate inventory turnover rates and related metrics"""
        turnover_metrics = {}
        
        # Calculate average inventory
        avg_inventory = inventory_data['Inventory_Level'].mean()
        
        # Calculate total inflows and outflows
        total_inflows = inflows_data['Quantity'].sum() if len(inflows_data) > 0 else 0
        total_outflows = outflows_data['Quantity'].sum() if len(outflows_data) > 0 else 0
        
        # Calculate turnover rate (based on outflows)
        if avg_inventory > 0:
            # Annualize based on data period
            data_days = (inventory_data['Date'].max() - inventory_data['Date'].min()).days
            annual_multiplier = 365 / data_days if data_days > 0 else 1
            
            turnover_rate = (total_outflows * annual_multiplier) / avg_inventory
            turnover_metrics['turnover_rate'] = turnover_rate
            turnover_metrics['days_sales_outstanding'] = 365 / turnover_rate if turnover_rate > 0 else np.inf
        else:
            turnover_metrics['turnover_rate'] = 0
            turnover_metrics['days_sales_outstanding'] = np.inf
        
        # Flow analysis
        turnover_metrics['total_inflows'] = total_inflows
        turnover_metrics['total_outflows'] = total_outflows
        turnover_metrics['net_change'] = total_inflows - total_outflows
        turnover_metrics['flow_ratio'] = total_inflows / total_outflows if total_outflows > 0 else np.inf
        
        # Monthly flow analysis
        if len(inflows_data) > 0 and len(outflows_data) > 0:
            inflows_monthly = inflows_data.groupby(inflows_data['Date'].dt.to_period('M'))['Quantity'].sum()
            outflows_monthly = outflows_data.groupby(outflows_data['Date'].dt.to_period('M'))['Quantity'].sum()
            
            turnover_metrics['monthly_inflows'] = inflows_monthly.to_dict()
            turnover_metrics['monthly_outflows'] = outflows_monthly.to_dict()
        
        return turnover_metrics
    
    def _analyze_distributions(self, inflows_data, outflows_data):
        """Analyze product and categorical distributions"""
        distributions = {}
        
        # Inflows distributions
        if len(inflows_data) > 0:
            distributions['inflows'] = {
                'category_distribution': inflows_data['Category'].value_counts().to_dict(),
                'brand_distribution': inflows_data['Brand'].value_counts().to_dict(),
                'warehouse_distribution': inflows_data['Warehouse'].value_counts().to_dict(),
                'vendor_distribution': inflows_data['Vendor'].value_counts().to_dict(),
                'grade_distribution': inflows_data['Grade'].value_counts().to_dict()
            }
            
            # GIK value analysis
            if 'Total_GIK' in inflows_data.columns:
                distributions['inflows']['gik_statistics'] = {
                    'mean_gik': inflows_data['Total_GIK'].mean(),
                    'median_gik': inflows_data['Total_GIK'].median(),
                    'total_gik': inflows_data['Total_GIK'].sum(),
                    'gik_by_category': inflows_data.groupby('Category')['Total_GIK'].sum().to_dict()
                }
        
        # Outflows distributions
        if len(outflows_data) > 0:
            distributions['outflows'] = {
                'category_distribution': outflows_data['Category'].value_counts().to_dict(),
                'brand_distribution': outflows_data['Brand'].value_counts().to_dict(),
                'warehouse_distribution': outflows_data['Warehouse'].value_counts().to_dict(),
                'partner_distribution': outflows_data['Partner'].value_counts().to_dict(),
                'grade_distribution': outflows_data['Grade'].value_counts().to_dict()
            }
            
            # Program analysis
            if 'Program' in outflows_data.columns:
                distributions['outflows']['program_distribution'] = outflows_data['Program'].value_counts().to_dict()
            
            # GIK value analysis
            if 'Total_GIK' in outflows_data.columns:
                distributions['outflows']['gik_statistics'] = {
                    'mean_gik': outflows_data['Total_GIK'].mean(),
                    'median_gik': outflows_data['Total_GIK'].median(),
                    'total_gik': outflows_data['Total_GIK'].sum(),
                    'gik_by_category': outflows_data.groupby('Category')['Total_GIK'].sum().to_dict()
                }
        
        return distributions
    
    def _analyze_temporal_patterns(self, inventory_data):
        """Analyze temporal patterns in inventory data"""
        temporal_patterns = {}
        
        inventory_data_sorted = inventory_data.sort_values('Date')
        
        # Daily patterns
        inventory_data_sorted['DayOfWeek'] = inventory_data_sorted['Date'].dt.day_name()
        inventory_data_sorted['Month'] = inventory_data_sorted['Date'].dt.month
        inventory_data_sorted['Quarter'] = inventory_data_sorted['Date'].dt.quarter
        inventory_data_sorted['Year'] = inventory_data_sorted['Date'].dt.year
        
        temporal_patterns['daily_patterns'] = inventory_data_sorted.groupby('DayOfWeek')['Inventory_Level'].mean().to_dict()
        temporal_patterns['monthly_patterns'] = inventory_data_sorted.groupby('Month')['Inventory_Level'].mean().to_dict()
        temporal_patterns['quarterly_patterns'] = inventory_data_sorted.groupby('Quarter')['Inventory_Level'].mean().to_dict()
        temporal_patterns['yearly_patterns'] = inventory_data_sorted.groupby('Year')['Inventory_Level'].mean().to_dict()
        
        # Identify peaks and troughs
        rolling_max = inventory_data_sorted['Inventory_Level'].rolling(30).max()
        rolling_min = inventory_data_sorted['Inventory_Level'].rolling(30).min()
        
        peaks = inventory_data_sorted[inventory_data_sorted['Inventory_Level'] == rolling_max]
        troughs = inventory_data_sorted[inventory_data_sorted['Inventory_Level'] == rolling_min]
        
        temporal_patterns['peaks'] = peaks[['Date', 'Inventory_Level']].to_dict('records')
        temporal_patterns['troughs'] = troughs[['Date', 'Inventory_Level']].to_dict('records')
        
        return temporal_patterns
    
    def _analyze_category_performance(self, inflows_data, outflows_data):
        """Analyze performance by product category"""
        category_performance = {}
        
        # Inflows by category
        if len(inflows_data) > 0:
            category_inflows = inflows_data.groupby('Category').agg({
                'Quantity': ['sum', 'mean', 'count'],
                'Total_GIK': 'sum' if 'Total_GIK' in inflows_data.columns else lambda x: 0
            }).round(2)
            
            category_performance['inflows'] = category_inflows.to_dict()
        
        # Outflows by category
        if len(outflows_data) > 0:
            category_outflows = outflows_data.groupby('Category').agg({
                'Quantity': ['sum', 'mean', 'count'],
                'Total_GIK': 'sum' if 'Total_GIK' in outflows_data.columns else lambda x: 0
            }).round(2)
            
            category_performance['outflows'] = category_outflows.to_dict()
        
        # Net flow by category
        if len(inflows_data) > 0 and len(outflows_data) > 0:
            inflow_totals = inflows_data.groupby('Category')['Quantity'].sum()
            outflow_totals = outflows_data.groupby('Category')['Quantity'].sum()
            
            all_categories = set(inflow_totals.index) | set(outflow_totals.index)
            net_flows = {}
            
            for category in all_categories:
                inflow = inflow_totals.get(category, 0)
                outflow = outflow_totals.get(category, 0)
                net_flows[category] = inflow - outflow
            
            category_performance['net_flows'] = net_flows
        
        return category_performance
    
    def _analyze_warehouse_performance(self, inflows_data, outflows_data):
        """Analyze performance by warehouse"""
        warehouse_performance = {}
        
        # Inflows by warehouse
        if len(inflows_data) > 0:
            warehouse_inflows = inflows_data.groupby('Warehouse').agg({
                'Quantity': ['sum', 'mean'],
                'Total_GIK': 'sum' if 'Total_GIK' in inflows_data.columns else lambda x: 0
            }).round(2)
            
            warehouse_performance['inflows'] = warehouse_inflows.to_dict()
        
        # Outflows by warehouse
        if len(outflows_data) > 0:
            warehouse_outflows = outflows_data.groupby('Warehouse').agg({
                'Quantity': ['sum', 'mean'],
                'Total_GIK': 'sum' if 'Total_GIK' in outflows_data.columns else lambda x: 0
            }).round(2)
            
            warehouse_performance['outflows'] = warehouse_outflows.to_dict()
        
        return warehouse_performance
    
    def _analyze_year_over_year(self, inventory_data):
        """Analyze year-over-year changes"""
        yoy_comparison = {}
        
        # Separate periods
        period1 = inventory_data[inventory_data['Date'] < '2019-01-01']  # 2017-2018
        period2 = inventory_data[inventory_data['Date'] >= '2023-01-01']  # 2023
        
        if len(period1) > 0 and len(period2) > 0:
            yoy_comparison['period1_avg'] = period1['Inventory_Level'].mean()
            yoy_comparison['period2_avg'] = period2['Inventory_Level'].mean()
            yoy_comparison['period1_std'] = period1['Inventory_Level'].std()
            yoy_comparison['period2_std'] = period2['Inventory_Level'].std()
            
            # Calculate percentage change
            change_pct = ((yoy_comparison['period2_avg'] - yoy_comparison['period1_avg']) / 
                         yoy_comparison['period1_avg']) * 100
            yoy_comparison['percentage_change'] = change_pct
            
            # Statistical significance test
            t_stat, p_value = stats.ttest_ind(period1['Inventory_Level'], period2['Inventory_Level'])
            yoy_comparison['statistical_significance'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return yoy_comparison
    
    def _calculate_key_metrics(self, analysis_results):
        """Calculate key summary metrics"""
        key_metrics = {}
        
        # Extract key metrics from analysis results
        if 'inventory_trends' in analysis_results:
            trends = analysis_results['inventory_trends']
            key_metrics['avg_inventory'] = trends['mean_level']
            key_metrics['peak_inventory'] = trends['max_level']
            key_metrics['min_inventory'] = trends['min_level']
            key_metrics['inventory_volatility'] = trends['volatility']['daily_volatility']
        
        if 'turnover_analysis' in analysis_results:
            turnover = analysis_results['turnover_analysis']
            key_metrics['turnover_rate'] = turnover['turnover_rate']
            key_metrics['days_sales_outstanding'] = turnover['days_sales_outstanding']
        
        if 'distribution_analysis' in analysis_results:
            distributions = analysis_results['distribution_analysis']
            if 'inflows' in distributions:
                key_metrics['category_distribution'] = distributions['inflows']['category_distribution']
                key_metrics['warehouse_distribution'] = distributions['inflows']['warehouse_distribution']
        
        if 'temporal_patterns' in analysis_results:
            key_metrics['temporal_patterns'] = analysis_results['temporal_patterns']
        
        return key_metrics
