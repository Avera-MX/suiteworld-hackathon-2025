import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Performs comprehensive statistical analysis of inventory data including
    trend analysis, seasonality detection, and distribution comparisons.
    """
    
    def __init__(self):
        pass
    
    def analyze_inventory_trends(self, train_inventory, tune_inventory):
        """
        Analyze inventory level trends and detect significant changes
        """
        analysis = {}
        
        # Basic statistics
        train_stats = self._calculate_basic_stats(train_inventory['Inventory_Level'])
        tune_stats = self._calculate_basic_stats(tune_inventory['Inventory_Level'])
        
        analysis['train_stats'] = train_stats
        analysis['tune_stats'] = tune_stats
        
        # Scale changes
        analysis['train_avg'] = train_stats['mean']
        analysis['tune_avg'] = tune_stats['mean']
        analysis['scale_change'] = tune_stats['mean'] / train_stats['mean']
        
        # Trend direction
        train_trend = self._calculate_trend(train_inventory)
        tune_trend = self._calculate_trend(tune_inventory)
        
        analysis['train_trend'] = train_trend
        analysis['tune_trend'] = tune_trend
        analysis['trend_direction'] = self._interpret_trend_change(train_trend, tune_trend)
        
        # Volatility analysis
        analysis['train_volatility'] = train_stats['std'] / train_stats['mean']
        analysis['tune_volatility'] = tune_stats['std'] / tune_stats['mean']
        analysis['volatility_change'] = analysis['tune_volatility'] / analysis['train_volatility']
        
        # Statistical significance tests
        analysis['distribution_change'] = self._test_distribution_change(
            train_inventory['Inventory_Level'], 
            tune_inventory['Inventory_Level']
        )
        
        # Seasonality detection
        analysis['train_seasonality'] = self._detect_seasonality(train_inventory)
        analysis['tune_seasonality'] = self._detect_seasonality(tune_inventory)
        
        return analysis
    
    def analyze_categories(self, train_data, tune_data):
        """
        Analyze product categories and their distributions
        """
        category_analysis = {}
        
        # Analyze training data categories
        train_categories = self._extract_category_data(train_data)
        tune_categories = self._extract_category_data(tune_data)
        
        # Product type analysis
        if 'Product_Type' in train_categories and 'Product_Type' in tune_categories:
            category_analysis['product_type'] = {
                'train_distribution': train_categories['Product_Type'].value_counts(normalize=True).to_dict(),
                'tune_distribution': tune_categories['Product_Type'].value_counts(normalize=True).to_dict(),
                'category_shift': self._calculate_distribution_shift(
                    train_categories['Product_Type'], tune_categories['Product_Type']
                )
            }
        
        # Category analysis
        if 'Category' in train_categories and 'Category' in tune_categories:
            category_analysis['category'] = {
                'train_distribution': train_categories['Category'].value_counts(normalize=True).to_dict(),
                'tune_distribution': tune_categories['Category'].value_counts(normalize=True).to_dict(),
                'category_shift': self._calculate_distribution_shift(
                    train_categories['Category'], tune_categories['Category']
                )
            }
        
        # Quantity analysis by category
        category_analysis['quantity_analysis'] = self._analyze_quantities_by_category(
            train_data, tune_data
        )
        
        return category_analysis
    
    def analyze_brands(self, train_data, tune_data):
        """
        Analyze brand distributions and preferences
        """
        brand_analysis = {}
        
        train_brands = self._extract_brand_data(train_data)
        tune_brands = self._extract_brand_data(tune_data)
        
        if not train_brands.empty and not tune_brands.empty:
            # Brand distribution analysis
            train_brand_dist = train_brands['Brand'].value_counts(normalize=True)
            tune_brand_dist = tune_brands['Brand'].value_counts(normalize=True)
            
            brand_analysis['train_distribution'] = train_brand_dist.to_dict()
            brand_analysis['tune_distribution'] = tune_brand_dist.to_dict()
            
            # Brand preference shifts
            brand_analysis['preference_shift'] = self._calculate_brand_preference_shift(
                train_brand_dist, tune_brand_dist
            )
            
            # Brand value analysis
            if 'Total_GIK' in train_brands.columns and 'Total_GIK' in tune_brands.columns:
                brand_analysis['value_analysis'] = self._analyze_brand_values(
                    train_brands, tune_brands
                )
        
        return brand_analysis
    
    def analyze_warehouses(self, train_data, tune_data):
        """
        Analyze warehouse utilization and distribution patterns
        """
        warehouse_analysis = {}
        
        train_warehouses = self._extract_warehouse_data(train_data)
        tune_warehouses = self._extract_warehouse_data(tune_data)
        
        if not train_warehouses.empty and not tune_warehouses.empty:
            # Warehouse utilization
            train_warehouse_util = train_warehouses.groupby('Warehouse').agg({
                'Quantity': 'sum',
                'Total_GIK': 'sum'
            }).reset_index()
            
            tune_warehouse_util = tune_warehouses.groupby('Warehouse').agg({
                'Quantity': 'sum',
                'Total_GIK': 'sum'
            }).reset_index()
            
            warehouse_analysis['train_utilization'] = train_warehouse_util.to_dict('records')
            warehouse_analysis['tune_utilization'] = tune_warehouse_util.to_dict('records')
            
            # Warehouse efficiency analysis
            warehouse_analysis['efficiency_analysis'] = self._analyze_warehouse_efficiency(
                train_warehouse_util, tune_warehouse_util
            )
        
        return warehouse_analysis
    
    def analyze_seasonality(self, train_data, tune_data):
        """
        Analyze seasonal patterns in inventory and flows
        """
        seasonality_analysis = {}
        
        # Inventory seasonality
        if 'inventory' in train_data and 'inventory' in tune_data:
            train_inv = train_data['inventory'].copy()
            tune_inv = tune_data['inventory'].copy()
            
            # Extract month and quarter patterns
            train_inv['Month'] = train_inv['Date'].dt.month
            train_inv['Quarter'] = train_inv['Date'].dt.quarter
            tune_inv['Month'] = tune_inv['Date'].dt.month  
            tune_inv['Quarter'] = tune_inv['Date'].dt.quarter
            
            seasonality_analysis['inventory'] = {
                'train_monthly': train_inv.groupby('Month')['Inventory_Level'].mean().to_dict(),
                'tune_monthly': tune_inv.groupby('Month')['Inventory_Level'].mean().to_dict(),
                'train_quarterly': train_inv.groupby('Quarter')['Inventory_Level'].mean().to_dict(),
                'tune_quarterly': tune_inv.groupby('Quarter')['Inventory_Level'].mean().to_dict()
            }
        
        # Flow seasonality
        for flow_type in ['inflows', 'outflows']:
            if flow_type in train_data or flow_type in tune_data:
                seasonality_analysis[flow_type] = self._analyze_flow_seasonality(
                    train_data.get(flow_type), tune_data.get(flow_type)
                )
        
        return seasonality_analysis
    
    def calculate_turnover_metrics(self, inventory_data, flow_data):
        """
        Calculate inventory turnover and related metrics
        """
        metrics = {}
        
        if inventory_data is not None and flow_data is not None:
            # Average inventory
            avg_inventory = inventory_data['Inventory_Level'].mean()
            
            # Total outflows (cost of goods sold equivalent)
            if 'Quantity' in flow_data.columns:
                total_outflows = flow_data['Quantity'].sum()
                
                # Calculate turnover ratio
                turnover_ratio = total_outflows / avg_inventory if avg_inventory > 0 else 0
                
                # Days of inventory
                days_in_period = (inventory_data['Date'].max() - inventory_data['Date'].min()).days
                days_of_inventory = days_in_period / turnover_ratio if turnover_ratio > 0 else days_in_period
                
                metrics['turnover_ratio'] = turnover_ratio
                metrics['days_of_inventory'] = days_of_inventory
                metrics['avg_inventory'] = avg_inventory
                metrics['total_outflows'] = total_outflows
        
        return metrics
    
    def _calculate_basic_stats(self, series):
        """Calculate basic statistical measures"""
        return {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
    
    def _calculate_trend(self, df):
        """Calculate trend using linear regression"""
        if len(df) < 2:
            return 0
        
        x = np.arange(len(df))
        y = df['Inventory_Level'].values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing'
        }
    
    def _interpret_trend_change(self, train_trend, tune_trend):
        """Interpret the change in trend between periods"""
        train_direction = train_trend.get('trend_direction', 'stable')
        tune_direction = tune_trend.get('trend_direction', 'stable')
        
        if train_direction == tune_direction:
            return f"Consistent {train_direction} trend"
        else:
            return f"Trend shift: {train_direction} → {tune_direction}"
    
    def _test_distribution_change(self, train_series, tune_series):
        """Test for significant distribution changes using statistical tests"""
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = stats.ks_2samp(train_series, tune_series)
            
            # Mann-Whitney U test
            mw_stat, mw_p_value = stats.mannwhitneyu(train_series, tune_series)
            
            return {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'ks_significant': ks_p_value < 0.05,
                'mw_statistic': mw_stat,
                'mw_p_value': mw_p_value,
                'mw_significant': mw_p_value < 0.05,
                'distribution_changed': ks_p_value < 0.05 or mw_p_value < 0.05
            }
        except:
            return {'distribution_changed': False}
    
    def _detect_seasonality(self, df):
        """Detect seasonal patterns in inventory data"""
        if len(df) < 30:  # Need sufficient data
            return {'seasonal': False}
        
        # Add time components
        df_temp = df.copy()
        df_temp['Month'] = df_temp['Date'].dt.month
        df_temp['DayOfWeek'] = df_temp['Date'].dt.dayofweek
        
        # Calculate monthly variation
        monthly_means = df_temp.groupby('Month')['Inventory_Level'].mean()
        monthly_cv = monthly_means.std() / monthly_means.mean()
        
        # Calculate weekly variation
        weekly_means = df_temp.groupby('DayOfWeek')['Inventory_Level'].mean()
        weekly_cv = weekly_means.std() / weekly_means.mean()
        
        return {
            'seasonal': monthly_cv > 0.1 or weekly_cv > 0.05,
            'monthly_variation': monthly_cv,
            'weekly_variation': weekly_cv,
            'monthly_pattern': monthly_means.to_dict(),
            'weekly_pattern': weekly_means.to_dict()
        }
    
    def _extract_category_data(self, data):
        """Extract category information from data"""
        category_data = pd.DataFrame()
        
        for data_type, df in data.items():
            if df is not None and 'Product_Type' in df.columns:
                temp_df = df[['Date', 'Product_Type', 'Category', 'Quantity']].copy()
                temp_df['data_type'] = data_type
                category_data = pd.concat([category_data, temp_df], ignore_index=True)
        
        return category_data
    
    def _extract_brand_data(self, data):
        """Extract brand information from data"""
        brand_data = pd.DataFrame()
        
        for data_type, df in data.items():
            if df is not None and 'Brand' in df.columns:
                relevant_cols = ['Date', 'Brand', 'Quantity']
                if 'Total_GIK' in df.columns:
                    relevant_cols.append('Total_GIK')
                
                temp_df = df[relevant_cols].copy()
                temp_df['data_type'] = data_type
                brand_data = pd.concat([brand_data, temp_df], ignore_index=True)
        
        return brand_data
    
    def _extract_warehouse_data(self, data):
        """Extract warehouse information from data"""
        warehouse_data = pd.DataFrame()
        
        for data_type, df in data.items():
            if df is not None and 'Warehouse' in df.columns:
                relevant_cols = ['Date', 'Warehouse', 'Quantity']
                if 'Total_GIK' in df.columns:
                    relevant_cols.append('Total_GIK')
                
                temp_df = df[relevant_cols].copy()
                temp_df['data_type'] = data_type
                warehouse_data = pd.concat([warehouse_data, temp_df], ignore_index=True)
        
        return warehouse_data
    
    def _calculate_distribution_shift(self, train_series, tune_series):
        """Calculate shift in categorical distributions"""
        train_dist = train_series.value_counts(normalize=True)
        tune_dist = tune_series.value_counts(normalize=True)
        
        # Calculate Jensen-Shannon divergence
        all_categories = set(train_dist.index) | set(tune_dist.index)
        
        train_probs = [train_dist.get(cat, 0) for cat in all_categories]
        tune_probs = [tune_dist.get(cat, 0) for cat in all_categories]
        
        # Calculate JS divergence
        js_div = self._jensen_shannon_divergence(train_probs, tune_probs)
        
        return {
            'js_divergence': js_div,
            'shift_magnitude': 'high' if js_div > 0.3 else 'medium' if js_div > 0.1 else 'low',
            'new_categories': set(tune_dist.index) - set(train_dist.index),
            'disappeared_categories': set(train_dist.index) - set(tune_dist.index)
        }
    
    def _jensen_shannon_divergence(self, p, q):
        """Calculate Jensen-Shannon divergence between two probability distributions"""
        p = np.array(p)
        q = np.array(q)
        
        # Normalize to ensure they sum to 1
        p = p / p.sum()
        q = q / q.sum()
        
        # Calculate M = (P + Q) / 2
        m = (p + q) / 2
        
        # Calculate KL divergences (with small epsilon to avoid log(0))
        epsilon = 1e-10
        kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
        kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))
        
        # Jensen-Shannon divergence
        js_div = 0.5 * kl_pm + 0.5 * kl_qm
        
        return js_div
    
    def _calculate_brand_preference_shift(self, train_dist, tune_dist):
        """Calculate brand preference shifts"""
        # Find brands that gained or lost preference
        all_brands = set(train_dist.index) | set(tune_dist.index)
        
        preference_changes = {}
        for brand in all_brands:
            train_pref = train_dist.get(brand, 0)
            tune_pref = tune_dist.get(brand, 0)
            change = tune_pref - train_pref
            preference_changes[brand] = {
                'train_preference': train_pref,
                'tune_preference': tune_pref,
                'change': change,
                'percent_change': (change / train_pref * 100) if train_pref > 0 else float('inf')
            }
        
        return preference_changes
    
    def _analyze_brand_values(self, train_brands, tune_brands):
        """Analyze brand values and changes"""
        value_analysis = {}
        
        train_brand_values = train_brands.groupby('Brand')['Total_GIK'].mean()
        tune_brand_values = tune_brands.groupby('Brand')['Total_GIK'].mean()
        
        all_brands = set(train_brand_values.index) | set(tune_brand_values.index)
        
        for brand in all_brands:
            train_val = train_brand_values.get(brand, 0)
            tune_val = tune_brand_values.get(brand, 0)
            
            value_analysis[brand] = {
                'train_avg_value': train_val,
                'tune_avg_value': tune_val,
                'value_change': tune_val - train_val,
                'value_change_pct': ((tune_val - train_val) / train_val * 100) if train_val > 0 else float('inf')
            }
        
        return value_analysis
    
    def _analyze_warehouse_efficiency(self, train_util, tune_util):
        """Analyze warehouse efficiency changes"""
        efficiency_analysis = {}
        
        # Convert to dictionaries for easier processing
        train_dict = {row['Warehouse']: row for row in train_util}
        tune_dict = {row['Warehouse']: row for row in tune_util}
        
        all_warehouses = set(train_dict.keys()) | set(tune_dict.keys())
        
        for warehouse in all_warehouses:
            train_data = train_dict.get(warehouse, {'Quantity': 0, 'Total_GIK': 0})
            tune_data = tune_dict.get(warehouse, {'Quantity': 0, 'Total_GIK': 0})
            
            # Calculate efficiency metrics (value per unit)
            train_efficiency = train_data['Total_GIK'] / train_data['Quantity'] if train_data['Quantity'] > 0 else 0
            tune_efficiency = tune_data['Total_GIK'] / tune_data['Quantity'] if tune_data['Quantity'] > 0 else 0
            
            efficiency_analysis[warehouse] = {
                'train_efficiency': train_efficiency,
                'tune_efficiency': tune_efficiency,
                'efficiency_change': tune_efficiency - train_efficiency,
                'quantity_change': tune_data['Quantity'] - train_data['Quantity'],
                'value_change': tune_data['Total_GIK'] - train_data['Total_GIK']
            }
        
        return efficiency_analysis
    
    def _analyze_flow_seasonality(self, train_flow, tune_flow):
        """Analyze seasonal patterns in flows"""
        seasonality = {}
        
        for period, df in [('train', train_flow), ('tune', tune_flow)]:
            if df is not None and 'Date' in df.columns and 'Quantity' in df.columns:
                df_temp = df.copy()
                df_temp['Month'] = df_temp['Date'].dt.month
                df_temp['Quarter'] = df_temp['Date'].dt.quarter
                
                monthly_flows = df_temp.groupby('Month')['Quantity'].sum()
                quarterly_flows = df_temp.groupby('Quarter')['Quantity'].sum()
                
                seasonality[f'{period}_monthly'] = monthly_flows.to_dict()
                seasonality[f'{period}_quarterly'] = quarterly_flows.to_dict()
        
        return seasonality
    
    def _analyze_quantities_by_category(self, train_data, tune_data):
        """Analyze quantity patterns by category"""
        quantity_analysis = {}
        
        # Combine all flow data
        train_flows = pd.DataFrame()
        tune_flows = pd.DataFrame()
        
        for flow_type in ['inflows', 'outflows']:
            if flow_type in train_data and train_data[flow_type] is not None:
                train_flows = pd.concat([train_flows, train_data[flow_type]], ignore_index=True)
            if flow_type in tune_data and tune_data[flow_type] is not None:
                tune_flows = pd.concat([tune_flows, tune_data[flow_type]], ignore_index=True)
        
        if not train_flows.empty and not tune_flows.empty and 'Category' in train_flows.columns:
            train_cat_qty = train_flows.groupby('Category')['Quantity'].agg(['sum', 'mean', 'count'])
            tune_cat_qty = tune_flows.groupby('Category')['Quantity'].agg(['sum', 'mean', 'count'])
            
            quantity_analysis = {
                'train_by_category': train_cat_qty.to_dict('index'),
                'tune_by_category': tune_cat_qty.to_dict('index')
            }
        
        return quantity_analysis
