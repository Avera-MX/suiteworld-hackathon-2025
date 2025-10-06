import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import zscore, ks_2samp
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = None
        self.scalers = {}
        
    def detect_anomalies(self, data_dict):
        """Comprehensive anomaly detection across all data types"""
        try:
            results = {
                'inventory_anomalies': [],
                'gik_outliers': [],
                'distribution_changes': {},
                'drift_detected': False,
                'anomaly_summary': {}
            }
            
            # Detect inventory level anomalies
            inventory_anomalies = self._detect_inventory_anomalies(data_dict['inventory'])
            results['inventory_anomalies'] = inventory_anomalies
            
            # Detect extreme GIK value outliers
            gik_outliers = self._detect_gik_outliers(data_dict['inflows'], data_dict['outflows'])
            results['gik_outliers'] = gik_outliers
            
            # Detect distribution changes between training and tune periods
            distribution_changes = self._detect_distribution_drift(data_dict)
            results['distribution_changes'] = distribution_changes
            results['drift_detected'] = len(distribution_changes) > 0
            
            # Generate anomaly summary
            results['anomaly_summary'] = self._generate_anomaly_summary(results)
            
            return results
            
        except Exception as e:
            raise Exception(f"Anomaly detection failed: {str(e)}")
    
    def _detect_inventory_anomalies(self, inventory_data):
        """Detect anomalies in inventory levels using multiple methods"""
        anomalies = []
        
        # Method 1: Statistical outliers using Z-score
        z_scores = np.abs(zscore(inventory_data['Inventory_Level']))
        statistical_outliers = inventory_data[z_scores > 3].copy()
        statistical_outliers['Anomaly_Type'] = 'Statistical_Outlier'
        statistical_outliers['Anomaly_Score'] = z_scores[z_scores > 3]
        
        # Method 2: IQR-based outliers
        Q1 = inventory_data['Inventory_Level'].quantile(0.25)
        Q3 = inventory_data['Inventory_Level'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = inventory_data[
            (inventory_data['Inventory_Level'] < lower_bound) | 
            (inventory_data['Inventory_Level'] > upper_bound)
        ].copy()
        iqr_outliers['Anomaly_Type'] = 'IQR_Outlier'
        iqr_outliers['Anomaly_Score'] = np.abs(
            (iqr_outliers['Inventory_Level'] - inventory_data['Inventory_Level'].median()) / 
            inventory_data['Inventory_Level'].std()
        )
        
        # Method 3: Isolation Forest for complex patterns
        if len(inventory_data) > 50:
            # Prepare features for Isolation Forest
            inventory_features = self._prepare_inventory_features(inventory_data)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(inventory_features)
            
            isolation_anomalies = inventory_data[anomaly_labels == -1].copy()
            isolation_anomalies['Anomaly_Type'] = 'Pattern_Anomaly'
            isolation_anomalies['Anomaly_Score'] = -iso_forest.decision_function(
                inventory_features[anomaly_labels == -1]
            )
            
            anomalies.extend([statistical_outliers, iqr_outliers, isolation_anomalies])
        else:
            anomalies.extend([statistical_outliers, iqr_outliers])
        
        # Combine and deduplicate anomalies
        if anomalies:
            combined_anomalies = pd.concat(anomalies, ignore_index=True)
            combined_anomalies = combined_anomalies.drop_duplicates(subset=['Date'])
            return combined_anomalies.sort_values('Anomaly_Score', ascending=False)
        
        return pd.DataFrame()
    
    def _prepare_inventory_features(self, inventory_data):
        """Prepare features for anomaly detection"""
        features_df = inventory_data.copy()
        
        # Time-based features
        features_df['DayOfYear'] = features_df['Date'].dt.dayofyear
        features_df['Month'] = features_df['Date'].dt.month
        features_df['Quarter'] = features_df['Date'].dt.quarter
        
        # Rolling statistics
        features_df['Rolling_Mean_7'] = features_df['Inventory_Level'].rolling(7).mean()
        features_df['Rolling_Std_7'] = features_df['Inventory_Level'].rolling(7).std()
        features_df['Rolling_Mean_30'] = features_df['Inventory_Level'].rolling(30).mean()
        
        # Lag features
        features_df['Inventory_Lag1'] = features_df['Inventory_Level'].shift(1)
        features_df['Inventory_Lag7'] = features_df['Inventory_Level'].shift(7)
        
        # Select numeric features only
        feature_cols = ['Inventory_Level', 'DayOfYear', 'Month', 'Quarter', 
                       'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_30',
                       'Inventory_Lag1', 'Inventory_Lag7']
        
        features = features_df[feature_cols].fillna(method='bfill').fillna(method='ffill')
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        return scaled_features
    
    def _detect_gik_outliers(self, inflows_data, outflows_data):
        """Detect extreme GIK value outliers"""
        gik_outliers = []
        
        # Analyze inflows GIK values
        if 'Total_GIK' in inflows_data.columns:
            inflows_outliers = self._find_gik_outliers(inflows_data, 'inflows')
            gik_outliers.append(inflows_outliers)
        
        # Analyze outflows GIK values
        if 'Total_GIK' in outflows_data.columns:
            outflows_outliers = self._find_gik_outliers(outflows_data, 'outflows')
            gik_outliers.append(outflows_outliers)
        
        if gik_outliers:
            return pd.concat(gik_outliers, ignore_index=True)
        
        return pd.DataFrame()
    
    def _find_gik_outliers(self, data, data_type):
        """Find GIK value outliers using modified Z-score"""
        if 'Total_GIK' not in data.columns or data['Total_GIK'].isna().all():
            return pd.DataFrame()
        
        # Calculate modified Z-score (more robust to outliers)
        median = data['Total_GIK'].median()
        mad = np.median(np.abs(data['Total_GIK'] - median))
        modified_z_scores = 0.6745 * (data['Total_GIK'] - median) / mad
        
        # Identify extreme outliers (modified Z-score > 3.5)
        outliers = data[np.abs(modified_z_scores) > 3.5].copy()
        outliers['Anomaly_Score'] = np.abs(modified_z_scores[np.abs(modified_z_scores) > 3.5])
        outliers['Data_Type'] = data_type
        outliers['Outlier_Type'] = 'Extreme_GIK_Value'
        
        return outliers[['Date', 'Product_Type', 'Brand', 'Total_GIK', 'Anomaly_Score', 
                        'Data_Type', 'Outlier_Type']]
    
    def _detect_distribution_drift(self, data_dict):
        """Detect distribution changes between different time periods"""
        distribution_changes = {}
        
        # Separate data by time periods (2017-2018 vs 2023)
        inventory_data = data_dict['inventory']
        training_period = inventory_data[inventory_data['Date'] < '2019-01-01']
        tune_period = inventory_data[inventory_data['Date'] >= '2023-01-01']
        
        if len(training_period) > 0 and len(tune_period) > 0:
            # Test for distribution changes in inventory levels
            ks_stat, p_value = ks_2samp(
                training_period['Inventory_Level'], 
                tune_period['Inventory_Level']
            )
            
            distribution_changes['inventory_levels'] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < 0.05,
                'training_mean': training_period['Inventory_Level'].mean(),
                'tune_mean': tune_period['Inventory_Level'].mean(),
                'scale_change': tune_period['Inventory_Level'].mean() / training_period['Inventory_Level'].mean()
            }
        
        # Check for categorical distribution changes in inflows/outflows
        inflows_data = data_dict['inflows']
        outflows_data = data_dict['outflows']
        
        # Analyze product category distributions
        if len(inflows_data) > 0:
            training_inflows = inflows_data[inflows_data['Date'] < '2019-01-01']
            tune_inflows = inflows_data[inflows_data['Date'] >= '2023-01-01']
            
            if len(training_inflows) > 0 and len(tune_inflows) > 0:
                distribution_changes['product_categories'] = self._compare_categorical_distributions(
                    training_inflows, tune_inflows, 'Category'
                )
                
                distribution_changes['brands'] = self._compare_categorical_distributions(
                    training_inflows, tune_inflows, 'Brand'
                )
                
                distribution_changes['warehouses'] = self._compare_categorical_distributions(
                    training_inflows, tune_inflows, 'Warehouse'
                )
        
        return distribution_changes
    
    def _compare_categorical_distributions(self, period1_data, period2_data, column):
        """Compare categorical distributions between two periods"""
        period1_dist = period1_data[column].value_counts(normalize=True)
        period2_dist = period2_data[column].value_counts(normalize=True)
        
        # Align distributions
        all_categories = set(period1_dist.index) | set(period2_dist.index)
        period1_aligned = pd.Series([period1_dist.get(cat, 0) for cat in all_categories], 
                                   index=all_categories)
        period2_aligned = pd.Series([period2_dist.get(cat, 0) for cat in all_categories], 
                                   index=all_categories)
        
        # Calculate chi-square test
        try:
            chi2_stat, p_value = stats.chisquare(period2_aligned, period1_aligned)
            
            return {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'drift_detected': p_value < 0.05,
                'period1_top_categories': period1_dist.head().to_dict(),
                'period2_top_categories': period2_dist.head().to_dict()
            }
        except:
            return {
                'chi2_statistic': np.nan,
                'p_value': np.nan,
                'drift_detected': False,
                'period1_top_categories': period1_dist.head().to_dict(),
                'period2_top_categories': period2_dist.head().to_dict()
            }
    
    def _generate_anomaly_summary(self, results):
        """Generate summary of anomaly detection results"""
        summary = {
            'total_inventory_anomalies': len(results['inventory_anomalies']),
            'total_gik_outliers': len(results['gik_outliers']),
            'distribution_drift_detected': results['drift_detected'],
            'severity_breakdown': {}
        }
        
        # Categorize anomalies by severity
        if len(results['inventory_anomalies']) > 0:
            high_severity = results['inventory_anomalies'][
                results['inventory_anomalies']['Anomaly_Score'] > 5
            ]
            medium_severity = results['inventory_anomalies'][
                (results['inventory_anomalies']['Anomaly_Score'] > 3) & 
                (results['inventory_anomalies']['Anomaly_Score'] <= 5)
            ]
            low_severity = results['inventory_anomalies'][
                results['inventory_anomalies']['Anomaly_Score'] <= 3
            ]
            
            summary['severity_breakdown'] = {
                'high': len(high_severity),
                'medium': len(medium_severity),
                'low': len(low_severity)
            }
        
        return summary
