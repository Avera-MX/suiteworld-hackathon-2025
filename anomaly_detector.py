import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self):
        self.outlier_threshold = 1.5  # IQR multiplier
        self.isolation_contamination = 0.1  # Expected proportion of outliers
    
    def detect_anomalies(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Comprehensive anomaly detection across all datasets
        """
        try:
            results = {
                'success': True,
                'outliers': {},
                'drift_analysis': {},
                'scale_shifts': {},
                'anomaly_summary': {}
            }
            
            # Detect outliers in different datasets
            results['outliers'] = self._detect_outliers(datasets)
            
            # Analyze data drift between training and tuning periods
            results['drift_analysis'] = self._analyze_data_drift(datasets)
            
            # Detect scale shifts
            results['scale_shifts'] = self._detect_scale_shifts(datasets)
            
            # Generate summary
            results['anomaly_summary'] = self._generate_anomaly_summary(results)
            
            return results
            
        except Exception as e:
            return {'success': False, 'error': f"Anomaly detection error: {str(e)}"}
    
    def _detect_outliers(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Detect extreme value outliers in various columns
        """
        outlier_results = {}
        
        # Check GIK values in inflows data
        gik_outliers = []
        for dataset_name in ['train_inflows', 'tune_inflows']:
            if dataset_name in datasets and datasets[dataset_name] is not None:
                df = datasets[dataset_name]
                if 'Total_GIK' in df.columns:
                    outliers = self._find_extreme_values(df, 'Total_GIK', dataset_name)
                    gik_outliers.extend(outliers)
        
        outlier_results['gik_outliers'] = gik_outliers
        
        # Check inventory level outliers
        inventory_outliers = []
        for dataset_name in ['train_inventory', 'tune_inventory']:
            if dataset_name in datasets and datasets[dataset_name] is not None:
                df = datasets[dataset_name]
                if 'Inventory_Level' in df.columns:
                    outliers = self._find_extreme_values(df, 'Inventory_Level', dataset_name)
                    inventory_outliers.extend(outliers)
        
        outlier_results['inventory_outliers'] = inventory_outliers
        
        # Check quantity outliers in flows
        quantity_outliers = []
        for dataset_name in ['train_inflows', 'tune_inflows', 'tune_outflows']:
            if dataset_name in datasets and datasets[dataset_name] is not None:
                df = datasets[dataset_name]
                if 'Quantity' in df.columns:
                    outliers = self._find_extreme_values(df, 'Quantity', dataset_name)
                    quantity_outliers.extend(outliers)
        
        outlier_results['quantity_outliers'] = quantity_outliers
        
        return outlier_results
    
    def _find_extreme_values(self, df: pd.DataFrame, column: str, dataset_name: str) -> List[Dict]:
        """
        Find extreme values using multiple methods
        """
        if column not in df.columns:
            return []
        
        outliers = []
        values = df[column].dropna()
        
        if len(values) == 0:
            return []
        
        # Method 1: IQR-based detection
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        iqr_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        # Method 2: Z-score based detection (for extreme values)
        z_scores = np.abs(stats.zscore(values))
        z_outliers = df[np.abs(stats.zscore(df[column].fillna(0))) > 3]
        
        # Method 3: Isolation Forest for complex patterns
        if len(values) > 10:
            iso_forest = IsolationForest(contamination=self.isolation_contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(values.values.reshape(-1, 1))
            iso_outliers = df[outlier_labels == -1]
        else:
            iso_outliers = pd.DataFrame()
        
        # Combine results
        all_outlier_indices = set()
        all_outlier_indices.update(iqr_outliers.index)
        all_outlier_indices.update(z_outliers.index)
        if not iso_outliers.empty:
            all_outlier_indices.update(iso_outliers.index)
        
        # Create outlier records
        for idx in all_outlier_indices:
            outlier_record = {
                'dataset': dataset_name,
                'column': column,
                'index': int(idx),
                'value': float(df.loc[idx, column]) if not pd.isna(df.loc[idx, column]) else None,
                'detection_methods': []
            }
            
            if idx in iqr_outliers.index:
                outlier_record['detection_methods'].append('IQR')
            if idx in z_outliers.index:
                outlier_record['detection_methods'].append('Z-Score')
            if not iso_outliers.empty and idx in iso_outliers.index:
                outlier_record['detection_methods'].append('Isolation_Forest')
            
            # Add date if available
            if 'Date' in df.columns:
                outlier_record['Date'] = df.loc[idx, 'Date']
            
            outliers.append(outlier_record)
        
        return outliers
    
    def _analyze_data_drift(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze data drift between training and tuning periods
        """
        drift_results = {}
        
        # Compare inventory levels
        if 'train_inventory' in datasets and 'tune_inventory' in datasets:
            train_inv = datasets['train_inventory']
            tune_inv = datasets['tune_inventory']
            
            if 'Inventory_Level' in train_inv.columns and 'Inventory_Level' in tune_inv.columns:
                drift_results['inventory_drift'] = self._calculate_distribution_drift(
                    train_inv['Inventory_Level'], tune_inv['Inventory_Level']
                )
        
        # Compare inflow patterns
        if 'train_inflows' in datasets and 'tune_inflows' in datasets:
            train_inflows = datasets['train_inflows']
            tune_inflows = datasets['tune_inflows']
            
            # Compare quantity distributions
            if 'Quantity' in train_inflows.columns and 'Quantity' in tune_inflows.columns:
                drift_results['quantity_drift'] = self._calculate_distribution_drift(
                    train_inflows['Quantity'], tune_inflows['Quantity']
                )
            
            # Compare GIK value distributions
            if 'Total_GIK' in train_inflows.columns and 'Total_GIK' in tune_inflows.columns:
                drift_results['gik_drift'] = self._calculate_distribution_drift(
                    train_inflows['Total_GIK'], tune_inflows['Total_GIK']
                )
            
            # Compare categorical distributions
            categorical_columns = ['Category', 'Brand', 'Warehouse']
            for col in categorical_columns:
                if col in train_inflows.columns and col in tune_inflows.columns:
                    drift_results[f'{col.lower()}_drift'] = self._calculate_categorical_drift(
                        train_inflows[col], tune_inflows[col]
                    )
        
        # Statistical tests for drift detection
        statistical_tests = self._perform_drift_statistical_tests(datasets)
        drift_results['statistical_tests'] = statistical_tests
        
        # Summary of distribution changes
        distribution_changes = {}
        for key, value in drift_results.items():
            if isinstance(value, dict) and 'js_divergence' in value:
                distribution_changes[key] = value['js_divergence']
        
        drift_results['distribution_changes'] = distribution_changes
        
        return drift_results
    
    def _calculate_distribution_drift(self, train_series: pd.Series, tune_series: pd.Series) -> Dict[str, float]:
        """
        Calculate distribution drift using multiple metrics
        """
        # Remove missing values
        train_clean = train_series.dropna()
        tune_clean = tune_series.dropna()
        
        if len(train_clean) == 0 or len(tune_clean) == 0:
            return {'js_divergence': 0, 'ks_statistic': 0, 'wasserstein_distance': 0}
        
        # Jensen-Shannon divergence
        js_div = self._jensen_shannon_divergence(train_clean, tune_clean)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(train_clean, tune_clean)
        
        # Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(train_clean, tune_clean)
        
        return {
            'js_divergence': js_div,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'wasserstein_distance': wasserstein_dist
        }
    
    def _jensen_shannon_divergence(self, p: pd.Series, q: pd.Series) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions
        """
        try:
            # Create histograms
            p_vals = p.values
            q_vals = q.values
            
            # Create common bins
            min_val = min(p_vals.min(), q_vals.min())
            max_val = max(p_vals.max(), q_vals.max())
            bins = np.linspace(min_val, max_val, 50)
            
            # Calculate histograms
            p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
            q_hist, _ = np.histogram(q_vals, bins=bins, density=True)
            
            # Normalize to probabilities
            p_hist = p_hist / p_hist.sum()
            q_hist = q_hist / q_hist.sum()
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p_hist = p_hist + epsilon
            q_hist = q_hist + epsilon
            
            # Calculate JS divergence
            m = 0.5 * (p_hist + q_hist)
            js_div = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)
            
            return js_div
            
        except:
            return 0.0
    
    def _calculate_categorical_drift(self, train_series: pd.Series, tune_series: pd.Series) -> Dict[str, Any]:
        """
        Calculate drift in categorical distributions
        """
        # Get value counts
        train_counts = train_series.value_counts(normalize=True)
        tune_counts = tune_series.value_counts(normalize=True)
        
        # Get all categories
        all_categories = set(train_counts.index) | set(tune_counts.index)
        
        # Create aligned distributions
        train_dist = [train_counts.get(cat, 0) for cat in all_categories]
        tune_dist = [tune_counts.get(cat, 0) for cat in all_categories]
        
        # Chi-square test
        try:
            chi2_stat, chi2_pvalue = stats.chisquare(tune_dist, train_dist)
        except:
            chi2_stat, chi2_pvalue = 0, 1
        
        # Calculate total variation distance
        tv_distance = 0.5 * sum(abs(p - q) for p, q in zip(train_dist, tune_dist))
        
        return {
            'chi2_statistic': chi2_stat,
            'chi2_pvalue': chi2_pvalue,
            'total_variation_distance': tv_distance,
            'new_categories': list(set(tune_counts.index) - set(train_counts.index)),
            'disappeared_categories': list(set(train_counts.index) - set(tune_counts.index))
        }
    
    def _perform_drift_statistical_tests(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform statistical tests for data drift
        """
        tests = {}
        
        # Mann-Whitney U test for inventory levels
        if 'train_inventory' in datasets and 'tune_inventory' in datasets:
            train_inv = datasets['train_inventory']['Inventory_Level'].dropna()
            tune_inv = datasets['tune_inventory']['Inventory_Level'].dropna()
            
            if len(train_inv) > 0 and len(tune_inv) > 0:
                u_stat, u_pvalue = stats.mannwhitneyu(train_inv, tune_inv, alternative='two-sided')
                tests['inventory_mannwhitney'] = {
                    'statistic': u_stat,
                    'p_value': u_pvalue,
                    'drift_detected': u_pvalue < 0.05
                }
        
        return tests
    
    def _detect_scale_shifts(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Detect scale shifts between training and tuning periods
        """
        scale_shifts = {}
        
        # Compare inventory scales
        if 'train_inventory' in datasets and 'tune_inventory' in datasets:
            train_inv = datasets['train_inventory']
            tune_inv = datasets['tune_inventory']
            
            if 'Inventory_Level' in train_inv.columns and 'Inventory_Level' in tune_inv.columns:
                train_mean = train_inv['Inventory_Level'].mean()
                tune_mean = tune_inv['Inventory_Level'].mean()
                
                scale_shifts['inventory_mean_ratio'] = tune_mean / train_mean if train_mean != 0 else 1
                
                train_std = train_inv['Inventory_Level'].std()
                tune_std = tune_inv['Inventory_Level'].std()
                
                scale_shifts['inventory_std_ratio'] = tune_std / train_std if train_std != 0 else 1
        
        # Compare inflow quantities
        if 'train_inflows' in datasets and 'tune_inflows' in datasets:
            train_inflows = datasets['train_inflows']
            tune_inflows = datasets['tune_inflows']
            
            if 'Quantity' in train_inflows.columns and 'Quantity' in tune_inflows.columns:
                train_qty_mean = train_inflows['Quantity'].mean()
                tune_qty_mean = tune_inflows['Quantity'].mean()
                
                scale_shifts['inflow_quantity_ratio'] = tune_qty_mean / train_qty_mean if train_qty_mean != 0 else 1
        
        return scale_shifts
    
    def _generate_anomaly_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of anomaly detection results
        """
        summary = {}
        
        # Count total outliers
        total_outliers = 0
        if 'outliers' in results:
            for outlier_type, outlier_list in results['outliers'].items():
                if isinstance(outlier_list, list):
                    total_outliers += len(outlier_list)
        
        summary['total_outliers_detected'] = total_outliers
        
        # Drift severity assessment
        drift_severity = 'Low'
        if 'drift_analysis' in results and 'distribution_changes' in results['drift_analysis']:
            max_drift = max(results['drift_analysis']['distribution_changes'].values()) if results['drift_analysis']['distribution_changes'] else 0
            
            if max_drift > 0.5:
                drift_severity = 'High'
            elif max_drift > 0.2:
                drift_severity = 'Medium'
        
        summary['data_drift_severity'] = drift_severity
        
        # Scale shift assessment
        scale_shift_severity = 'Low'
        if 'scale_shifts' in results:
            max_ratio = max([abs(1 - ratio) for ratio in results['scale_shifts'].values()]) if results['scale_shifts'] else 0
            
            if max_ratio > 0.5:  # >50% change
                scale_shift_severity = 'High'
            elif max_ratio > 0.2:  # >20% change
                scale_shift_severity = 'Medium'
        
        summary['scale_shift_severity'] = scale_shift_severity
        
        return summary
