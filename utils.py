import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
from scipy import stats
import logging

warnings.filterwarnings('ignore')

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DateUtils:
    """Utility functions for date handling and manipulation"""
    
    @staticmethod
    def parse_date_flexible(date_input: Union[str, pd.Timestamp, datetime]) -> Optional[pd.Timestamp]:
        """
        Parse date with flexible input formats
        """
        if pd.isna(date_input):
            return None
        
        try:
            return pd.to_datetime(date_input)
        except:
            return None
    
    @staticmethod
    def get_date_range_info(dates: pd.Series) -> Dict[str, Any]:
        """
        Get comprehensive date range information
        """
        clean_dates = dates.dropna()
        
        if len(clean_dates) == 0:
            return {'error': 'No valid dates found'}
        
        min_date = clean_dates.min()
        max_date = clean_dates.max()
        total_days = (max_date - min_date).days
        
        # Calculate gaps
        date_diffs = clean_dates.sort_values().diff().dropna()
        avg_gap = date_diffs.mean().days if len(date_diffs) > 0 else 0
        max_gap = date_diffs.max().days if len(date_diffs) > 0 else 0
        
        return {
            'start_date': min_date,
            'end_date': max_date,
            'total_days': total_days,
            'data_points': len(clean_dates),
            'average_gap_days': avg_gap,
            'maximum_gap_days': max_gap,
            'has_gaps': max_gap > 1
        }
    
    @staticmethod
    def fill_date_gaps(df: pd.DataFrame, date_col: str = 'Date', 
                      value_cols: List[str] = None, method: str = 'ffill') -> pd.DataFrame:
        """
        Fill gaps in date series
        """
        if date_col not in df.columns:
            raise DataValidationError(f"Date column '{date_col}' not found")
        
        # Create continuous date range
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create complete dataframe
        complete_df = pd.DataFrame({date_col: date_range})
        
        # Merge with original data
        result_df = complete_df.merge(df, on=date_col, how='left')
        
        # Fill missing values
        if value_cols:
            for col in value_cols:
                if col in result_df.columns:
                    if method == 'ffill':
                        result_df[col] = result_df[col].fillna(method='ffill')
                    elif method == 'bfill':
                        result_df[col] = result_df[col].fillna(method='bfill')
                    elif method == 'interpolate':
                        result_df[col] = result_df[col].interpolate()
                    elif method == 'zero':
                        result_df[col] = result_df[col].fillna(0)
        
        return result_df

class StatisticalUtils:
    """Utility functions for statistical calculations and analysis"""
    
    @staticmethod
    def detect_outliers_multiple_methods(data: pd.Series, methods: List[str] = None) -> Dict[str, Any]:
        """
        Detect outliers using multiple statistical methods
        """
        if methods is None:
            methods = ['iqr', 'zscore', 'modified_zscore']
        
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return {'outliers': [], 'summary': 'No data available'}
        
        outlier_results = {}
        all_outliers = set()
        
        # IQR method
        if 'iqr' in methods:
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            outlier_results['iqr'] = {
                'outliers': iqr_outliers.index.tolist(),
                'count': len(iqr_outliers),
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            all_outliers.update(iqr_outliers.index)
        
        # Z-score method
        if 'zscore' in methods:
            z_scores = np.abs(stats.zscore(clean_data))
            zscore_outliers = clean_data[z_scores > 3]
            outlier_results['zscore'] = {
                'outliers': zscore_outliers.index.tolist(),
                'count': len(zscore_outliers),
                'threshold': 3
            }
            all_outliers.update(zscore_outliers.index)
        
        # Modified Z-score method
        if 'modified_zscore' in methods:
            median = clean_data.median()
            mad = np.median(np.abs(clean_data - median))
            modified_z_scores = 0.6745 * (clean_data - median) / mad
            mod_zscore_outliers = clean_data[np.abs(modified_z_scores) > 3.5]
            outlier_results['modified_zscore'] = {
                'outliers': mod_zscore_outliers.index.tolist(),
                'count': len(mod_zscore_outliers),
                'threshold': 3.5
            }
            all_outliers.update(mod_zscore_outliers.index)
        
        return {
            'methods': outlier_results,
            'all_outliers': list(all_outliers),
            'total_outliers': len(all_outliers),
            'outlier_percentage': (len(all_outliers) / len(clean_data)) * 100
        }
    
    @staticmethod
    def winsorize_data(data: pd.Series, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
        """
        Winsorize data to handle extreme values
        """
        from scipy.stats import mstats
        
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return data
        
        winsorized = mstats.winsorize(clean_data, limits=limits)
        
        # Create result series maintaining original index
        result = data.copy()
        result.loc[clean_data.index] = winsorized
        
        return result
    
    @staticmethod
    def calculate_distribution_metrics(data: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive distribution metrics
        """
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return {}
        
        metrics = {
            'count': len(clean_data),
            'mean': clean_data.mean(),
            'median': clean_data.median(),
            'std': clean_data.std(),
            'var': clean_data.var(),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'range': clean_data.max() - clean_data.min(),
            'q1': clean_data.quantile(0.25),
            'q3': clean_data.quantile(0.75),
            'iqr': clean_data.quantile(0.75) - clean_data.quantile(0.25),
            'skewness': clean_data.skew(),
            'kurtosis': clean_data.kurtosis(),
            'cv': clean_data.std() / clean_data.mean() if clean_data.mean() != 0 else 0
        }
        
        # Add percentiles
        percentiles = [5, 10, 90, 95, 99]
        for p in percentiles:
            metrics[f'p{p}'] = clean_data.quantile(p/100)
        
        return metrics
    
    @staticmethod
    def compare_distributions(data1: pd.Series, data2: pd.Series) -> Dict[str, Any]:
        """
        Compare two distributions using various statistical tests
        """
        clean_data1 = data1.dropna()
        clean_data2 = data2.dropna()
        
        if len(clean_data1) == 0 or len(clean_data2) == 0:
            return {'error': 'Insufficient data for comparison'}
        
        comparison = {}
        
        # Basic statistics comparison
        comparison['descriptive'] = {
            'data1': StatisticalUtils.calculate_distribution_metrics(clean_data1),
            'data2': StatisticalUtils.calculate_distribution_metrics(clean_data2)
        }
        
        # Statistical tests
        try:
            # Two-sample t-test
            t_stat, t_pvalue = stats.ttest_ind(clean_data1, clean_data2)
            comparison['t_test'] = {'statistic': t_stat, 'p_value': t_pvalue}
        except:
            comparison['t_test'] = {'error': 'T-test failed'}
        
        try:
            # Mann-Whitney U test
            u_stat, u_pvalue = stats.mannwhitneyu(clean_data1, clean_data2, alternative='two-sided')
            comparison['mann_whitney'] = {'statistic': u_stat, 'p_value': u_pvalue}
        except:
            comparison['mann_whitney'] = {'error': 'Mann-Whitney test failed'}
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(clean_data1, clean_data2)
            comparison['ks_test'] = {'statistic': ks_stat, 'p_value': ks_pvalue}
        except:
            comparison['ks_test'] = {'error': 'KS test failed'}
        
        return comparison

class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with default value for zero denominator
        """
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """
        Calculate percentage change with safe handling of zero values
        """
        if pd.isna(old_value) or pd.isna(new_value) or old_value == 0:
            return 0.0
        return ((new_value - old_value) / abs(old_value)) * 100
    
    @staticmethod
    def calculate_compound_growth_rate(start_value: float, end_value: float, periods: int) -> float:
        """
        Calculate compound annual growth rate (CAGR)
        """
        if start_value <= 0 or end_value <= 0 or periods <= 0:
            return 0.0
        return (pow(end_value / start_value, 1/periods) - 1) * 100
    
    @staticmethod
    def normalize_data(data: pd.Series, method: str = 'minmax') -> pd.Series:
        """
        Normalize data using various methods
        """
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return data
        
        if method == 'minmax':
            min_val = clean_data.min()
            max_val = clean_data.max()
            if max_val == min_val:
                return pd.Series(0.5, index=data.index)
            normalized = (data - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            mean_val = clean_data.mean()
            std_val = clean_data.std()
            if std_val == 0:
                return pd.Series(0, index=data.index)
            normalized = (data - mean_val) / std_val
            
        elif method == 'robust':
            median_val = clean_data.median()
            mad = np.median(np.abs(clean_data - median_val))
            if mad == 0:
                return pd.Series(0, index=data.index)
            normalized = (data - median_val) / mad
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized

class DataQualityUtils:
    """Utilities for assessing and improving data quality"""
    
    @staticmethod
    def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment
        """
        quality_report = {
            'overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'missing_values': {},
            'duplicates': {},
            'data_types': {},
            'outliers': {},
            'consistency': {}
        }
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        quality_report['missing_values'] = {
            'by_column': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'total_missing_cells': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        quality_report['duplicates'] = {
            'total_duplicate_rows': int(duplicate_rows),
            'percentage': float((duplicate_rows / len(df)) * 100)
        }
        
        # Data types analysis
        quality_report['data_types'] = df.dtypes.value_counts().to_dict()
        
        # Outlier analysis for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            outlier_info = StatisticalUtils.detect_outliers_multiple_methods(df[col])
            quality_report['outliers'][col] = {
                'count': outlier_info['total_outliers'],
                'percentage': outlier_info['outlier_percentage']
            }
        
        # Consistency checks
        quality_report['consistency'] = DataQualityUtils._check_data_consistency(df)
        
        return quality_report
    
    @staticmethod
    def _check_data_consistency(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check various aspects of data consistency
        """
        consistency_issues = {}
        
        # Check for negative values where they shouldn't be
        negative_check_columns = ['Quantity', 'Inventory_Level', 'Total_GIK', 'GIK_Per_Unit']
        for col in negative_check_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    consistency_issues[f'{col}_negative_values'] = int(negative_count)
        
        # Check for extremely large values (potential data entry errors)
        extreme_value_threshold = 1e6  # 1 million
        for col in df.select_dtypes(include=[np.number]).columns:
            extreme_count = (df[col] > extreme_value_threshold).sum()
            if extreme_count > 0:
                consistency_issues[f'{col}_extreme_values'] = int(extreme_count)
        
        # Check date consistency
        if 'Date' in df.columns:
            date_series = pd.to_datetime(df['Date'], errors='coerce')
            future_dates = (date_series > datetime.now()).sum()
            if future_dates > 0:
                consistency_issues['future_dates'] = int(future_dates)
        
        return consistency_issues
    
    @staticmethod
    def suggest_data_cleaning_actions(quality_report: Dict[str, Any]) -> List[str]:
        """
        Suggest data cleaning actions based on quality assessment
        """
        suggestions = []
        
        # Missing values suggestions
        missing_data = quality_report.get('missing_values', {})
        high_missing_columns = [col for col, pct in missing_data.get('percentages', {}).items() if pct > 50]
        
        if high_missing_columns:
            suggestions.append(f"Consider removing columns with >50% missing values: {high_missing_columns}")
        
        moderate_missing_columns = [col for col, pct in missing_data.get('percentages', {}).items() if 10 < pct <= 50]
        if moderate_missing_columns:
            suggestions.append(f"Consider imputation for columns with moderate missing values: {moderate_missing_columns}")
        
        # Duplicate suggestions
        duplicates = quality_report.get('duplicates', {})
        if duplicates.get('total_duplicate_rows', 0) > 0:
            suggestions.append(f"Remove {duplicates['total_duplicate_rows']} duplicate rows")
        
        # Outlier suggestions
        outliers = quality_report.get('outliers', {})
        high_outlier_columns = [col for col, info in outliers.items() if info.get('percentage', 0) > 5]
        if high_outlier_columns:
            suggestions.append(f"Review outliers in columns: {high_outlier_columns}")
        
        # Consistency suggestions
        consistency = quality_report.get('consistency', {})
        if consistency:
            suggestions.append("Address data consistency issues: " + ", ".join(consistency.keys()))
        
        return suggestions

class PerformanceUtils:
    """Utilities for performance monitoring and optimization"""
    
    @staticmethod
    def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_pred_lower: Optional[np.ndarray] = None,
                                  y_pred_upper: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive forecast performance metrics
        """
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid data points for metric calculation'}
        
        metrics = {}
        
        # Basic error metrics
        errors = y_true_clean - y_pred_clean
        metrics['mae'] = np.mean(np.abs(errors))
        metrics['mse'] = np.mean(errors**2)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Percentage-based metrics
        non_zero_mask = y_true_clean != 0
        if np.any(non_zero_mask):
            percentage_errors = np.abs(errors[non_zero_mask] / y_true_clean[non_zero_mask]) * 100
            metrics['mape'] = np.mean(percentage_errors)
            metrics['median_ape'] = np.median(percentage_errors)
        else:
            metrics['mape'] = 0
            metrics['median_ape'] = 0
        
        # Direction accuracy
        if len(y_true_clean) > 1:
            true_direction = np.diff(y_true_clean) > 0
            pred_direction = np.diff(y_pred_clean) > 0
            metrics['direction_accuracy'] = np.mean(true_direction == pred_direction) * 100
        
        # R-squared
        ss_res = np.sum((y_true_clean - y_pred_clean)**2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean))**2)
        metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Confidence interval metrics if available
        if y_pred_lower is not None and y_pred_upper is not None:
            y_pred_lower_clean = y_pred_lower[mask]
            y_pred_upper_clean = y_pred_upper[mask]
            
            # Coverage probability
            within_interval = (y_true_clean >= y_pred_lower_clean) & (y_true_clean <= y_pred_upper_clean)
            metrics['coverage_probability'] = np.mean(within_interval) * 100
            
            # Average interval width
            metrics['avg_interval_width'] = np.mean(y_pred_upper_clean - y_pred_lower_clean)
        
        return metrics
    
    @staticmethod
    def benchmark_model_performance(metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Benchmark model performance against industry standards
        """
        benchmarks = {}
        
        # MAPE benchmarks
        mape = metrics.get('mape', 0)
        if mape < 10:
            benchmarks['mape_rating'] = 'Excellent'
        elif mape < 20:
            benchmarks['mape_rating'] = 'Good'
        elif mape < 30:
            benchmarks['mape_rating'] = 'Acceptable'
        else:
            benchmarks['mape_rating'] = 'Poor'
        
        # R-squared benchmarks
        r_squared = metrics.get('r_squared', 0)
        if r_squared > 0.9:
            benchmarks['fit_quality'] = 'Excellent'
        elif r_squared > 0.7:
            benchmarks['fit_quality'] = 'Good'
        elif r_squared > 0.5:
            benchmarks['fit_quality'] = 'Moderate'
        else:
            benchmarks['fit_quality'] = 'Poor'
        
        # Direction accuracy benchmarks
        direction_accuracy = metrics.get('direction_accuracy', 0)
        if direction_accuracy > 80:
            benchmarks['trend_prediction'] = 'Excellent'
        elif direction_accuracy > 60:
            benchmarks['trend_prediction'] = 'Good'
        elif direction_accuracy > 50:
            benchmarks['trend_prediction'] = 'Acceptable'
        else:
            benchmarks['trend_prediction'] = 'Poor'
        
        return benchmarks

class ConfigUtils:
    """Configuration and settings utilities"""
    
    DEFAULT_CONFIG = {
        'outlier_threshold': 1.5,
        'missing_value_threshold': 0.3,
        'forecast_confidence_level': 0.95,
        'min_data_points': 30,
        'max_forecast_horizon': 365,
        'default_lead_time_days': 7,
        'default_service_level': 0.95
    }
    
    @staticmethod
    def get_config_value(key: str, default=None):
        """
        Get configuration value with fallback to default
        """
        return ConfigUtils.DEFAULT_CONFIG.get(key, default)
    
    @staticmethod
    def validate_forecast_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate forecasting configuration
        """
        errors = []
        
        # Required fields
        required_fields = ['periods', 'confidence_level']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Value ranges
        if 'periods' in config:
            periods = config['periods']
            if not isinstance(periods, int) or periods <= 0:
                errors.append("Periods must be a positive integer")
            elif periods > ConfigUtils.get_config_value('max_forecast_horizon'):
                errors.append(f"Periods cannot exceed {ConfigUtils.get_config_value('max_forecast_horizon')}")
        
        if 'confidence_level' in config:
            conf_level = config['confidence_level']
            if not isinstance(conf_level, (int, float)) or conf_level <= 0 or conf_level >= 1:
                errors.append("Confidence level must be between 0 and 1")
        
        return len(errors) == 0, errors

# Logging utility
def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Set up logger with consistent formatting
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Error handling decorator
def handle_errors(default_return=None, log_errors=True):
    """
    Decorator for consistent error handling
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = setup_logger(func.__module__)
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                
                if default_return is not None:
                    return default_return
                else:
                    raise
        return wrapper
    return decorator

# Data type conversion utilities
def safe_convert_numeric(value, default=0.0):
    """
    Safely convert value to numeric with default fallback
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_convert_int(value, default=0):
    """
    Safely convert value to integer with default fallback
    """
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def safe_convert_string(value, default=""):
    """
    Safely convert value to string with default fallback
    """
    try:
        return str(value) if value is not None else default
    except:
        return default

# Memory management utilities
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type != 'object':
            if str(col_type).startswith('int'):
                # Downcast integers
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
            elif str(col_type).startswith('float'):
                # Downcast floats
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    return df_optimized
