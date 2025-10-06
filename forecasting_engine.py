import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Optional, List
import logging

class ForecastingEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.trained = False
    
    def generate_forecasts(self, train_data: pd.DataFrame, tune_data: pd.DataFrame, 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate forecasts using specified methods
        """
        try:
            results = {'success': True, 'forecasts': {}, 'metrics': {}}
            
            # Prepare data
            train_ts = self._prepare_time_series(train_data)
            tune_ts = self._prepare_time_series(tune_data)
            
            if train_ts is None or tune_ts is None:
                return {'success': False, 'error': 'Failed to prepare time series data'}
            
            # Apply preprocessing
            if config.get('handle_outliers', True):
                train_ts = self._handle_outliers(train_ts)
                tune_ts = self._handle_outliers(tune_ts)
            
            if config.get('scale_normalization', True):
                train_ts, tune_ts, scaler = self._apply_scaling(train_ts, tune_ts)
                self.scalers['inventory'] = scaler
            
            # Generate forecasts based on selected method
            method = config.get('method', 'Prophet')
            
            if method in ['Prophet', 'Both'] and PROPHET_AVAILABLE:
                prophet_result = self._generate_prophet_forecast(
                    train_ts, tune_ts, config
                )
                if prophet_result['success']:
                    results['forecasts']['Prophet'] = prophet_result
            
            if method in ['SARIMA', 'Both']:
                sarima_result = self._generate_sarima_forecast(
                    train_ts, tune_ts, config
                )
                if sarima_result['success']:
                    results['forecasts']['SARIMA'] = sarima_result
            
            # Add stability report
            results['stability_report'] = self._generate_stability_report(
                train_ts, tune_ts, results['forecasts']
            )
            
            return results
            
        except Exception as e:
            return {'success': False, 'error': f"Forecasting error: {str(e)}"}
    
    def _prepare_time_series(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare time series data for forecasting
        """
        try:
            if 'Date' not in df.columns or 'Inventory_Level' not in df.columns:
                return None
            
            ts_df = df[['Date', 'Inventory_Level']].copy()
            ts_df = ts_df.rename(columns={'Date': 'ds', 'Inventory_Level': 'y'})
            
            # Convert date column
            ts_df['ds'] = pd.to_datetime(ts_df['ds'])
            
            # Sort and remove duplicates
            ts_df = ts_df.sort_values('ds').drop_duplicates(subset=['ds'], keep='last')
            
            # Remove missing values
            ts_df = ts_df.dropna()
            
            return ts_df
            
        except Exception as e:
            print(f"Error preparing time series: {str(e)}")
            return None
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using IQR method
        """
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        df['y'] = df['y'].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _apply_scaling(self, train_df: pd.DataFrame, tune_df: pd.DataFrame):
        """
        Apply scaling to time series data
        """
        scaler = StandardScaler()
        
        # Fit on training data
        train_scaled = train_df.copy()
        train_scaled['y'] = scaler.fit_transform(train_df[['y']])
        
        # Transform tuning data
        tune_scaled = tune_df.copy()
        tune_scaled['y'] = scaler.transform(tune_df[['y']])
        
        return train_scaled, tune_scaled, scaler
    
    def _generate_prophet_forecast(self, train_ts: pd.DataFrame, tune_ts: pd.DataFrame, 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Prophet forecast
        """
        if not PROPHET_AVAILABLE:
            return {'success': False, 'error': 'Prophet not available'}
        
        try:
            # Initialize Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=config.get('confidence_level', 0.95)
            )
            
            # Add custom seasonalities if trend detection is enabled
            if config.get('trend_detection', True):
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
            
            # Fit model
            model.fit(train_ts)
            
            # Create future dataframe
            periods = config.get('periods', 90)
            future = model.make_future_dataframe(periods=periods)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Calculate metrics on tune data if available
            metrics = self._calculate_metrics(tune_ts, forecast, model_name='Prophet')
            
            # Inverse transform if scaling was applied
            if 'inventory' in self.scalers:
                forecast[['yhat', 'yhat_lower', 'yhat_upper']] = self.scalers['inventory'].inverse_transform(
                    forecast[['yhat', 'yhat_lower', 'yhat_upper']]
                )
                train_ts['y'] = self.scalers['inventory'].inverse_transform(train_ts[['y']])
            
            return {
                'success': True,
                'model': model,
                'forecast': forecast,
                'historical': train_ts,
                'metrics': metrics
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Prophet forecast error: {str(e)}"}
    
    def _generate_sarima_forecast(self, train_ts: pd.DataFrame, tune_ts: pd.DataFrame, 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SARIMA forecast
        """
        try:
            # Prepare data
            y_train = train_ts['y'].values
            
            # Auto-determine SARIMA parameters
            sarima_order, seasonal_order = self._determine_sarima_params(y_train)
            
            # Fit SARIMA model
            model = SARIMAX(
                y_train,
                order=sarima_order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False, maxiter=100)
            
            # Generate forecast
            periods = config.get('periods', 90)
            forecast = fitted_model.get_forecast(steps=periods)
            forecast_df = pd.DataFrame({
                'ds': pd.date_range(start=train_ts['ds'].max() + timedelta(days=1), periods=periods),
                'yhat': forecast.predicted_mean,
                'yhat_lower': forecast.conf_int().iloc[:, 0],
                'yhat_upper': forecast.conf_int().iloc[:, 1]
            })
            
            # Calculate metrics
            metrics = self._calculate_metrics(tune_ts, forecast_df, model_name='SARIMA')
            
            # Inverse transform if scaling was applied
            if 'inventory' in self.scalers:
                forecast_df[['yhat', 'yhat_lower', 'yhat_upper']] = self.scalers['inventory'].inverse_transform(
                    forecast_df[['yhat', 'yhat_lower', 'yhat_upper']]
                )
                train_ts['y'] = self.scalers['inventory'].inverse_transform(train_ts[['y']])
            
            return {
                'success': True,
                'model': fitted_model,
                'forecast': forecast_df,
                'historical': train_ts,
                'metrics': metrics
            }
            
        except Exception as e:
            return {'success': False, 'error': f"SARIMA forecast error: {str(e)}"}
    
    def _determine_sarima_params(self, y: np.ndarray) -> tuple:
        """
        Determine SARIMA parameters using simple heuristics
        """
        # Use simple default parameters that often work well
        # In a production system, you'd use more sophisticated parameter selection
        sarima_order = (1, 1, 1)  # (p, d, q)
        seasonal_order = (1, 1, 1, 12)  # (P, D, Q, s)
        
        return sarima_order, seasonal_order
    
    def _calculate_metrics(self, actual_ts: pd.DataFrame, forecast_df: pd.DataFrame, 
                          model_name: str) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics
        """
        try:
            if actual_ts is None or len(actual_ts) == 0:
                return {'rmse': 0, 'mae': 0, 'mape': 0}
            
            # Find overlapping dates for validation
            actual_dates = set(actual_ts['ds'].dt.date)
            forecast_dates = set(forecast_df['ds'].dt.date)
            common_dates = actual_dates.intersection(forecast_dates)
            
            if len(common_dates) == 0:
                return {'rmse': 0, 'mae': 0, 'mape': 0}
            
            # Filter to common dates
            actual_filtered = actual_ts[actual_ts['ds'].dt.date.isin(common_dates)].copy()
            forecast_filtered = forecast_df[forecast_df['ds'].dt.date.isin(common_dates)].copy()
            
            if len(actual_filtered) == 0 or len(forecast_filtered) == 0:
                return {'rmse': 0, 'mae': 0, 'mape': 0}
            
            # Merge on date
            merged = actual_filtered.merge(forecast_filtered, on='ds', suffixes=('_actual', '_forecast'))
            
            if len(merged) == 0:
                return {'rmse': 0, 'mae': 0, 'mape': 0}
            
            y_true = merged['y'].values
            y_pred = merged['yhat'].values
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate MAPE, avoiding division by zero
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
            
            return {'rmse': rmse, 'mae': mae, 'mape': mape}
            
        except Exception as e:
            print(f"Error calculating metrics for {model_name}: {str(e)}")
            return {'rmse': 0, 'mae': 0, 'mape': 0}
    
    def _generate_stability_report(self, train_ts: pd.DataFrame, tune_ts: pd.DataFrame, 
                                  forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate model stability report
        """
        try:
            report = {}
            
            # Time period coverage
            train_days = (train_ts['ds'].max() - train_ts['ds'].min()).days
            tune_days = (tune_ts['ds'].max() - tune_ts['ds'].min()).days
            report['coverage'] = f"{train_days + tune_days} days total"
            
            # Scale adaptation
            train_mean = train_ts['y'].mean()
            tune_mean = tune_ts['y'].mean()
            scale_ratio = tune_mean / train_mean if train_mean != 0 else 1
            
            if 0.8 <= scale_ratio <= 1.2:
                report['scale_adaptation'] = "Good"
            elif 0.5 <= scale_ratio <= 2.0:
                report['scale_adaptation'] = "Moderate"
            else:
                report['scale_adaptation'] = "Poor"
            
            # Prediction consistency
            model_count = len([f for f in forecasts.values() if f.get('success', False)])
            if model_count >= 2:
                report['consistency'] = "Multiple models available"
            elif model_count == 1:
                report['consistency'] = "Single model"
            else:
                report['consistency'] = "No successful models"
            
            return report
            
        except Exception as e:
            return {'error': f"Stability report generation failed: {str(e)}"}
    
    def detect_trend_changes(self, ts_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect trend changes in time series
        """
        try:
            if len(ts_df) < 24:  # Need at least 2 years of data
                return {'trend_detected': False, 'reason': 'Insufficient data'}
            
            # Perform seasonal decomposition
            y_series = ts_df.set_index('ds')['y']
            
            # Use additive decomposition
            decomposition = seasonal_decompose(y_series, model='additive', period=365)
            
            trend = decomposition.trend.dropna()
            
            # Detect trend changes using simple slope analysis
            mid_point = len(trend) // 2
            early_trend = np.polyfit(range(mid_point), trend.values[:mid_point], 1)[0]
            late_trend = np.polyfit(range(mid_point), trend.values[-mid_point:], 1)[0]
            
            trend_change = abs(late_trend - early_trend) > (trend.std() * 0.1)
            
            return {
                'trend_detected': trend_change,
                'early_slope': early_trend,
                'late_slope': late_trend,
                'change_magnitude': abs(late_trend - early_trend)
            }
            
        except Exception as e:
            return {'trend_detected': False, 'error': str(e)}
