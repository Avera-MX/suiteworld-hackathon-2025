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

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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
            
            # Extract warehouse features from config
            train_warehouse_features = config.get('train_warehouse_features')
            tune_warehouse_features = config.get('tune_warehouse_features')
            
            # Extract category features from config
            train_category_features = config.get('train_category_features')
            tune_category_features = config.get('tune_category_features')
            
            # Prepare data with warehouse and category features
            train_ts = self._prepare_time_series(train_data, train_warehouse_features, train_category_features)
            tune_ts = self._prepare_time_series(tune_data, tune_warehouse_features, tune_category_features)
            
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
            
            if method in ['Prophet', 'Both', 'All'] and PROPHET_AVAILABLE:
                prophet_result = self._generate_prophet_forecast(
                    train_ts, tune_ts, config
                )
                if prophet_result['success']:
                    results['forecasts']['Prophet'] = prophet_result
            
            if method in ['SARIMA', 'Both', 'All']:
                sarima_result = self._generate_sarima_forecast(
                    train_ts, tune_ts, config
                )
                if sarima_result['success']:
                    results['forecasts']['SARIMA'] = sarima_result
            
            if method in ['XGBoost', 'All'] and XGBOOST_AVAILABLE:
                xgboost_result = self._generate_xgboost_forecast(
                    train_ts, tune_ts, config
                )
                if xgboost_result['success']:
                    results['forecasts']['XGBoost'] = xgboost_result
            
            # Add stability report
            results['stability_report'] = self._generate_stability_report(
                train_ts, tune_ts, results['forecasts']
            )
            
            # Add confidence scoring for each forecast
            results['confidence_scores'] = self._calculate_confidence_scores(
                train_ts, tune_ts, results['forecasts']
            )
            
            return results
            
        except Exception as e:
            return {'success': False, 'error': f"Forecasting error: {str(e)}"}
    
    def _prepare_time_series(self, df: pd.DataFrame, warehouse_features=None, category_features=None) -> Optional[pd.DataFrame]:
        """
        Prepare time series data for forecasting with optional warehouse and category features
        """
        try:
            if 'Date' not in df.columns or 'Inventory_Level' not in df.columns:
                return None
            
            ts_df = df[['Date', 'Inventory_Level']].copy()
            ts_df = ts_df.rename(columns={'Date': 'ds', 'Inventory_Level': 'y'})
            
            # Convert date column
            ts_df['ds'] = pd.to_datetime(ts_df['ds'])
            
            # Merge warehouse features if available
            if warehouse_features is not None and not warehouse_features.empty:
                warehouse_features_copy = warehouse_features.copy()
                warehouse_features_copy['Date'] = pd.to_datetime(warehouse_features_copy['Date'])
                warehouse_features_copy = warehouse_features_copy.rename(columns={'Date': 'ds'})
                ts_df = pd.merge(ts_df, warehouse_features_copy, on='ds', how='left')
                
                # Fill missing warehouse features with 0
                warehouse_cols = ['Total_Daily_Inflows', 'Total_Daily_Outflows', 'Active_Warehouses', 'Warehouse_Diversity']
                for col in warehouse_cols:
                    if col in ts_df.columns:
                        ts_df[col] = ts_df[col].fillna(0)
            
            # Merge category features if available
            if category_features is not None and not category_features.empty:
                category_features_copy = category_features.copy()
                category_features_copy['Date'] = pd.to_datetime(category_features_copy['Date'])
                category_features_copy = category_features_copy.rename(columns={'Date': 'ds'})
                ts_df = pd.merge(ts_df, category_features_copy, on='ds', how='left')
                
                # Fill missing category features with 0
                category_cols = ['Total_Category_Inflows', 'Total_Category_Outflows', 'Active_Categories', 'Category_Diversity']
                for col in category_cols:
                    if col in ts_df.columns:
                        ts_df[col] = ts_df[col].fillna(0)
            
            # Sort and remove duplicates
            ts_df = ts_df.sort_values('ds').drop_duplicates(subset=['ds'], keep='last')
            
            # Remove missing values in critical columns
            ts_df = ts_df.dropna(subset=['ds', 'y'])
            
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
        Generate Prophet forecast with warehouse features
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
            
            # Add warehouse features as regressors if available
            warehouse_features = ['Total_Daily_Inflows', 'Total_Daily_Outflows', 'Active_Warehouses', 'Warehouse_Diversity']
            available_warehouse_features = [col for col in warehouse_features if col in train_ts.columns]
            for col in available_warehouse_features:
                model.add_regressor(col)
            
            # Add category features as regressors if available
            category_features = ['Total_Category_Inflows', 'Total_Category_Outflows', 'Active_Categories', 'Category_Diversity']
            available_category_features = [col for col in category_features if col in train_ts.columns]
            for col in available_category_features:
                model.add_regressor(col)
            
            # Fit model
            model.fit(train_ts)
            
            # Create future dataframe
            periods = config.get('periods', 90)
            future = model.make_future_dataframe(periods=periods)
            
            # Merge warehouse features from historical data and extend to future
            for col in available_warehouse_features:
                if col in train_ts.columns:
                    # Merge historical values from train_ts
                    future = future.merge(
                        train_ts[['ds', col]], 
                        on='ds', 
                        how='left'
                    )
                    # For future dates (where merge resulted in NaN), use forward fill or last value
                    future[col] = future[col].fillna(method='ffill')
                    # If still NaN (shouldn't happen), use 0
                    future[col] = future[col].fillna(0)
            
            # Merge category features from historical data and extend to future
            for col in available_category_features:
                if col in train_ts.columns:
                    # Merge historical values from train_ts
                    future = future.merge(
                        train_ts[['ds', col]], 
                        on='ds', 
                        how='left'
                    )
                    # For future dates (where merge resulted in NaN), use forward fill
                    future[col] = future[col].fillna(method='ffill')
                    # If still NaN, use 0
                    future[col] = future[col].fillna(0)
            
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
    
    def _generate_xgboost_forecast(self, train_ts: pd.DataFrame, tune_ts: pd.DataFrame, 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate XGBoost forecast using engineered features
        """
        if not XGBOOST_AVAILABLE:
            return {'success': False, 'error': 'XGBoost not available'}
        
        try:
            # Create features for training
            train_features = self._create_time_series_features(train_ts)
            
            if train_features is None or len(train_features) < 10:
                return {'success': False, 'error': 'Insufficient data for XGBoost training'}
            
            # Prepare training data
            feature_cols = [col for col in train_features.columns if col not in ['ds', 'y']]
            X_train = train_features[feature_cols].values
            y_train = train_features['y'].values
            
            # Train XGBoost model
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            
            # Generate future dates
            periods = config.get('periods', 90)
            last_date = train_ts['ds'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
            
            # Create future dataframe with features
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Maintain history for rolling features - use last 30 days from training
            history_window = train_ts['y'].tail(30).tolist()
            
            predictions = []
            prediction_std = train_features['y'].std()
            
            for i, date in enumerate(future_dates):
                # Create features for future date
                features = {
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'dayofweek': date.dayofweek,
                    'dayofyear': date.dayofyear,
                    'quarter': date.quarter,
                    'weekofyear': date.isocalendar()[1]
                }
                
                # Add lag features using recent predictions and history
                if i > 0:
                    features['lag_1'] = predictions[-1]
                    if i >= 7:
                        features['lag_7'] = predictions[i-7]
                    else:
                        idx_from_history = len(history_window) - (7 - i)
                        features['lag_7'] = history_window[idx_from_history] if idx_from_history >= 0 else history_window[0]
                else:
                    features['lag_1'] = history_window[-1] if len(history_window) > 0 else 0
                    features['lag_7'] = history_window[-7] if len(history_window) >= 7 else history_window[0]
                
                # Calculate rolling statistics using history + predictions
                all_values = history_window + predictions
                recent_7 = all_values[-7:] if len(all_values) >= 7 else all_values
                
                features['rolling_mean_7'] = np.mean(recent_7) if len(recent_7) > 0 else 0
                features['rolling_std_7'] = np.std(recent_7) if len(recent_7) > 1 else 0
                
                # Create feature vector in same order as training
                X_future = np.array([[features.get(col, 0) for col in feature_cols]])
                
                # Predict
                pred = model.predict(X_future)[0]
                predictions.append(pred)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': predictions,
                'yhat_lower': np.array(predictions) - 1.96 * prediction_std,
                'yhat_upper': np.array(predictions) + 1.96 * prediction_std
            })
            
            # Calculate metrics on tune data if available
            metrics = self._calculate_metrics(tune_ts, forecast_df, model_name='XGBoost')
            
            # Inverse transform if scaling was applied
            if 'inventory' in self.scalers:
                forecast_df[['yhat', 'yhat_lower', 'yhat_upper']] = self.scalers['inventory'].inverse_transform(
                    forecast_df[['yhat', 'yhat_lower', 'yhat_upper']]
                )
                train_ts['y'] = self.scalers['inventory'].inverse_transform(train_ts[['y']])
            
            # Store feature importance
            feature_importance = {
                feature: float(importance) 
                for feature, importance in zip(feature_cols, model.feature_importances_)
            }
            
            return {
                'success': True,
                'model': model,
                'forecast': forecast_df,
                'historical': train_ts,
                'metrics': metrics,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            return {'success': False, 'error': f"XGBoost forecast error: {str(e)}"}
    
    def _create_time_series_features(self, ts_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Create features from time series data for XGBoost
        """
        try:
            df = ts_df.copy()
            
            # Date features
            df['year'] = df['ds'].dt.year
            df['month'] = df['ds'].dt.month
            df['day'] = df['ds'].dt.day
            df['dayofweek'] = df['ds'].dt.dayofweek
            df['dayofyear'] = df['ds'].dt.dayofyear
            df['quarter'] = df['ds'].dt.quarter
            df['weekofyear'] = df['ds'].dt.isocalendar().week
            
            # Lag features
            df['lag_1'] = df['y'].shift(1)
            df['lag_7'] = df['y'].shift(7)
            
            # Rolling statistics
            df['rolling_mean_7'] = df['y'].rolling(window=7, min_periods=1).mean()
            df['rolling_std_7'] = df['y'].rolling(window=7, min_periods=1).std()
            
            # Drop rows with NaN values from lag features
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error creating features: {str(e)}")
            return None
    
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
    
    def _calculate_confidence_scores(self, train_ts: pd.DataFrame, tune_ts: pd.DataFrame,
                                     forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate real-time confidence scores for forecast predictions
        """
        try:
            confidence_data = {}
            
            for method, result in forecasts.items():
                if not result.get('success', False):
                    continue
                    
                forecast_df = result.get('forecast')
                if forecast_df is None or len(forecast_df) == 0:
                    continue
                
                # Calculate confidence based on multiple factors
                scores = []
                
                for idx, row in forecast_df.iterrows():
                    # Factor 1: Prediction interval width (narrower = higher confidence)
                    if 'yhat_lower' in forecast_df and 'yhat_upper' in forecast_df:
                        interval_width = row['yhat_upper'] - row['yhat_lower']
                        pred_value = row['yhat']
                        if pred_value != 0:
                            relative_width = interval_width / abs(pred_value)
                            interval_score = max(0, 1 - (relative_width / 2))
                        else:
                            interval_score = 0.5
                    else:
                        interval_score = 0.5
                    
                    # Factor 2: Distance from training data (closer = higher confidence)
                    days_ahead = idx + 1
                    distance_score = max(0, 1 - (days_ahead / 180))
                    
                    # Factor 3: Historical model performance
                    if 'metrics' in result:
                        mae = result['metrics'].get('mae', 0)
                        train_mean = train_ts['y'].mean()
                        if train_mean > 0:
                            error_ratio = mae / train_mean
                            performance_score = max(0, 1 - error_ratio)
                        else:
                            performance_score = 0.5
                    else:
                        performance_score = 0.5
                    
                    # Factor 4: Data stability (how stable was training data)
                    train_cv = train_ts['y'].std() / train_ts['y'].mean() if train_ts['y'].mean() > 0 else 1
                    stability_score = max(0, 1 - min(train_cv, 1))
                    
                    # Weighted combination
                    confidence = (
                        0.3 * interval_score +
                        0.25 * distance_score +
                        0.25 * performance_score +
                        0.2 * stability_score
                    )
                    
                    # Convert to percentage and category
                    confidence_pct = confidence * 100
                    
                    if confidence_pct >= 80:
                        category = "High"
                    elif confidence_pct >= 60:
                        category = "Medium"
                    else:
                        category = "Low"
                    
                    scores.append({
                        'date': row['ds'],
                        'confidence': confidence_pct,
                        'category': category,
                        'factors': {
                            'interval': interval_score * 100,
                            'distance': distance_score * 100,
                            'performance': performance_score * 100,
                            'stability': stability_score * 100
                        }
                    })
                
                confidence_data[method] = {
                    'scores': scores,
                    'average_confidence': np.mean([s['confidence'] for s in scores]),
                    'high_confidence_days': len([s for s in scores if s['category'] == 'High']),
                    'medium_confidence_days': len([s for s in scores if s['category'] == 'Medium']),
                    'low_confidence_days': len([s for s in scores if s['category'] == 'Low'])
                }
            
            return confidence_data
            
        except Exception as e:
            print(f"Error calculating confidence scores: {str(e)}")
            return {}
    
    def recommend_best_model(self, train_ts: pd.DataFrame, tune_ts: pd.DataFrame) -> Dict[str, Any]:
        """
        Automatically recommend best forecasting model based on data characteristics
        """
        try:
            recommendations = {
                'recommended_model': None,
                'reasons': [],
                'data_characteristics': {},
                'all_scores': {}
            }
            
            # Analyze data characteristics
            data_size = len(train_ts)
            data_range_days = (train_ts['ds'].max() - train_ts['ds'].min()).days
            
            # Calculate variability
            cv = train_ts['y'].std() / train_ts['y'].mean() if train_ts['y'].mean() > 0 else 0
            
            # Calculate trend strength
            from scipy import stats
            time_idx = np.arange(len(train_ts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, train_ts['y'].values)
            trend_strength = abs(r_value)
            
            # Store characteristics
            recommendations['data_characteristics'] = {
                'data_points': data_size,
                'time_span_days': data_range_days,
                'variability_cv': cv,
                'trend_strength': trend_strength
            }
            
            # Score each model based on data characteristics
            scores = {}
            
            # Prophet scoring
            prophet_score = 0
            prophet_reasons = []
            if PROPHET_AVAILABLE:
                if data_size >= 100:
                    prophet_score += 30
                    prophet_reasons.append("Sufficient data for Prophet's Bayesian approach")
                if trend_strength > 0.5:
                    prophet_score += 25
                    prophet_reasons.append("Strong trend detected, Prophet excels at trend modeling")
                if data_range_days > 365:
                    prophet_score += 20
                    prophet_reasons.append("Long time series benefits from Prophet's seasonality detection")
                if cv < 0.5:
                    prophet_score += 15
                    prophet_reasons.append("Moderate variability suits Prophet's robust fitting")
                prophet_score += 10
                scores['Prophet'] = {'score': prophet_score, 'reasons': prophet_reasons}
            
            # SARIMA scoring
            sarima_score = 0
            sarima_reasons = []
            if data_size >= 50 and data_size <= 500:
                sarima_score += 30
                sarima_reasons.append("Data size optimal for SARIMA")
            if 0.3 < cv < 0.7:
                sarima_score += 25
                sarima_reasons.append("Variability level suitable for SARIMA")
            if data_range_days >= 180:
                sarima_score += 20
                sarima_reasons.append("Sufficient history for seasonal pattern detection")
            sarima_score += 5
            scores['SARIMA'] = {'score': sarima_score, 'reasons': sarima_reasons}
            
            # XGBoost scoring
            xgboost_score = 0
            xgboost_reasons = []
            if XGBOOST_AVAILABLE:
                if data_size >= 50:
                    xgboost_score += 30
                    xgboost_reasons.append("Adequate data for gradient boosting")
                if cv > 0.5:
                    xgboost_score += 25
                    xgboost_reasons.append("High variability benefits from XGBoost's non-linear modeling")
                if trend_strength < 0.3:
                    xgboost_score += 20
                    xgboost_reasons.append("Complex patterns suit XGBoost's flexibility")
                xgboost_score += 5
                scores['XGBoost'] = {'score': xgboost_score, 'reasons': xgboost_reasons}
            
            recommendations['all_scores'] = scores
            
            # Select best model
            if scores:
                best_model = max(scores.items(), key=lambda x: x[1]['score'])
                recommendations['recommended_model'] = best_model[0]
                recommendations['reasons'] = best_model[1]['reasons']
                recommendations['confidence'] = min(100, best_model[1]['score'])
            
            return recommendations
            
        except Exception as e:
            return {
                'error': f"Model recommendation failed: {str(e)}",
                'recommended_model': 'Prophet',
                'reasons': ['Default recommendation due to error']
            }
    
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
    
    def generate_ensemble_forecast(self, forecasts: Dict[str, Any], method: str = 'weighted') -> Dict[str, Any]:
        """
        Generate ensemble forecast combining multiple models
        
        Args:
            forecasts: Dictionary of model forecasts
            method: 'weighted', 'average', or 'best'
        
        Returns:
            Ensemble forecast result
        """
        try:
            # Filter successful forecasts
            successful_forecasts = {k: v for k, v in forecasts.items() 
                                   if v.get('success', False) and 'forecast' in v}
            
            if len(successful_forecasts) < 2:
                return {'success': False, 'error': 'Need at least 2 models for ensemble'}
            
            # Extract forecast dataframes
            forecast_dfs = []
            model_names = []
            weights = []
            
            for model_name, result in successful_forecasts.items():
                forecast_dfs.append(result['forecast'])
                model_names.append(model_name)
                
                # Calculate weight based on performance metrics
                metrics = result.get('metrics', {})
                mape = metrics.get('mape', None)
                
                # Handle missing or invalid MAPE values
                if mape is None or mape <= 0 or np.isnan(mape) or np.isinf(mape):
                    # Default weight if metrics are unavailable
                    weight = 1.0
                else:
                    # Lower MAPE = higher weight (inverse relationship)
                    # Use exponential decay for weighting
                    weight = np.exp(-mape / 10)
                
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights)
            weights_sum = weights.sum()
            
            # Ensure we don't divide by zero
            if weights_sum > 0:
                weights = weights / weights_sum
            else:
                # Equal weights if all are zero
                weights = np.ones(len(weights)) / len(weights)
            
            # Create ensemble forecast
            if method == 'weighted':
                ensemble_forecast = self._weighted_ensemble(forecast_dfs, weights)
            elif method == 'average':
                ensemble_forecast = self._average_ensemble(forecast_dfs)
            elif method == 'best':
                # Select best performing model
                best_idx = np.argmax(weights)
                ensemble_forecast = forecast_dfs[best_idx].copy()
            else:
                ensemble_forecast = self._weighted_ensemble(forecast_dfs, weights)
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_ensemble_metrics(successful_forecasts, weights)
            
            return {
                'success': True,
                'forecast': ensemble_forecast,
                'method': method,
                'model_weights': {model_names[i]: weights[i] for i in range(len(model_names))},
                'metrics': ensemble_metrics,
                'component_models': model_names
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Ensemble forecast error: {str(e)}"}
    
    def _weighted_ensemble(self, forecast_dfs: List[pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
        """
        Create weighted ensemble forecast
        """
        # Merge all forecasts on date
        ensemble_df = forecast_dfs[0][['ds']].copy()
        
        # Calculate weighted predictions
        ensemble_df['yhat'] = 0
        ensemble_df['yhat_lower'] = 0
        ensemble_df['yhat_upper'] = 0
        
        for i, forecast_df in enumerate(forecast_dfs):
            merged = ensemble_df.merge(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                                      on='ds', how='left', suffixes=('', f'_{i}'))
            
            ensemble_df['yhat'] += merged['yhat'].fillna(0) * weights[i]
            ensemble_df['yhat_lower'] += merged['yhat_lower'].fillna(0) * weights[i]
            ensemble_df['yhat_upper'] += merged['yhat_upper'].fillna(0) * weights[i]
        
        return ensemble_df
    
    def _average_ensemble(self, forecast_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Create simple average ensemble forecast
        """
        n_models = len(forecast_dfs)
        weights = np.ones(n_models) / n_models
        return self._weighted_ensemble(forecast_dfs, weights)
    
    def _calculate_ensemble_metrics(self, forecasts: Dict[str, Any], weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate ensemble performance metrics
        """
        # Weight individual model metrics
        weighted_rmse = 0
        weighted_mae = 0
        weighted_mape = 0
        
        for i, (model_name, result) in enumerate(forecasts.items()):
            metrics = result.get('metrics', {})
            weighted_rmse += metrics.get('rmse', 0) * weights[i]
            weighted_mae += metrics.get('mae', 0) * weights[i]
            weighted_mape += metrics.get('mape', 0) * weights[i]
        
        return {
            'rmse': weighted_rmse,
            'mae': weighted_mae,
            'mape': weighted_mape,
            'ensemble_method': 'weighted_average'
        }
    
    def analyze_whatif_scenario(self, base_forecast: pd.DataFrame, scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze what-if scenarios by adjusting inventory parameters
        
        Args:
            base_forecast: Base forecast dataframe
            scenario_params: Dictionary with scenario parameters
                - reorder_point: Reorder point level
                - safety_stock: Safety stock level
                - lead_time_days: Lead time in days
                - order_quantity: Order quantity
                - initial_inventory: Starting inventory level
        
        Returns:
            Scenario analysis results with projected inventory levels
        """
        try:
            # Extract parameters
            reorder_point = scenario_params.get('reorder_point', 0)
            safety_stock = scenario_params.get('safety_stock', 0)
            lead_time_days = scenario_params.get('lead_time_days', 7)
            order_quantity = scenario_params.get('order_quantity', 10000)
            initial_inventory = scenario_params.get('initial_inventory')
            
            # If no initial inventory provided, use first forecast value
            if initial_inventory is None:
                initial_inventory = base_forecast['yhat'].iloc[0]
            
            # Simulate inventory levels with the given parameters
            simulated_inventory = []
            current_inventory = initial_inventory
            pending_orders = []  # Track pending orders with (arrival_date, quantity)
            
            for idx, row in base_forecast.iterrows():
                date = row['ds']
                forecast_demand = abs(row['yhat'])  # Use absolute value as demand
                
                # Process pending orders
                arrived_orders = [order for order in pending_orders if order[0] <= date]
                for order in arrived_orders:
                    current_inventory += order[1]
                    pending_orders.remove(order)
                
                # Check if reorder point is reached
                if current_inventory <= reorder_point and len(pending_orders) == 0:
                    # Place order
                    arrival_date = date + pd.Timedelta(days=lead_time_days)
                    pending_orders.append((arrival_date, order_quantity))
                
                # Simulate demand (use a portion of forecast as daily demand)
                daily_demand = forecast_demand * 0.01  # Assume 1% of forecast level as daily demand
                current_inventory = max(0, current_inventory - daily_demand)
                
                # Ensure safety stock is maintained
                if current_inventory < safety_stock and len(pending_orders) == 0:
                    # Emergency order
                    arrival_date = date + pd.Timedelta(days=lead_time_days)
                    pending_orders.append((arrival_date, order_quantity))
                
                simulated_inventory.append({
                    'ds': date,
                    'inventory_level': current_inventory,
                    'forecast_demand': forecast_demand,
                    'pending_orders': len(pending_orders),
                    'stockout': 1 if current_inventory < safety_stock else 0
                })
            
            # Create result dataframe
            result_df = pd.DataFrame(simulated_inventory)
            
            # Calculate scenario metrics
            stockout_days = result_df['stockout'].sum()
            avg_inventory = result_df['inventory_level'].mean()
            min_inventory = result_df['inventory_level'].min()
            max_inventory = result_df['inventory_level'].max()
            total_orders = len([order for _, orders in result_df.groupby('ds')['pending_orders'].sum().items() if orders > 0])
            
            return {
                'success': True,
                'scenario_data': result_df,
                'metrics': {
                    'stockout_days': int(stockout_days),
                    'avg_inventory': avg_inventory,
                    'min_inventory': min_inventory,
                    'max_inventory': max_inventory,
                    'total_orders_placed': total_orders,
                    'service_level': ((len(result_df) - stockout_days) / len(result_df)) * 100
                },
                'parameters': scenario_params
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Scenario analysis error: {str(e)}"}
    
    def compare_scenarios(self, base_forecast: pd.DataFrame, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple what-if scenarios
        
        Args:
            base_forecast: Base forecast dataframe
            scenarios: List of scenario parameter dictionaries
        
        Returns:
            Comparison results with metrics for each scenario
        """
        try:
            results = []
            
            for i, scenario_params in enumerate(scenarios):
                scenario_name = scenario_params.get('name', f'Scenario {i+1}')
                scenario_result = self.analyze_whatif_scenario(base_forecast, scenario_params)
                
                if scenario_result['success']:
                    results.append({
                        'name': scenario_name,
                        'result': scenario_result
                    })
            
            if not results:
                return {'success': False, 'error': 'No successful scenario analyses'}
            
            # Create comparison summary
            comparison_summary = []
            for scenario in results:
                comparison_summary.append({
                    'scenario': scenario['name'],
                    **scenario['result']['metrics']
                })
            
            return {
                'success': True,
                'scenarios': results,
                'comparison_summary': pd.DataFrame(comparison_summary)
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Scenario comparison error: {str(e)}"}
