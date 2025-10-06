import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class ForecastingModel:
    """
    Adaptive ML forecasting using Prophet/SARIMA with automatic trend detection
    and model retraining capabilities for handling temporal gaps and scale shifts.
    """
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.forecasts = {}
        self.training_data = None
        self.tuning_data = None
        
    def train_adaptive_models(self, train_data, tune_data, model_type="Prophet", 
                            include_external=True, auto_retrain=True):
        """
        Train adaptive forecasting models that can handle scale shifts and temporal gaps
        """
        results = {}
        
        # Store data for reference
        self.training_data = train_data
        self.tuning_data = tune_data
        
        # Prepare inventory time series
        train_ts = self._prepare_time_series(train_data['inventory'])
        tune_ts = self._prepare_time_series(tune_data['inventory'])
        
        # Detect scale differences
        scale_adjustment = self._detect_scale_adjustment(train_ts, tune_ts)
        results['scale_adjustment'] = scale_adjustment
        
        # Train baseline model on training data
        baseline_model = self._train_baseline_model(train_ts, model_type, include_external)
        
        # Evaluate baseline model on tune data
        baseline_performance = self._evaluate_model_on_tune_data(baseline_model, tune_ts)
        results['before_rmse'] = baseline_performance['rmse']
        results['before_mae'] = baseline_performance['mae'] 
        results['before_mape'] = baseline_performance['mape']
        
        # Adapt model for scale shifts
        adapted_model = self._adapt_model_for_scale_shift(
            baseline_model, train_ts, tune_ts, scale_adjustment, model_type
        )
        
        # Evaluate adapted model
        adapted_performance = self._evaluate_model_on_tune_data(adapted_model, tune_ts)
        results['rmse'] = adapted_performance['rmse']
        results['mae'] = adapted_performance['mae']
        results['mape'] = adapted_performance['mape']
        
        # Store models
        self.models['baseline'] = baseline_model
        self.models['adapted'] = adapted_model
        self.model_performance = results
        
        # Generate stability metrics
        stability_metrics = self._calculate_stability_metrics(adapted_model, train_ts, tune_ts)
        results['stability_metrics'] = stability_metrics
        
        # Feature importance (if applicable)
        if hasattr(adapted_model, 'feature_importances_') or hasattr(adapted_model, 'params'):
            results['feature_importance'] = self._extract_feature_importance(adapted_model)
        
        return results
    
    def generate_forecasts(self, horizon=90):
        """
        Generate forecasts using the best available model
        """
        if 'adapted' not in self.models:
            raise ValueError("Models must be trained first")
        
        model = self.models['adapted']
        
        # Get the latest date from tune data
        latest_date = self.tuning_data['inventory']['Date'].max()
        
        # Generate future dates
        future_dates = pd.date_range(
            start=latest_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Generate forecasts based on model type
        if isinstance(model, dict) and 'model_type' in model:
            if model['model_type'] == 'prophet' and PROPHET_AVAILABLE:
                forecasts = self._generate_prophet_forecasts(model, future_dates)
            elif model['model_type'] == 'sarima' and STATSMODELS_AVAILABLE:
                forecasts = self._generate_sarima_forecasts(model, future_dates)
            else:
                forecasts = self._generate_hybrid_forecasts(model, future_dates)
        else:
            forecasts = self._generate_sklearn_forecasts(model, future_dates)
        
        # Add confidence intervals
        forecasts = self._add_confidence_intervals(forecasts, horizon)
        
        self.forecasts = forecasts
        return forecasts
    
    def _prepare_time_series(self, inventory_df):
        """
        Prepare inventory data as a proper time series
        """
        ts_df = inventory_df.copy()
        ts_df = ts_df.sort_values('Date')
        
        # Create complete date range
        start_date = ts_df['Date'].min()
        end_date = ts_df['Date'].max()
        
        complete_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Reindex to complete date range
        ts_df = ts_df.set_index('Date').reindex(complete_dates)
        ts_df['Inventory_Level'] = ts_df['Inventory_Level'].fillna(method='ffill')
        ts_df['Inventory_Level'] = ts_df['Inventory_Level'].fillna(method='bfill')
        ts_df = ts_df.reset_index()
        ts_df.rename(columns={'index': 'Date'}, inplace=True)
        
        # Add time features
        ts_df['DayOfWeek'] = ts_df['Date'].dt.dayofweek
        ts_df['Month'] = ts_df['Date'].dt.month
        ts_df['Quarter'] = ts_df['Date'].dt.quarter
        ts_df['DayOfYear'] = ts_df['Date'].dt.dayofyear
        
        # Add lag features
        for lag in [1, 7, 30]:
            ts_df[f'Inventory_Lag_{lag}'] = ts_df['Inventory_Level'].shift(lag)
        
        # Add moving averages
        for window in [7, 30]:
            ts_df[f'MA_{window}'] = ts_df['Inventory_Level'].rolling(window=window).mean()
        
        return ts_df
    
    def _detect_scale_adjustment(self, train_ts, tune_ts):
        """
        Detect scale differences between training and tuning periods
        """
        train_mean = train_ts['Inventory_Level'].mean()
        tune_mean = tune_ts['Inventory_Level'].mean()
        
        train_std = train_ts['Inventory_Level'].std()
        tune_std = tune_ts['Inventory_Level'].std()
        
        scale_factor = tune_mean / train_mean
        variance_factor = tune_std / train_std
        
        return {
            'scale_factor': scale_factor,
            'variance_factor': variance_factor,
            'train_mean': train_mean,
            'tune_mean': tune_mean,
            'train_std': train_std,
            'tune_std': tune_std,
            'requires_adjustment': abs(scale_factor - 1) > 0.1
        }
    
    def _train_baseline_model(self, train_ts, model_type, include_external):
        """
        Train baseline model on training data
        """
        if model_type == "Prophet" and PROPHET_AVAILABLE:
            return self._train_prophet_model(train_ts, include_external)
        elif model_type == "SARIMA" and STATSMODELS_AVAILABLE:
            return self._train_sarima_model(train_ts)
        elif model_type == "Hybrid Ensemble":
            return self._train_hybrid_model(train_ts, include_external)
        else:
            # Fallback to sklearn models
            return self._train_sklearn_model(train_ts, include_external)
    
    def _train_prophet_model(self, train_ts, include_external):
        """
        Train Prophet model
        """
        if not PROPHET_AVAILABLE:
            return self._train_sklearn_model(train_ts, include_external)
        
        # Prepare data for Prophet
        prophet_df = train_ts[['Date', 'Inventory_Level']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize Prophet with appropriate parameters
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # Add regressors if including external factors
        if include_external:
            for col in ['DayOfWeek', 'Month', 'Quarter']:
                if col in train_ts.columns:
                    model.add_regressor(col)
                    prophet_df[col] = train_ts[col]
        
        # Fit model
        model.fit(prophet_df)
        
        return {
            'model': model,
            'model_type': 'prophet',
            'training_data': prophet_df,
            'include_external': include_external
        }
    
    def _train_sarima_model(self, train_ts):
        """
        Train SARIMA model
        """
        if not STATSMODELS_AVAILABLE:
            return self._train_sklearn_model(train_ts, False)
        
        # Prepare data
        y = train_ts['Inventory_Level'].dropna()
        
        # Auto-determine SARIMA parameters (simplified approach)
        try:
            # Try common SARIMA parameters
            model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fitted_model = model.fit(disp=False)
            
            return {
                'model': fitted_model,
                'model_type': 'sarima',
                'training_data': y
            }
        except:
            # Fallback to simpler ARIMA
            try:
                model = ARIMA(y, order=(1, 1, 1))
                fitted_model = model.fit()
                
                return {
                    'model': fitted_model,
                    'model_type': 'arima',
                    'training_data': y
                }
            except:
                return self._train_sklearn_model(train_ts, False)
    
    def _train_sklearn_model(self, train_ts, include_external):
        """
        Train sklearn-based model as fallback
        """
        # Prepare features
        feature_cols = []
        
        # Time features
        time_features = ['DayOfWeek', 'Month', 'Quarter', 'DayOfYear']
        feature_cols.extend([col for col in time_features if col in train_ts.columns])
        
        # Lag features
        lag_features = [col for col in train_ts.columns if 'Lag' in col or 'MA' in col]
        feature_cols.extend(lag_features)
        
        if not feature_cols:
            # Create basic features if none exist
            train_ts['trend'] = np.arange(len(train_ts))
            feature_cols = ['trend']
        
        # Prepare training data
        X = train_ts[feature_cols].fillna(0)
        y = train_ts['Inventory_Level']
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X, y)
        
        return {
            'model': model,
            'model_type': 'sklearn',
            'feature_columns': feature_cols,
            'training_data': train_ts
        }
    
    def _train_hybrid_model(self, train_ts, include_external):
        """
        Train hybrid ensemble model
        """
        models = []
        
        # Train multiple models
        if PROPHET_AVAILABLE:
            prophet_model = self._train_prophet_model(train_ts, include_external)
            models.append(('prophet', prophet_model, 0.4))
        
        if STATSMODELS_AVAILABLE:
            sarima_model = self._train_sarima_model(train_ts)
            models.append(('sarima', sarima_model, 0.3))
        
        sklearn_model = self._train_sklearn_model(train_ts, include_external)
        models.append(('sklearn', sklearn_model, 0.3))
        
        return {
            'model_type': 'hybrid',
            'models': models,
            'training_data': train_ts
        }
    
    def _adapt_model_for_scale_shift(self, baseline_model, train_ts, tune_ts, scale_adjustment, model_type):
        """
        Adapt model to handle scale shifts between training and tuning periods
        """
        if not scale_adjustment['requires_adjustment']:
            return baseline_model
        
        # For significant scale shifts, retrain with combined data
        if abs(scale_adjustment['scale_factor'] - 1) > 0.5:
            # Combine training and tuning data with scaling
            combined_ts = self._combine_and_scale_data(train_ts, tune_ts, scale_adjustment)
            
            # Retrain model on combined data
            adapted_model = self._train_baseline_model(combined_ts, model_type, True)
            
        else:
            # For moderate shifts, apply post-processing adjustment
            adapted_model = baseline_model.copy() if isinstance(baseline_model, dict) else baseline_model
            adapted_model['scale_adjustment'] = scale_adjustment
        
        return adapted_model
    
    def _combine_and_scale_data(self, train_ts, tune_ts, scale_adjustment):
        """
        Combine training and tuning data with appropriate scaling
        """
        # Scale training data to match tuning scale
        train_scaled = train_ts.copy()
        train_scaled['Inventory_Level'] = (
            train_scaled['Inventory_Level'] * scale_adjustment['scale_factor']
        )
        
        # Combine datasets
        combined_ts = pd.concat([train_scaled, tune_ts], ignore_index=True)
        combined_ts = combined_ts.sort_values('Date').reset_index(drop=True)
        
        return combined_ts
    
    def _evaluate_model_on_tune_data(self, model, tune_ts):
        """
        Evaluate model performance on tuning data
        """
        try:
            if isinstance(model, dict):
                if model['model_type'] == 'prophet' and PROPHET_AVAILABLE:
                    predictions = self._predict_prophet(model, tune_ts)
                elif model['model_type'] in ['sarima', 'arima'] and STATSMODELS_AVAILABLE:
                    predictions = self._predict_sarima(model, tune_ts)
                elif model['model_type'] == 'sklearn':
                    predictions = self._predict_sklearn(model, tune_ts)
                elif model['model_type'] == 'hybrid':
                    predictions = self._predict_hybrid(model, tune_ts)
                else:
                    # Fallback prediction
                    predictions = np.full(len(tune_ts), tune_ts['Inventory_Level'].mean())
            else:
                predictions = model.predict(tune_ts[['Inventory_Level']].fillna(0))
            
            actual = tune_ts['Inventory_Level'].values
            
            # Handle any remaining NaN values
            valid_mask = ~(np.isnan(predictions) | np.isnan(actual))
            predictions = predictions[valid_mask]
            actual = actual[valid_mask]
            
            if len(predictions) == 0 or len(actual) == 0:
                return {'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf')}
            
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            
            # Calculate MAPE with safety check
            try:
                mape = mean_absolute_percentage_error(actual, predictions) * 100
            except:
                # Manual MAPE calculation
                mape_values = np.abs((actual - predictions) / (actual + 1e-10)) * 100
                mape = np.mean(mape_values)
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'predictions': predictions,
                'actual': actual
            }
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf')}
    
    def _predict_prophet(self, model_dict, test_ts):
        """
        Make predictions using Prophet model
        """
        model = model_dict['model']
        
        # Prepare future dataframe
        future_df = test_ts[['Date']].copy()
        future_df.columns = ['ds']
        
        # Add regressors if they were used in training
        if model_dict.get('include_external', False):
            for col in ['DayOfWeek', 'Month', 'Quarter']:
                if col in test_ts.columns:
                    future_df[col] = test_ts[col]
        
        # Make predictions
        forecast = model.predict(future_df)
        return forecast['yhat'].values
    
    def _predict_sarima(self, model_dict, test_ts):
        """
        Make predictions using SARIMA model
        """
        model = model_dict['model']
        n_periods = len(test_ts)
        
        # Generate forecasts
        forecast = model.forecast(steps=n_periods)
        return forecast
    
    def _predict_sklearn(self, model_dict, test_ts):
        """
        Make predictions using sklearn model
        """
        model = model_dict['model']
        feature_cols = model_dict['feature_columns']
        
        X_test = test_ts[feature_cols].fillna(0)
        predictions = model.predict(X_test)
        
        return predictions
    
    def _predict_hybrid(self, model_dict, test_ts):
        """
        Make predictions using hybrid ensemble
        """
        predictions = []
        weights = []
        
        for model_name, model, weight in model_dict['models']:
            try:
                if model_name == 'prophet':
                    pred = self._predict_prophet(model, test_ts)
                elif model_name == 'sarima':
                    pred = self._predict_sarima(model, test_ts)
                elif model_name == 'sklearn':
                    pred = self._predict_sklearn(model, test_ts)
                else:
                    continue
                
                predictions.append(pred)
                weights.append(weight)
            except:
                continue
        
        if not predictions:
            return np.full(len(test_ts), test_ts['Inventory_Level'].mean())
        
        # Weighted average of predictions
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        final_prediction = np.average(predictions, axis=0, weights=weights)
        return final_prediction
    
    def _calculate_stability_metrics(self, model, train_ts, tune_ts):
        """
        Calculate model stability metrics across different time periods
        """
        # Split tune data into multiple periods
        tune_periods = self._split_into_periods(tune_ts, n_periods=3)
        
        stability_metrics = {
            'period_performance': [],
            'performance_variance': 0,
            'trend_consistency': 0
        }
        
        for i, period_data in enumerate(tune_periods):
            if len(period_data) > 0:
                performance = self._evaluate_model_on_tune_data(model, period_data)
                stability_metrics['period_performance'].append({
                    'period': i+1,
                    'rmse': performance['rmse'],
                    'mae': performance['mae'],
                    'mape': performance['mape']
                })
        
        # Calculate variance in performance
        if len(stability_metrics['period_performance']) > 1:
            rmse_values = [p['rmse'] for p in stability_metrics['period_performance']]
            stability_metrics['performance_variance'] = np.var(rmse_values)
        
        return stability_metrics
    
    def _split_into_periods(self, ts_data, n_periods=3):
        """
        Split time series data into multiple periods
        """
        period_size = len(ts_data) // n_periods
        periods = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = start_idx + period_size if i < n_periods - 1 else len(ts_data)
            period_data = ts_data.iloc[start_idx:end_idx].copy()
            periods.append(period_data)
        
        return periods
    
    def _extract_feature_importance(self, model):
        """
        Extract feature importance from trained models
        """
        if isinstance(model, dict):
            if model['model_type'] == 'sklearn':
                sklearn_model = model['model']
                if hasattr(sklearn_model, 'feature_importances_'):
                    feature_names = model['feature_columns']
                    importances = sklearn_model.feature_importances_
                    
                    return {
                        'features': feature_names,
                        'importances': importances.tolist(),
                        'feature_importance_pairs': list(zip(feature_names, importances))
                    }
        
        return {'features': [], 'importances': [], 'feature_importance_pairs': []}
    
    def _generate_prophet_forecasts(self, model_dict, future_dates):
        """
        Generate forecasts using Prophet model
        """
        model = model_dict['model']
        
        # Create future dataframe
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Add external regressors if needed
        if model_dict.get('include_external', False):
            future_df['DayOfWeek'] = future_df['ds'].dt.dayofweek
            future_df['Month'] = future_df['ds'].dt.month
            future_df['Quarter'] = future_df['ds'].dt.quarter
        
        # Generate forecast
        forecast = model.predict(future_df)
        
        return {
            'dates': future_dates,
            'forecast': forecast['yhat'].values,
            'lower_bound': forecast['yhat_lower'].values,
            'upper_bound': forecast['yhat_upper'].values,
            'model_type': 'prophet'
        }
    
    def _generate_sarima_forecasts(self, model_dict, future_dates):
        """
        Generate forecasts using SARIMA model
        """
        model = model_dict['model']
        n_periods = len(future_dates)
        
        # Generate forecast
        forecast_result = model.forecast(steps=n_periods, alpha=0.05)
        
        if hasattr(forecast_result, 'predicted_mean'):
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            lower_bound = conf_int.iloc[:, 0]
            upper_bound = conf_int.iloc[:, 1]
        else:
            forecast = forecast_result
            lower_bound = forecast * 0.9  # Simple confidence interval
            upper_bound = forecast * 1.1
        
        return {
            'dates': future_dates,
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'model_type': 'sarima'
        }
    
    def _generate_sklearn_forecasts(self, model_dict, future_dates):
        """
        Generate forecasts using sklearn model
        """
        # This is more complex as we need to create features for future dates
        # For now, use a simple trend extrapolation
        last_value = self.tuning_data['inventory']['Inventory_Level'].iloc[-1]
        trend = np.linspace(0, len(future_dates), len(future_dates))
        forecast = last_value + trend * 0.01  # Simple trend
        
        return {
            'dates': future_dates,
            'forecast': forecast,
            'lower_bound': forecast * 0.95,
            'upper_bound': forecast * 1.05,
            'model_type': 'sklearn'
        }
    
    def _generate_hybrid_forecasts(self, model_dict, future_dates):
        """
        Generate forecasts using hybrid ensemble
        """
        forecasts = []
        weights = []
        
        for model_name, model, weight in model_dict['models']:
            try:
                if model_name == 'prophet':
                    forecast = self._generate_prophet_forecasts(model, future_dates)
                elif model_name == 'sarima':
                    forecast = self._generate_sarima_forecasts(model, future_dates)
                elif model_name == 'sklearn':
                    forecast = self._generate_sklearn_forecasts(model, future_dates)
                else:
                    continue
                
                forecasts.append(forecast['forecast'])
                weights.append(weight)
            except:
                continue
        
        if not forecasts:
            return self._generate_sklearn_forecasts(model_dict, future_dates)
        
        # Weighted average
        forecasts = np.array(forecasts)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        final_forecast = np.average(forecasts, axis=0, weights=weights)
        
        return {
            'dates': future_dates,
            'forecast': final_forecast,
            'lower_bound': final_forecast * 0.9,
            'upper_bound': final_forecast * 1.1,
            'model_type': 'hybrid'
        }
    
    def _add_confidence_intervals(self, forecasts, horizon):
        """
        Add confidence intervals to forecasts if not already present
        """
        if 'lower_bound' not in forecasts or 'upper_bound' not in forecasts:
            forecast_values = forecasts['forecast']
            
            # Calculate confidence intervals based on historical volatility
            if self.tuning_data is not None:
                historical_volatility = self.tuning_data['inventory']['Inventory_Level'].std()
                
                # Widen confidence intervals for longer horizons
                volatility_factor = 1 + (np.arange(horizon) / horizon) * 0.5
                confidence_width = historical_volatility * volatility_factor
                
                forecasts['lower_bound'] = forecast_values - 1.96 * confidence_width
                forecasts['upper_bound'] = forecast_values + 1.96 * confidence_width
        
        return forecasts
