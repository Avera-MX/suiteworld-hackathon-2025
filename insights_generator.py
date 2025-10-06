import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

class InsightsGenerator:
    def __init__(self):
        self.risk_thresholds = {
            'inventory_low': 0.1,  # Below 10% of normal levels
            'inventory_high': 2.0,  # Above 200% of normal levels
            'gik_extreme': 3.0,    # 3 standard deviations from mean
            'forecast_uncertainty': 0.3  # 30% uncertainty in forecasts
        }
    
    def generate_insights(self, datasets: Dict[str, pd.DataFrame], 
                         forecast_results: Optional[Dict[str, Any]] = None,
                         anomaly_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive business insights and recommendations
        """
        try:
            insights = {
                'success': True,
                'recommendations': [],
                'risk_alerts': [],
                'inventory_optimization': {},
                'category_insights': {},
                'performance_metrics': {}
            }
            
            # Generate strategic recommendations
            insights['recommendations'] = self._generate_recommendations(
                datasets, forecast_results, anomaly_results
            )
            
            # Identify risk alerts
            insights['risk_alerts'] = self._identify_risk_alerts(
                datasets, forecast_results, anomaly_results
            )
            
            # Generate inventory optimization recommendations
            insights['inventory_optimization'] = self._generate_inventory_optimization(
                datasets, forecast_results
            )
            
            # Analyze categories and products
            insights['category_insights'] = self._analyze_categories_and_products(datasets)
            
            # Calculate key performance metrics
            insights['performance_metrics'] = self._calculate_performance_metrics(datasets)
            
            return insights
            
        except Exception as e:
            return {'success': False, 'error': f"Insight generation error: {str(e)}"}
    
    def _generate_recommendations(self, datasets: Dict[str, pd.DataFrame], 
                                 forecast_results: Optional[Dict[str, Any]],
                                 anomaly_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate strategic business recommendations
        """
        recommendations = []
        
        # Recommendation 1: Address data quality issues
        if anomaly_results and anomaly_results.get('success'):
            outlier_count = 0
            if 'anomaly_summary' in anomaly_results:
                outlier_count = anomaly_results['anomaly_summary'].get('total_outliers_detected', 0)
            
            if outlier_count > 50:
                recommendations.append({
                    'title': 'Improve Data Quality and Governance',
                    'description': f'Detected {outlier_count} outliers across datasets. Implement data validation rules to prevent extreme values and ensure data consistency.',
                    'priority': 'High',
                    'impact': 'High',
                    'action_items': [
                        'Establish data validation rules for GIK values and quantities',
                        'Implement automated outlier detection in data pipeline',
                        'Review and standardize data collection processes',
                        'Train staff on proper data entry procedures'
                    ]
                })
        
        # Recommendation 2: Inventory scale adaptation
        scale_shift_detected = False
        if anomaly_results and 'scale_shifts' in anomaly_results:
            for ratio in anomaly_results['scale_shifts'].values():
                if abs(1 - ratio) > 0.3:  # >30% change
                    scale_shift_detected = True
                    break
        
        if scale_shift_detected:
            recommendations.append({
                'title': 'Adapt Inventory Management for Scale Changes',
                'description': 'Significant inventory scale changes detected between 2017-2018 and 2023. Adjust forecasting models and inventory policies to accommodate new scale.',
                'priority': 'High',
                'impact': 'High',
                'action_items': [
                    'Recalibrate inventory management parameters',
                    'Update safety stock calculations for new scale',
                    'Retrain forecasting models with recent data',
                    'Review warehouse capacity and operational procedures'
                ]
            })
        
        # Recommendation 3: Forecast model optimization
        if forecast_results and forecast_results.get('success'):
            prophet_available = 'Prophet' in forecast_results.get('forecasts', {})
            sarima_available = 'SARIMA' in forecast_results.get('forecasts', {})
            
            if prophet_available and sarima_available:
                recommendations.append({
                    'title': 'Implement Ensemble Forecasting',
                    'description': 'Both Prophet and SARIMA models are available. Combine predictions from multiple models to improve forecast accuracy and robustness.',
                    'priority': 'Medium',
                    'impact': 'Medium',
                    'action_items': [
                        'Develop ensemble forecasting methodology',
                        'Weight models based on historical performance',
                        'Implement automated model selection based on data characteristics',
                        'Monitor and compare model performance continuously'
                    ]
                })
        
        # Recommendation 4: Category diversification
        category_insights = self._analyze_categories_and_products(datasets)
        dominant_category = self._find_dominant_category(category_insights)
        
        if dominant_category and dominant_category['percentage'] > 60:
            recommendations.append({
                'title': 'Diversify Product Category Portfolio',
                'description': f'{dominant_category["category"]} represents {dominant_category["percentage"]:.1f}% of inventory. Consider diversifying to reduce concentration risk.',
                'priority': 'Medium',
                'impact': 'Medium',
                'action_items': [
                    'Analyze demand for underrepresented categories',
                    'Develop sourcing strategies for diverse product types',
                    'Monitor market trends and donor preferences',
                    'Balance inventory across multiple categories'
                ]
            })
        
        # Recommendation 5: Seasonal planning
        recommendations.append({
            'title': 'Implement Seasonal Inventory Planning',
            'description': 'Develop seasonal inventory strategies to optimize stock levels throughout the year and improve resource allocation.',
            'priority': 'Medium',
            'impact': 'High',
            'action_items': [
                'Analyze historical seasonal patterns',
                'Plan inventory buildup for high-demand periods',
                'Coordinate with donation campaigns and outreach programs',
                'Develop seasonal workforce planning'
            ]
        })
        
        return recommendations
    
    def _identify_risk_alerts(self, datasets: Dict[str, pd.DataFrame], 
                             forecast_results: Optional[Dict[str, Any]],
                             anomaly_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify operational and financial risks
        """
        risk_alerts = []
        
        # Risk 1: Extreme GIK values
        if anomaly_results and 'outliers' in anomaly_results:
            gik_outliers = anomaly_results['outliers'].get('gik_outliers', [])
            extreme_gik_count = len([o for o in gik_outliers if o.get('value', 0) > 1000000])
            
            if extreme_gik_count > 0:
                risk_alerts.append({
                    'level': 'high',
                    'message': f'{extreme_gik_count} donations with extremely high GIK values (>$1M) detected. Review for data entry errors or special handling requirements.'
                })
        
        # Risk 2: Data drift impact
        if anomaly_results and 'anomaly_summary' in anomaly_results:
            drift_severity = anomaly_results['anomaly_summary'].get('data_drift_severity', 'Low')
            
            if drift_severity == 'High':
                risk_alerts.append({
                    'level': 'high',
                    'message': 'High data drift detected between training and current periods. Forecasting accuracy may be compromised.'
                })
            elif drift_severity == 'Medium':
                risk_alerts.append({
                    'level': 'medium',
                    'message': 'Moderate data drift detected. Monitor forecast performance closely and consider model retraining.'
                })
        
        # Risk 3: Inventory volatility
        inventory_volatility = self._calculate_inventory_volatility(datasets)
        if inventory_volatility > 0.3:  # >30% coefficient of variation
            risk_alerts.append({
                'level': 'medium',
                'message': f'High inventory volatility detected (CV: {inventory_volatility:.2f}). Consider improving demand forecasting and supply chain stability.'
            })
        
        # Risk 4: Forecast uncertainty
        if forecast_results:
            high_uncertainty_detected = self._check_forecast_uncertainty(forecast_results)
            if high_uncertainty_detected:
                risk_alerts.append({
                    'level': 'medium',
                    'message': 'High uncertainty in forecasting models detected. Consider collecting additional data or improving model parameters.'
                })
        
        # Risk 5: Operational capacity
        capacity_risk = self._assess_capacity_risk(datasets)
        if capacity_risk:
            risk_alerts.append({
                'level': 'medium',
                'message': capacity_risk
            })
        
        return risk_alerts
    
    def _generate_inventory_optimization(self, datasets: Dict[str, pd.DataFrame], 
                                       forecast_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate inventory optimization recommendations
        """
        optimization = {}
        
        # Calculate reorder points
        reorder_points = self._calculate_reorder_points(datasets)
        if reorder_points:
            optimization['reorder_points'] = reorder_points
        
        # Calculate safety stock levels
        safety_stock = self._calculate_safety_stock(datasets, forecast_results)
        if safety_stock:
            optimization['safety_stock'] = safety_stock
        
        # Inventory turnover analysis
        turnover_analysis = self._analyze_inventory_turnover(datasets)
        if turnover_analysis:
            optimization['turnover_analysis'] = turnover_analysis
        
        return optimization
    
    def _calculate_reorder_points(self, datasets: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Calculate recommended reorder points by category
        """
        reorder_points = []
        
        # Analyze by category if outflow data is available
        if 'tune_outflows' in datasets and datasets['tune_outflows'] is not None:
            outflows = datasets['tune_outflows']
            
            if 'Category' in outflows.columns and 'Quantity' in outflows.columns:
                # Calculate average daily demand by category
                category_demand = outflows.groupby('Category')['Quantity'].agg(['mean', 'std']).reset_index()
                
                for _, row in category_demand.iterrows():
                    # Simple reorder point calculation: lead time * demand + safety stock
                    lead_time_days = 7  # Assume 7-day lead time
                    safety_factor = 1.5  # Safety factor
                    
                    reorder_point = (lead_time_days * row['mean']) + (safety_factor * row['std'])
                    
                    reorder_points.append({
                        'category': row['Category'],
                        'avg_daily_demand': round(row['mean'], 2),
                        'demand_std': round(row['std'], 2),
                        'recommended_reorder_point': round(reorder_point, 0)
                    })
        
        return reorder_points
    
    def _calculate_safety_stock(self, datasets: Dict[str, pd.DataFrame], 
                              forecast_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate recommended safety stock levels
        """
        safety_stock_recommendations = []
        
        # Use forecast uncertainty if available
        if forecast_results and 'forecasts' in forecast_results:
            for method, forecast_data in forecast_results['forecasts'].items():
                if forecast_data and 'metrics' in forecast_data:
                    mae = forecast_data['metrics'].get('mae', 0)
                    
                    # Service level approach
                    service_levels = [0.95, 0.98, 0.99]
                    z_scores = [1.65, 2.05, 2.33]  # Z-scores for service levels
                    
                    for service_level, z_score in zip(service_levels, z_scores):
                        safety_stock = z_score * mae
                        
                        safety_stock_recommendations.append({
                            'model': method,
                            'service_level': f'{service_level*100}%',
                            'safety_stock': round(safety_stock, 0),
                            'description': f'Safety stock for {service_level*100}% service level'
                        })
        
        return safety_stock_recommendations
    
    def _analyze_categories_and_products(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze categories and products for insights
        """
        category_insights = {}
        
        # Analyze inflow patterns by category
        for dataset_name in ['tune_inflows', 'train_inflows']:
            if dataset_name in datasets and datasets[dataset_name] is not None:
                df = datasets[dataset_name]
                
                if 'Category' in df.columns:
                    # Category distribution
                    category_dist = df['Category'].value_counts()
                    
                    # Category value analysis
                    if 'Total_GIK' in df.columns:
                        category_value = df.groupby('Category')['Total_GIK'].agg(['sum', 'mean', 'count']).reset_index()
                        
                        for _, row in category_value.iterrows():
                            category = row['Category']
                            
                            if category not in category_insights:
                                category_insights[category] = {}
                            
                            category_insights[category][f'{dataset_name}_summary'] = {
                                'total_value': row['sum'],
                                'avg_value': row['mean'],
                                'volume': row['count']
                            }
                    
                    # Brand analysis within categories
                    if 'Brand' in df.columns:
                        brand_category = df.groupby(['Category', 'Brand']).size().reset_index(name='count')
                        
                        for category in brand_category['Category'].unique():
                            if category not in category_insights:
                                category_insights[category] = {}
                            
                            category_brands = brand_category[brand_category['Category'] == category]
                            top_brands = category_brands.nlargest(3, 'count')['Brand'].tolist()
                            
                            category_insights[category][f'{dataset_name}_top_brands'] = top_brands
        
        # Generate summaries for each category
        for category, data in category_insights.items():
            summary_parts = []
            
            # Volume analysis
            tune_volume = data.get('tune_inflows_summary', {}).get('volume', 0)
            train_volume = data.get('train_inflows_summary', {}).get('volume', 0)
            
            if tune_volume > 0 and train_volume > 0:
                volume_growth = ((tune_volume - train_volume) / train_volume) * 100
                summary_parts.append(f"Volume changed by {volume_growth:.1f}% from 2017-2018 to 2023")
            
            # Value analysis
            tune_avg_value = data.get('tune_inflows_summary', {}).get('avg_value', 0)
            train_avg_value = data.get('train_inflows_summary', {}).get('avg_value', 0)
            
            if tune_avg_value > 0 and train_avg_value > 0:
                value_growth = ((tune_avg_value - train_avg_value) / train_avg_value) * 100
                summary_parts.append(f"Average value changed by {value_growth:.1f}%")
            
            # Brand analysis
            tune_brands = data.get('tune_inflows_top_brands', [])
            if tune_brands:
                summary_parts.append(f"Top brands: {', '.join(tune_brands[:3])}")
            
            data['summary'] = '. '.join(summary_parts) if summary_parts else 'Limited data available for analysis'
            data['metrics'] = {
                'volume': tune_volume,
                'avg_value': tune_avg_value,
                'growth_rate': volume_growth if 'volume_growth' in locals() else 0
            }
        
        return category_insights
    
    def _find_dominant_category(self, category_insights: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find the dominant category by volume
        """
        if not category_insights:
            return None
        
        total_volume = sum([data.get('metrics', {}).get('volume', 0) for data in category_insights.values()])
        
        if total_volume == 0:
            return None
        
        max_volume = 0
        dominant_category = None
        
        for category, data in category_insights.items():
            volume = data.get('metrics', {}).get('volume', 0)
            if volume > max_volume:
                max_volume = volume
                dominant_category = category
        
        if dominant_category:
            percentage = (max_volume / total_volume) * 100
            return {'category': dominant_category, 'percentage': percentage}
        
        return None
    
    def _calculate_performance_metrics(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate key performance metrics
        """
        metrics = {}
        
        # Inventory efficiency metrics
        if 'tune_inventory' in datasets and datasets['tune_inventory'] is not None:
            inventory_df = datasets['tune_inventory']
            
            if 'Inventory_Level' in inventory_df.columns:
                avg_inventory = inventory_df['Inventory_Level'].mean()
                max_inventory = inventory_df['Inventory_Level'].max()
                min_inventory = inventory_df['Inventory_Level'].min()
                
                metrics['inventory_efficiency'] = {
                    'average_level': avg_inventory,
                    'peak_level': max_inventory,
                    'minimum_level': min_inventory,
                    'coefficient_of_variation': inventory_df['Inventory_Level'].std() / avg_inventory
                }
        
        # Donation volume trends
        if 'tune_inflows' in datasets and datasets['tune_inflows'] is not None:
            inflows_df = datasets['tune_inflows']
            
            if 'Quantity' in inflows_df.columns:
                total_donations = inflows_df['Quantity'].sum()
                avg_donation_size = inflows_df['Quantity'].mean()
                
                metrics['donation_metrics'] = {
                    'total_volume': total_donations,
                    'average_donation_size': avg_donation_size,
                    'number_of_donations': len(inflows_df)
                }
        
        return metrics
    
    def _calculate_inventory_volatility(self, datasets: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate inventory volatility (coefficient of variation)
        """
        if 'tune_inventory' in datasets and datasets['tune_inventory'] is not None:
            inventory_df = datasets['tune_inventory']
            
            if 'Inventory_Level' in inventory_df.columns:
                mean_inventory = inventory_df['Inventory_Level'].mean()
                std_inventory = inventory_df['Inventory_Level'].std()
                
                if mean_inventory > 0:
                    return std_inventory / mean_inventory
        
        return 0.0
    
    def _check_forecast_uncertainty(self, forecast_results: Dict[str, Any]) -> bool:
        """
        Check if forecast uncertainty is high
        """
        if 'forecasts' in forecast_results:
            for method, forecast_data in forecast_results['forecasts'].items():
                if forecast_data and 'metrics' in forecast_data:
                    mape = forecast_data['metrics'].get('mape', 0)
                    if mape > 30:  # >30% MAPE indicates high uncertainty
                        return True
        
        return False
    
    def _assess_capacity_risk(self, datasets: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        Assess operational capacity risks
        """
        # Check for rapid inventory growth
        if 'tune_inventory' in datasets and datasets['tune_inventory'] is not None:
            inventory_df = datasets['tune_inventory']
            
            if 'Inventory_Level' in inventory_df.columns and len(inventory_df) > 30:
                recent_avg = inventory_df.tail(30)['Inventory_Level'].mean()
                early_avg = inventory_df.head(30)['Inventory_Level'].mean()
                
                if early_avg > 0:
                    growth_rate = (recent_avg - early_avg) / early_avg
                    
                    if growth_rate > 0.5:  # >50% growth
                        return f'Rapid inventory growth detected ({growth_rate*100:.1f}%). Assess warehouse capacity and processing capabilities.'
        
        return None
    
    def _analyze_inventory_turnover(self, datasets: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Analyze inventory turnover patterns
        """
        if 'tune_inventory' in datasets and 'tune_outflows' in datasets:
            inventory_df = datasets['tune_inventory']
            outflows_df = datasets['tune_outflows']
            
            if (inventory_df is not None and 'Inventory_Level' in inventory_df.columns and
                outflows_df is not None and 'Quantity' in outflows_df.columns):
                
                avg_inventory = inventory_df['Inventory_Level'].mean()
                total_outflows = outflows_df['Quantity'].sum()
                
                if avg_inventory > 0:
                    # Annualized turnover ratio
                    days_in_period = len(inventory_df)
                    annual_outflows = total_outflows * (365 / days_in_period)
                    turnover_ratio = annual_outflows / avg_inventory
                    
                    return {
                        'turnover_ratio': turnover_ratio,
                        'days_of_inventory': 365 / turnover_ratio if turnover_ratio > 0 else 0,
                        'interpretation': self._interpret_turnover_ratio(turnover_ratio)
                    }
        
        return None
    
    def _interpret_turnover_ratio(self, ratio: float) -> str:
        """
        Interpret inventory turnover ratio
        """
        if ratio > 12:
            return 'Very high turnover - inventory moves quickly'
        elif ratio > 6:
            return 'High turnover - efficient inventory management'
        elif ratio > 3:
            return 'Moderate turnover - reasonable efficiency'
        elif ratio > 1:
            return 'Low turnover - inventory may be excessive'
        else:
            return 'Very low turnover - significant inventory accumulation'
