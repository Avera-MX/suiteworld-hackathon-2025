import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InsightsEngine:
    def __init__(self):
        self.safety_stock_multiplier = 2.0  # Standard safety stock multiplier
        self.reorder_point_multiplier = 1.5  # Multiplier for reorder points
        
    def generate_insights(self, data_dict, forecast_results=None):
        """Generate comprehensive business insights and recommendations"""
        try:
            insights = {
                'reorder_recommendations': [],
                'safety_stock_levels': [],
                'risk_alerts': [],
                'growing_categories': [],
                'declining_categories': [],
                'warehouse_efficiency': [],
                'demand_forecasts': {},
                'optimization_opportunities': []
            }
            
            # Generate reorder point recommendations
            insights['reorder_recommendations'] = self._calculate_reorder_points(data_dict)
            
            # Calculate safety stock levels
            insights['safety_stock_levels'] = self._calculate_safety_stock(data_dict)
            
            # Generate risk alerts
            insights['risk_alerts'] = self._generate_risk_alerts(data_dict, forecast_results)
            
            # Analyze category trends
            category_trends = self._analyze_category_trends(data_dict)
            insights['growing_categories'] = category_trends['growing']
            insights['declining_categories'] = category_trends['declining']
            
            # Warehouse efficiency analysis
            insights['warehouse_efficiency'] = self._analyze_warehouse_efficiency(data_dict)
            
            # Demand forecasts by category and warehouse
            if forecast_results:
                insights['demand_forecasts'] = self._generate_demand_forecasts(data_dict, forecast_results)
            
            # Optimization opportunities
            insights['optimization_opportunities'] = self._identify_optimization_opportunities(data_dict)
            
            return insights
            
        except Exception as e:
            raise Exception(f"Insights generation failed: {str(e)}")
    
    def _calculate_reorder_points(self, data_dict):
        """Calculate reorder point recommendations by category and warehouse"""
        reorder_recommendations = []
        
        inflows_data = data_dict['inflows']
        outflows_data = data_dict['outflows']
        inventory_data = data_dict['inventory']
        
        if len(outflows_data) == 0:
            return []
        
        # Group by category and warehouse
        category_warehouse_groups = outflows_data.groupby(['Category', 'Warehouse'])
        
        for (category, warehouse), group in category_warehouse_groups:
            # Calculate average daily demand
            total_days = (group['Date'].max() - group['Date'].min()).days
            if total_days <= 0:
                continue
                
            total_demand = group['Quantity'].sum()
            avg_daily_demand = total_demand / total_days
            
            # Calculate demand variability
            daily_demand = group.groupby('Date')['Quantity'].sum()
            demand_std = daily_demand.std() if len(daily_demand) > 1 else avg_daily_demand * 0.1
            
            # Calculate lead time (assumed average)
            lead_time_days = 7  # Default assumption
            
            # Calculate reorder point
            reorder_point = (avg_daily_demand * lead_time_days) + (demand_std * self.reorder_point_multiplier)
            
            # Get current inventory level (approximate)
            current_inventory = inventory_data['Inventory_Level'].iloc[-1] if len(inventory_data) > 0 else 0
            
            reorder_recommendations.append({
                'Category': category,
                'Warehouse': warehouse,
                'Current_Inventory_Estimate': current_inventory,
                'Recommended_Reorder_Point': round(reorder_point),
                'Average_Daily_Demand': round(avg_daily_demand, 2),
                'Lead_Time_Days': lead_time_days,
                'Demand_Variability': round(demand_std, 2),
                'Status': 'REORDER_NEEDED' if current_inventory < reorder_point else 'OK'
            })
        
        return sorted(reorder_recommendations, key=lambda x: x['Recommended_Reorder_Point'], reverse=True)
    
    def _calculate_safety_stock(self, data_dict):
        """Calculate safety stock recommendations"""
        safety_stock_levels = []
        
        outflows_data = data_dict['outflows']
        
        if len(outflows_data) == 0:
            return []
        
        # Group by category and warehouse
        category_warehouse_groups = outflows_data.groupby(['Category', 'Warehouse'])
        
        for (category, warehouse), group in category_warehouse_groups:
            # Calculate demand statistics
            daily_demand = group.groupby('Date')['Quantity'].sum()
            
            if len(daily_demand) <= 1:
                continue
            
            avg_demand = daily_demand.mean()
            demand_std = daily_demand.std()
            
            # Calculate safety stock using standard formula
            service_level_z = 1.96  # 97.5% service level
            lead_time_days = 7  # Assumed lead time
            
            safety_stock = service_level_z * demand_std * np.sqrt(lead_time_days)
            
            safety_stock_levels.append({
                'Category': category,
                'Warehouse': warehouse,
                'Recommended_Safety_Stock': round(safety_stock),
                'Average_Daily_Demand': round(avg_demand, 2),
                'Demand_Standard_Deviation': round(demand_std, 2),
                'Service_Level': '97.5%',
                'Lead_Time_Days': lead_time_days
            })
        
        return sorted(safety_stock_levels, key=lambda x: x['Recommended_Safety_Stock'], reverse=True)
    
    def _generate_risk_alerts(self, data_dict, forecast_results=None):
        """Generate risk alerts based on inventory patterns"""
        risk_alerts = []
        
        inventory_data = data_dict['inventory']
        inflows_data = data_dict['inflows']
        outflows_data = data_dict['outflows']
        
        # Alert 1: Low inventory levels
        if len(inventory_data) > 0:
            current_level = inventory_data['Inventory_Level'].iloc[-1]
            avg_level = inventory_data['Inventory_Level'].mean()
            
            if current_level < avg_level * 0.5:
                risk_alerts.append({
                    'severity': 'HIGH',
                    'type': 'LOW_INVENTORY',
                    'message': f'Current inventory level ({current_level:,.0f}) is significantly below average ({avg_level:,.0f})',
                    'recommendation': 'Consider increasing procurement to avoid stockouts'
                })
        
        # Alert 2: Declining inflow trends
        if len(inflows_data) > 0:
            recent_inflows = inflows_data[inflows_data['Date'] >= inflows_data['Date'].max() - timedelta(days=30)]
            older_inflows = inflows_data[inflows_data['Date'] < inflows_data['Date'].max() - timedelta(days=30)]
            
            if len(recent_inflows) > 0 and len(older_inflows) > 0:
                recent_avg = recent_inflows['Quantity'].mean()
                older_avg = older_inflows['Quantity'].mean()
                
                if recent_avg < older_avg * 0.7:
                    risk_alerts.append({
                        'severity': 'MEDIUM',
                        'type': 'DECLINING_INFLOWS',
                        'message': f'Recent inflows ({recent_avg:.0f}/day) are 30% below historical average ({older_avg:.0f}/day)',
                        'recommendation': 'Review vendor relationships and procurement strategies'
                    })
        
        # Alert 3: Increasing outflow demands
        if len(outflows_data) > 0:
            recent_outflows = outflows_data[outflows_data['Date'] >= outflows_data['Date'].max() - timedelta(days=30)]
            older_outflows = outflows_data[outflows_data['Date'] < outflows_data['Date'].max() - timedelta(days=30)]
            
            if len(recent_outflows) > 0 and len(older_outflows) > 0:
                recent_avg = recent_outflows['Quantity'].mean()
                older_avg = older_outflows['Quantity'].mean()
                
                if recent_avg > older_avg * 1.3:
                    risk_alerts.append({
                        'severity': 'MEDIUM',
                        'type': 'INCREASING_DEMAND',
                        'message': f'Recent outflows ({recent_avg:.0f}/day) are 30% above historical average ({older_avg:.0f}/day)',
                        'recommendation': 'Prepare for increased demand by adjusting procurement levels'
                    })
        
        # Alert 4: Forecast-based alerts
        if forecast_results and 'forecasts' in forecast_results:
            forecasts_df = forecast_results['forecasts']
            if len(forecasts_df) > 0:
                min_forecast = forecasts_df['Forecast'].min()
                current_level = inventory_data['Inventory_Level'].iloc[-1] if len(inventory_data) > 0 else 0
                
                if min_forecast < current_level * 0.3:
                    risk_alerts.append({
                        'severity': 'HIGH',
                        'type': 'FORECAST_LOW_INVENTORY',
                        'message': f'Forecast predicts inventory levels will drop to {min_forecast:,.0f}',
                        'recommendation': 'Immediate action needed to prevent stockout'
                    })
        
        return risk_alerts
    
    def _analyze_category_trends(self, data_dict):
        """Analyze trends by product category"""
        inflows_data = data_dict['inflows']
        outflows_data = data_dict['outflows']
        
        growing_categories = []
        declining_categories = []
        
        if len(inflows_data) == 0 and len(outflows_data) == 0:
            return {'growing': [], 'declining': []}
        
        # Analyze inflow trends
        if len(inflows_data) > 0:
            # Split data into periods
            mid_date = inflows_data['Date'].median()
            early_period = inflows_data[inflows_data['Date'] <= mid_date]
            late_period = inflows_data[inflows_data['Date'] > mid_date]
            
            if len(early_period) > 0 and len(late_period) > 0:
                early_category_totals = early_period.groupby('Category')['Quantity'].sum()
                late_category_totals = late_period.groupby('Category')['Quantity'].sum()
                
                for category in early_category_totals.index:
                    if category in late_category_totals.index:
                        early_total = early_category_totals[category]
                        late_total = late_category_totals[category]
                        
                        if early_total > 0:
                            growth_rate = ((late_total - early_total) / early_total) * 100
                            
                            if growth_rate > 20:
                                growing_categories.append({
                                    'Category': category,
                                    'Growth_Rate': f"{growth_rate:.1f}%",
                                    'Early_Period_Volume': early_total,
                                    'Late_Period_Volume': late_total
                                })
                            elif growth_rate < -20:
                                declining_categories.append({
                                    'Category': category,
                                    'Decline_Rate': f"{abs(growth_rate):.1f}%",
                                    'Early_Period_Volume': early_total,
                                    'Late_Period_Volume': late_total
                                })
        
        return {
            'growing': sorted(growing_categories, key=lambda x: float(x['Growth_Rate'].rstrip('%')), reverse=True)[:5],
            'declining': sorted(declining_categories, key=lambda x: float(x['Decline_Rate'].rstrip('%')), reverse=True)[:5]
        }
    
    def _analyze_warehouse_efficiency(self, data_dict):
        """Analyze warehouse efficiency metrics"""
        inflows_data = data_dict['inflows']
        outflows_data = data_dict['outflows']
        warehouse_efficiency = []
        
        if len(inflows_data) == 0 or len(outflows_data) == 0:
            return []
        
        # Get unique warehouses
        all_warehouses = set(inflows_data['Warehouse'].unique()) | set(outflows_data['Warehouse'].unique())
        
        for warehouse in all_warehouses:
            warehouse_inflows = inflows_data[inflows_data['Warehouse'] == warehouse]
            warehouse_outflows = outflows_data[outflows_data['Warehouse'] == warehouse]
            
            total_inflows = warehouse_inflows['Quantity'].sum()
            total_outflows = warehouse_outflows['Quantity'].sum()
            
            # Calculate efficiency metrics
            throughput_ratio = total_outflows / total_inflows if total_inflows > 0 else 0
            
            # Average processing time (simplified)
            avg_inflow_quantity = warehouse_inflows['Quantity'].mean() if len(warehouse_inflows) > 0 else 0
            avg_outflow_quantity = warehouse_outflows['Quantity'].mean() if len(warehouse_outflows) > 0 else 0
            
            efficiency_score = min(throughput_ratio, 1.0) * 100  # Cap at 100%
            
            warehouse_efficiency.append({
                'Warehouse': warehouse,
                'Total_Inflows': total_inflows,
                'Total_Outflows': total_outflows,
                'Throughput_Ratio': f"{throughput_ratio:.2f}",
                'Efficiency_Score': f"{efficiency_score:.1f}%",
                'Avg_Inflow_Quantity': round(avg_inflow_quantity),
                'Avg_Outflow_Quantity': round(avg_outflow_quantity),
                'Status': 'High' if efficiency_score > 80 else 'Medium' if efficiency_score > 60 else 'Low'
            })
        
        return sorted(warehouse_efficiency, key=lambda x: float(x['Efficiency_Score'].rstrip('%')), reverse=True)
    
    def _generate_demand_forecasts(self, data_dict, forecast_results):
        """Generate demand forecasts by category and warehouse"""
        demand_forecasts = {}
        
        outflows_data = data_dict['outflows']
        
        if len(outflows_data) == 0:
            return {}
        
        # Group by category
        category_groups = outflows_data.groupby('Category')
        
        for category, group in category_groups:
            # Calculate trend and seasonality
            daily_demand = group.groupby('Date')['Quantity'].sum()
            
            if len(daily_demand) > 1:
                # Simple linear trend calculation
                x = np.arange(len(daily_demand))
                y = daily_demand.values
                trend_slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                
                # Forecast next 30 days
                avg_demand = daily_demand.mean()
                forecast_30_days = avg_demand + (trend_slope * 30)
                
                demand_forecasts[category] = {
                    'current_avg_daily_demand': round(avg_demand, 2),
                    'trend_slope': round(trend_slope, 3),
                    '30_day_forecast': max(0, round(forecast_30_days, 2)),
                    'confidence': 'Medium'  # Simplified confidence level
                }
        
        return demand_forecasts
    
    def _identify_optimization_opportunities(self, data_dict):
        """Identify opportunities for inventory optimization"""
        opportunities = []
        
        inflows_data = data_dict['inflows']
        outflows_data = data_dict['outflows']
        inventory_data = data_dict['inventory']
        
        # Opportunity 1: Slow-moving inventory
        if len(outflows_data) > 0:
            category_outflows = outflows_data.groupby('Category')['Quantity'].sum().sort_values()
            slow_categories = category_outflows.head(3)
            
            for category, quantity in slow_categories.items():
                opportunities.append({
                    'type': 'SLOW_MOVING_INVENTORY',
                    'category': category,
                    'description': f'Category "{category}" has low outflow volume ({quantity} units)',
                    'recommendation': 'Consider reducing procurement for this category or finding alternative distribution channels',
                    'priority': 'Medium'
                })
        
        # Opportunity 2: High variability categories
        if len(outflows_data) > 0:
            category_std = outflows_data.groupby('Category')['Quantity'].std().sort_values(ascending=False)
            high_variability = category_std.head(2)
            
            for category, std_dev in high_variability.items():
                opportunities.append({
                    'type': 'HIGH_DEMAND_VARIABILITY',
                    'category': category,
                    'description': f'Category "{category}" shows high demand variability (std dev: {std_dev:.1f})',
                    'recommendation': 'Implement demand smoothing strategies or increase safety stock levels',
                    'priority': 'High'
                })
        
        # Opportunity 3: Inventory level optimization
        if len(inventory_data) > 0:
            current_level = inventory_data['Inventory_Level'].iloc[-1]
            avg_level = inventory_data['Inventory_Level'].mean()
            
            if current_level > avg_level * 1.5:
                opportunities.append({
                    'type': 'EXCESS_INVENTORY',
                    'category': 'Overall',
                    'description': f'Current inventory ({current_level:,.0f}) is 50% above average ({avg_level:,.0f})',
                    'recommendation': 'Consider accelerating distribution or reducing procurement temporarily',
                    'priority': 'Medium'
                })
        
        return opportunities[:10]  # Return top 10 opportunities
