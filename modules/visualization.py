import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class VisualizationEngine:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def plot_inventory_trends(self, inventory_trends):
        """Plot inventory level trends over time"""
        # This would normally use the actual inventory data
        # For now, creating a sample trend visualization structure
        fig = go.Figure()
        
        # Add trend information as text annotations
        fig.add_annotation(
            text=f"Average Level: {inventory_trends['mean_level']:,.0f}<br>" +
                 f"Peak Level: {inventory_trends['max_level']:,.0f}<br>" +
                 f"Min Level: {inventory_trends['min_level']:,.0f}<br>" +
                 f"Trend: {inventory_trends['linear_trend']['trend_direction']}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        fig.update_layout(
            title="Inventory Level Trends",
            xaxis_title="Time",
            yaxis_title="Inventory Level",
            height=400
        )
        
        return fig
    
    def plot_category_distribution(self, category_distribution):
        """Plot product category distribution"""
        if not category_distribution:
            # Return empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No category distribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Product Category Distribution")
            return fig
        
        categories = list(category_distribution.keys())[:10]  # Top 10
        values = [category_distribution[cat] for cat in categories]
        
        fig = px.pie(
            values=values,
            names=categories,
            title="Product Category Distribution",
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
    
    def plot_warehouse_distribution(self, warehouse_distribution):
        """Plot warehouse distribution"""
        if not warehouse_distribution:
            fig = go.Figure()
            fig.add_annotation(
                text="No warehouse distribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Warehouse Distribution")
            return fig
        
        warehouses = list(warehouse_distribution.keys())
        values = list(warehouse_distribution.values())
        
        fig = px.bar(
            x=warehouses,
            y=values,
            title="Warehouse Distribution",
            color=warehouses,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title="Warehouse",
            yaxis_title="Volume",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_temporal_patterns(self, temporal_patterns):
        """Plot temporal patterns"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Patterns', 'Quarterly Patterns', 'Daily Patterns', 'Yearly Patterns'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Monthly patterns
        if 'monthly_patterns' in temporal_patterns:
            months = list(temporal_patterns['monthly_patterns'].keys())
            monthly_values = list(temporal_patterns['monthly_patterns'].values())
            fig.add_trace(
                go.Bar(x=months, y=monthly_values, name="Monthly", marker_color=self.color_palette[0]),
                row=1, col=1
            )
        
        # Quarterly patterns
        if 'quarterly_patterns' in temporal_patterns:
            quarters = list(temporal_patterns['quarterly_patterns'].keys())
            quarterly_values = list(temporal_patterns['quarterly_patterns'].values())
            fig.add_trace(
                go.Bar(x=quarters, y=quarterly_values, name="Quarterly", marker_color=self.color_palette[1]),
                row=1, col=2
            )
        
        # Daily patterns
        if 'daily_patterns' in temporal_patterns:
            days = list(temporal_patterns['daily_patterns'].keys())
            daily_values = list(temporal_patterns['daily_patterns'].values())
            fig.add_trace(
                go.Bar(x=days, y=daily_values, name="Daily", marker_color=self.color_palette[2]),
                row=2, col=1
            )
        
        # Yearly patterns
        if 'yearly_patterns' in temporal_patterns:
            years = list(temporal_patterns['yearly_patterns'].keys())
            yearly_values = list(temporal_patterns['yearly_patterns'].values())
            fig.add_trace(
                go.Bar(x=years, y=yearly_values, name="Yearly", marker_color=self.color_palette[3]),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Temporal Patterns Analysis",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_forecast(self, forecast_results):
        """Plot forecasting results"""
        if 'forecasts' not in forecast_results:
            fig = go.Figure()
            fig.add_annotation(
                text="No forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Forecast Results")
            return fig
        
        forecasts_df = forecast_results['forecasts']
        
        fig = go.Figure()
        
        # Add historical data if available
        if 'historical_data' in forecast_results:
            historical = forecast_results['historical_data']
            fig.add_trace(
                go.Scatter(
                    x=historical.index,
                    y=historical['Inventory_Level'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue')
                )
            )
        
        # Add forecast
        fig.add_trace(
            go.Scatter(
                x=forecasts_df['Date'],
                y=forecasts_df['Forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecasts_df['Date'], forecasts_df['Date'][::-1]]),
                y=pd.concat([forecasts_df['Upper_CI'], forecasts_df['Lower_CI'][::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=False
            )
        )
        
        fig.update_layout(
            title="Inventory Forecast",
            xaxis_title="Date",
            yaxis_title="Inventory Level",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_anomalies(self, anomalies_df):
        """Plot inventory anomalies"""
        if len(anomalies_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No anomalies detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Inventory Anomalies")
            return fig
        
        fig = go.Figure()
        
        # Group anomalies by type
        anomaly_types = anomalies_df['Anomaly_Type'].unique()
        colors = ['red', 'orange', 'purple']
        
        for i, anomaly_type in enumerate(anomaly_types):
            type_data = anomalies_df[anomalies_df['Anomaly_Type'] == anomaly_type]
            
            fig.add_trace(
                go.Scatter(
                    x=type_data['Date'],
                    y=type_data['Inventory_Level'],
                    mode='markers',
                    name=anomaly_type,
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=type_data['Anomaly_Score'] * 2,  # Size based on severity
                        symbol='diamond'
                    )
                )
            )
        
        fig.update_layout(
            title="Inventory Level Anomalies",
            xaxis_title="Date",
            yaxis_title="Inventory Level",
            height=500
        )
        
        return fig
    
    def plot_distribution_drift(self, distribution_changes):
        """Plot distribution drift analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Inventory Levels', 'Product Categories', 'Brands', 'Warehouses'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Inventory levels drift
        if 'inventory_levels' in distribution_changes:
            inv_data = distribution_changes['inventory_levels']
            fig.add_trace(
                go.Bar(
                    x=['2017-2018', '2023'],
                    y=[inv_data['training_mean'], inv_data['tune_mean']],
                    name="Avg Inventory",
                    marker_color=['blue', 'red']
                ),
                row=1, col=1
            )
        
        # Other distribution comparisons would be added here
        # For brevity, showing structure
        
        fig.update_layout(
            title="Distribution Drift Analysis",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_prediction_comparison(self, forecast_results):
        """Plot prediction vs actual comparison"""
        fig = go.Figure()
        
        # This would compare predictions to actual values
        # For now, showing the structure
        fig.add_annotation(
            text="Prediction vs Actual comparison would be displayed here",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        
        fig.update_layout(
            title="Predictions vs Actuals",
            xaxis_title="Time",
            yaxis_title="Inventory Level",
            height=500
        )
        
        return fig
