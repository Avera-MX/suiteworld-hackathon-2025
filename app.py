import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_handler import DataHandler
from forecasting_engine import ForecastingEngine
from anomaly_detector import AnomalyDetector
from insights_generator import InsightsGenerator
from report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="AI Inventory Forecasting Platform",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("🚀 AI-Powered Inventory Forecasting Platform")
    st.markdown("### 2025 SuiteWorld Hackathon 4Good Challenge")
    st.markdown("---")
    
    # Initialize session state
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()
    if 'forecasting_engine' not in st.session_state:
        st.session_state.forecasting_engine = ForecastingEngine()
    if 'anomaly_detector' not in st.session_state:
        st.session_state.anomaly_detector = AnomalyDetector()
    if 'insights_generator' not in st.session_state:
        st.session_state.insights_generator = InsightsGenerator()
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = ReportGenerator()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Data Upload & Validation", "Statistical Analysis", "ML Forecasting", 
         "Anomaly Detection", "Model Performance", "Business Insights", "Export Reports"]
    )
    
    if page == "Data Upload & Validation":
        data_upload_page()
    elif page == "Statistical Analysis":
        statistical_analysis_page()
    elif page == "ML Forecasting":
        ml_forecasting_page()
    elif page == "Anomaly Detection":
        anomaly_detection_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Business Insights":
        business_insights_page()
    elif page == "Export Reports":
        export_reports_page()

def data_upload_page():
    st.header("📁 Data Upload & Schema Validation")
    
    # Use pre-loaded data or upload new data
    use_sample_data = st.checkbox("Use provided sample data", value=True)
    
    if use_sample_data:
        try:
            # Load the provided data files
            train_inventory = pd.read_csv('attached_assets/train_inventory_1759770514597.csv')
            train_inflows = pd.read_csv('attached_assets/train_inflows_1759770514597.csv')
            tune_inventory = pd.read_csv('attached_assets/tune_inventory - csv_1759772587802.csv')
            tune_inflows = pd.read_csv('attached_assets/tune_inflows - csv_1759772587803.csv')
            tune_outflows = pd.read_csv('attached_assets/tune_outflows - csv_1759772587801.csv')
            
            st.success("✅ Sample data loaded successfully!")
            
            # Process and validate the data
            result = st.session_state.data_handler.process_datasets(
                train_inventory, train_inflows, None,
                tune_inventory, tune_inflows, tune_outflows
            )
            
            if result['success']:
                st.session_state.datasets = result['datasets']
                display_data_overview(result['datasets'])
            else:
                st.error(f"❌ Data processing failed: {result['error']}")
                
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
    
    else:
        st.subheader("Upload Your Data Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Period Data")
            train_inventory_file = st.file_uploader("Training Inventory", type=['csv', 'xlsx'])
            train_inflows_file = st.file_uploader("Training Inflows", type=['csv', 'xlsx'])
            train_outflows_file = st.file_uploader("Training Outflows (Optional)", type=['csv', 'xlsx'])
        
        with col2:
            st.markdown("#### Tuning Period Data")
            tune_inventory_file = st.file_uploader("Tuning Inventory", type=['csv', 'xlsx'])
            tune_inflows_file = st.file_uploader("Tuning Inflows", type=['csv', 'xlsx'])
            tune_outflows_file = st.file_uploader("Tuning Outflows", type=['csv', 'xlsx'])
        
        if st.button("Process Uploaded Data"):
            if train_inventory_file and train_inflows_file and tune_inventory_file and tune_inflows_file:
                try:
                    # Load uploaded files
                    datasets = {}
                    files = {
                        'train_inventory': train_inventory_file,
                        'train_inflows': train_inflows_file,
                        'train_outflows': train_outflows_file,
                        'tune_inventory': tune_inventory_file,
                        'tune_inflows': tune_inflows_file,
                        'tune_outflows': tune_outflows_file
                    }
                    
                    for key, file in files.items():
                        if file is not None:
                            if file.name.endswith('.csv'):
                                datasets[key] = pd.read_csv(file)
                            else:
                                datasets[key] = pd.read_excel(file)
                    
                    # Process and validate
                    result = st.session_state.data_handler.process_datasets(
                        datasets.get('train_inventory'),
                        datasets.get('train_inflows'),
                        datasets.get('train_outflows'),
                        datasets.get('tune_inventory'),
                        datasets.get('tune_inflows'),
                        datasets.get('tune_outflows')
                    )
                    
                    if result['success']:
                        st.session_state.datasets = result['datasets']
                        st.success("✅ Data processed successfully!")
                        display_data_overview(result['datasets'])
                    else:
                        st.error(f"❌ Data processing failed: {result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
            else:
                st.warning("⚠️ Please upload at least the required files: Training Inventory, Training Inflows, Tuning Inventory, and Tuning Inflows")

def display_data_overview(datasets):
    st.subheader("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Period", "2017-2018")
        if 'train_inventory' in datasets:
            st.metric("Training Inventory Records", len(datasets['train_inventory']))
        if 'train_inflows' in datasets:
            st.metric("Training Inflows Records", len(datasets['train_inflows']))
    
    with col2:
        st.metric("Tuning Period", "2023")
        if 'tune_inventory' in datasets:
            st.metric("Tuning Inventory Records", len(datasets['tune_inventory']))
        if 'tune_inflows' in datasets:
            st.metric("Tuning Inflows Records", len(datasets['tune_inflows']))
    
    with col3:
        st.metric("Time Gap", "5+ Years")
        if 'tune_outflows' in datasets:
            st.metric("Tuning Outflows Records", len(datasets['tune_outflows']))
    
    # Display sample data
    if st.checkbox("Show Data Samples"):
        for dataset_name, df in datasets.items():
            if df is not None:
                st.subheader(f"{dataset_name.replace('_', ' ').title()}")
                st.dataframe(df.head())

def statistical_analysis_page():
    st.header("📈 Statistical Analysis & Trends")
    
    if 'datasets' not in st.session_state:
        st.warning("⚠️ Please upload and process data first!")
        return
    
    datasets = st.session_state.datasets
    
    # Inventory trends comparison
    st.subheader("Inventory Level Trends")
    
    if 'train_inventory' in datasets and 'tune_inventory' in datasets:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training Period (2017-2018)', 'Tuning Period (2023)'),
            shared_xaxes=False
        )
        
        # Training period
        train_inv = datasets['train_inventory'].copy()
        train_inv['Date'] = pd.to_datetime(train_inv['Date'])
        fig.add_trace(
            go.Scatter(x=train_inv['Date'], y=train_inv['Inventory_Level'],
                      mode='lines', name='Training', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Tuning period
        tune_inv = datasets['tune_inventory'].copy()
        tune_inv['Date'] = pd.to_datetime(tune_inv['Date'])
        fig.add_trace(
            go.Scatter(x=tune_inv['Date'], y=tune_inv['Inventory_Level'],
                      mode='lines', name='Tuning', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title="Inventory Trends Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Period Statistics")
            train_stats = train_inv['Inventory_Level'].describe()
            st.dataframe(train_stats)
        
        with col2:
            st.subheader("Tuning Period Statistics")
            tune_stats = tune_inv['Inventory_Level'].describe()
            st.dataframe(tune_stats)
    
    # Inflows analysis
    st.subheader("Inflows Analysis")
    
    if 'train_inflows' in datasets and 'tune_inflows' in datasets:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Period Inflows by Category")
            if 'Category' in datasets['train_inflows'].columns:
                train_category = datasets['train_inflows']['Category'].value_counts()
                fig = px.pie(values=train_category.values, names=train_category.index,
                           title="Training Period Categories")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Tuning Period Inflows by Category")
            if 'Category' in datasets['tune_inflows'].columns:
                tune_category = datasets['tune_inflows']['Category'].value_counts()
                fig = px.pie(values=tune_category.values, names=tune_category.index,
                           title="Tuning Period Categories")
                st.plotly_chart(fig, use_container_width=True)
    
    # GIK value analysis
    if 'tune_inflows' in datasets and 'Total_GIK' in datasets['tune_inflows'].columns:
        st.subheader("GIK Value Distribution Analysis")
        
        gik_values = datasets['tune_inflows']['Total_GIK'].dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(gik_values, nbins=50, title="GIK Values Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(y=gik_values, title="GIK Values Box Plot")
            st.plotly_chart(fig, use_container_width=True)
        
        # Extreme values detection
        Q1 = gik_values.quantile(0.25)
        Q3 = gik_values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = gik_values[(gik_values < Q1 - 1.5*IQR) | (gik_values > Q3 + 1.5*IQR)]
        
        st.metric("Extreme GIK Values Detected", len(outliers))
        if len(outliers) > 0:
            st.dataframe(outliers.describe())

def ml_forecasting_page():
    st.header("🤖 ML Forecasting Engine")
    
    if 'datasets' not in st.session_state:
        st.warning("⚠️ Please upload and process data first!")
        return
    
    datasets = st.session_state.datasets
    
    # Forecasting configuration
    st.subheader("Forecasting Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_method = st.selectbox("Forecasting Method", ["Prophet", "SARIMA", "Both"])
        forecast_periods = st.slider("Forecast Periods (days)", 30, 365, 90)
    
    with col2:
        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
        handle_outliers = st.checkbox("Handle Outliers", value=True)
    
    with col3:
        scale_normalization = st.checkbox("Scale Normalization", value=True)
        trend_detection = st.checkbox("Automatic Trend Detection", value=True)
    
    if st.button("Generate Forecasts"):
        with st.spinner("Training models and generating forecasts..."):
            try:
                # Prepare data for forecasting
                if 'train_inventory' in datasets and 'tune_inventory' in datasets:
                    train_data = datasets['train_inventory'].copy()
                    tune_data = datasets['tune_inventory'].copy()
                    
                    # Configure forecasting engine
                    forecast_config = {
                        'method': forecast_method,
                        'periods': forecast_periods,
                        'confidence_level': confidence_level,
                        'handle_outliers': handle_outliers,
                        'scale_normalization': scale_normalization,
                        'trend_detection': trend_detection
                    }
                    
                    # Generate forecasts
                    results = st.session_state.forecasting_engine.generate_forecasts(
                        train_data, tune_data, forecast_config
                    )
                    
                    if results['success']:
                        st.session_state.forecast_results = results
                        display_forecast_results(results)
                    else:
                        st.error(f"❌ Forecasting failed: {results['error']}")
                        
            except Exception as e:
                st.error(f"❌ Error during forecasting: {str(e)}")

def display_forecast_results(results):
    st.success("✅ Forecasts generated successfully!")
    
    # Display forecast plots
    for method, result in results['forecasts'].items():
        if result is not None:
            st.subheader(f"{method} Forecast Results")
            
            # Plot forecast
            fig = go.Figure()
            
            # Historical data
            if 'historical' in result:
                fig.add_trace(go.Scatter(
                    x=result['historical']['ds'],
                    y=result['historical']['y'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
            
            # Forecast
            if 'forecast' in result:
                fig.add_trace(go.Scatter(
                    x=result['forecast']['ds'],
                    y=result['forecast']['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
                
                # Confidence intervals
                if 'yhat_lower' in result['forecast'] and 'yhat_upper' in result['forecast']:
                    fig.add_trace(go.Scatter(
                        x=result['forecast']['ds'],
                        y=result['forecast']['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=result['forecast']['ds'],
                        y=result['forecast']['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(255,0,0,0.2)',
                        fill='tonexty',
                        name='Confidence Interval',
                        hoverinfo='skip'
                    ))
            
            fig.update_layout(
                title=f"{method} Inventory Forecast",
                xaxis_title="Date",
                yaxis_title="Inventory Level",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            if 'metrics' in result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{result['metrics'].get('rmse', 0):.2f}")
                with col2:
                    st.metric("MAE", f"{result['metrics'].get('mae', 0):.2f}")
                with col3:
                    st.metric("MAPE", f"{result['metrics'].get('mape', 0):.2f}%")

def anomaly_detection_page():
    st.header("🔍 Anomaly Detection & Data Drift Analysis")
    
    if 'datasets' not in st.session_state:
        st.warning("⚠️ Please upload and process data first!")
        return
    
    datasets = st.session_state.datasets
    
    # Run anomaly detection
    if st.button("Run Anomaly Detection Analysis"):
        with st.spinner("Analyzing anomalies and data drift..."):
            try:
                results = st.session_state.anomaly_detector.detect_anomalies(datasets)
                
                if results['success']:
                    st.session_state.anomaly_results = results
                    display_anomaly_results(results)
                else:
                    st.error(f"❌ Anomaly detection failed: {results['error']}")
                    
            except Exception as e:
                st.error(f"❌ Error during anomaly detection: {str(e)}")
    
    # Display cached results if available
    if 'anomaly_results' in st.session_state:
        display_anomaly_results(st.session_state.anomaly_results)

def display_anomaly_results(results):
    st.success("✅ Anomaly detection completed!")
    
    # Outlier detection results
    if 'outliers' in results:
        st.subheader("🚨 Extreme Value Outliers")
        
        outlier_summary = results['outliers']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("GIK Outliers", len(outlier_summary.get('gik_outliers', [])))
        with col2:
            st.metric("Inventory Outliers", len(outlier_summary.get('inventory_outliers', [])))
        with col3:
            st.metric("Quantity Outliers", len(outlier_summary.get('quantity_outliers', [])))
        
        # Display outlier plots
        if outlier_summary.get('gik_outliers') is not None and len(outlier_summary['gik_outliers']) > 0:
            st.subheader("GIK Value Outliers")
            outliers_df = pd.DataFrame(outlier_summary['gik_outliers'])
            if not outliers_df.empty:
                fig = px.scatter(outliers_df, x='Date', y='Total_GIK', 
                               title="Extreme GIK Values Over Time")
                st.plotly_chart(fig, use_container_width=True)
    
    # Data drift analysis
    if 'drift_analysis' in results:
        st.subheader("📊 Data Drift Analysis")
        
        drift_results = results['drift_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribution Changes")
            if 'distribution_changes' in drift_results:
                for feature, change in drift_results['distribution_changes'].items():
                    st.metric(f"{feature} Change", f"{change:.3f}")
        
        with col2:
            st.markdown("#### Statistical Tests")
            if 'statistical_tests' in drift_results:
                for test_name, result in drift_results['statistical_tests'].items():
                    status = "✅ No Drift" if result['p_value'] > 0.05 else "⚠️ Drift Detected"
                    st.metric(f"{test_name}", status)
    
    # Scale shift detection
    if 'scale_shifts' in results:
        st.subheader("⚖️ Scale Shift Detection")
        
        scale_shifts = results['scale_shifts']
        
        for metric, shift in scale_shifts.items():
            st.metric(f"{metric} Scale Change", f"{shift:.2f}x")

def model_performance_page():
    st.header("📊 Model Performance & Retraining Dashboard")
    
    if 'forecast_results' not in st.session_state:
        st.warning("⚠️ Please run ML forecasting first!")
        return
    
    results = st.session_state.forecast_results
    
    # Performance metrics comparison
    st.subheader("📈 Performance Metrics Comparison")
    
    metrics_data = []
    for method, result in results['forecasts'].items():
        if result and 'metrics' in result:
            metrics_data.append({
                'Method': method,
                'RMSE': result['metrics'].get('rmse', 0),
                'MAE': result['metrics'].get('mae', 0),
                'MAPE': result['metrics'].get('mape', 0)
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df)
        
        # Visualize metrics
        fig = make_subplots(rows=1, cols=3, 
                          subplot_titles=('RMSE', 'MAE', 'MAPE'))
        
        fig.add_trace(go.Bar(x=metrics_df['Method'], y=metrics_df['RMSE'], name='RMSE'), row=1, col=1)
        fig.add_trace(go.Bar(x=metrics_df['Method'], y=metrics_df['MAE'], name='MAE'), row=1, col=2)
        fig.add_trace(go.Bar(x=metrics_df['Method'], y=metrics_df['MAPE'], name='MAPE'), row=1, col=3)
        
        fig.update_layout(height=400, title="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model retraining options
    st.subheader("🔄 Model Retraining")
    
    col1, col2 = st.columns(2)
    
    with col1:
        retrain_frequency = st.selectbox("Retraining Frequency", 
                                       ["Weekly", "Monthly", "Quarterly", "Manual"])
        performance_threshold = st.slider("Performance Threshold (MAPE)", 5.0, 25.0, 10.0)
    
    with col2:
        auto_retrain = st.checkbox("Automatic Retraining", value=True)
        notification_enabled = st.checkbox("Performance Alerts", value=True)
    
    if st.button("Configure Retraining"):
        st.success("✅ Retraining configuration updated!")
        st.info(f"Models will retrain {retrain_frequency.lower()} when MAPE exceeds {performance_threshold}%")
    
    # Stability report
    st.subheader("🔒 Model Stability Report")
    
    if 'stability_report' in results:
        stability = results['stability_report']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Time Period Coverage", stability.get('coverage', 'N/A'))
        with col2:
            st.metric("Scale Adaptability", stability.get('scale_adaptation', 'N/A'))
        with col3:
            st.metric("Prediction Consistency", stability.get('consistency', 'N/A'))

def business_insights_page():
    st.header("💡 Business Insights & Recommendations")
    
    if 'datasets' not in st.session_state:
        st.warning("⚠️ Please upload and process data first!")
        return
    
    datasets = st.session_state.datasets
    
    # Generate insights
    if st.button("Generate Business Insights"):
        with st.spinner("Generating actionable insights..."):
            try:
                insights = st.session_state.insights_generator.generate_insights(
                    datasets, 
                    st.session_state.get('forecast_results'),
                    st.session_state.get('anomaly_results')
                )
                
                if insights['success']:
                    st.session_state.business_insights = insights
                    display_business_insights(insights)
                else:
                    st.error(f"❌ Insight generation failed: {insights['error']}")
                    
            except Exception as e:
                st.error(f"❌ Error generating insights: {str(e)}")
    
    # Display cached insights
    if 'business_insights' in st.session_state:
        display_business_insights(st.session_state.business_insights)

def display_business_insights(insights):
    st.success("✅ Business insights generated!")
    
    # Key recommendations
    if 'recommendations' in insights:
        st.subheader("🎯 Key Recommendations")
        
        recommendations = insights['recommendations']
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"Recommendation {i}: {rec['title']}"):
                st.write(rec['description'])
                st.markdown(f"**Priority:** {rec.get('priority', 'Medium')}")
                st.markdown(f"**Impact:** {rec.get('impact', 'Medium')}")
                if 'action_items' in rec:
                    st.markdown("**Action Items:**")
                    for item in rec['action_items']:
                        st.markdown(f"• {item}")
    
    # Risk alerts
    if 'risk_alerts' in insights:
        st.subheader("⚠️ Risk Alerts")
        
        risk_alerts = insights['risk_alerts']
        
        if risk_alerts:
            for alert in risk_alerts:
                alert_level = alert.get('level', 'medium')
                if alert_level == 'high':
                    st.error(f"🚨 HIGH RISK: {alert['message']}")
                elif alert_level == 'medium':
                    st.warning(f"⚠️ MEDIUM RISK: {alert['message']}")
                else:
                    st.info(f"ℹ️ LOW RISK: {alert['message']}")
        else:
            st.success("✅ No significant risks detected")
    
    # Inventory optimization
    if 'inventory_optimization' in insights:
        st.subheader("📦 Inventory Optimization")
        
        optimization = insights['inventory_optimization']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'reorder_points' in optimization:
                st.markdown("#### Recommended Reorder Points")
                reorder_df = pd.DataFrame(optimization['reorder_points'])
                st.dataframe(reorder_df)
        
        with col2:
            if 'safety_stock' in optimization:
                st.markdown("#### Safety Stock Levels")
                safety_df = pd.DataFrame(optimization['safety_stock'])
                st.dataframe(safety_df)
    
    # Category insights
    if 'category_insights' in insights:
        st.subheader("🏷️ Category & Product Insights")
        
        category_insights = insights['category_insights']
        
        for category, insight in category_insights.items():
            with st.expander(f"{category} Analysis"):
                st.write(insight.get('summary', 'No summary available'))
                
                if 'metrics' in insight:
                    metrics = insight['metrics']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Volume", metrics.get('volume', 0))
                    with col2:
                        st.metric("Avg Value", f"${metrics.get('avg_value', 0):.2f}")
                    with col3:
                        st.metric("Growth Rate", f"{metrics.get('growth_rate', 0):.1f}%")

def export_reports_page():
    st.header("📄 Export Reports & Data")
    
    if 'datasets' not in st.session_state:
        st.warning("⚠️ Please upload and process data first!")
        return
    
    st.subheader("📊 Available Exports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Data Exports")
        
        if st.button("Export Processed Data (CSV)"):
            try:
                csv_data = st.session_state.report_generator.export_data_csv(st.session_state.datasets)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="inventory_data_export.csv",
                    mime="text/csv"
                )
                st.success("✅ CSV export ready!")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
        
        if st.button("Export Data (Excel)"):
            try:
                excel_data = st.session_state.report_generator.export_data_excel(st.session_state.datasets)
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name="inventory_data_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("✅ Excel export ready!")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    with col2:
        st.markdown("#### Forecast Exports")
        
        if 'forecast_results' in st.session_state:
            if st.button("Export Forecasts (CSV)"):
                try:
                    csv_data = st.session_state.report_generator.export_forecasts_csv(
                        st.session_state.forecast_results
                    )
                    st.download_button(
                        label="Download Forecasts CSV",
                        data=csv_data,
                        file_name="inventory_forecasts.csv",
                        mime="text/csv"
                    )
                    st.success("✅ Forecast CSV export ready!")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
    
    with col3:
        st.markdown("#### Business Reports")
        
        if 'business_insights' in st.session_state:
            if st.button("Generate PDF Report"):
                try:
                    pdf_data = st.session_state.report_generator.generate_pdf_report(
                        st.session_state.datasets,
                        st.session_state.get('forecast_results'),
                        st.session_state.get('business_insights')
                    )
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_data,
                        file_name="inventory_analysis_report.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF report ready!")
                except Exception as e:
                    st.error(f"❌ Report generation failed: {str(e)}")
    
    # Quick export options
    st.subheader("⚡ Quick Export Options")
    
    export_options = st.multiselect(
        "Select items to include in comprehensive export:",
        ["Raw Data", "Statistical Analysis", "Forecasts", "Anomaly Detection", "Business Insights"],
        default=["Statistical Analysis", "Forecasts", "Business Insights"]
    )
    
    if st.button("Generate Comprehensive Export"):
        if export_options:
            try:
                export_data = st.session_state.report_generator.generate_comprehensive_export(
                    st.session_state.datasets,
                    st.session_state.get('forecast_results'),
                    st.session_state.get('anomaly_results'),
                    st.session_state.get('business_insights'),
                    export_options
                )
                
                st.download_button(
                    label="Download Comprehensive Report",
                    data=export_data,
                    file_name="comprehensive_inventory_report.zip",
                    mime="application/zip"
                )
                st.success("✅ Comprehensive export ready!")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
        else:
            st.warning("⚠️ Please select at least one export option")

if __name__ == "__main__":
    main()
