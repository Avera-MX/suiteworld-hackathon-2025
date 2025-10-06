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
import data_storage

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
         "Model Performance", "Actual vs Predicted", "What-If Scenarios", "Anomaly Detection", 
         "Business Insights", "Partner Analytics", "Automated Alerts", "Export Reports"]
    )
    
    if page == "Data Upload & Validation":
        data_upload_page()
    elif page == "Statistical Analysis":
        statistical_analysis_page()
    elif page == "ML Forecasting":
        ml_forecasting_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Actual vs Predicted":
        actual_vs_predicted_page()
    elif page == "What-If Scenarios":
        whatif_scenarios_page()
    elif page == "Anomaly Detection":
        anomaly_detection_page()
    elif page == "Business Insights":
        business_insights_page()
    elif page == "Partner Analytics":
        partner_analytics_page()
    elif page == "Automated Alerts":
        automated_alerts_page()
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
                
                # Save data to API storage
                save_results = data_storage.save_all_datasets(result['datasets'])
                saved_count = sum(1 for v in save_results.values() if v)
                if saved_count > 0:
                    st.info(f"💾 Saved {saved_count} datasets for API access")
                
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
        
        st.markdown("---")
        st.markdown("#### Test Period Data (For Model Comparison)")
        st.info("📝 Upload test data to compare actual vs predicted inventory levels")
        
        col3, col4 = st.columns(2)
        with col3:
            test_inventory_file = st.file_uploader("Test Inventory (Actual)", type=['csv', 'xlsx'], key='test_inv')
        with col4:
            test_inflows_file = st.file_uploader("Test Inflows (Optional)", type=['csv', 'xlsx'], key='test_inf')
            test_outflows_file = st.file_uploader("Test Outflows (Optional)", type=['csv', 'xlsx'], key='test_out')
        
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
                        'tune_outflows': tune_outflows_file,
                        'test_inventory': test_inventory_file,
                        'test_inflows': test_inflows_file,
                        'test_outflows': test_outflows_file
                    }
                    
                    for key, file in files.items():
                        if file is not None:
                            if file.name.endswith('.csv'):
                                datasets[key] = pd.read_csv(file)
                            else:
                                datasets[key] = pd.read_excel(file)
                    
                    # Process and validate train/tune data
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
                        
                        # Process test data separately if provided
                        if datasets.get('test_inventory') is not None:
                            test_result = st.session_state.data_handler._validate_and_clean_dataset(
                                datasets['test_inventory'], 'inventory', 'test_inventory'
                            )
                            if test_result['success']:
                                st.session_state.datasets['test_inventory'] = test_result['data']
                                st.success("✅ Data processed successfully (including test data)!")
                            else:
                                st.warning(f"⚠️ Test inventory validation failed: {test_result['error']}")
                                st.success("✅ Training and tuning data processed successfully!")
                        else:
                            st.success("✅ Data processed successfully!")
                        
                        # Save data to API storage
                        save_results = data_storage.save_all_datasets(st.session_state.datasets)
                        saved_count = sum(1 for v in save_results.values() if v)
                        if saved_count > 0:
                            st.info(f"💾 Saved {saved_count} datasets for API access")
                        
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
    
    # Extract unique warehouses and categories from the data
    warehouses = set()
    categories = set()
    
    for dataset_name in ['train_inflows', 'tune_inflows', 'train_outflows', 'tune_outflows']:
        if dataset_name in datasets:
            df = datasets[dataset_name]
            if 'Warehouse' in df.columns:
                warehouses.update(df['Warehouse'].dropna().unique())
            if 'Category' in df.columns:
                categories.update(df['Category'].dropna().unique())
    
    # Filter controls
    if warehouses or categories:
        st.subheader("🔍 Filter by Dimension")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if warehouses:
                warehouse_options = ['All Warehouses'] + sorted(list(warehouses))
                selected_warehouses = st.multiselect(
                    "Select Warehouses",
                    options=warehouse_options,
                    default=['All Warehouses'],
                    help="Forecast for specific warehouses"
                )
            else:
                selected_warehouses = ['All Warehouses']
        
        with col2:
            if categories:
                category_options = ['All Categories'] + sorted(list(categories))
                selected_categories = st.multiselect(
                    "Select Categories",
                    options=category_options,
                    default=['All Categories'],
                    help="Forecast for specific product categories"
                )
            else:
                selected_categories = ['All Categories']
        
        with col3:
            st.markdown("#### Active Filters")
            filter_warehouse = 'All Warehouses' not in selected_warehouses and selected_warehouses
            filter_category = 'All Categories' not in selected_categories and selected_categories
            
            if filter_warehouse:
                st.metric("Warehouses", len(selected_warehouses))
            if filter_category:
                st.metric("Categories", len(selected_categories))
            if not filter_warehouse and not filter_category:
                st.info("No filters active")
        
        st.divider()
        
        # Store filter state for later use
        st.session_state.ml_forecast_filters = {
            'active': filter_warehouse or filter_category,
            'warehouses': selected_warehouses if filter_warehouse else ['All Warehouses'],
            'categories': selected_categories if filter_category else ['All Categories']
        }
    else:
        # No filters available
        st.session_state.ml_forecast_filters = {
            'active': False,
            'warehouses': ['All Warehouses'],
            'categories': ['All Categories']
        }
    
    # Create working copy of datasets with filters applied if needed
    working_datasets = datasets.copy()
    if st.session_state.ml_forecast_filters['active']:
        filter_warehouse = st.session_state.ml_forecast_filters['warehouses'] != ['All Warehouses']
        filter_category = st.session_state.ml_forecast_filters['categories'] != ['All Categories']
        
        for dataset_name in ['train_inflows', 'tune_inflows', 'train_outflows', 'tune_outflows']:
            if dataset_name in working_datasets:
                df = working_datasets[dataset_name].copy()
                
                # Apply warehouse filter
                if filter_warehouse and 'Warehouse' in df.columns:
                    df = df[df['Warehouse'].isin(st.session_state.ml_forecast_filters['warehouses'])]
                
                # Apply category filter
                if filter_category and 'Category' in df.columns:
                    df = df[df['Category'].isin(st.session_state.ml_forecast_filters['categories'])]
                
                working_datasets[dataset_name] = df
        
        st.info(f"📊 Forecasting for {len(st.session_state.ml_forecast_filters['warehouses']) if filter_warehouse else 'all'} warehouse(s) and {len(st.session_state.ml_forecast_filters['categories']) if filter_category else 'all'} category(ies)")
    
    # Model recommendation
    st.subheader("📋 Automated Model Selection")
    if st.button("🔍 Get Model Recommendation"):
        with st.spinner("Analyzing data characteristics..."):
            try:
                if 'train_inventory' in working_datasets and 'tune_inventory' in working_datasets:
                    train_data = working_datasets['train_inventory'].copy()
                    tune_data = working_datasets['tune_inventory'].copy()
                    
                    train_ts = st.session_state.forecasting_engine._prepare_time_series(train_data)
                    tune_ts = st.session_state.forecasting_engine._prepare_time_series(tune_data)
                    
                    if train_ts is not None:
                        recommendation = st.session_state.forecasting_engine.recommend_best_model(train_ts, tune_ts)
                        
                        if 'error' not in recommendation:
                            st.success(f"✅ Recommended Model: **{recommendation['recommended_model']}** (Confidence: {recommendation['confidence']:.0f}%)")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Reasons:**")
                                for reason in recommendation['reasons']:
                                    st.write(f"• {reason}")
                            
                            with col2:
                                st.markdown("**Data Characteristics:**")
                                chars = recommendation['data_characteristics']
                                st.write(f"• Data points: {chars['data_points']}")
                                st.write(f"• Time span: {chars['time_span_days']} days")
                                st.write(f"• Variability: {chars['variability_cv']:.2f}")
                                st.write(f"• Trend strength: {chars['trend_strength']:.2f}")
                            
                            # Show all model scores
                            if recommendation['all_scores']:
                                scores_df = pd.DataFrame([
                                    {'Model': model, 'Score': data['score']}
                                    for model, data in recommendation['all_scores'].items()
                                ]).sort_values('Score', ascending=False)
                                
                                fig = px.bar(scores_df, x='Model', y='Score', 
                                           title='Model Suitability Scores',
                                           color='Score',
                                           color_continuous_scale='Viridis')
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"⚠️ {recommendation['error']}")
            except Exception as e:
                st.error(f"❌ Error getting recommendation: {str(e)}")
    
    st.divider()
    
    # Forecasting configuration
    st.subheader("Forecasting Configuration")
    
    # Calculate optimal forecast period based on available data
    forecast_periods = 90  # default
    if 'train_inventory' in working_datasets and 'tune_inventory' in working_datasets:
        try:
            train_data = working_datasets['train_inventory'].copy()
            train_data['Date'] = pd.to_datetime(train_data['Date'])
            
            # Calculate historical data span
            data_span_days = (train_data['Date'].max() - train_data['Date'].min()).days
            
            # Set forecast period as 25% of historical data span
            # with reasonable min/max bounds (30-365 days)
            calculated_periods = int(data_span_days * 0.25)
            forecast_periods = max(30, min(365, calculated_periods))
            
            st.info(f"📊 **Auto-calculated Forecast Window:** {forecast_periods} days (based on {data_span_days} days of historical data)")
        except Exception as e:
            st.warning(f"⚠️ Using default forecast period of 90 days. Error calculating from data: {str(e)}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_method = st.selectbox("Forecasting Method", ["Prophet", "SARIMA", "XGBoost", "Both", "All"])
        # Allow manual override if needed
        manual_override = st.checkbox("Manual Override", value=False)
        if manual_override:
            forecast_periods = st.slider("Forecast Periods (days)", 30, 365, forecast_periods)
    
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
                if 'train_inventory' in working_datasets and 'tune_inventory' in working_datasets:
                    train_data = working_datasets['train_inventory'].copy()
                    tune_data = working_datasets['tune_inventory'].copy()
                    
                    # Include warehouse and category features if available
                    train_warehouse_features = working_datasets.get('train_warehouse_features')
                    tune_warehouse_features = working_datasets.get('tune_warehouse_features')
                    train_category_features = working_datasets.get('train_category_features')
                    tune_category_features = working_datasets.get('tune_category_features')
                    
                    # Configure forecasting engine
                    forecast_config = {
                        'method': forecast_method,
                        'periods': forecast_periods,
                        'confidence_level': confidence_level,
                        'handle_outliers': handle_outliers,
                        'scale_normalization': scale_normalization,
                        'trend_detection': trend_detection,
                        'train_warehouse_features': train_warehouse_features,
                        'tune_warehouse_features': tune_warehouse_features,
                        'train_category_features': train_category_features,
                        'tune_category_features': tune_category_features
                    }
                    
                    # Generate forecasts
                    results = st.session_state.forecasting_engine.generate_forecasts(
                        train_data, tune_data, forecast_config
                    )
                    
                    if results['success']:
                        # Store filters with forecast results
                        if warehouses or categories:
                            filter_warehouse = 'All Warehouses' not in selected_warehouses and selected_warehouses
                            filter_category = 'All Categories' not in selected_categories and selected_categories
                            results['filters'] = {
                                'warehouses': selected_warehouses if filter_warehouse else ['All Warehouses'],
                                'categories': selected_categories if filter_category else ['All Categories']
                            }
                        st.session_state.forecast_results = results
                        display_forecast_results(results)
                    else:
                        st.error(f"❌ Forecasting failed: {results['error']}")
                        
            except Exception as e:
                st.error(f"❌ Error during forecasting: {str(e)}")

def display_forecast_results(results):
    st.success("✅ Forecasts generated successfully!")
    
    # Display active filters if present
    if 'filters' in results:
        filters = results['filters']
        warehouses = filters.get('warehouses', ['All Warehouses'])
        categories = filters.get('categories', ['All Categories'])
        
        if warehouses != ['All Warehouses'] or categories != ['All Categories']:
            with st.expander("🔍 Active Filters & Forecast Scope", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📦 Warehouses:**")
                    for wh in warehouses:
                        st.markdown(f"• {wh}")
                with col2:
                    st.markdown("**🏷️ Categories:**")
                    for cat in categories:
                        st.markdown(f"• {cat}")
    
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
            
            # Display feature importance for XGBoost
            if method == 'XGBoost' and 'feature_importance' in result:
                st.markdown("#### 🔍 Feature Importance")
                
                importance_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v} 
                    for k, v in result['feature_importance'].items()
                ]).sort_values('Importance', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(importance_df, x='Importance', y='Feature', 
                                orientation='h',
                                title='Feature Importance for Prediction')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(importance_df, use_container_width=True)
    
    # Display confidence scores
    if 'confidence_scores' in results:
        st.divider()
        st.subheader("📊 Forecast Confidence Scoring")
        st.write("Real-time reliability assessment for each prediction point")
        
        for method, conf_data in results['confidence_scores'].items():
            if conf_data and 'scores' in conf_data:
                st.markdown(f"#### {method} Confidence")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Confidence", f"{conf_data['average_confidence']:.1f}%")
                with col2:
                    st.metric("High Confidence Days", conf_data['high_confidence_days'])
                with col3:
                    st.metric("Medium Confidence Days", conf_data['medium_confidence_days'])
                with col4:
                    st.metric("Low Confidence Days", conf_data['low_confidence_days'])
                
                # Create confidence timeline
                scores_df = pd.DataFrame(conf_data['scores'])
                
                fig = go.Figure()
                
                # Add confidence line
                fig.add_trace(go.Scatter(
                    x=scores_df['date'],
                    y=scores_df['confidence'],
                    mode='lines+markers',
                    name='Confidence Score',
                    line=dict(color='green'),
                    marker=dict(
                        size=6,
                        color=scores_df['confidence'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Confidence %")
                    )
                ))
                
                # Add threshold lines
                fig.add_hline(y=80, line_dash="dash", line_color="green", 
                             annotation_text="High Confidence")
                fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                             annotation_text="Medium Confidence")
                
                fig.update_layout(
                    title=f"{method} Prediction Confidence Over Time",
                    xaxis_title="Date",
                    yaxis_title="Confidence (%)",
                    yaxis_range=[0, 100],
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Ensemble forecasting section
    if 'forecast_results' in st.session_state and len(results.get('forecasts', {})) >= 2:
        st.divider()
        st.subheader("🎯 Ensemble Forecasting")
        st.write("Combine multiple models for improved accuracy and stability")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            ensemble_method = st.selectbox(
                "Ensemble Method",
                ["weighted", "average", "best"],
                format_func=lambda x: {
                    'weighted': 'Weighted Average',
                    'average': 'Simple Average',
                    'best': 'Best Model'
                }[x]
            )
        
        with col2:
            if st.button("Generate Ensemble Forecast"):
                with st.spinner("Creating ensemble forecast..."):
                    try:
                        ensemble_result = st.session_state.forecasting_engine.generate_ensemble_forecast(
                            results['forecasts'], 
                            method=ensemble_method
                        )
                        
                        if ensemble_result['success']:
                            st.session_state.ensemble_result = ensemble_result
                            display_ensemble_results(ensemble_result)
                        else:
                            st.error(f"❌ Ensemble generation failed: {ensemble_result['error']}")
                    except Exception as e:
                        st.error(f"❌ Error generating ensemble: {str(e)}")
        
        # Display cached ensemble results
        if 'ensemble_result' in st.session_state:
            display_ensemble_results(st.session_state.ensemble_result)

def display_ensemble_results(ensemble_result):
    st.success("✅ Ensemble forecast generated!")
    
    # Model weights
    st.subheader("Model Contribution Weights")
    weights = ensemble_result['model_weights']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        weights_df = pd.DataFrame({
            'Model': list(weights.keys()),
            'Weight': list(weights.values())
        })
        st.dataframe(weights_df, use_container_width=True)
    
    with col2:
        fig = px.pie(weights_df, values='Weight', names='Model', 
                     title='Ensemble Model Weights')
        st.plotly_chart(fig, use_container_width=True)
    
    # Ensemble forecast plot
    st.subheader("Ensemble Forecast")
    
    fig = go.Figure()
    
    # Add forecast trace
    forecast_df = ensemble_result['forecast']
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Ensemble Forecast',
        line=dict(color='purple', width=3)
    ))
    
    # Add confidence intervals
    if 'yhat_upper' in forecast_df and 'yhat_lower' in forecast_df:
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(128,0,128,0.2)',
            fill='tonexty',
            name='Confidence Interval',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f"Ensemble Forecast ({ensemble_result['method'].title()})",
        xaxis_title="Date",
        yaxis_title="Inventory Level",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Ensemble metrics
    if 'metrics' in ensemble_result:
        st.subheader("Ensemble Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("RMSE", f"{ensemble_result['metrics'].get('rmse', 0):.2f}")
        with col2:
            st.metric("MAE", f"{ensemble_result['metrics'].get('mae', 0):.2f}")
        with col3:
            st.metric("MAPE", f"{ensemble_result['metrics'].get('mape', 0):.2f}%")

def whatif_scenarios_page():
    st.header("🔮 What-If Scenario Analysis")
    
    if 'datasets' not in st.session_state:
        st.warning("⚠️ Please upload and process data first!")
        return
    
    if 'forecast_results' not in st.session_state:
        st.warning("⚠️ Please generate forecasts first before running scenario analysis!")
        st.info("Navigate to the 'ML Forecasting' page to generate forecasts.")
        return
    
    st.write("Test different inventory strategies and see their projected impact on inventory levels and service.")
    
    # Get base forecast (use ensemble if available, otherwise use first available model)
    base_forecast = None
    if 'ensemble_result' in st.session_state:
        base_forecast = st.session_state.ensemble_result['forecast']
        st.info("Using ensemble forecast as baseline for scenario analysis")
    else:
        # Use first available forecast
        for model_name, forecast_data in st.session_state.forecast_results['forecasts'].items():
            if forecast_data and 'forecast' in forecast_data:
                base_forecast = forecast_data['forecast']
                st.info(f"Using {model_name} forecast as baseline for scenario analysis")
                break
    
    if base_forecast is None:
        st.error("No forecast data available for scenario analysis")
        return
    
    # Calculate baseline stats
    if 'tune_inventory' in st.session_state.datasets:
        baseline_inventory = st.session_state.datasets['tune_inventory']['Inventory_Level'].mean()
    else:
        baseline_inventory = base_forecast['yhat'].mean()
    
    st.subheader("📋 Configure Scenarios")
    
    # Create multiple scenarios
    num_scenarios = st.slider("Number of Scenarios to Compare", 1, 4, 2)
    
    scenarios = []
    
    cols = st.columns(num_scenarios)
    
    for i in range(num_scenarios):
        with cols[i]:
            st.markdown(f"### Scenario {i+1}")
            
            scenario_name = st.text_input(f"Scenario Name", value=f"Strategy {i+1}", key=f"name_{i}")
            
            reorder_point = st.number_input(
                "Reorder Point",
                min_value=0,
                max_value=int(baseline_inventory * 2),
                value=int(baseline_inventory * 0.3),
                step=1000,
                key=f"reorder_{i}",
                help="Inventory level that triggers a new order"
            )
            
            safety_stock = st.number_input(
                "Safety Stock",
                min_value=0,
                max_value=int(baseline_inventory),
                value=int(baseline_inventory * 0.15),
                step=1000,
                key=f"safety_{i}",
                help="Minimum inventory level to maintain"
            )
            
            lead_time = st.slider(
                "Lead Time (days)",
                min_value=1,
                max_value=30,
                value=7 + (i * 3),
                key=f"lead_{i}",
                help="Time between order placement and arrival"
            )
            
            order_qty = st.number_input(
                "Order Quantity",
                min_value=1000,
                max_value=int(baseline_inventory * 3),
                value=int(baseline_inventory * 0.5),
                step=5000,
                key=f"order_{i}",
                help="Quantity to order each time"
            )
            
            scenarios.append({
                'name': scenario_name,
                'reorder_point': reorder_point,
                'safety_stock': safety_stock,
                'lead_time_days': lead_time,
                'order_quantity': order_qty,
                'initial_inventory': baseline_inventory
            })
    
    # Analyze scenarios
    if st.button("Run Scenario Analysis", type="primary"):
        with st.spinner("Simulating inventory scenarios..."):
            try:
                comparison_result = st.session_state.forecasting_engine.compare_scenarios(
                    base_forecast, scenarios
                )
                
                if comparison_result['success']:
                    st.session_state.scenario_results = comparison_result
                    display_scenario_results(comparison_result)
                else:
                    st.error(f"❌ Scenario analysis failed: {comparison_result['error']}")
                    
            except Exception as e:
                st.error(f"❌ Error during scenario analysis: {str(e)}")
    
    # Display cached results
    if 'scenario_results' in st.session_state:
        display_scenario_results(st.session_state.scenario_results)

def display_scenario_results(results):
    st.success("✅ Scenario analysis completed!")
    
    # Comparison summary table
    st.subheader("📊 Scenario Comparison Summary")
    
    comparison_df = results['comparison_summary']
    
    # Style the dataframe
    styled_df = comparison_df.style.highlight_max(
        subset=['service_level', 'avg_inventory'], color='lightgreen'
    ).highlight_min(
        subset=['stockout_days', 'total_orders_placed'], color='lightgreen'
    ).format({
        'service_level': '{:.2f}%',
        'avg_inventory': '{:.0f}',
        'min_inventory': '{:.0f}',
        'max_inventory': '{:.0f}',
        'stockout_days': '{:.0f}',
        'total_orders_placed': '{:.0f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Key metrics comparison
    st.subheader("🎯 Key Metrics Comparison")
    
    num_scenarios = len(results['scenarios'])
    cols = st.columns(num_scenarios)
    
    for i, scenario in enumerate(results['scenarios']):
        with cols[i]:
            st.markdown(f"#### {scenario['name']}")
            
            metrics = scenario['result']['metrics']
            
            st.metric("Service Level", f"{metrics['service_level']:.1f}%")
            st.metric("Avg Inventory", f"{metrics['avg_inventory']:,.0f}")
            st.metric("Stockout Days", f"{metrics['stockout_days']}")
            st.metric("Orders Placed", f"{metrics['total_orders_placed']}")
            
            # Display parameters
            with st.expander("View Parameters"):
                params = scenario['result']['parameters']
                st.write(f"**Reorder Point:** {params['reorder_point']:,}")
                st.write(f"**Safety Stock:** {params['safety_stock']:,}")
                st.write(f"**Lead Time:** {params['lead_time_days']} days")
                st.write(f"**Order Quantity:** {params['order_quantity']:,}")
    
    # Inventory level comparison chart
    st.subheader("📈 Projected Inventory Levels Comparison")
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, scenario in enumerate(results['scenarios']):
        scenario_data = scenario['result']['scenario_data']
        
        fig.add_trace(go.Scatter(
            x=scenario_data['ds'],
            y=scenario_data['inventory_level'],
            mode='lines',
            name=scenario['name'],
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title="Inventory Level Projections Under Different Scenarios",
        xaxis_title="Date",
        yaxis_title="Inventory Level",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stockout comparison
    st.subheader("⚠️ Stockout Risk Analysis")
    
    stockout_data = []
    for scenario in results['scenarios']:
        scenario_data = scenario['result']['scenario_data']
        stockout_dates = scenario_data[scenario_data['stockout'] == 1]
        stockout_data.append({
            'Scenario': scenario['name'],
            'Stockout Days': len(stockout_dates),
            'Risk Level': 'High' if len(stockout_dates) > 10 else 'Medium' if len(stockout_dates) > 5 else 'Low'
        })
    
    stockout_df = pd.DataFrame(stockout_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(stockout_df, x='Scenario', y='Stockout Days',
                    title='Stockout Days by Scenario',
                    color='Risk Level',
                    color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(stockout_df, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Scenario Recommendations")
    
    # Find best scenario by service level
    best_service = comparison_df.loc[comparison_df['service_level'].idxmax()]
    best_inventory = comparison_df.loc[comparison_df['avg_inventory'].idxmin()]
    best_overall = comparison_df.loc[
        (comparison_df['service_level'] * 0.6 + 
         (100 - comparison_df['stockout_days']) * 0.4).idxmax()
    ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Best Service Level:** {best_service['scenario']}")
        st.write(f"Service Level: {best_service['service_level']:.1f}%")
    
    with col2:
        st.info(f"**Lowest Inventory Cost:** {best_inventory['scenario']}")
        st.write(f"Avg Inventory: {best_inventory['avg_inventory']:,.0f}")
    
    with col3:
        st.success(f"**Recommended Scenario:** {best_overall['scenario']}")
        st.write(f"Balanced performance across metrics")

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
            if not outliers_df.empty and 'value' in outliers_df.columns:
                fig = px.scatter(outliers_df, x='Date', y='value', 
                               title="Extreme GIK Values Over Time",
                               labels={'value': 'Total GIK'},
                               hover_data=['dataset', 'detection_methods'])
                st.plotly_chart(fig, use_container_width=True, key="gik_outliers_chart")
                
                # Show detailed table
                with st.expander("View Outlier Details"):
                    display_df = outliers_df[['Date', 'value', 'dataset', 'detection_methods']].copy()
                    display_df.columns = ['Date', 'GIK Value', 'Dataset', 'Detection Methods']
                    st.dataframe(display_df, use_container_width=True)
        
        # Display inventory outliers
        if outlier_summary.get('inventory_outliers') is not None and len(outlier_summary['inventory_outliers']) > 0:
            st.subheader("Inventory Level Outliers")
            outliers_df = pd.DataFrame(outlier_summary['inventory_outliers'])
            if not outliers_df.empty and 'value' in outliers_df.columns:
                fig = px.scatter(outliers_df, x='Date', y='value', 
                               title="Extreme Inventory Levels Over Time",
                               labels={'value': 'Inventory Level'},
                               hover_data=['dataset', 'detection_methods'])
                st.plotly_chart(fig, use_container_width=True, key="inventory_outliers_chart")
                
                with st.expander("View Outlier Details"):
                    display_df = outliers_df[['Date', 'value', 'dataset', 'detection_methods']].copy()
                    display_df.columns = ['Date', 'Inventory Level', 'Dataset', 'Detection Methods']
                    st.dataframe(display_df, use_container_width=True)
        
        # Display quantity outliers
        if outlier_summary.get('quantity_outliers') is not None and len(outlier_summary['quantity_outliers']) > 0:
            st.subheader("Quantity Outliers")
            outliers_df = pd.DataFrame(outlier_summary['quantity_outliers'])
            if not outliers_df.empty and 'value' in outliers_df.columns:
                fig = px.scatter(outliers_df, x='Date', y='value', 
                               title="Extreme Quantity Values Over Time",
                               labels={'value': 'Quantity'},
                               hover_data=['dataset', 'detection_methods'])
                st.plotly_chart(fig, use_container_width=True, key="quantity_outliers_chart")
                
                with st.expander("View Outlier Details"):
                    display_df = outliers_df[['Date', 'value', 'dataset', 'detection_methods']].copy()
                    display_df.columns = ['Date', 'Quantity', 'Dataset', 'Detection Methods']
                    st.dataframe(display_df, use_container_width=True)
    
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

def actual_vs_predicted_page():
    st.header("📊 Actual vs Predicted Comparison")
    
    # Check if test data exists
    if 'datasets' not in st.session_state or 'test_inventory' not in st.session_state.datasets:
        st.warning("⚠️ Please upload test data first in the Data Upload page!")
        st.info("Test data is used to compare actual inventory levels against model predictions.")
        return
    
    # Check if training data exists
    if 'train_inventory' not in st.session_state.datasets or 'tune_inventory' not in st.session_state.datasets:
        st.warning("⚠️ Please upload and process training and tuning data first!")
        return
    
    test_data = st.session_state.datasets['test_inventory']
    datasets = st.session_state.datasets
    
    st.subheader("📈 Generate Predictions for Test Period")
    
    # Model selection
    available_models = ['Prophet', 'SARIMA', 'XGBoost', 'All']
    selected_model = st.selectbox("Select Model for Comparison", available_models)
    
    if st.button("Generate Test Predictions"):
        with st.spinner(f"Generating {selected_model} predictions for test period..."):
            try:
                # Combine training and tuning data for forecasting
                # This ensures forecasts start from the most recent data point (tuning end)
                train_data = datasets['train_inventory'].copy()
                tune_data = datasets['tune_inventory'].copy()
                
                # Combine train and tune data to use as the complete historical dataset
                combined_data = pd.concat([train_data, tune_data], ignore_index=True)
                combined_data['Date'] = pd.to_datetime(combined_data['Date'])
                combined_data = combined_data.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
                
                # Get test period date range
                test_df = test_data.copy()
                test_df['Date'] = pd.to_datetime(test_df['Date'])
                test_start = test_df['Date'].min()
                test_end = test_df['Date'].max()
                
                # Get the end date of combined data
                combined_end = pd.to_datetime(combined_data['Date'].max())
                
                # Calculate periods needed to reach from combined data end to test end
                forecast_periods = (test_end - combined_end).days
                
                if forecast_periods <= 0:
                    st.error(f"❌ Test period must be after the latest training/tuning data. Latest data: {combined_end.date()}, Test starts: {test_start.date()}")
                    return
                
                st.info(f"Test period: {test_start.date()} to {test_end.date()}")
                st.info(f"Forecasting {forecast_periods} days from latest data ({combined_end.date()}) to test end ({test_end.date()})")
                
                # Extract features for training and tuning periods
                train_warehouse_features = datasets.get('train_warehouse_features')
                tune_warehouse_features = datasets.get('tune_warehouse_features')
                train_category_features = datasets.get('train_category_features')
                tune_category_features = datasets.get('tune_category_features')
                
                # Configure forecasting for test period
                forecast_config = {
                    'method': selected_model,
                    'periods': forecast_periods,
                    'confidence_level': 0.95,
                    'handle_outliers': True,
                    'scale_normalization': True,
                    'trend_detection': True,
                    'train_warehouse_features': train_warehouse_features,
                    'tune_warehouse_features': tune_warehouse_features,
                    'train_category_features': train_category_features,
                    'tune_category_features': tune_category_features
                }
                
                # Generate forecasts using combined data
                # Use combined data as both train and tune to ensure forecasts start from latest point
                results = st.session_state.forecasting_engine.generate_forecasts(
                    combined_data, combined_data, forecast_config
                )
                
                if results['success']:
                    st.session_state.test_forecast_results = results
                    st.success("✅ Test predictions generated successfully!")
                else:
                    st.error(f"❌ Forecast generation failed: {results.get('error')}")
                    return
                    
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                return
    
    # Display comparison if forecasts are available
    if 'test_forecast_results' not in st.session_state:
        st.info("👆 Click 'Generate Test Predictions' to create forecasts for comparison")
        return
    
    forecast_results = st.session_state.test_forecast_results
    
    st.subheader("📊 Comparison Results")
    
    # Get forecast results for selected model
    model_key = selected_model if selected_model in forecast_results['forecasts'] else list(forecast_results['forecasts'].keys())[0]
    model_result = forecast_results['forecasts'][model_key]
    forecast_df = model_result['forecast'].copy()
    
    # Prepare test data
    test_df = test_data.copy()
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    
    # Prepare forecast data
    if 'ds' in forecast_df.columns:
        forecast_df['Date'] = pd.to_datetime(forecast_df['ds'])
    
    # Get only future predictions (after the historical data)
    # Get the latest date from combined training+tuning data
    train_data = datasets['train_inventory'].copy()
    tune_data = datasets['tune_inventory'].copy()
    combined_data = pd.concat([train_data, tune_data], ignore_index=True)
    historical_end = pd.to_datetime(combined_data['Date'].max())
    forecast_df = forecast_df[forecast_df['Date'] > historical_end]
    
    # Merge actual and predicted
    comparison_df = test_df.merge(
        forecast_df[['Date', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='Date', 
        how='inner',
        suffixes=('', '_forecast')
    )
    
    if len(comparison_df) == 0:
        st.error("❌ No overlapping dates between test data and forecast")
        st.info(f"Test data range: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
        if len(forecast_df) > 0:
            st.info(f"Forecast range (future only): {forecast_df['Date'].min().date()} to {forecast_df['Date'].max().date()}")
        st.info(f"Historical data ended: {historical_end.date()}")
        return
    
    comparison_df = comparison_df.rename(columns={
        'Inventory_Level': 'Actual',
        'yhat': 'Predicted'
    })
    
    # Calculate metrics
    actual = comparison_df['Actual'].values
    predicted = comparison_df['Predicted'].values
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Comparison Points", len(comparison_df))
    with col2:
        st.metric("MAE", f"{mae:.2f}")
    with col3:
        st.metric("RMSE", f"{rmse:.2f}")
    with col4:
        st.metric("MAPE", f"{mape:.2f}%")
    
    # Visualization
    st.subheader("📉 Actual vs Predicted Over Time")
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['Actual'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['Predicted'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
    ))
    
    # Confidence intervals if available
    if 'yhat_lower' in comparison_df.columns and 'yhat_upper' in comparison_df.columns:
            fig.add_trace(go.Scatter(
                x=comparison_df['Date'],
                y=comparison_df['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                x=comparison_df['Date'],
                y=comparison_df['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255,0,0,0.2)',
                fill='tonexty',
                showlegend=True,
                name='Confidence Interval'
            ))
    
    fig.update_layout(
            title=f'{selected_model} Model: Actual vs Predicted Inventory Levels',
            xaxis_title='Date',
            yaxis_title='Inventory Level',
            hovermode='x unified',
            height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error analysis
    st.subheader("🔍 Error Analysis")
    
    comparison_df['Error'] = comparison_df['Actual'] - comparison_df['Predicted']
    comparison_df['Absolute_Error'] = np.abs(comparison_df['Error'])
    comparison_df['Percentage_Error'] = (comparison_df['Error'] / np.where(comparison_df['Actual'] != 0, comparison_df['Actual'], 1)) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
            # Error distribution
            fig_error = go.Figure()
            fig_error.add_trace(go.Histogram(
                x=comparison_df['Error'],
                nbinsx=30,
                name='Error Distribution'
            ))
            fig_error.update_layout(
                title='Prediction Error Distribution',
                xaxis_title='Error (Actual - Predicted)',
                yaxis_title='Frequency',
                height=400
            )
            st.plotly_chart(fig_error, use_container_width=True)
    
    with col2:
            # Error over time
            fig_error_time = go.Figure()
            fig_error_time.add_trace(go.Scatter(
                x=comparison_df['Date'],
                y=comparison_df['Error'],
                mode='lines+markers',
                name='Error',
                line=dict(color='orange', width=2)
            ))
            fig_error_time.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_error_time.update_layout(
                title='Prediction Error Over Time',
                xaxis_title='Date',
                yaxis_title='Error',
                height=400
            )
            st.plotly_chart(fig_error_time, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison Table")
    
    display_df = comparison_df[['Date', 'Actual', 'Predicted', 'Error', 'Absolute_Error', 'Percentage_Error']].copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df = display_df.round(2)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
            label="📥 Download Comparison Data",
            data=csv,
            file_name=f"actual_vs_predicted_{selected_model}.csv",
            mime="text/csv"
        )

def business_insights_page():
    st.header("💡 Business Insights & Recommendations")
    
    if 'datasets' not in st.session_state:
        st.warning("⚠️ Please upload and process data first!")
        return
    
    datasets = st.session_state.datasets
    
    # Extract unique warehouses and categories from the data
    warehouses = set()
    categories = set()
    
    for dataset_name in ['train_inflows', 'tune_inflows', 'train_outflows', 'tune_outflows']:
        if dataset_name in datasets:
            df = datasets[dataset_name]
            if 'Warehouse' in df.columns:
                warehouses.update(df['Warehouse'].dropna().unique())
            if 'Category' in df.columns:
                categories.update(df['Category'].dropna().unique())
    
    # Filter controls
    st.subheader("🔍 Filter by Dimension")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if warehouses:
            warehouse_options = ['All Warehouses'] + sorted(list(warehouses))
            selected_warehouses = st.multiselect(
                "Select Warehouses",
                options=warehouse_options,
                default=['All Warehouses'],
                help="Compare insights across specific warehouses"
            )
        else:
            selected_warehouses = ['All Warehouses']
            st.info("No warehouse data available")
    
    with col2:
        if categories:
            category_options = ['All Categories'] + sorted(list(categories))
            selected_categories = st.multiselect(
                "Select Categories",
                options=category_options,
                default=['All Categories'],
                help="Compare insights across specific product categories"
            )
        else:
            selected_categories = ['All Categories']
            st.info("No category data available")
    
    with col3:
        st.markdown("#### Active Filters")
        filter_warehouse = 'All Warehouses' not in selected_warehouses and selected_warehouses
        filter_category = 'All Categories' not in selected_categories and selected_categories
        
        if filter_warehouse:
            st.metric("Warehouses", len(selected_warehouses))
        if filter_category:
            st.metric("Categories", len(selected_categories))
        if not filter_warehouse and not filter_category:
            st.info("No filters active")
    
    st.divider()
    
    # Apply filters to datasets
    filtered_datasets = datasets.copy()
    
    if filter_warehouse or filter_category:
        for dataset_name in ['train_inflows', 'tune_inflows', 'train_outflows', 'tune_outflows']:
            if dataset_name in filtered_datasets:
                df = filtered_datasets[dataset_name].copy()
                
                # Apply warehouse filter
                if filter_warehouse and 'Warehouse' in df.columns:
                    df = df[df['Warehouse'].isin(selected_warehouses)]
                
                # Apply category filter
                if filter_category and 'Category' in df.columns:
                    df = df[df['Category'].isin(selected_categories)]
                
                filtered_datasets[dataset_name] = df
        
        # Show filter summary
        st.info(f"📊 Generating insights for {len(selected_warehouses) if filter_warehouse else 'all'} warehouse(s) and {len(selected_categories) if filter_category else 'all'} category(ies)")
    
    # Generate insights
    if st.button("Generate Business Insights", type="primary"):
        with st.spinner("Generating actionable insights..."):
            try:
                insights = st.session_state.insights_generator.generate_insights(
                    filtered_datasets, 
                    st.session_state.get('forecast_results'),
                    st.session_state.get('anomaly_results')
                )
                
                if insights['success']:
                    # Store filters with insights
                    insights['filters'] = {
                        'warehouses': selected_warehouses if filter_warehouse else ['All Warehouses'],
                        'categories': selected_categories if filter_category else ['All Categories']
                    }
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
    
    # Display active filters if present
    if 'filters' in insights:
        filters = insights['filters']
        warehouses = filters.get('warehouses', ['All Warehouses'])
        categories = filters.get('categories', ['All Categories'])
        
        if warehouses != ['All Warehouses'] or categories != ['All Categories']:
            with st.expander("🔍 Active Filters & Comparison Scope", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📦 Warehouses:**")
                    for wh in warehouses:
                        st.markdown(f"• {wh}")
                with col2:
                    st.markdown("**🏷️ Categories:**")
                    for cat in categories:
                        st.markdown(f"• {cat}")
    
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

def partner_analytics_page():
    st.header("🤝 Partner Analytics Dashboard")
    
    if 'datasets' not in st.session_state:
        st.warning("⚠️ Please upload and process data first!")
        return
    
    datasets = st.session_state.datasets
    
    # Check if outflows data is available
    if 'tune_outflows' not in datasets:
        st.error("❌ Partner analytics requires outflows data. Please upload tuning outflows data.")
        return
    
    outflows_df = datasets['tune_outflows'].copy()
    
    # Validate required columns
    required_columns = ['Partner', 'Date', 'Quantity']
    missing_columns = [col for col in required_columns if col not in outflows_df.columns]
    
    if missing_columns:
        st.error(f"❌ Outflows data is missing required columns: {', '.join(missing_columns)}")
        st.info("Partner analytics requires at minimum: Partner, Date, and Quantity columns.")
        return
    
    # Convert date column with error handling
    try:
        outflows_df['Date'] = pd.to_datetime(outflows_df['Date'])
    except Exception as e:
        st.error(f"❌ Error parsing Date column: {str(e)}")
        st.info("Please ensure the Date column contains valid date values.")
        return
    
    # Validate numeric columns
    try:
        outflows_df['Quantity'] = pd.to_numeric(outflows_df['Quantity'], errors='coerce')
        if outflows_df['Quantity'].isna().all():
            st.error("❌ Quantity column does not contain valid numeric values.")
            return
        
        # Remove negative and NaN quantities
        initial_count = len(outflows_df)
        outflows_df = outflows_df[outflows_df['Quantity'] > 0].dropna(subset=['Quantity'])
        
        if len(outflows_df) == 0:
            st.error("❌ No valid positive quantity values found in the data.")
            return
        
        if len(outflows_df) < initial_count * 0.5:
            st.warning(f"⚠️ {initial_count - len(outflows_df)} rows with invalid/negative quantities were removed.")
            
    except Exception as e:
        st.error(f"❌ Error processing Quantity column: {str(e)}")
        return
    
    # Validate optional numeric columns
    if 'Total_GIK' in outflows_df.columns:
        try:
            outflows_df['Total_GIK'] = pd.to_numeric(outflows_df['Total_GIK'], errors='coerce')
        except:
            st.warning("⚠️ Total_GIK column could not be processed as numeric. It will be excluded from analysis.")
            outflows_df = outflows_df.drop(columns=['Total_GIK'])
    
    if 'Total_EI' in outflows_df.columns:
        try:
            outflows_df['Total_EI'] = pd.to_numeric(outflows_df['Total_EI'], errors='coerce')
        except:
            st.warning("⚠️ Total_EI column could not be processed as numeric. It will be excluded from analysis.")
            outflows_df = outflows_df.drop(columns=['Total_EI'])
    
    # Overview metrics
    st.subheader("📊 Partner Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_partners = outflows_df['Partner'].nunique()
        st.metric("Total Partners", total_partners)
    
    with col2:
        total_shipments = len(outflows_df)
        st.metric("Total Shipments", f"{total_shipments:,}")
    
    with col3:
        total_quantity = outflows_df['Quantity'].sum()
        st.metric("Total Quantity Shipped", f"{total_quantity:,.0f}")
    
    with col4:
        avg_shipment_size = outflows_df['Quantity'].mean()
        st.metric("Avg Shipment Size", f"{avg_shipment_size:,.0f}")
    
    st.divider()
    
    # Top partners analysis
    st.subheader("🏆 Top Partners by Volume")
    
    partner_volumes = outflows_df.groupby('Partner')['Quantity'].agg(['sum', 'count', 'mean']).reset_index()
    partner_volumes.columns = ['Partner', 'Total_Quantity', 'Shipment_Count', 'Avg_Shipment']
    partner_volumes = partner_volumes.sort_values('Total_Quantity', ascending=False)
    
    # Show top 10 partners
    top_n = st.slider("Number of top partners to display", 5, 20, 10)
    top_partners = partner_volumes.head(top_n)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(top_partners, x='Partner', y='Total_Quantity',
                    title=f'Top {top_n} Partners by Total Quantity',
                    labels={'Total_Quantity': 'Total Quantity Shipped'},
                    color='Total_Quantity',
                    color_continuous_scale='Blues')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(top_partners, x='Partner', y='Shipment_Count',
                    title=f'Top {top_n} Partners by Shipment Count',
                    labels={'Shipment_Count': 'Number of Shipments'},
                    color='Shipment_Count',
                    color_continuous_scale='Greens')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Partner share pie chart
    st.markdown("#### Market Share Distribution")
    fig = px.pie(top_partners, values='Total_Quantity', names='Partner',
                title=f'Top {top_n} Partners Volume Share')
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Temporal patterns
    st.subheader("📈 Shipment Trends Over Time")
    
    # Select partner for detailed analysis
    selected_partner = st.selectbox("Select partner for detailed analysis", 
                                    ['All Partners'] + sorted(outflows_df['Partner'].unique().tolist()))
    
    if selected_partner == 'All Partners':
        filtered_df = outflows_df.copy()
        title_suffix = "All Partners"
    else:
        filtered_df = outflows_df[outflows_df['Partner'] == selected_partner].copy()
        title_suffix = selected_partner
    
    if len(filtered_df) == 0:
        st.warning(f"⚠️ No data available for {title_suffix}")
    else:
        # Daily shipment trends
        daily_shipments = filtered_df.groupby('Date')['Quantity'].sum().reset_index()
        
        if len(daily_shipments) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_shipments['Date'],
                y=daily_shipments['Quantity'],
                mode='lines+markers',
                name='Daily Shipments',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f'Daily Shipment Volume - {title_suffix}',
                xaxis_title='Date',
                yaxis_title='Quantity Shipped',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly aggregation
        filtered_df['Month'] = filtered_df['Date'].dt.to_period('M').astype(str)
        monthly_shipments = filtered_df.groupby('Month')['Quantity'].sum().reset_index()
        
        if len(monthly_shipments) > 0:
            fig = px.bar(monthly_shipments, x='Month', y='Quantity',
                        title=f'Monthly Shipment Volume - {title_suffix}',
                        labels={'Quantity': 'Total Quantity'},
                        color='Quantity',
                        color_continuous_scale='Viridis')
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Partner comparison
    st.subheader("⚖️ Partner Comparison")
    
    # Select multiple partners for comparison
    compare_partners = st.multiselect(
        "Select partners to compare (max 5)",
        sorted(outflows_df['Partner'].unique().tolist()),
        default=top_partners['Partner'].head(3).tolist()[:5]
    )
    
    if compare_partners:
        comparison_df = outflows_df[outflows_df['Partner'].isin(compare_partners)].copy()
        
        if len(comparison_df) == 0:
            st.warning("⚠️ No data available for selected partners.")
        else:
            comparison_df['Month'] = comparison_df['Date'].dt.to_period('M').astype(str)
            
            monthly_comparison = comparison_df.groupby(['Month', 'Partner'])['Quantity'].sum().reset_index()
            
            if len(monthly_comparison) > 0:
                fig = px.line(monthly_comparison, x='Month', y='Quantity', color='Partner',
                             title='Partner Shipment Trends Comparison',
                             labels={'Quantity': 'Monthly Quantity'},
                             markers=True)
                
                st.plotly_chart(fig, use_container_width=True)
        
            # Partner performance metrics
            st.markdown("#### Partner Performance Metrics")
            
            partner_metrics = []
            for partner in compare_partners:
                partner_data = outflows_df[outflows_df['Partner'] == partner]
                
                if len(partner_data) > 0:
                    metrics = {
                        'Partner': partner,
                        'Total Shipments': len(partner_data),
                        'Total Quantity': partner_data['Quantity'].sum(),
                        'Avg Shipment Size': partner_data['Quantity'].mean()
                    }
                    
                    # Add optional columns if they exist
                    if 'Total_GIK' in outflows_df.columns and 'Total_GIK' in partner_data.columns:
                        try:
                            gik_sum = pd.to_numeric(partner_data['Total_GIK'], errors='coerce').sum()
                            if not pd.isna(gik_sum):
                                metrics['Total GIK Value'] = gik_sum
                        except:
                            pass
                    
                    if 'Total_EI' in outflows_df.columns and 'Total_EI' in partner_data.columns:
                        try:
                            ei_sum = pd.to_numeric(partner_data['Total_EI'], errors='coerce').sum()
                            if not pd.isna(ei_sum):
                                metrics['Total EI Value'] = ei_sum
                        except:
                            pass
                    
                    partner_metrics.append(metrics)
            
            if len(partner_metrics) > 0:
                metrics_df = pd.DataFrame(partner_metrics)
                
                # Build format dict dynamically based on available columns
                format_dict = {
                    'Total Quantity': '{:,.0f}',
                    'Avg Shipment Size': '{:,.0f}'
                }
                if 'Total GIK Value' in metrics_df.columns:
                    format_dict['Total GIK Value'] = '${:,.2f}'
                if 'Total EI Value' in metrics_df.columns:
                    format_dict['Total EI Value'] = '${:,.2f}'
                
                st.dataframe(metrics_df.style.format(format_dict), use_container_width=True)
            else:
                st.info("No metrics available for selected partners.")
    
    st.divider()
    
    # Detailed partner table
    st.subheader("📋 All Partners Summary")
    st.dataframe(partner_volumes.style.format({
        'Total_Quantity': '{:,.0f}',
        'Avg_Shipment': '{:,.0f}'
    }), use_container_width=True)

def automated_alerts_page():
    st.header("🚨 Automated Inventory Alerts")
    
    if 'datasets' not in st.session_state:
        st.warning("⚠️ Please upload and process data first!")
        return
    
    if 'forecast_results' not in st.session_state:
        st.warning("⚠️ Please generate forecasts first before configuring alerts!")
        st.info("Navigate to the 'ML Forecasting' page to generate forecasts.")
        return
    
    st.write("Configure automated alerts for predicted stockout and overstock situations.")
    
    # Alert configuration
    st.subheader("⚙️ Alert Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stockout_threshold = st.slider(
            "Stockout Threshold (%)",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            help="Alert when predicted inventory falls below this % of average"
        )
        
        critical_threshold = st.slider(
            "Critical Threshold (%)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Critical alert when inventory falls below this % of average"
        )
    
    with col2:
        overstock_threshold = st.slider(
            "Overstock Threshold (%)",
            min_value=150,
            max_value=400,
            value=250,
            step=10,
            help="Alert when predicted inventory exceeds this % of average"
        )
        
        warning_days = st.slider(
            "Warning Window (days)",
            min_value=7,
            max_value=60,
            value=14,
            step=1,
            help="Generate alerts for predictions within this many days"
        )
    
    with col3:
        st.markdown("#### Alert Options")
        enable_critical = st.checkbox("Enable Critical Alerts", value=True)
        enable_stockout = st.checkbox("Enable Stockout Warnings", value=True)
        enable_overstock = st.checkbox("Enable Overstock Warnings", value=True)
    
    # Generate alerts button
    if st.button("Generate Inventory Alerts", type="primary"):
        with st.spinner("Analyzing forecasts and generating alerts..."):
            try:
                alert_config = {
                    'stockout_threshold_ratio': stockout_threshold / 100,
                    'critical_stockout_ratio': critical_threshold / 100,
                    'overstock_threshold_ratio': overstock_threshold / 100,
                    'days_ahead_warning': warning_days
                }
                
                alerts = st.session_state.insights_generator.generate_inventory_alerts(
                    st.session_state.forecast_results,
                    st.session_state.datasets,
                    alert_config
                )
                
                if alerts['success']:
                    st.session_state.inventory_alerts = alerts
                    display_inventory_alerts(alerts, enable_critical, enable_stockout, enable_overstock)
                else:
                    st.error(f"❌ Alert generation failed: {alerts['error']}")
                    
            except Exception as e:
                st.error(f"❌ Error generating alerts: {str(e)}")
    
    # Display cached alerts
    if 'inventory_alerts' in st.session_state:
        alerts = st.session_state.inventory_alerts
        display_inventory_alerts(
            alerts,
            st.session_state.get('enable_critical', True),
            st.session_state.get('enable_stockout', True),
            st.session_state.get('enable_overstock', True)
        )

def display_inventory_alerts(alerts, show_critical=True, show_stockout=True, show_overstock=True):
    st.success("✅ Inventory alerts generated!")
    
    # Summary metrics
    st.subheader("📊 Alert Summary")
    
    summary = alerts['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Critical Alerts", summary['total_critical_alerts'], 
                 delta="High Priority" if summary['total_critical_alerts'] > 0 else None,
                 delta_color="inverse")
    
    with col2:
        st.metric("Stockout Warnings", summary['total_stockout_alerts'])
    
    with col3:
        st.metric("Overstock Warnings", summary['total_overstock_alerts'])
    
    with col4:
        st.metric("Total Alerts", 
                 summary['total_critical_alerts'] + summary['total_stockout_alerts'] + summary['total_overstock_alerts'])
    
    # Display thresholds
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Baseline Average", f"{summary['baseline_average']:,.0f}")
    with col2:
        st.metric("Stockout Threshold", f"{summary['stockout_threshold']:,.0f}")
    with col3:
        st.metric("Overstock Threshold", f"{summary['overstock_threshold']:,.0f}")
    
    # Critical alerts
    if show_critical and alerts['critical_alerts']:
        st.subheader("🚨 Critical Stockout Alerts")
        
        critical_df = pd.DataFrame(alerts['critical_alerts'])
        critical_df = critical_df.sort_values('days_until')
        
        for _, alert in critical_df.iterrows():
            st.error(f"**CRITICAL:** {alert['message']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"📅 Date: {alert['date']}")
            with col2:
                st.write(f"⏱️ Days Until: {alert['days_until']}")
            with col3:
                st.write(f"🤖 Model: {alert['model']}")
    
    # Stockout alerts
    if show_stockout and alerts['stockout_alerts']:
        st.subheader("⚠️ Stockout Warnings")
        
        stockout_df = pd.DataFrame(alerts['stockout_alerts'])
        
        # Create timeline visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=stockout_df['date'],
            y=stockout_df['predicted_level'],
            mode='markers',
            name='Predicted Stockouts',
            marker=dict(size=10, color='orange'),
            text=stockout_df['message'],
            hoverinfo='text'
        ))
        
        fig.add_hline(y=summary['stockout_threshold'], line_dash="dash", 
                     line_color="red", annotation_text="Stockout Threshold")
        
        fig.update_layout(
            title="Predicted Stockout Timeline",
            xaxis_title="Date",
            yaxis_title="Predicted Inventory Level",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert table
        st.dataframe(
            stockout_df[['date', 'predicted_level', 'threshold', 'days_until', 'model']],
            use_container_width=True
        )
    
    # Overstock alerts
    if show_overstock and alerts['overstock_alerts']:
        st.subheader("📦 Overstock Warnings")
        
        overstock_df = pd.DataFrame(alerts['overstock_alerts'])
        
        # Create timeline visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=overstock_df['date'],
            y=overstock_df['predicted_level'],
            mode='markers',
            name='Predicted Overstocks',
            marker=dict(size=10, color='blue'),
            text=overstock_df['message'],
            hoverinfo='text'
        ))
        
        fig.add_hline(y=summary['overstock_threshold'], line_dash="dash", 
                     line_color="purple", annotation_text="Overstock Threshold")
        
        fig.update_layout(
            title="Predicted Overstock Timeline",
            xaxis_title="Date",
            yaxis_title="Predicted Inventory Level",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert table
        st.dataframe(
            overstock_df[['date', 'predicted_level', 'threshold', 'days_until', 'model']],
            use_container_width=True
        )
    
    # Recommendations
    if alerts['recommendations']:
        st.subheader("💡 Recommended Actions")
        
        for i, recommendation in enumerate(alerts['recommendations'], 1):
            if 'URGENT' in recommendation:
                st.error(f"{i}. {recommendation}")
            elif 'Multiple' in recommendation:
                st.warning(f"{i}. {recommendation}")
            else:
                st.info(f"{i}. {recommendation}")

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
