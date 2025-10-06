# AI-Powered Inventory Forecasting Platform

## Overview

This is an AI-powered inventory forecasting platform built for the 2025 SuiteWorld Hackathon 4Good Challenge. The application analyzes inventory data, inflows, and outflows to generate predictive forecasts, detect anomalies, and provide actionable business insights for inventory optimization. It supports multiple machine learning models including Prophet, SARIMA, and ensemble methods to forecast inventory levels while handling temporal gaps and scale shifts in data.

The platform features a Streamlit-based web interface for data upload, validation, statistical analysis, forecasting with warehouse and category dimensions, test data evaluation, actual vs predicted comparison, scenario modeling, anomaly detection, and automated reporting.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### API Layer
**Technology**: FastAPI REST API
- **Runs on port 8000** alongside the Streamlit interface (port 5000)
- **RESTful endpoints** for programmatic access to all datasets:
  - `GET /` - API information and endpoint listing
  - `GET /health` - Health check endpoint
  - `GET /datasets` - List all available datasets with metadata
  - `GET /summary` - Aggregate statistics across all periods
  - `GET /inventory/{period}` - Retrieve inventory data with pagination and date filtering
  - `GET /inflows/{period}` - Retrieve inflows data with filters (warehouse, category, date range)
  - `GET /outflows/{period}` - Retrieve outflows data with filters (warehouse, category, partner, date range)
- **Query parameters** support: limit, offset, warehouse, category, partner, date_from, date_to
- **JSON response format** with metadata (total_records, returned_records, filters applied)
- **Data persistence** via JSON files in api_data/ directory shared with Streamlit

### Frontend Architecture
**Technology**: Streamlit web framework
- **Multi-page application structure** with navigation sidebar for different functional areas
- **Interactive visualizations** using Plotly for time series charts, distributions, and forecasting results
- **Session state management** to persist data and models across page navigation
- **Real-time data validation** and error handling with user-friendly feedback
- **File upload interface** supporting CSV and Excel formats for training, tuning, and test datasets
- **Actual vs Predicted comparison** with test data evaluation, performance metrics (MAE, RMSE, MAPE), time series visualization, error analysis, and downloadable results

**Design Pattern**: Component-based architecture where each major functionality (data upload, forecasting, insights, etc.) is implemented as a separate page function, promoting modularity and maintainability.

### Backend Architecture
**Core Components**:

1. **Data Processing Layer** (`DataHandler`, `DataProcessor`, `DataLoader`)
   - **Schema validation** with predefined rules for inventory, inflows, and outflows datasets
   - **Data cleaning pipeline** handling missing values, duplicates, date parsing, and outlier removal
   - **Multi-period support** for training, tuning, and test datasets to handle temporal gaps
   - **Feature extraction** for warehouse and category dimensions:
     - **Warehouse features**: Daily aggregates (Total_Warehouse_Inflows, Total_Warehouse_Outflows, Active_Warehouses, Warehouse_Count)
     - **Category features**: Daily aggregates (Total_Category_Inflows, Total_Category_Outflows, Active_Categories, Category_Diversity)
   - Uses pandas for data manipulation with openpyxl for Excel support

2. **Forecasting Engine** (`ForecastingEngine`, `ForecastingModel`)
   - **Multi-model approach**: Prophet (time series with seasonality), SARIMA (statistical forecasting), ARIMA, XGBoost, and Linear/Random Forest models
   - **Adaptive training** with automatic model selection and retraining capabilities
   - **Handles temporal gaps and scale shifts** between training (2017-2018) and tuning (2023) periods
   - **Warehouse and category features as regressors**: 
     - Prophet models use warehouse and category features as external regressors with historical value preservation
     - XGBoost models automatically incorporate these features through the dataframe
   - **Test period forecasting**: Combines training+tuning data to anchor forecasts from the latest historical point (2023) for accurate test period predictions
   - **Time series preparation** with outlier handling and preprocessing using StandardScaler/MinMaxScaler
   - **Performance metrics** including MAE, MSE, MAPE for model evaluation
   - **Confidence intervals** for forecast uncertainty quantification

3. **Anomaly Detection** (`AnomalyDetector`)
   - **Outlier detection** using IQR method and Isolation Forest algorithm
   - **Data drift analysis** comparing training vs tuning period distributions using statistical tests (KS test, z-scores)
   - **Scale shift detection** to identify significant changes in data magnitude
   - **Multiple detection methods**: extreme value detection, distribution change analysis, and pattern anomalies

4. **Insights & Analytics** (`InsightsGenerator`, `InsightsEngine`)
   - **Risk assessment** with configurable thresholds for inventory levels and forecast uncertainty
   - **Business recommendations** including reorder points, safety stock calculations, and optimization opportunities
   - **Category and warehouse analysis** for performance benchmarking
   - **Partner analytics** for outflow optimization

5. **Statistical Analysis** (`StatisticalAnalyzer`)
   - **Trend analysis** using linear regression and seasonal decomposition
   - **Distribution analysis** with statistical tests for comparing periods
   - **Year-over-year comparisons** and temporal pattern detection
   - **Turnover metrics** and inventory efficiency calculations

6. **Reporting** (`ReportGenerator`)
   - **Multi-format export**: CSV, Excel (multi-sheet), and PDF reports
   - **Automated report generation** with timestamp tracking
   - **Data aggregation** from multiple sources with source indicators

**Design Rationale**: 
- Separation of concerns with distinct modules for data processing, modeling, analysis, and reporting
- Graceful degradation when optional libraries (Prophet, XGBoost, FPDF) are unavailable
- Error handling with try-catch blocks and validation results dictionaries
- Flexible configuration via dictionary-based parameters

### Data Storage Solutions
**Current Implementation**: Hybrid approach with session state and JSON persistence
- **Session state**: Data persists in Streamlit during user sessions for UI operations
- **JSON persistence**: Datasets automatically saved to `api_data/` directory for API access
- **Format**: Each dataset stored as {period}_{type}.json (e.g., train_inventory.json)
- **Automatic sync**: Data saved whenever uploaded or loaded through Streamlit interface

**Data Flow**:
1. User uploads CSV/Excel files or loads sample data
2. Data validated against predefined schemas
3. Cleaned and stored in session state
4. **Automatically persisted to JSON files** for API consumption
5. Accessed by forecasting, analysis, and insights modules
6. Available via REST API for external integrations
7. Results exported as downloadable files

**Future Consideration**: Architecture supports adding database persistence (e.g., PostgreSQL with Drizzle ORM) for multi-user scenarios and historical analysis.

### Processing Pipeline
**Multi-stage data processing**:
1. **Validation**: Schema checking, required columns verification
2. **Cleaning**: Date parsing, missing value handling, duplicate removal
3. **Preprocessing**: Outlier detection, normalization/scaling
4. **Feature Engineering**: Time series preparation, external regressor integration
5. **Modeling**: Training, validation, and forecasting
6. **Analysis**: Statistical testing, anomaly detection, insights generation

## External Dependencies

### Core Libraries
- **pandas**: Data manipulation and time series handling
- **numpy**: Numerical computations and array operations
- **scipy**: Statistical tests and scientific computing
- **scikit-learn**: Machine learning models (Isolation Forest, Random Forest, Linear Regression) and preprocessing (StandardScaler, RobustScaler, MinMaxScaler)

### Machine Learning & Forecasting
- **Prophet** (optional): Facebook's time series forecasting with seasonality support
- **statsmodels**: ARIMA, SARIMA models and seasonal decomposition
- **XGBoost** (optional): Gradient boosting for ensemble forecasting

### Visualization
- **Plotly Express & Graph Objects**: Interactive charts and dashboards
- **Streamlit**: Web application framework

### File Processing
- **openpyxl**: Excel file reading and writing
- **FPDF** (optional): PDF report generation

### Utilities
- **warnings**: Suppress library warnings for cleaner output
- **logging**: Error tracking and debugging (infrastructure present but minimal usage)
- **io/zipfile**: In-memory file operations and compression

### Design Note
The application uses conditional imports with availability flags (e.g., `PROPHET_AVAILABLE`, `XGBOOST_AVAILABLE`) to handle missing optional dependencies gracefully, ensuring core functionality works even without advanced ML libraries.