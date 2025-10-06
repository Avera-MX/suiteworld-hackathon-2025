# 🚀 AI-Powered Inventory Forecasting Platform

> Built for the 2025 SuiteWorld Hackathon 4Good Challenge

An intelligent inventory management system that uses advanced machine learning models to forecast inventory levels, detect anomalies, and provide actionable business insights. The platform combines powerful forecasting algorithms with an intuitive web interface and a comprehensive REST API.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![Prophet](https://img.shields.io/badge/Prophet-ML-orange.svg)

---

## ✨ Features

### 📊 Web Interface (Streamlit)
- **Data Upload & Validation** - Support for CSV and Excel files with automatic schema validation
- **Statistical Analysis** - Comprehensive trend analysis, distribution comparisons, and temporal patterns
- **ML Forecasting** - Prophet-based time series forecasting with confidence intervals
- **Actual vs Predicted** - Compare test data against predictions with performance metrics (MAE, RMSE, MAPE)
- **Anomaly Detection** - Outlier detection using IQR, Z-scores, and Isolation Forest algorithms
- **Business Insights** - Automated recommendations for reorder points, safety stock, and optimization
- **Partner Analytics** - Performance analysis across distribution partners
- **Interactive Visualizations** - Plotly-powered charts and dashboards

### 🔌 REST API (FastAPI)
- **Data Endpoints** - Access inventory, inflows, and outflows data with filtering and pagination
- **Forecast Generation** - Programmatic access to ML forecasting with configurable parameters
- **Summary Statistics** - Aggregate metrics across all time periods
- **Health Monitoring** - Health check and status endpoints

### 🤖 Machine Learning
- **Prophet Model** - Time series forecasting with seasonality detection
- **Warehouse Features** - Daily aggregates (inflows, outflows, active warehouses)
- **Category Features** - Product category-level analysis and forecasting
- **Confidence Intervals** - Uncertainty quantification for all predictions
- **Multi-period Support** - Handles temporal gaps between training (2017-2018) and tuning (2023) periods

---

## 🛠️ Tech Stack

### Backend
- **Python 3.11** - Core programming language
- **FastAPI** - High-performance REST API framework
- **Streamlit** - Interactive web application framework

### Machine Learning & Data Processing
- **Prophet** - Facebook's time series forecasting library
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning utilities (Isolation Forest, preprocessing)
- **statsmodels** - Statistical models (ARIMA, SARIMA)
- **XGBoost** (optional) - Gradient boosting for ensemble forecasting

### Visualization
- **Plotly** - Interactive charts and dashboards
- **Plotly Express** - High-level plotting interface

### Data Persistence
- **JSON** - File-based storage for API data sharing
- **Session State** - In-memory storage for Streamlit sessions

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd inventory-forecasting-platform
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
- fastapi
- uvicorn
- streamlit
- pandas
- numpy
- plotly
- prophet
- scikit-learn
- statsmodels
- xgboost (optional)
- openpyxl
- fpdf (optional)

3. **Prepare sample data**

Place your data files in the `attached_assets/` directory:
- Training inventory: `train_inventory.csv`
- Training inflows: `train_inflows.csv`
- Training outflows: `train_outflows.csv` (optional)
- Tuning inventory: `tune_inventory.csv`
- Tuning inflows: `tune_inflows.csv`
- Tuning outflows: `tune_outflows.csv`

### Running the Application

#### Start the Web Interface (Streamlit)
```bash
streamlit run app.py --server.port 5000
```

Access the application at: `http://localhost:5000`

#### Start the API Server (FastAPI)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

API documentation available at: `http://localhost:8000/docs`

#### Run Both Services
The project is configured to run both services simultaneously:
- **Streamlit UI**: Port 5000
- **FastAPI Server**: Port 8000

---

## 📁 Project Structure

```
inventory-forecasting-platform/
├── app.py                      # Main Streamlit application
├── api.py                      # FastAPI REST API server
├── data_handler.py             # Data loading, validation, and processing
├── data_storage.py             # JSON persistence layer
├── forecasting_engine.py       # ML forecasting orchestration
├── forecasting_model.py        # Model implementations (Prophet, SARIMA, etc.)
├── anomaly_detector.py         # Anomaly detection algorithms
├── insights_generator.py       # Business insights and recommendations
├── report_generator.py         # Report export functionality
├── utils.py                    # Utility functions
├── api_data/                   # JSON data storage for API
├── attached_assets/            # Sample data files
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

### Key Modules

#### `data_handler.py`
- Schema validation for inventory, inflows, and outflows
- Data cleaning pipeline (missing values, duplicates, outliers)
- Feature extraction for warehouse and category dimensions

#### `forecasting_engine.py`
- Multi-model forecasting support
- Warehouse and category feature integration
- Test period prediction with combined train+tune data

#### `anomaly_detector.py`
- IQR-based outlier detection
- Isolation Forest for pattern anomalies
- Data drift analysis between periods

#### `api.py`
- RESTful endpoints for all datasets
- Forecast generation with validation
- Pagination and filtering support

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-06T20:00:00.000000"
}
```

#### 2. List All Datasets
```http
GET /datasets
```

**Response:**
```json
{
  "datasets": [
    {
      "period": "train",
      "type": "inventory",
      "filename": "train_inventory.json",
      "size_bytes": 142414,
      "row_count": 2191
    }
  ],
  "count": 5
}
```

#### 3. Get Inventory Data
```http
GET /inventory/{period}?limit=100&offset=0&date_from=2023-01-01&date_to=2023-12-31
```

**Parameters:**
- `period` (path): train, tune, or test
- `limit` (query): Maximum records to return
- `offset` (query): Pagination offset
- `date_from` (query): Start date filter (YYYY-MM-DD)
- `date_to` (query): End date filter (YYYY-MM-DD)

**Response:**
```json
{
  "period": "train",
  "type": "inventory",
  "total_records": 2191,
  "returned_records": 100,
  "data": [
    {
      "Date": "2017-01-01",
      "Inventory_Level": 29154
    }
  ]
}
```

#### 4. Get Inflows Data
```http
GET /inflows/{period}?warehouse=WarehouseA&category=Electronics
```

**Additional Parameters:**
- `warehouse` (query): Filter by warehouse name
- `category` (query): Filter by product category

#### 5. Get Outflows Data
```http
GET /outflows/{period}?partner=Partner1
```

**Additional Parameters:**
- `partner` (query): Filter by distribution partner

#### 6. Get Summary Statistics
```http
GET /summary
```

**Response:**
```json
{
  "inventory": {
    "train": {
      "record_count": 2191,
      "avg_inventory": 608101.73,
      "min_inventory": 29154.0,
      "max_inventory": 971128.0
    }
  },
  "inflows": {...},
  "outflows": {...}
}
```

#### 7. Generate Forecast
```http
POST /forecast
Content-Type: application/json

{
  "model": "prophet",
  "periods": 30,
  "use_tuning_data": true,
  "confidence_level": 0.95
}
```

**Request Parameters:**
- `model` (string): Forecasting model - currently "prophet" supported
- `periods` (integer): Number of days to forecast (1-730)
- `use_tuning_data` (boolean): Include tuning period data
- `confidence_level` (float): Prediction interval confidence (0.5-0.99)

**Response:**
```json
{
  "success": true,
  "model": "Prophet",
  "periods": 30,
  "forecast_start_date": "2017-01-01",
  "forecast_end_date": "2023-01-30",
  "metrics": {
    "rmse": 0.99,
    "mae": 0.93,
    "mape": 48.35
  },
  "forecast_data": [
    {
      "Date": "2023-01-01",
      "yhat": 936547.82,
      "yhat_lower": 818234.56,
      "yhat_upper": 1054861.08
    }
  ]
}
```

### Example API Calls

**Using cURL:**
```bash
# Get inventory data
curl "http://localhost:8000/inventory/train?limit=10"

# Generate 60-day forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prophet",
    "periods": 60,
    "use_tuning_data": true,
    "confidence_level": 0.95
  }'
```

**Using Python:**
```python
import requests

# Get summary statistics
response = requests.get("http://localhost:8000/summary")
summary = response.json()

# Generate forecast
forecast_request = {
    "model": "prophet",
    "periods": 30,
    "use_tuning_data": True,
    "confidence_level": 0.95
}
response = requests.post(
    "http://localhost:8000/forecast",
    json=forecast_request
)
forecast = response.json()
```

---

## 📊 Usage Examples

### Web Interface Workflow

1. **Upload Data**
   - Navigate to "Data Upload & Validation"
   - Upload CSV/Excel files or use sample data
   - View data overview and validation results

2. **Analyze Data**
   - Go to "Statistical Analysis" for trends and distributions
   - Check "Anomaly Detection" for outliers and data drift

3. **Generate Forecasts**
   - Visit "ML Forecasting" page
   - Select forecast periods and model parameters
   - View interactive forecast visualizations

4. **Compare Predictions**
   - Upload test data in "Data Upload & Validation"
   - Navigate to "Actual vs Predicted"
   - Generate predictions and view performance metrics

5. **Get Insights**
   - Check "Business Insights" for recommendations
   - Review "Partner Analytics" for distribution optimization

6. **Export Reports**
   - Go to "Export Reports"
   - Download forecasts and analysis in CSV, Excel, or PDF format

### API Integration

```python
import requests
import pandas as pd

# Initialize API client
API_BASE = "http://localhost:8000"

# 1. Check API health
health = requests.get(f"{API_BASE}/health").json()
print(f"API Status: {health['status']}")

# 2. Get available datasets
datasets = requests.get(f"{API_BASE}/datasets").json()
print(f"Available datasets: {datasets['count']}")

# 3. Fetch inventory data
inventory = requests.get(
    f"{API_BASE}/inventory/train",
    params={"limit": 100, "date_from": "2017-01-01"}
).json()
df = pd.DataFrame(inventory['data'])

# 4. Generate forecast
forecast_params = {
    "model": "prophet",
    "periods": 90,
    "use_tuning_data": True,
    "confidence_level": 0.95
}
forecast = requests.post(
    f"{API_BASE}/forecast",
    json=forecast_params
).json()

if forecast['success']:
    forecast_df = pd.DataFrame(forecast['forecast_data'])
    print(f"Generated {len(forecast_df)} forecast points")
    print(f"MAPE: {forecast['metrics']['mape']:.2f}%")
```

---

## 🔧 Configuration

### Streamlit Configuration (`.streamlit/config.toml`)

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### Environment Variables

The application uses the following environment variables (optional):

- `DATABASE_URL` - PostgreSQL connection string (if using database)
- `PGPORT` - PostgreSQL port
- `PGUSER` - PostgreSQL username
- `PGPASSWORD` - PostgreSQL password
- `PGDATABASE` - PostgreSQL database name
- `PGHOST` - PostgreSQL host

---

## 📈 Data Format

### Inventory Dataset
```csv
Date,Inventory_Level
2017-01-01,29154
2017-01-02,558393
```

### Inflows Dataset
```csv
Date,Vendor,Quantity,Product_Type,Category,Brand,Grade,Size,Warehouse,Vendor_Address,GIK_Per_Unit,Total_GIK
2017-01-01,Vendor A,1000,Type1,Electronics,BrandX,A,M,Warehouse1,Address,10.5,10500
```

### Outflows Dataset
```csv
Date,Quantity,Product_Type,Category,Brand,Grade,Size,Warehouse,GIK_Per_Unit,Total_GIK,Partner,EI_Per_Unit,Total_EI,Partner_Address,Shipment_Type,Program
2017-01-01,500,Type1,Electronics,BrandX,A,M,Warehouse1,10.5,5250,Partner1,8.0,4000,Address,Standard,Program1
```

---

## 🧪 Testing

### Test API Endpoints

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test data retrieval
curl "http://localhost:8000/inventory/train?limit=5"

# Test forecast generation
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"model": "prophet", "periods": 30}'
```

### Verify Streamlit Application

1. Start Streamlit: `streamlit run app.py --server.port 5000`
2. Open browser to `http://localhost:5000`
3. Load sample data
4. Navigate through all pages to verify functionality

---

## 🎯 Key Features Explained

### 1. Warehouse & Category Dimensions

The system extracts daily aggregated features:
- **Warehouse Features**: Total inflows/outflows, active warehouses, warehouse count
- **Category Features**: Total inflows/outflows, active categories, category diversity

These features are used as regressors in Prophet models to improve forecast accuracy.

### 2. Multi-Period Handling

The platform handles temporal gaps between data periods:
- **Training Period**: 2017-2018 (historical baseline)
- **Tuning Period**: 2023 (recent data for model adjustment)
- **Test Period**: Future data for validation

Forecasts combine training and tuning data to ensure predictions reach test period dates.

### 3. Anomaly Detection

Multiple detection methods:
- **IQR Method**: Statistical outlier detection
- **Z-Score**: Extreme value identification
- **Isolation Forest**: Pattern-based anomaly detection
- **Data Drift**: Distribution change analysis between periods

### 4. Automatic Data Sync

Data uploaded through Streamlit is automatically saved to JSON files, making it immediately available through the API without manual export/import.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project was developed for the 2025 SuiteWorld Hackathon 4Good Challenge.

---

## 🙏 Acknowledgments

- **Prophet** - Facebook's open-source forecasting library
- **Streamlit** - For the amazing web framework
- **FastAPI** - For the high-performance API framework
- **Plotly** - For interactive visualizations

---

## 📞 Support

For questions, issues, or feedback:
- Open an issue in the repository
- Contact the development team

---

## 🗺️ Roadmap

Future enhancements:
- [ ] Additional ML models (SARIMA, XGBoost, LSTM)
- [ ] PostgreSQL database integration
- [ ] User authentication and multi-tenancy
- [ ] Real-time data ingestion
- [ ] Mobile-responsive design
- [ ] Automated retraining pipeline
- [ ] Advanced alert system with notifications

---

**Built with ❤️ for the 2025 SuiteWorld Hackathon 4Good Challenge**
