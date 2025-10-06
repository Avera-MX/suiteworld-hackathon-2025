import pandas as pd
import numpy as np
import io
import zipfile
from typing import Dict, Any, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

class ReportGenerator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_data_csv(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """
        Export processed data as CSV
        """
        try:
            output = io.StringIO()
            
            # Combine all datasets with source indicator
            combined_data = []
            
            for dataset_name, df in datasets.items():
                if df is not None and not df.empty:
                    df_copy = df.copy()
                    df_copy['Dataset_Source'] = dataset_name
                    combined_data.append(df_copy)
            
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
                combined_df.to_csv(output, index=False)
                return output.getvalue()
            else:
                return "No data available for export"
                
        except Exception as e:
            return f"Error exporting CSV: {str(e)}"
    
    def export_data_excel(self, datasets: Dict[str, pd.DataFrame]) -> bytes:
        """
        Export processed data as Excel with multiple sheets
        """
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for dataset_name, df in datasets.items():
                    if df is not None and not df.empty:
                        # Clean sheet name (Excel sheet names have restrictions)
                        sheet_name = dataset_name.replace('_', ' ').title()[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Add summary sheet
                self._create_summary_sheet(writer, datasets)
            
            return output.getvalue()
            
        except Exception as e:
            return f"Error exporting Excel: {str(e)}".encode()
    
    def export_forecasts_csv(self, forecast_results: Dict[str, Any]) -> str:
        """
        Export forecast results as CSV
        """
        try:
            output = io.StringIO()
            
            # Combine forecasts from all models
            combined_forecasts = []
            
            if 'forecasts' in forecast_results:
                for method, result in forecast_results['forecasts'].items():
                    if result and 'forecast' in result:
                        forecast_df = result['forecast'].copy()
                        forecast_df['Model'] = method
                        
                        # Add metrics as columns
                        if 'metrics' in result:
                            for metric_name, metric_value in result['metrics'].items():
                                forecast_df[f'{metric_name.upper()}'] = metric_value
                        
                        combined_forecasts.append(forecast_df)
            
            if combined_forecasts:
                combined_df = pd.concat(combined_forecasts, ignore_index=True, sort=False)
                combined_df.to_csv(output, index=False)
                return output.getvalue()
            else:
                return "No forecast data available for export"
                
        except Exception as e:
            return f"Error exporting forecasts CSV: {str(e)}"
    
    def generate_pdf_report(self, datasets: Dict[str, pd.DataFrame], 
                           forecast_results: Optional[Dict[str, Any]] = None,
                           business_insights: Optional[Dict[str, Any]] = None) -> bytes:
        """
        Generate comprehensive PDF report
        """
        if not FPDF_AVAILABLE:
            return b"PDF generation not available - fpdf library not installed"
        
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            
            # Title page
            pdf.cell(0, 10, 'AI-Powered Inventory Forecasting Report', 0, 1, 'C')
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
            pdf.ln(20)
            
            # Executive Summary
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Executive Summary', 0, 1)
            pdf.set_font('Arial', '', 11)
            
            # Data overview
            self._add_data_overview_to_pdf(pdf, datasets)
            
            # Key insights
            if business_insights and business_insights.get('success'):
                self._add_insights_to_pdf(pdf, business_insights)
            
            # Forecast summary
            if forecast_results and forecast_results.get('success'):
                self._add_forecast_summary_to_pdf(pdf, forecast_results)
            
            return pdf.output(dest='S').encode('latin-1')
            
        except Exception as e:
            return f"Error generating PDF: {str(e)}".encode()
    
    def generate_comprehensive_export(self, datasets: Dict[str, pd.DataFrame],
                                     forecast_results: Optional[Dict[str, Any]] = None,
                                     anomaly_results: Optional[Dict[str, Any]] = None,
                                     business_insights: Optional[Dict[str, Any]] = None,
                                     export_options: list = None) -> bytes:
        """
        Generate comprehensive export package as ZIP file
        """
        try:
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                
                # Raw data export
                if "Raw Data" in (export_options or []):
                    csv_data = self.export_data_csv(datasets)
                    zip_file.writestr(f'raw_data_{self.timestamp}.csv', csv_data)
                    
                    excel_data = self.export_data_excel(datasets)
                    zip_file.writestr(f'raw_data_{self.timestamp}.xlsx', excel_data)
                
                # Statistical analysis
                if "Statistical Analysis" in (export_options or []):
                    stats_report = self._generate_statistical_analysis_report(datasets)
                    zip_file.writestr(f'statistical_analysis_{self.timestamp}.json', stats_report)
                
                # Forecasts
                if "Forecasts" in (export_options or []) and forecast_results:
                    forecast_csv = self.export_forecasts_csv(forecast_results)
                    zip_file.writestr(f'forecasts_{self.timestamp}.csv', forecast_csv)
                    
                    forecast_json = json.dumps(forecast_results, default=str, indent=2)
                    zip_file.writestr(f'forecast_results_{self.timestamp}.json', forecast_json)
                
                # Anomaly detection
                if "Anomaly Detection" in (export_options or []) and anomaly_results:
                    anomaly_json = json.dumps(anomaly_results, default=str, indent=2)
                    zip_file.writestr(f'anomaly_analysis_{self.timestamp}.json', anomaly_json)
                
                # Business insights
                if "Business Insights" in (export_options or []) and business_insights:
                    insights_json = json.dumps(business_insights, default=str, indent=2)
                    zip_file.writestr(f'business_insights_{self.timestamp}.json', insights_json)
                    
                    # Generate insights summary report
                    insights_summary = self._generate_insights_summary_report(business_insights)
                    zip_file.writestr(f'insights_summary_{self.timestamp}.txt', insights_summary)
                
                # Comprehensive PDF report
                pdf_report = self.generate_pdf_report(datasets, forecast_results, business_insights)
                zip_file.writestr(f'comprehensive_report_{self.timestamp}.pdf', pdf_report)
                
                # Add metadata
                metadata = self._generate_export_metadata(datasets, export_options)
                zip_file.writestr('export_metadata.json', json.dumps(metadata, default=str, indent=2))
            
            return zip_buffer.getvalue()
            
        except Exception as e:
            return f"Error generating comprehensive export: {str(e)}".encode()
    
    def _create_summary_sheet(self, writer, datasets: Dict[str, pd.DataFrame]):
        """
        Create summary sheet for Excel export
        """
        try:
            summary_data = []
            
            for dataset_name, df in datasets.items():
                if df is not None:
                    date_range = "N/A"
                    if 'Date' in df.columns:
                        min_date = df['Date'].min()
                        max_date = df['Date'].max()
                        date_range = f"{min_date} to {max_date}"
                    
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Rows': len(df),
                        'Columns': len(df.columns),
                        'Date Range': date_range,
                        'Missing Values': df.isnull().sum().sum()
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
        except Exception as e:
            print(f"Error creating summary sheet: {str(e)}")
    
    def _add_data_overview_to_pdf(self, pdf, datasets: Dict[str, pd.DataFrame]):
        """
        Add data overview section to PDF
        """
        try:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Data Overview', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            for dataset_name, df in datasets.items():
                if df is not None:
                    pdf.cell(0, 6, f'{dataset_name.replace("_", " ").title()}: {len(df)} records', 0, 1)
            
            pdf.ln(5)
            
        except Exception as e:
            print(f"Error adding data overview to PDF: {str(e)}")
    
    def _add_insights_to_pdf(self, pdf, business_insights: Dict[str, Any]):
        """
        Add business insights section to PDF
        """
        try:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Key Business Insights', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            # Add recommendations
            if 'recommendations' in business_insights:
                recommendations = business_insights['recommendations'][:3]  # Top 3 recommendations
                
                for i, rec in enumerate(recommendations, 1):
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 6, f"{i}. {rec.get('title', 'N/A')}", 0, 1)
                    pdf.set_font('Arial', '', 10)
                    
                    # Split long descriptions
                    description = rec.get('description', '')[:200] + "..." if len(rec.get('description', '')) > 200 else rec.get('description', '')
                    pdf.multi_cell(0, 5, description)
                    pdf.ln(2)
            
            pdf.ln(5)
            
        except Exception as e:
            print(f"Error adding insights to PDF: {str(e)}")
    
    def _add_forecast_summary_to_pdf(self, pdf, forecast_results: Dict[str, Any]):
        """
        Add forecast summary section to PDF
        """
        try:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Forecast Performance Summary', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            if 'forecasts' in forecast_results:
                for method, result in forecast_results['forecasts'].items():
                    if result and 'metrics' in result:
                        metrics = result['metrics']
                        pdf.cell(0, 6, f'{method} Model:', 0, 1)
                        pdf.cell(20, 6, '', 0, 0)  # Indent
                        pdf.cell(0, 6, f'RMSE: {metrics.get("rmse", 0):.2f}, MAE: {metrics.get("mae", 0):.2f}, MAPE: {metrics.get("mape", 0):.2f}%', 0, 1)
            
            pdf.ln(5)
            
        except Exception as e:
            print(f"Error adding forecast summary to PDF: {str(e)}")
    
    def _generate_statistical_analysis_report(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """
        Generate statistical analysis report as JSON
        """
        try:
            analysis = {}
            
            for dataset_name, df in datasets.items():
                if df is not None:
                    dataset_analysis = {}
                    
                    # Basic statistics
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        dataset_analysis['descriptive_stats'] = df[numeric_columns].describe().to_dict()
                    
                    # Missing values analysis
                    dataset_analysis['missing_values'] = df.isnull().sum().to_dict()
                    
                    # Date range analysis
                    if 'Date' in df.columns:
                        dataset_analysis['date_range'] = {
                            'start': str(df['Date'].min()),
                            'end': str(df['Date'].max()),
                            'total_days': str((df['Date'].max() - df['Date'].min()).days)
                        }
                    
                    # Categorical analysis
                    categorical_columns = df.select_dtypes(include=['object']).columns
                    categorical_analysis = {}
                    for col in categorical_columns:
                        if col != 'Date':
                            value_counts = df[col].value_counts().head(10)  # Top 10 values
                            categorical_analysis[col] = value_counts.to_dict()
                    
                    if categorical_analysis:
                        dataset_analysis['categorical_distribution'] = categorical_analysis
                    
                    analysis[dataset_name] = dataset_analysis
            
            return json.dumps(analysis, default=str, indent=2)
            
        except Exception as e:
            return json.dumps({'error': f'Statistical analysis failed: {str(e)}'})
    
    def _generate_insights_summary_report(self, business_insights: Dict[str, Any]) -> str:
        """
        Generate business insights summary report as text
        """
        try:
            report_lines = []
            report_lines.append("BUSINESS INSIGHTS SUMMARY REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Key Recommendations
            if 'recommendations' in business_insights:
                report_lines.append("KEY RECOMMENDATIONS:")
                report_lines.append("-" * 30)
                
                for i, rec in enumerate(business_insights['recommendations'], 1):
                    report_lines.append(f"{i}. {rec.get('title', 'N/A')}")
                    report_lines.append(f"   Priority: {rec.get('priority', 'N/A')}")
                    report_lines.append(f"   Impact: {rec.get('impact', 'N/A')}")
                    report_lines.append(f"   Description: {rec.get('description', 'N/A')}")
                    
                    if 'action_items' in rec:
                        report_lines.append("   Action Items:")
                        for action in rec['action_items']:
                            report_lines.append(f"   - {action}")
                    report_lines.append("")
            
            # Risk Alerts
            if 'risk_alerts' in business_insights:
                report_lines.append("RISK ALERTS:")
                report_lines.append("-" * 20)
                
                for alert in business_insights['risk_alerts']:
                    level = alert.get('level', 'unknown').upper()
                    message = alert.get('message', 'N/A')
                    report_lines.append(f"[{level}] {message}")
                
                report_lines.append("")
            
            # Inventory Optimization
            if 'inventory_optimization' in business_insights:
                optimization = business_insights['inventory_optimization']
                
                if 'reorder_points' in optimization:
                    report_lines.append("REORDER POINT RECOMMENDATIONS:")
                    report_lines.append("-" * 35)
                    
                    for item in optimization['reorder_points']:
                        report_lines.append(f"Category: {item.get('category', 'N/A')}")
                        report_lines.append(f"  Avg Daily Demand: {item.get('avg_daily_demand', 0)}")
                        report_lines.append(f"  Recommended Reorder Point: {item.get('recommended_reorder_point', 0)}")
                        report_lines.append("")
                
                if 'safety_stock' in optimization:
                    report_lines.append("SAFETY STOCK RECOMMENDATIONS:")
                    report_lines.append("-" * 35)
                    
                    for item in optimization['safety_stock']:
                        report_lines.append(f"Model: {item.get('model', 'N/A')}")
                        report_lines.append(f"  Service Level: {item.get('service_level', 'N/A')}")
                        report_lines.append(f"  Safety Stock: {item.get('safety_stock', 0)}")
                        report_lines.append("")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Error generating insights summary: {str(e)}"
    
    def _generate_export_metadata(self, datasets: Dict[str, pd.DataFrame], export_options: list) -> Dict[str, Any]:
        """
        Generate metadata for export package
        """
        try:
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'export_options': export_options or [],
                'datasets_included': list(datasets.keys()),
                'file_descriptions': {
                    'raw_data': 'Processed datasets in CSV and Excel formats',
                    'statistical_analysis': 'Descriptive statistics and data quality metrics',
                    'forecasts': 'ML model predictions and performance metrics',
                    'anomaly_analysis': 'Outlier detection and data drift analysis',
                    'business_insights': 'Strategic recommendations and risk alerts',
                    'comprehensive_report': 'Complete analysis in PDF format'
                },
                'dataset_summary': {}
            }
            
            # Add dataset summaries
            for dataset_name, df in datasets.items():
                if df is not None:
                    metadata['dataset_summary'][dataset_name] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'date_range': str(df['Date'].min()) + ' to ' + str(df['Date'].max()) if 'Date' in df.columns else 'N/A'
                    }
            
            return metadata
            
        except Exception as e:
            return {'error': f'Metadata generation failed: {str(e)}'}
