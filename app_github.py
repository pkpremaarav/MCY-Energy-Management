"""
MCY Energy Management - GitHub Cloud-Ready Flask Application
Auto-refresh from GitHub CSV with Team Sharing
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
from pathlib import Path
import requests
from io import StringIO
import threading
import time

# Import analysis engine
import sys
sys.path.insert(0, os.path.dirname(__file__))
from backend_analysis_engine import (
    CSVAnalyzer, EnergyAnalytics, WeatherAnalyzer,
    OUTPUT_DIR, MODELS_DIR, QATAR_LAT, QATAR_LON
)

app = Flask(__name__)
CORS(app)
app.secret_key = 'mcy-energy-secret-2024'

# ============================================================================
# GITHUB CONFIGURATION
# ============================================================================

GITHUB_USERNAME = os.getenv('GITHUB_USERNAME', 'your-username')
GITHUB_REPO = os.getenv('GITHUB_REPO', 'mcy-energy-data')
GITHUB_CSV_PATH = os.getenv('GITHUB_CSV_PATH', 'data/MCY_WEST.csv')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')  # Optional: for private repos

# Build GitHub raw URL
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/{GITHUB_CSV_PATH}"

# Global state
CURRENT_ANALYSIS = None
CURRENT_DATA = None
LAST_REFRESH = None
REFRESH_INTERVAL = 900  # 15 minutes in seconds
AUTO_REFRESH_ENABLED = True
DATA_REFRESH_THREAD = None
LAST_GITHUB_CHECK = None
GITHUB_DATA_UPDATED = False

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# GITHUB DATA FETCHING & CACHING
# ============================================================================

def fetch_csv_from_github():
    """Fetch CSV directly from GitHub raw content URL."""
    global CURRENT_DATA, LAST_GITHUB_CHECK, GITHUB_DATA_UPDATED
    
    try:
        headers = {}
        if GITHUB_TOKEN:
            headers['Authorization'] = f'token {GITHUB_TOKEN}'
        
        print(f"[{datetime.now()}] Fetching CSV from GitHub: {GITHUB_RAW_URL}")
        
        response = requests.get(GITHUB_RAW_URL, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Convert response to DataFrame
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        print(f"[{datetime.now()}] ✅ Successfully fetched {len(df)} rows from GitHub")
        
        CURRENT_DATA = df
        LAST_GITHUB_CHECK = datetime.now()
        GITHUB_DATA_UPDATED = True
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now()}] ❌ GitHub fetch error: {str(e)}")
        if CURRENT_DATA is not None:
            print(f"[{datetime.now()}] Using cached data ({len(CURRENT_DATA)} rows)")
            return CURRENT_DATA
        else:
            raise Exception(f"Failed to fetch from GitHub and no cached data available: {str(e)}")


def save_csv_temporarily(df):
    """Save DataFrame to temporary CSV file for processing."""
    temp_file = OUTPUT_DIR / 'temp_github_data.csv'
    df.to_csv(temp_file, index=False)
    return str(temp_file)


def run_analysis(csv_file):
    """Run complete analysis on CSV data."""
    global CURRENT_ANALYSIS
    
    try:
        print(f"[{datetime.now()}] Starting analysis...")
        
        analyzer = CSVAnalyzer(csv_file)
        df = analyzer.load_csv()
        timestamp_col = analyzer.detect_timestamp_column()
        
        buildings_data = analyzer.extract_building_data(df, timestamp_col)
        analytics = EnergyAnalytics(buildings_data)
        weather = WeatherAnalyzer()
        
        # Run all analyses
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'building_count': len(buildings_data),
            'data_rows': len(df),
            'buildings': {}
        }
        
        for building, data in buildings_data.items():
            if len(data) >= 50:  # Minimum data requirement
                valve_modes = analytics.detect_valve_operation_mode(data, building)
                leakage = analytics.detect_leakage_indicators(data, building)
                anomalies = analytics.detect_anomalies(data, building)
                
                analysis_results['buildings'][building] = {
                    'valve_modes': valve_modes,
                    'leakage_indicators': leakage,
                    'anomalies': anomalies,
                    'data_points': len(data)
                }
        
        # ML Training (on thread)
        try:
            ml_results = analytics.train_predictive_models(buildings_data)
            analysis_results['ml_models'] = ml_results
        except:
            print(f"[{datetime.now()}] ML training skipped (insufficient data)")
        
        # Weather data
        try:
            weather_data = weather.get_current_weather()
            analysis_results['weather'] = weather_data
        except:
            print(f"[{datetime.now()}] Weather fetch skipped")
        
        CURRENT_ANALYSIS = analysis_results
        print(f"[{datetime.now()}] ✅ Analysis complete for {len(buildings_data)} buildings")
        
        return analysis_results
        
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Analysis error: {str(e)}")
        raise


def auto_refresh_worker():
    """Background worker for auto-refreshing data from GitHub."""
    global LAST_REFRESH
    
    while AUTO_REFRESH_ENABLED:
        try:
            if LAST_REFRESH is None or (datetime.now() - LAST_REFRESH).seconds >= REFRESH_INTERVAL:
                print(f"\n{'='*60}")
                print(f"[{datetime.now()}] 🔄 AUTO-REFRESH CYCLE STARTED")
                print(f"{'='*60}")
                
                # Fetch from GitHub
                df = fetch_csv_from_github()
                
                # Save temporarily
                temp_file = save_csv_temporarily(df)
                
                # Run analysis
                run_analysis(temp_file)
                
                LAST_REFRESH = datetime.now()
                
                print(f"[{datetime.now()}] ✅ AUTO-REFRESH COMPLETE")
                print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"[{datetime.now()}] ⚠️ Auto-refresh error: {str(e)}")
        
        time.sleep(30)  # Check every 30 seconds


def start_auto_refresh():
    """Start background auto-refresh thread."""
    global DATA_REFRESH_THREAD
    
    if DATA_REFRESH_THREAD is None or not DATA_REFRESH_THREAD.is_alive():
        DATA_REFRESH_THREAD = threading.Thread(target=auto_refresh_worker, daemon=True)
        DATA_REFRESH_THREAD.start()
        print(f"[{datetime.now()}] ✅ Auto-refresh worker started")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_number(value, decimals=2):
    """Format number for display."""
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def get_analysis_summary():
    """Get cached analysis or run new analysis."""
    global CURRENT_ANALYSIS
    
    if CURRENT_ANALYSIS is None:
        try:
            df = fetch_csv_from_github()
            temp_file = save_csv_temporarily(df)
            CURRENT_ANALYSIS = run_analysis(temp_file)
        except Exception as e:
            return {'error': str(e), 'message': 'Failed to fetch and analyze data from GitHub'}
    
    return CURRENT_ANALYSIS


# ============================================================================
# ROUTES - PAGES
# ============================================================================

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html', 
                          github_repo=GITHUB_REPO,
                          github_username=GITHUB_USERNAME)


@app.route('/analysis')
def analysis():
    """Analysis page with filters."""
    return render_template('analysis.html',
                          github_repo=GITHUB_REPO)


@app.route('/data')
def data_viewer():
    """Raw data viewer page."""
    return render_template('data.html',
                          github_repo=GITHUB_REPO)


@app.route('/reports')
def reports():
    """Reports generation page."""
    return render_template('reports.html',
                          github_repo=GITHUB_REPO)


@app.route('/settings')
def settings():
    """Settings page - now shows GitHub configuration."""
    return render_template('settings.html',
                          github_url=GITHUB_RAW_URL,
                          github_username=GITHUB_USERNAME,
                          github_repo=GITHUB_REPO,
                          refresh_interval=REFRESH_INTERVAL)


@app.route('/api/status')
def api_status():
    """Get system status including GitHub connection."""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'github': {
            'username': GITHUB_USERNAME,
            'repo': GITHUB_REPO,
            'csv_path': GITHUB_CSV_PATH,
            'url': GITHUB_RAW_URL,
            'last_check': LAST_GITHUB_CHECK.isoformat() if LAST_GITHUB_CHECK else None,
            'data_updated': GITHUB_DATA_UPDATED
        },
        'analysis': {
            'last_refresh': LAST_REFRESH.isoformat() if LAST_REFRESH else None,
            'auto_refresh_enabled': AUTO_REFRESH_ENABLED,
            'refresh_interval_seconds': REFRESH_INTERVAL,
            'cached_buildings': len(CURRENT_ANALYSIS['buildings']) if CURRENT_ANALYSIS else 0
        },
        'data': {
            'rows': len(CURRENT_DATA) if CURRENT_DATA is not None else 0,
            'cached': CURRENT_DATA is not None
        }
    })


# ============================================================================
# API ENDPOINTS - ANALYSIS
# ============================================================================

@app.route('/api/analysis/summary', methods=['GET', 'POST'])
def api_analysis_summary():
    """Get analysis summary for all buildings."""
    try:
        analysis = get_analysis_summary()
        
        if 'error' in analysis:
            return jsonify(analysis), 500
        
        # Format response
        buildings_summary = []
        if 'buildings' in analysis:
            for building, data in analysis['buildings'].items():
                buildings_summary.append({
                    'building': building,
                    'valve_modes': data.get('valve_modes', {}),
                    'leakage_indicators': data.get('leakage_indicators', {}),
                    'anomalies_count': len(data.get('anomalies', [])),
                    'data_points': data.get('data_points', 0)
                })
        
        return jsonify({
            'status': 'success',
            'timestamp': analysis.get('timestamp'),
            'building_count': analysis.get('building_count'),
            'buildings': buildings_summary,
            'data_source': 'GitHub',
            'last_refresh': LAST_REFRESH.isoformat() if LAST_REFRESH else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/compare', methods=['POST'])
def api_analysis_compare():
    """Compare multiple buildings."""
    try:
        payload = request.get_json()
        buildings_to_compare = payload.get('buildings', [])
        
        analysis = get_analysis_summary()
        
        if 'error' in analysis:
            return jsonify(analysis), 500
        
        comparison = {}
        for building in buildings_to_compare:
            if building in analysis.get('buildings', {}):
                comparison[building] = analysis['buildings'][building]
        
        return jsonify({
            'status': 'success',
            'comparison': comparison,
            'data_source': 'GitHub'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/trends', methods=['POST'])
def api_analysis_trends():
    """Get time-series trends."""
    try:
        payload = request.get_json()
        building = payload.get('building', '')
        period = payload.get('period', 'daily')  # daily, weekly, monthly
        
        if CURRENT_DATA is None:
            df = fetch_csv_from_github()
        else:
            df = CURRENT_DATA
        
        # Filter for building columns
        building_cols = [col for col in df.columns if col.startswith(building)]
        
        if not building_cols:
            return jsonify({'error': f'No data for building {building}'}), 404
        
        # Aggregate by period
        if 'timestamp' in df.columns or 'Date & Time' in df.columns:
            time_col = 'timestamp' if 'timestamp' in df.columns else 'Date & Time'
            df[time_col] = pd.to_datetime(df[time_col])
            
            if period == 'daily':
                trends = df.groupby(df[time_col].dt.date)[building_cols].mean()
            elif period == 'weekly':
                trends = df.groupby(df[time_col].dt.isocalendar().week)[building_cols].mean()
            else:  # monthly
                trends = df.groupby(df[time_col].dt.to_period('M'))[building_cols].mean()
            
            return jsonify({
                'status': 'success',
                'building': building,
                'period': period,
                'trends': trends.to_dict(orient='index'),
                'data_source': 'GitHub'
            })
        
        return jsonify({'error': 'No timestamp column found'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/refresh', methods=['POST'])
def api_manual_refresh():
    """Manually trigger refresh from GitHub."""
    try:
        print(f"\n[{datetime.now()}] 🔄 MANUAL REFRESH TRIGGERED")
        
        df = fetch_csv_from_github()
        temp_file = save_csv_temporarily(df)
        analysis = run_analysis(temp_file)
        
        return jsonify({
            'status': 'success',
            'message': 'Data refreshed from GitHub',
            'buildings_analyzed': len(analysis.get('buildings', {})),
            'timestamp': analysis.get('timestamp'),
            'data_source': 'GitHub'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - DATA
# ============================================================================

@app.route('/api/data/raw', methods=['POST'])
def api_data_raw():
    """Get paginated raw data."""
    try:
        payload = request.get_json()
        page = payload.get('page', 1)
        per_page = payload.get('per_page', 50)
        building = payload.get('building', None)
        
        if CURRENT_DATA is None:
            df = fetch_csv_from_github()
        else:
            df = CURRENT_DATA
        
        # Filter by building if specified
        if building:
            cols = [col for col in df.columns if col.startswith(building) or col in ['Date & Time', 'timestamp', 'Timestamp']]
            df = df[cols]
        
        total_rows = len(df)
        start = (page - 1) * per_page
        end = start + per_page
        
        paginated_data = df.iloc[start:end]
        
        return jsonify({
            'status': 'success',
            'page': page,
            'per_page': per_page,
            'total_rows': total_rows,
            'total_pages': (total_rows + per_page - 1) // per_page,
            'columns': list(df.columns),
            'data': paginated_data.values.tolist(),
            'data_source': 'GitHub'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/buildings', methods=['GET'])
def api_data_buildings():
    """Get list of available buildings from current data."""
    try:
        if CURRENT_DATA is None:
            df = fetch_csv_from_github()
        else:
            df = CURRENT_DATA
        
        # Extract building names
        buildings = set()
        for col in df.columns:
            if any(col.startswith(zone) for zone in ['J', 'A', 'B', 'C', 'D']):
                building = col.split('B')[0]  # Extract zone + unit
                if len(building) > 0:
                    buildings.add(building)
        
        return jsonify({
            'status': 'success',
            'buildings': sorted(list(buildings)),
            'count': len(buildings),
            'data_source': 'GitHub'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - REPORTS
# ============================================================================

@app.route('/api/reports/executive', methods=['POST'])
def api_reports_executive():
    """Generate executive summary report."""
    try:
        payload = request.get_json()
        building = payload.get('building', 'All')
        
        analysis = get_analysis_summary()
        
        if 'error' in analysis:
            return jsonify(analysis), 500
        
        report = {
            'title': 'Executive Summary Report',
            'generated': datetime.now().isoformat(),
            'building': building,
            'total_buildings': analysis.get('building_count'),
            'data_rows_analyzed': analysis.get('data_rows'),
            'key_findings': [],
            'recommendations': [],
            'data_source': 'GitHub'
        }
        
        # Add findings from analysis
        critical_buildings = []
        for bldg, data in analysis.get('buildings', {}).items():
            if data.get('leakage_indicators', {}).get('severity') == 'CRITICAL':
                critical_buildings.append(bldg)
        
        if critical_buildings:
            report['key_findings'].append({
                'type': 'Leakage Risk',
                'severity': 'CRITICAL',
                'buildings': critical_buildings,
                'action': 'Immediate investigation required'
            })
        
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/technical', methods=['POST'])
def api_reports_technical():
    """Generate technical deep-dive report."""
    try:
        payload = request.get_json()
        building = payload.get('building', 'All')
        
        analysis = get_analysis_summary()
        
        if 'error' in analysis:
            return jsonify(analysis), 500
        
        report = {
            'title': 'Technical Deep-Dive Report',
            'generated': datetime.now().isoformat(),
            'building': building,
            'analysis_details': analysis,
            'data_source': 'GitHub'
        }
        
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# INITIALIZATION
# ============================================================================

@app.before_request
def before_request():
    """Initialize on first request."""
    pass


if __name__ == '__main__':
    # Start auto-refresh worker
    start_auto_refresh()
    
    # Initial fetch from GitHub
    try:
        print(f"[{datetime.now()}] 🚀 Starting MCY Energy Management with GitHub Integration")
        print(f"[{datetime.now()}] GitHub: {GITHUB_USERNAME}/{GITHUB_REPO}")
        print(f"[{datetime.now()}] CSV: {GITHUB_CSV_PATH}")
        df = fetch_csv_from_github()
        temp_file = save_csv_temporarily(df)
        run_analysis(temp_file)
        LAST_REFRESH = datetime.now()
        print(f"[{datetime.now()}] ✅ Initial load complete")
    except Exception as e:
        print(f"[{datetime.now()}] ⚠️ Initial load error: {str(e)}")
    
    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
