# Stock Dashboard - Candlestick Pattern Analysis

A Flask-based web application for analyzing candlestick patterns and stock data using natural language queries.

## Quick Start

To run the application with one command:

```bash
./run_app.sh
```

This script will:
1. Create a Python virtual environment (if it doesn't exist)
2. Install all required dependencies
3. Start the Flask server on port 5001

## Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python run_app.py --port 5001
```

## Direct Python Execution

You can also run the application directly:

```bash
# From project root
python run_app.py --port 5001

# Or from src directory (not recommended due to import issues)
cd src && PYTHONPATH=.. python app.py --port 5001
```

## Usage

Once running, open your browser to: http://localhost:5001

The application provides:
- Stock price data retrieval
- Candlestick pattern detection
- Natural language query processing
- RESTful API endpoints

## API Endpoints

- `GET /` - Main dashboard
- `GET /api/stock/<ticker>` - Get stock data
- `POST /api/stocks/batch` - Get multiple stocks data
- `POST /api/query` - Natural language queries
- `GET /api/stock/<ticker>/info` - Stock company information
- `GET /api/stock/<ticker>/latest` - Latest price data
- `GET /api/stock/<ticker>/price` - Current price
- `GET /api/cache/stats` - Cache statistics
- `POST /api/cache/clear` - Clear cache

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Development

The main Flask application is in `src/app.py`.
Command-line interface is available in `app.py` (root level).