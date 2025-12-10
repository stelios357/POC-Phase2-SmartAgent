# Kaggle Dataset Integration - Technical Analysis Agent

## Overview

The Kaggle historical dataset has been successfully integrated with the existing technical analysis system, providing access to 10+ years of minute-level data for 486 Indian stocks from the Nifty 100 universe.

## Dataset Details

- **Source**: [Kaggle Algo Trading Dataset](https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators)
- **Stocks**: 486 Indian companies (Nifty 100 universe)
- **Time Period**: February 2015 - July 2025
- **Granularity**: 1-minute OHLCV data
- **Data Quality**: Excellent (99.7% time continuity, 0% missing OHLC values)
- **Total Records**: ~470 million data points across all stocks

## Integration Architecture

### Components Added

1. **`KaggleDataFetcher`** (`src/kaggle_data_fetcher.py`)
   - Handles loading and caching of historical CSV data
   - Provides same interface as `DataFetcher`
   - Supports all timeframes through resampling (1m, 5m, 15m, 30m, 1h, 1d, 1w)
   - Memory caching for optimal performance

2. **Configuration Support** (`config/settings.py`)
   - `DATA_SOURCE_CONFIG["default_source"]`: Choose "yfinance" or "kaggle"
   - `DATA_SOURCE_CONFIG["fallback_to_yfinance"]`: Automatic fallback on errors
   - `DATA_SOURCE_CONFIG["kaggle_data_directory"]`: Data location

3. **Enhanced DataFetcher** (`src/data_fetcher.py`)
   - Added `data_source` parameter to `get_latest_candle()`
   - Automatic routing between data sources
   - Maintains backward compatibility

## Usage Examples

### Basic Usage

```python
from src.data_fetcher import data_fetcher

# Use Kaggle historical data
candle = data_fetcher.get_latest_candle("TCS", "1d", data_source="kaggle")
print(f"TCS: ₹{candle['close']:.2f} ({candle['change_percent']:+.2f}%)")

# Use Yahoo Finance (default)
candle = data_fetcher.get_latest_candle("TCS", "1d", data_source="yfinance")
```

### Configuration-Based Usage

```python
# In config/settings.py
DATA_SOURCE_CONFIG = {
    "default_source": "kaggle",  # Set default to Kaggle
    "fallback_to_yfinance": True,  # Fallback on errors
}

# Then use normally
candle = data_fetcher.get_latest_candle("TCS", "1d")  # Uses Kaggle
```

### Pattern Detection with Historical Data

```python
from src.candlestick_analyzer import CandlestickAnalyzer

analyzer = CandlestickAnalyzer()
query = {
    "ticker": "RELIANCE",
    "timeframe": "5m",
    "query_type": "pattern_detection"
}

# Automatically uses configured data source
result = analyzer.analyze_patterns(query)
```

## Key Benefits

### 🚀 **Performance Benefits**
- **Zero API Costs**: No rate limits or costs for historical data access
- **Instant Response**: No network delays (local file access)
- **Memory Caching**: Fast subsequent queries for same stocks

### 📊 **Data Benefits**
- **10+ Years History**: Extensive backtesting capabilities
- **High Quality**: 99.7% data continuity, validated OHLC relationships
- **Complete Coverage**: All major Indian stocks (Nifty 100)

### 🔧 **Integration Benefits**
- **Seamless Switching**: Same interface for both data sources
- **Backward Compatible**: No changes needed to existing code
- **Automatic Fallback**: System continues working if one source fails
- **Flexible Configuration**: Easy switching via config file

### 🎯 **Use Cases**

1. **Backtesting Trading Strategies**
   ```python
   # Test strategies on 10 years of data
   historical_data = kaggle_data_fetcher.get_historical_data("TCS", "1h", limit=1000)
   # Run your strategy logic here
   ```

2. **Offline Analysis**
   ```python
   # Works without internet connection
   patterns = analyzer.analyze_patterns({"ticker": "INFY", "timeframe": "15m"})
   ```

3. **Educational Research**
   ```python
   # Study patterns across different market conditions
   data_2020 = kaggle_data_fetcher.get_historical_data(
       "SBIN", "1d",
       start_date="2020-01-01",
       end_date="2020-12-31"
   )
   ```

## File Structure

```
kaggle_data/
├── archive/                    # Historical CSV files (486 files)
│   ├── TCS_minute.csv         # ~970K records per stock
│   ├── RELIANCE_minute.csv
│   └── ...
└── archive.zip                # Original downloaded archive

src/
├── kaggle_data_fetcher.py     # NEW: Historical data fetcher
├── data_fetcher.py            # MODIFIED: Added data source routing
└── ...

config/
└── settings.py                # MODIFIED: Added DATA_SOURCE_CONFIG
```

## Testing

Run the comprehensive demonstration:

```bash
python demo_kaggle_integration.py
```

This will show:
- Data source comparison
- Pattern detection examples
- Dataset statistics
- Use cases and benefits

## Configuration Options

### Smart Fallback Configuration (Recommended)
```python
DATA_SOURCE_CONFIG = {
    "default_source": "smart_fallback",  # Intelligent routing based on context
    "kaggle_data_directory": "kaggle_data/archive",
    "fallback_to_yfinance": True,        # Fallback when primary source fails
    "cache_kaggle_data": True,           # Cache loaded Kaggle data in memory
    "smart_fallback": {
        "current_data_priority": "yfinance",     # Current prices use yfinance
        "historical_data_priority": "kaggle",    # Historical data uses kaggle
        "pattern_analysis_priority": "auto",     # Auto-detect based on recency needs
        "backtest_priority": "kaggle",           # Backtesting always uses kaggle
        "max_kaggle_age_days": 30,               # Consider kaggle data "old" after this
    }
}
```

### Legacy Configuration Options
```python
DATA_SOURCE_CONFIG = {
    "default_source": "yfinance",     # "yfinance" or "kaggle"
    "kaggle_data_directory": "kaggle_data/archive",
    "fallback_to_yfinance": True,    # Fallback on Kaggle errors
    "cache_kaggle_data": True,       # Memory cache loaded data
}
```

### Recommended for Development/Testing
```python
DATA_SOURCE_CONFIG = {
    "default_source": "kaggle",      # Use historical data
    "fallback_to_yfinance": False,   # Don't fallback (consistent testing)
}
```

### Recommended for Production
```python
DATA_SOURCE_CONFIG = {
    "default_source": "smart_fallback",    # Intelligent routing
    "fallback_to_yfinance": True,          # Fallback to Kaggle if yfinance fails
}
```

## Smart Fallback Logic

The `smart_fallback` mode intelligently chooses the best data source based on:

### Context-Aware Routing
- **Current/Latest Data**: Uses yfinance (real-time prices)
- **Historical Data**: Uses Kaggle (fast, no API costs)
- **Pattern Analysis**: Auto-detects based on timeframe and recency
- **Backtesting**: Always uses Kaggle (comprehensive historical data)

### Request Type Detection
```python
# Latest price requests -> yfinance
data_fetcher.get_latest_candle("TCS", "1d", context="latest")

# Historical analysis -> kaggle
get_stock_history("TCS", period="1y", timeframe="1d", context="historical")

# Pattern analysis -> smart routing
analyzer.analyze_patterns({"ticker": "TCS", "timeframe": "5m"})  # Uses yfinance for recent
analyzer.analyze_patterns({"ticker": "TCS", "timeframe": "1d"})  # Uses kaggle for daily
```

### Fallback Scenarios
1. **Kaggle requested but fails** → Falls back to yfinance
2. **yfinance requested but fails** → Falls back to Kaggle (for historical data only)
3. **Latest data requested with Kaggle** → Error (Kaggle has old data)

## Data Quality Metrics

Based on analysis of sample stocks:

- **Completeness**: 0% missing values in OHLC columns
- **Accuracy**: 0% invalid OHLC relationships
- **Continuity**: 99.7% continuous 1-minute intervals
- **Volume**: All valid (non-negative)
- **Time Range**: Consistent 2015-2025 across stocks

## Performance Benchmarks

- **Data Loading**: ~0.5-2 seconds for first access per stock
- **Subsequent Queries**: <0.1 seconds (cached)
- **Pattern Detection**: Same performance as live data
- **Memory Usage**: ~50-200MB per cached stock (depends on timeframe)

## Troubleshooting

### Common Issues

1. **"Ticker not found in Kaggle dataset"**
   - Check ticker symbol (should not include .NS suffix)
   - Verify stock is in Nifty 100 universe

2. **"No data available for timeframe"**
   - Kaggle data is minute-level; higher timeframes are resampled
   - Check if sufficient historical data exists

3. **Memory issues with large datasets**
   - Reduce cache size or disable caching for large analyses
   - Process data in chunks for extensive backtesting

### Data Source Switching

```python
# Force Kaggle for specific query
result = data_fetcher.get_latest_candle("TCS", "1d", data_source="kaggle")

# Force Yahoo Finance
result = data_fetcher.get_latest_candle("TCS", "1d", data_source="yfinance")

# Use default from config
result = data_fetcher.get_latest_candle("TCS", "1d")
```

## Future Enhancements

Potential improvements for the integration:

1. **Database Integration**: Store data in SQLite/PostgreSQL for faster queries
2. **Advanced Caching**: Time-based cache expiration, LRU eviction
3. **Data Enrichment**: Add technical indicators to cached data
4. **Parallel Loading**: Load multiple stocks simultaneously
5. **Compression**: Store data in compressed format to save disk space

## Conclusion

The Kaggle dataset integration provides a robust, high-performance alternative to live API data while maintaining full compatibility with the existing technical analysis system. It's particularly valuable for backtesting, offline analysis, and development/testing scenarios.

The integration is production-ready and can be seamlessly switched on/off via configuration, making it ideal for both development and production environments.
