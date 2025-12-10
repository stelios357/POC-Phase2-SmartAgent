**Problem Statement:**  
Build an agent that can process natural language queries about stock data (OHLCV) and single candlestick patterns for a given stock and timeframe.

The agent should:

1. Understand the following in a query:  
   * Stock symbol (e.g., "TCS", "RELIANCE")  
   * Timeframe (e.g., "5-minute", "daily", "weekly")  
   * Type of query (e.g., current price, OHLCV, or specific candle pattern)  
2. For OHLCV queries, return:  
   * The current price (or the latest close) and the OHLCV data for the specified timeframe.  
3. For candle pattern queries, detect the following **single candle patterns**:  
   * Doji (and its types: common, dragonfly, gravestone, long-legged)  
   * Hammer  
   * Shooting Star  
   * Marubozu (Bullish and Bearish)  
4. The agent should be able to handle at least the following timeframes:  
   * Intraday: 1m, 5m, 15m, 30m, 1h  
   * Daily: 1d  
   * Weekly: 1w  
5. The response should be in a structured format (e.g., JSON) and also a natural language response.

Example Queries:

* "What is the current price of TCS on the 5-minute chart?"  
* "Show me the OHLCV for RELIANCE today."  
* "Is there a Doji on the daily chart of INFY?"  
* "Check for Hammer patterns on the 1-hour chart of HDFCBANK."

Data Source:

* Use Yahoo Finance API (with yfinance library) for fetching OHLCV data.  
* Or use datasets from Kaggle such as [https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators](https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators)

Technical Requirements:

* Use Python as the primary language.  
* The system should have a modular design so that we can extend it later for more patterns and timeframes.

Deliverables:

1. A working script that can take a natural language query and return the result.  
2. The script should be able to fetch real-time data for the given stock and timeframe.  
3. The script should detect the specified candle patterns for the latest candle in the given timeframe.  
4. The response should include:  
   * For OHLCV: the open, high, low, close, volume, and the change from the previous close.  
   * For patterns: the pattern name

Bonus:

* The agent also has a (at least a minimal) web frontend.  
* The system can handle multiple stocks in one query (e.g., "Compare TCS and INFY on daily chart").  
* The system can handle multiple timeframes in one query (e.g., "Show me weekly and monthly of RELIANCE").

Tenets:

* Understand what you create  
* Prioritise Buy over Build  
* Think User First  
* DON’T OPTIMIZE FOR A LIGHTWEIGHT IMPLEMENTATION